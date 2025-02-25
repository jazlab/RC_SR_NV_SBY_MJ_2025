"""
This script is used to make plot for the angle bwetween actor and observer 
outcome dimensions and the switch evidence direction.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="is_categorical_dtype is deprecated",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="use_inf_as_na option is deprecated",
)

# Configure pandas to convert inf to NaN
pd.options.mode.use_inf_as_na = True

from utils.plots import beutify
from utils.LoadSession import findrootdir
from polar_plot_common import plot_angles, make_combined_files
from compute_angle import compute_angle


def select_dimensions(coef1, coef2, type="positive"):
    if type == "positive":
        dims = np.where(np.sign(coef1) == np.sign(coef2))[0]
    else:
        dims = np.where(np.sign(coef1) != np.sign(coef2))[0]
    return dims


def compute_angles_decompose(coef_1, coef_2):
    dim_p = select_dimensions(coef_1, coef_2, "positive")
    dim_n = select_dimensions(coef_1, coef_2, "negative")
    theta_p = compute_angle(coef_1[dim_p], coef_2[dim_p])
    theta_n = compute_angle(coef_1[dim_n], coef_2[dim_n])
    return theta_p, theta_n, dim_p, dim_n


def make_polar_plots_outcome_evidence(results):
    # plot self-out and other-out vs. switch evidence in a single polar plot
    coef_self = np.vstack([results["coef_self"]])
    coef_other = np.vstack([results["coef_other"]])
    coef_switch = np.vstack([results["coef_switch"]])
    theta_self_list = []
    theta_other_list = []
    n_shuffles = coef_self.shape[1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection="polar"))
    for i_shuffle in range(n_shuffles):
        coef_self_i = coef_self[:, i_shuffle]
        coef_other_i = coef_other[:, i_shuffle]
        coef_switch_i = coef_switch[:, i_shuffle]
        theta_self_list.append(compute_angle(coef_self_i, coef_switch_i))
        theta_other_list.append(compute_angle(coef_other_i, coef_switch_i))
    plot_mean_and_sd(ax, theta_self_list, "b")
    plot_mean_and_sd(ax, theta_other_list, "r")
    ax.set_title("Outcome vs. Evidence Angles")
    # Set the angle limits
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(
        [
            0,
            np.pi / 6,
            np.pi / 3,
            np.pi / 2,
            2 * np.pi / 3,
            5 * np.pi / 6,
            np.pi,
        ]
    )
    return fig


# Function to calculate mean and SD, and plot
def plot_mean_and_sd(ax, data, color):
    mean_theta = np.mean(data)
    std_theta = np.std(data)

    # Convert to radians
    mean_theta_rad = np.deg2rad(mean_theta)
    std_theta_rad = np.deg2rad(std_theta)

    # Plot mean as a line from 0 to 1
    ax.plot([0, mean_theta_rad], [0, 1], color=color, linewidth=0.5, zorder=3)

    # Plot SEM as a shaded area
    theta = np.linspace(
        mean_theta_rad - std_theta_rad, mean_theta_rad + std_theta_rad, 100
    )
    ax.fill_between(
        theta, np.ones_like(theta), color=color, alpha=0.2, zorder=2
    )

    # Add a text annotation with mean and SEM values
    ax.text(
        mean_theta_rad,
        1.1,
        f"{mean_theta:.2f}° ± {std_theta:.2f}°",
        color=color,
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    return mean_theta, std_theta


def make_polar_plot_with_sd(results, angle="self_other"):
    if angle == "self_other":
        coef_1 = np.vstack(results["coef_self"])  # n_units x n_shuffles
        coef_2 = np.vstack(results["coef_other"])
    elif angle == "self_switch":
        coef_1 = np.vstack(results["coef_self"])
        coef_2 = np.vstack(results["coef_switch"])
    elif angle == "switch_other":
        coef_1 = np.vstack(results["coef_switch"])
        coef_2 = np.vstack(results["coef_other"])
    n_shuffles = coef_1.shape[1]
    theta_p_list = []
    theta_n_list = []
    theta_list = []
    for i_shuffle in range(n_shuffles):
        coef_1_i = coef_1[:, i_shuffle]
        coef_2_i = coef_2[:, i_shuffle]
        theta_p, theta_n, dim_p, dim_n = compute_angles_decompose(
            coef_1_i, coef_2_i
        )
        theta = compute_angle(coef_1_i, coef_2_i)
        theta_p_list.append(theta_p)
        theta_n_list.append(theta_n)
        theta_list.append(theta)
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection="polar"))

    # Define bin size and create bins
    bin_size = 1
    bins = np.arange(0, 180 + bin_size, bin_size)

    # Usage in the main function:
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection="polar"))

    mean_p, std_p = plot_mean_and_sd(ax, theta_p_list, "g")
    mean_n, std_n = plot_mean_and_sd(ax, theta_n_list, "m")
    mean_all, std_all = plot_mean_and_sd(ax, theta_list, "k")
    # report the mean and sd of theta
    print(f"Theta_positive: {mean_p} ± {std_p}")
    print(f"Theta_negative: {mean_n} ± {std_n}")
    print(f"Theta_all: {mean_all} ± {std_all}")

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, 1.2)  # Adjust y-limit to accommodate text annotations
    ax.set_yticks([])  # Remove radial ticks as they're no longer relevant
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Add radial lines for reference
    for angle in [0, 30, 60, 90, 120, 150, 180]:
        ax.plot(
            [np.deg2rad(angle), np.deg2rad(angle)],
            [0, 1],
            color="gray",
            linestyle="--",
            alpha=0.5,
        )

    plt.tight_layout()
    return fig


def make_polar_plot_with_magnitude(results, angle="self_other"):
    # results should contain coef_self, coef_other, coef_switch
    if angle == "self_other":
        coef_1 = np.vstack(results["coef_self"])  # n_units x n_shuffles
        coef_2 = np.vstack(results["coef_other"])
    elif angle == "self_switch":
        coef_1 = np.vstack(results["coef_self"])
        coef_2 = np.vstack(results["coef_switch"])
    elif angle == "switch_other":
        coef_1 = np.vstack(results["coef_switch"])
        coef_2 = np.vstack(results["coef_other"])
    n_shuffles = coef_1.shape[1]

    # save thetas for reporting
    theta_list = []
    theta_p_list = []
    theta_n_list = []

    fig = plt.figure(figsize=(4, 4), facecolor="white")
    ax = fig.add_subplot(111, polar=True)

    for i_shuffle in range(n_shuffles):
        coef_1_i = coef_1[:, i_shuffle]
        coef_2_i = coef_2[:, i_shuffle]
        theta_p, theta_n, dim_p, dim_n = compute_angles_decompose(
            coef_1_i, coef_2_i
        )
        theta = compute_angle(coef_1_i, coef_2_i)
        theta_list.append(theta)
        theta_p_list.append(theta_p)
        theta_n_list.append(theta_n)
        n_units_p = len(dim_p)
        n_units_n = len(dim_n)
        n_units = len(coef_1_i)
        mag_p = np.abs(np.dot(coef_1_i[dim_p], coef_2_i[dim_p]))  # / n_units_p
        mag_n = np.abs(np.dot(coef_1_i[dim_n], coef_2_i[dim_n]))  # / n_units_n
        magnitude = np.abs(np.dot(coef_1_i, coef_2_i))  # / n_units
        plot_angles(ax, theta_p, mag_p, color="#54B06C", alpha=0.5)
        plot_angles(ax, theta_n, mag_n, color="#935DA6", alpha=0.5)
        plot_angles(ax, theta, magnitude, color="k", alpha=0.5)
    # set y limit
    ax.set_ylim([0, 2])
    ax.set_yticks([0, 1, 2])
    if angle == "self_other":
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])
    ax.set_title(f"{angle}")
    # set guidline for theta in polar plot to be 30, 60, 90, 120, 150, 180 degrees
    ax.set_xticks(
        [
            0,
            np.pi / 6,
            np.pi / 3,
            np.pi / 2,
            2 * np.pi / 3,
            5 * np.pi / 6,
            np.pi,
        ]
    )

    return fig


# make a polar plot of the angles between self and other directions
def make_polar_plot(animal, event, angle="self_other"):
    datadir = findrootdir()
    angle_directory = f"{datadir}/stats_paper"
    plot_directory = f"{datadir}/plots_paper"
    file_name_b = (
        f"{datadir}/stats_paper/{animal}_{event}_act_obs_dimensions.json"
    )
    # CHECK IF FILE EXISTS
    if not os.path.exists(file_name_b):
        print(f"File not found: {file_name_b}")
        return

    results_b = json.load(open(file_name_b, "r"))

    fig = make_polar_plot_with_sd(results_b, angle)
    print(f"Animal: {animal} Event: {event} Angle: {angle}")
    if animal == "both" and angle == "self_other":
        fig_name_polar = (
            f"{plot_directory}/Fig4O_{animal}_{event}_{angle}_polar_plot_sd.pdf"
        )
    elif angle == "self_other":
        fig_name_polar = f"{plot_directory}/FigS7GH_{animal}_{event}_{angle}_polar_plot_sd.pdf"
    else:
        fig_name_polar = (
            f"{angle_directory}/{animal}_{event}_{angle}_polar_plot_sd.pdf"
        )
    fig.savefig(fig_name_polar, dpi=300)
    plt.close()


def main():
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"

    # Close all previously opened figures
    plt.close("all")

    # return
    # load data
    for event in ["fdbk", "prechoice"]:
        make_combined_files(
            event,
            item="act_obs_dimensions",
        )
        for animal in ["O", "L", "both"]:
            for angle in ["self_other", "self_switch", "switch_other"]:
                make_polar_plot(animal, event, angle)


if __name__ == "__main__":
    main()
