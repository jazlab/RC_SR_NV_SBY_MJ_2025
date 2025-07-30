"""
This script is used to make plot for the cross-validation results of the
actor and observer trial's projectin on the switch evidence direction.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.plots import beutify
from utils.LoadSession import findrootdir
from polar_plot_common import make_combined_files


def plot_projection_by_history(results):
    # Calculate mean and standard error
    # select dimensions that are significant in predicting single trial behavior
    proj_actor = np.array(results["proj_act_shuff_actor"])
    proj_observer = np.array(results["proj_act_shuff_observer"])

    n_shuff = proj_actor.shape[0]
    actor_1r = np.nanmean(proj_actor[:, 0].reshape(n_shuff, -1), axis=1)
    actor_1nr = np.nanmean(proj_actor[:, 1].reshape(n_shuff, -1), axis=1)
    actor_2nr = np.nanmean(proj_actor[:, 2].reshape(n_shuff, -1), axis=1)
    observer_1r = np.nanmean(proj_observer[:, 0].reshape(n_shuff, -1), axis=1)
    observer_1nr = np.nanmean(proj_observer[:, 1].reshape(n_shuff, -1), axis=1)
    observer_2nr = np.nanmean(proj_observer[:, 2].reshape(n_shuff, -1), axis=1)
    # remove nans
    actor_1r = actor_1r[~np.isnan(actor_1r)]
    actor_1nr = actor_1nr[~np.isnan(actor_1nr)]
    actor_2nr = actor_2nr[~np.isnan(actor_2nr)]
    observer_1r = observer_1r[~np.isnan(observer_1r)]
    observer_1nr = observer_1nr[~np.isnan(observer_1nr)]
    observer_2nr = observer_2nr[~np.isnan(observer_2nr)]
    # stats comparing 1R, 1NR, 2NR using paired t-test
    from scipy.stats import ttest_rel

    # 1R vs 1NR
    t1, p1 = ttest_rel(actor_1r, actor_1nr)
    # 1R vs 2NR
    t2, p2 = ttest_rel(actor_1r, actor_2nr)
    # 1NR vs 2NR
    t3, p3 = ttest_rel(actor_1nr, actor_2nr)
    print(f"1R vs 1NR: t={t1}, p={p1}")
    print(f"1R vs 2NR: t={t2}, p={p2}")
    print(f"1NR vs 2NR: t={t3}, p={p3}")
    # observer
    # 1R vs 1NR
    t1, p1 = ttest_rel(observer_1r, observer_1nr)
    # 1R vs 2NR
    t2, p2 = ttest_rel(observer_1r, observer_2nr)
    # 1NR vs 2NR
    t3, p3 = ttest_rel(observer_1nr, observer_2nr)
    print(f"1R vs 1NR: t={t1}, p={p1}")
    print(f"1R vs 2NR: t={t2}, p={p2}")
    print(f"1NR vs 2NR: t={t3}, p={p3}")

    # create dataframe for plotting
    data = pd.DataFrame(
        {
            "projection": np.concatenate(
                [
                    actor_1r,
                    actor_1nr,
                    actor_2nr,
                    observer_1r,
                    observer_1nr,
                    observer_2nr,
                ]
            ),
            "condition": (
                ["1R"] * len(actor_1r)
                + ["1NR"] * len(actor_1nr)
                + ["2NR"] * len(actor_2nr)
                + ["1R"] * len(observer_1r)
                + ["1NR"] * len(observer_1nr)
                + ["2NR"] * len(observer_2nr)
            ),
            "actor_observer": ["Actor"] * len(actor_1r)
            + ["Actor"] * len(actor_1nr)
            + ["Actor"] * len(actor_2nr)
            + ["Observer"] * len(observer_1r)
            + ["Observer"] * len(observer_1nr)
            + ["Observer"] * len(observer_2nr),
        }
    )
    # Set up the figure
    figsize = (2, 2)
    fig = plt.figure(figsize=figsize, facecolor="white")
    data_actor = data[data["actor_observer"] == "Actor"]
    data_observer = data[data["actor_observer"] == "Observer"]
    ax = sns.lineplot(
        x="condition",
        y="projection",
        data=data_actor,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        color="black",
        marker="o",
        linestyle="-",
        markersize=10,
    )
    sns.lineplot(
        x="condition",
        y="projection",
        data=data_observer,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        color="black",
        marker="o",
        linestyle="-",
        markersize=10,
        markerfacecolor="none",
        markeredgewidth=1,
        markeredgecolor="black",
    )
    # Get the lines and customize markers
    lines = ax.get_lines()
    for line in lines:
        if line.get_label() == "Observer":
            line.set_markerfacecolor(
                "none"
            )  # Make the marker hollow for Observer

    # Customize the aesthetics of the plot
    sns.set_theme(font_scale=1.4)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["bottom"].set_color("gray")
    ax.set_facecolor("white")
    ax.grid(visible=True, which="major", axis="y", color="0.9", linestyle="-")
    beutify(ax)
    return fig


def make_plots_from_results(results, animal, event, angle_directory, suffix=""):
    # plot pswitch for each unit on a scatter plot
    fig = plot_projection_by_history(results)
    fignames = {
        "both": "Fig4D",
        "O": "FigS7E",
        "L": "FigS7F",
    }
    fig_name_scatter = f"{angle_directory}/{fignames[animal]}_{animal}_{event}_projection_by_history{suffix}.pdf"
    fig.savefig(fig_name_scatter, dpi=300)
    plt.close()


def make_plots_per_animal(animal, event, suffix=""):
    datadir = findrootdir()
    file_name = f"{datadir}/stats_paper/{animal}_{event}_act_obs_projections{suffix}.json"
    angle_directory = f"{datadir}/plots_paper"
    # CHECK IF FILE EXISTS
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return
    with open(file_name, "r") as file:
        results = json.load(file)
    make_plots_from_results(
        results, animal, event, angle_directory, suffix=suffix
    )


def main():
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    # load data
    for event in ["fdbk"]:
        for suffix in ["", "_equalNSwitch_equalNNeurons"]:
            make_combined_files(
                event,
                item=f"act_obs_projections{suffix}",
            )
            for animal in ["O", "L", "both"]:
                make_plots_per_animal(animal, event, suffix=suffix)


if __name__ == "__main__":
    main()
