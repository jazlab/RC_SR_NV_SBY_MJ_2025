"""
This script plots the result of the cross-validation of the switch evidence 
direction.
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


def plot_projection_train_and_test(results):
    n_shuff = 100
    proj_test = np.array(results["proj_act_shuff_test"])
    proj_train = np.array(results["proj_act_shuff_train"])
    proj_random = np.array(results["proj_act_shuff_random_test"])
    test_0 = proj_test[:, 0].reshape(n_shuff, -1).mean(axis=1)
    test_1 = proj_test[:, 1].reshape(n_shuff, -1).mean(axis=1)
    test_2 = proj_test[:, 2].reshape(n_shuff, -1).mean(axis=1)
    train_0 = proj_train[:, 0].reshape(n_shuff, -1).mean(axis=1)
    train_1 = proj_train[:, 1].reshape(n_shuff, -1).mean(axis=1)
    train_2 = proj_train[:, 2].reshape(n_shuff, -1).mean(axis=1)
    random_0 = proj_random[:, 0].reshape(n_shuff, -1).mean(axis=1)
    random_1 = proj_random[:, 1].reshape(n_shuff, -1).mean(axis=1)
    random_2 = proj_random[:, 2].reshape(n_shuff, -1).mean(axis=1)
    data = pd.DataFrame(
        {
            "projection": np.concatenate(
                [
                    test_0,
                    test_1,
                    test_2,
                    train_0,
                    train_1,
                    train_2,
                    random_0,
                    random_1,
                    random_2,
                ]
            ),
            "condition": (
                ["3"] * len(test_0) + ["2"] * len(test_1) + ["1"] * len(test_2)
            )
            * 3,
            "train_test": ["Test"] * 3 * n_shuff
            + ["Train"] * 3 * n_shuff
            + ["Random"] * 3 * n_shuff,
        }
    )
    data_train = data[data["train_test"] == "Train"]
    data_test = data[data["train_test"] == "Test"]
    data_random = data[data["train_test"] == "Random"]
    figsize = (3, 3)
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = sns.lineplot(
        x="condition",
        y="projection",
        data=data_train,
        estimator="mean",
        marker="o",
        color="black",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linestyle="-",
        legend=False,
        markersize=10,
    )
    sns.lineplot(
        x="condition",
        y="projection",
        data=data_test,
        estimator="mean",
        marker="o",
        color="black",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linestyle="-",
        legend=False,
        markersize=10,
        markerfacecolor="none",
        markeredgewidth=1,
        markeredgecolor="black",
    )
    sns.lineplot(
        x="condition",
        y="projection",
        data=data_random,
        estimator="mean",
        marker="o",
        color="black",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linestyle="--",
        legend=False,
        markersize=10,
        markerfacecolor="none",
        markeredgewidth=1,
        markeredgecolor="black",
    )
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


def make_plots_from_results(results, animal, save_dir):
    fig = plot_projection_train_and_test(results)
    fignames = {
        "both": "Fig4B",
        "O": "FigS7A",
        "L": "FigS7B",
    }
    fig_name_scatter = (
        f"{save_dir}/{fignames[animal]}_{animal}_projection_test.pdf"
    )
    fig.savefig(fig_name_scatter, dpi=300)


def make_plots_per_animal(animal, event):
    datadir = findrootdir()
    file_name = (
        f"{datadir}/stats_paper/{animal}_{event}_swe_cross_validation.json"
    )
    save_dir = f"{datadir}/plots_paper"
    # CHECK IF FILE EXISTS
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return
    # make save dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(file_name, "r") as file:
        results = json.load(file)
    make_plots_from_results(results, animal, save_dir)


def main():
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    # return
    # load data
    for event in ["fdbk"]:
        make_combined_files(
            event,
            item="swe_cross_validation",
        )
        for animal in ["O", "L", "both"]:
            make_plots_per_animal(animal, event)


if __name__ == "__main__":
    main()
