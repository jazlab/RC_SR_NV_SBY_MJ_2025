"""
This script makes behavior plots for the paper, for the single player monkey task.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils.plots import beutify
from utils.get_session_info import findrootdir, find_behavior_sessions


def plot_accuracy_by_session(plot_file_name, df):
    # plot accuracy over trial.

    fig = plt.figure(figsize=(9, 9), facecolor="white")

    ax = sns.lineplot(
        x="date",
        y="correct_a0",
        hue="player",
        data=df,
        palette=["green", "purple"],
        estimator="mean",
        err_style="band",
        linestyle="-",
    )

    # print the mean and standard error of the mean of correct_a0
    df_O = df[df["player"] == "Offenbach"]
    df_L = df[df["player"] == "Lalo"]
    mean_corr_O = df_O.groupby("date")["correct_a0"].mean().mean()
    se_corr_O = df_O.groupby("date")["correct_a0"].mean().std() / 6
    mean_corr_L = df_L.groupby("date")["correct_a0"].mean().mean()
    se_corr_L = df_L.groupby("date")["correct_a0"].mean().std() / 6
    print(f"Mean correct O: {mean_corr_O} +/- {se_corr_O}")
    print(f"Mean correct L: {mean_corr_L} +/- {se_corr_L}")

    ax.tick_params(axis="y", colors="black")

    # add a legend of 'Accuracy' in black and 'Reward' in green
    ax.set_xticks([0, 9])
    ax.set_xticklabels([1, 10])
    # Remove x and y axis lines
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Change x and y labels
    ax.set_xlabel("Session number")
    ax.set_ylabel("Probability")

    # Add horizontal grid lines
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ylim = [0.45, 1.05]
    ax.set_ylim(ylim)
    plt.grid(axis="y")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


def combine_dfs_from_sessions_single(
    basedir, file_pattern="*_switches", subject=None
):
    bhv_dirs = find_behavior_sessions(basedir, subject)
    dfs = []
    for bhv_dir in bhv_dirs:
        # check number of sessions from trial_info.csv
        csv_files = list(bhv_dir.glob(f"trial_info.csv"))
        if len(csv_files) == 0:
            print(f"No trial_info files found in {bhv_dir}")
            continue
        df = pd.read_csv(csv_files[0])
        # infer date from bhv_dir
        date = bhv_dir.parts[-3]
        # load .csv to a dataframe and add to list
        csv_files = list(bhv_dir.glob(f"{file_pattern}.csv"))
        if len(csv_files) == 0:
            print(f"No switch files found in {bhv_dir}")
            continue
        if subject is not None and file_pattern != "trial_info":
            if subject != "Both":
                csv_files = [f for f in csv_files if subject in f.name]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df["date"] = date
            dfs.append(df)
    # concatenate list of dataframes into a single dataframe
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("date")
    unique_dates = df["date"].unique()[-14:]
    # convert date to the position of date in unique dates
    df["date"] = np.array(
        [np.where(unique_dates == d)[0][0] for d in df["date"]]
    )
    return df


def plot_accuracy_by_switch(plot_file_name, df, option="combined"):
    # plot accuracy over trial after switch measure. make a mean plot
    # from all sessions in black, then individual session in gray.

    # print the mean and std of accuracy overall for each "player"
    print(df.groupby("player")["accuracy"].mean())

    fig = plt.figure(figsize=(6, 6))
    if option == "combined":
        ax = sns.lineplot(
            x="trial_after",
            y="accuracy",
            data=df,
            color="black",
            estimator="mean",
        )
    elif option == "animals":
        ax = sns.lineplot(
            x="trial_after",
            y="accuracy",
            hue="player",
            palette=["green", "purple"],
            data=df,
            estimator="mean",
            marker="o",
            markersize=8,
            linewidth=2,
        )
    elif option == "sessions":
        ax = sns.lineplot(
            x="trial_after",
            y="accuracy",
            hue="date",
            data=df,
            estimator="mean",
        )
    sns.set_theme(font_scale=1.4)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Change x and y labels
    ax.set_xlabel("N (trials after switch)")
    ax.set_ylabel("Choice accuracy")
    plt.legend(frameon=False)  # Remove the box around the legend
    ax.set_xlim(-4, 10)
    ax.set_xticks([-4, 0, 4, 8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.grid(visible=True, which="major", axis="y", color="0.9", linestyle="-")
    # Customize the figure and axes to have a transparent background
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


def main():
    mpl.rcParams["pdf.fonttype"] = 42
    rootdir = findrootdir()
    behv_plot_dir = f"{rootdir}/plots_paper"
    df_accuracy_O = combine_dfs_from_sessions_single(
        rootdir, file_pattern="*accuracy", subject="Offenbach"
    )
    df_accuracy_O["player"] = "Offenbach"
    df_accuracy_L = combine_dfs_from_sessions_single(
        rootdir, file_pattern="*accuracy", subject="Lalo"
    )
    df_accuracy_L["player"] = "Lalo"
    df_accuracy = pd.concat([df_accuracy_O, df_accuracy_L], ignore_index=True)
    plot_file_name = f"{behv_plot_dir}/Fig1J_combined_accuracy_single.pdf"
    plot_accuracy_by_switch(plot_file_name, df_accuracy, option="animals")

    # make plot of reward rate by session
    df_O = combine_dfs_from_sessions_single(
        rootdir, file_pattern="trial_info", subject="Offenbach"
    )
    df_O["correct_a0"] = df_O["correct_a0"].map({1: 1, -1: 0})
    mean_corr = df_O.groupby("date")["correct_a0"].mean().mean()
    sd_corr = df_O.groupby("date")["correct_a0"].mean().std()
    n_sessions = len(df_O["date"].unique())
    print(
        f"O Mean correct from {n_sessions} sessions: {mean_corr} +/- {sd_corr}"
    )
    df_O["player"] = "Offenbach"
    df_L = combine_dfs_from_sessions_single(
        rootdir, file_pattern="trial_info", subject="Lalo"
    )
    df_L["player"] = "Lalo"
    df_L["correct_a0"] = df_L["correct_a0"].map({1: 1, -1: 0})
    mean_corr = df_L.groupby("date")["correct_a0"].mean().mean()
    sd_corr = df_L.groupby("date")["correct_a0"].mean().std()
    n_sessions = len(df_L["date"].unique())
    print(
        f"L Mean correct from {n_sessions} sessions: {mean_corr} +/- {sd_corr}"
    )
    df = pd.concat([df_O, df_L], ignore_index=True)
    file_name = f"{behv_plot_dir}/Fig1J_accuracy_by_session_single.pdf"
    plot_accuracy_by_session(file_name, df)


if __name__ == "__main__":
    main()
