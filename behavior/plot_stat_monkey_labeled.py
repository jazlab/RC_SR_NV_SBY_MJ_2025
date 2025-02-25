"""
This script makes behavior plots for the paper, for the two-player monkey task.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import seaborn as sns
from utils.plots import beutify
from behavior.plot_switch_social import (
    plot_data_outcome,
    plot_data_early_and_late,
    plot_data_by_tokens,
    plot_data_condition,
    plot_data_by_attention_lvl,
    plot_distribution_attention,
    plot_attention_stacked,
)
from behavior.trial_info_utils import (
    combine_dfs_from_sessions,
)
from behavior.congruency_plots import (
    plot_switch_by_congruency,
)
from behavior.plot_slopes import plot_slopes_by_session

# print location of plot_slopes_by_session
print(sys.modules["behavior.plot_slopes"])
import warnings

warnings.filterwarnings(action="ignore", category=FutureWarning)


def add_performance_level(df):
    """
    Categorizes trials based on the number of gates touched.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'Gates_touched_total' with the number of gates touched per trial.

    Returns:
        pd.DataFrame: DataFrame with an additional 'performance_level' column, categorized as 'Low', 'Mid', or 'High'.
    """
    # create a new column to indicate whether the trial has less or more gates
    df["performance_level"] = "Mid"
    gates_per_trial_threshold_low = 9
    gates_per_trial_threshold = 10
    df.loc[
        (df["Gates_touched_total"] > gates_per_trial_threshold),
        "performance_level",
    ] = "High"
    df.loc[
        (df["Gates_touched_total"] < gates_per_trial_threshold_low),
        "performance_level",
    ] = "Low"

    return df


def plot_outcome_histogram(plot_file_name, df):
    """
    Plots and saves a histogram of trial outcomes.

    Args:
        plot_file_name (str or Path): Path to save the plot.
        df (pd.DataFrame): DataFrame containing an 'Outcome' column.

    Returns:
        None
    """
    fig = plt.figure(figsize=(2, 2))
    ax = sns.histplot(
        df["Outcome"],
        color="grey",
        stat="count",
    )
    plt.grid(False)
    ax.set_yscale("log")

    # Remove minor ticks
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    beutify(ax)
    plt.xlabel("Outcome")
    plt.ylabel("N (trials)")
    # plt.show()
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_accuracy_by_switch(plot_file_name, df, option="combined"):
    """
    Plots accuracy over trials after a switch event, optionally grouped by animal or session.

    Args:
        plot_file_name (str or Path): Path to save the plot.
        df (pd.DataFrame): DataFrame containing 'trial_after' and 'accuracy' columns.
        option (str, optional): Grouping option ('combined', 'animals', or 'sessions'). Defaults to 'combined'.

    Returns:
        None
    """

    fig = plt.figure(figsize=(2, 2))
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
            markersize=8,
            markers="o",
            data=df,
            estimator="mean",
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

    ax.set_xlim([-4, 10])
    # Change x and y labels
    ax.set_xlabel("N (trials after switch)")
    ax.set_ylabel("Choice accuracy")
    plt.legend(frameon=False)  # Remove the box around the legend
    # Add horizontal grid lines
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.grid(visible=True, which="major", axis="y", color="0.9", linestyle="-")
    # Customize the figure and axes to have a transparent background
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_reward_by_session(plot_file_name, df):
    """
    Plots the probability of correct responses over sessions.

    Args:
        plot_file_name (str or Path): Path to save the plot.
        df (pd.DataFrame): DataFrame containing session-wise performance data.

    Returns:
        None
    """
    # add a column called date
    df["date"] = df["date_time"].apply(lambda x: "-".join(x.split("_")[:3]))
    df = df.sort_values("date")
    # map value of correct_a0 and correct_a1 from 1,-1 to 1,0
    df["correct_a0"] = df["correct_a0"].map({1: 1, -1: 0})
    df["correct_a1"] = df["correct_a1"].map({1: 1, -1: 0})
    fig = plt.figure(figsize=(9, 9), facecolor="white")

    ax = sns.lineplot(
        x="date",
        y="correct_a0",
        data=df,
        color="green",
        estimator="mean",
        err_style="band",
        linestyle="-",
    )
    sns.lineplot(
        x="date",
        y="correct_a1",
        data=df,
        color="purple",
        estimator="mean",
        err_style="band",
        linestyle="-",
    )

    # print the mean and standard error of correct_a0 and a1
    mean_corr_a0 = df.groupby("date")["correct_a0"].mean().mean()
    sd_corr_a0 = df.groupby("date")["correct_a0"].mean().std()
    mean_corr_a1 = df.groupby("date")["correct_a1"].mean().mean()
    sd_corr_a1 = df.groupby("date")["correct_a1"].mean().std()
    print(f"Mean correct_a0: {mean_corr_a0} +/- {sd_corr_a0}")
    print(f"Mean correct_a1: {mean_corr_a1} +/- {sd_corr_a1}")

    ax.tick_params(axis="y", colors="black")

    # add a legend of 'Accuracy' in black and 'Reward' in green
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xticklabels([1, 10, 20, 30])
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


def plot_block_length_histogram(plot_file_name, df, field="pos_in_block"):
    """
    Plots and saves a histogram of block lengths within sessions.

    Args:
        plot_file_name (str or Path): Path to save the plot.
        df (pd.DataFrame): DataFrame containing trial progression data.
        field (str, optional): Column name used to determine block lengths. Defaults to 'pos_in_block'.

    Returns:
        None
    """
    # Calculate block lengths
    values = df[field].values
    trial_nums = df["trial_num"].values
    block_lengths = []
    current_length = 1
    first_block = True

    for i in range(1, len(values)):
        if trial_nums[i] < trial_nums[i - 1]:
            # New session starts, ignore current block and reset
            first_block = True
            current_length = 1
        elif values[i] == 1 and values[i - 1] != 1:
            if first_block:
                # ignore the first block because active side is undefined
                first_block = False
            else:
                block_lengths.append(current_length)
            current_length = 1
        else:
            current_length += 1

    # Add the last block length if it's not already added
    if current_length > 1 or (len(block_lengths) == 0 and current_length == 1):
        block_lengths.append(current_length)

    # Remove the last block as it might be incomplete
    if len(block_lengths) > 1:
        block_lengths = block_lengths[:-1]

    # Histogram plot for block length
    fig = plt.figure(figsize=(2, 2))
    sns.set(style="whitegrid")

    # Create a histogram with bins from 1 to 30
    bins = np.arange(1, 31)
    ax = sns.histplot(
        block_lengths,
        color="grey",
        bins=bins,
        stat="count",
    )
    plt.grid(False)
    ax.set_xlim([1, 30])
    plt.xticks([1, 10, 20, 30])
    beutify(ax)
    plt.xlabel("Block length")
    plt.ylabel("N(trials)")
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()

    # Print some statistics about block lengths:
    # range, median, 25th, and 75th percentiles
    print(f" stats for {field} ")
    print(
        f"Range (min to max): {np.min(block_lengths)} to {np.max(block_lengths)}"
    )
    print(f"Median block length: {np.median(block_lengths)}")
    print(f"25th percentile: {np.percentile(block_lengths, 25)}")
    print(f"75th percentile: {np.percentile(block_lengths, 75)}")


def plot_performance_by_difficulty(plot_file_name, df):
    """
    Plots performance as a function of task difficulty.

    Args:
        plot_file_name (str or Path): Path to save the plot.
        df (pd.DataFrame): DataFrame containing 'difficulty' and 'touched_polls' columns.

    Returns:
        None
    """
    df = df[df["history"] != "1R"]
    # print mean and sd of touched_polls
    mean_touched_polls = df["touched_polls"].mean()
    sd_touched_polls = df["touched_polls"].std()
    print(f"Mean touched polls: {mean_touched_polls}")
    print(f"SD touched polls: {sd_touched_polls}")
    # difficulty is continuous, we first quantize it to 0.1
    df["difficulty_qt"] = np.round(df["difficulty"], 1)
    # then, calculate the number of datapoints for each bin from 0 to 4
    bins = np.arange(0, 4.1, 0.2)
    for bin in bins:
        n_trials = len(df[df["difficulty_qt"] == bin])
        if n_trials < 5:
            df = df[~np.isclose(df["difficulty_qt"], bin)]
    fig = plt.figure(figsize=(2, 2))
    ax = sns.lineplot(
        x="difficulty_qt",
        y="touched_polls",
        data=df,
        color="black",
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
    )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([0, 1, 2, 3])
    ax.set_yticks([5, 10, 15])
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    beutify(ax)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Performance")
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)


def main():
    """
    Main function to generate and save various behavioral analysis plots.

    - Loads session data.
    - Computes derived statistics.
    - Generates and saves multiple figures.

    Args:
        None

    Returns:
        None
    """
    mpl.rcParams["pdf.fonttype"] = 42
    from utils.LoadSession import findrootdir

    root_dir = findrootdir()
    root_dir = Path(root_dir)
    behv_plot_dir = root_dir / "plots_paper"
    if not behv_plot_dir.exists():
        behv_plot_dir.mkdir()

    for subject in ["O", "L"]:
        combined_df = combine_dfs_from_sessions(root_dir, subject=subject)
        # print number of sessions
        # first, remove all nans in the attention field - only for attention plots
        combined_df = combined_df.dropna(subset=["Attention"])
        n_sessions = len(combined_df["date"].unique())
        print(f"Number of sessions for {subject}: {n_sessions}")
        # plot switch by attention
        plot_file_name = (
            behv_plot_dir / f"FigS1J_M_switch_by_attention_{subject}.pdf"
        )
        plot_data_by_attention_lvl(plot_file_name, combined_df)

        # plot attention by condition
        plot_file_name = (
            behv_plot_dir / f"FigS1H_K_attention_by_condition_{subject}.pdf"
        )
        plot_attention_stacked(plot_file_name, combined_df)

        # plot attention value distribution
        plot_file_name = (
            behv_plot_dir / f"FigS1I_L_attention_distribution_{subject}.pdf"
        )
        plot_distribution_attention(plot_file_name, combined_df)

    # make plot of performance by difficulty
    df = combine_dfs_from_sessions(root_dir, file_pattern="trial_info")
    plot_file_name = (
        behv_plot_dir / "FigS2H_combined_performance_by_difficulty.pdf"
    )
    plot_performance_by_difficulty(plot_file_name, df)

    # make plot of slopes by session
    for subject in ["Offenbach", "Lalo", "Both"]:
        df = combine_dfs_from_sessions(root_dir, subject=subject)
        df = add_performance_level(df)
        plot_slopes_by_session(root_dir, df, subject)

    df = combine_dfs_from_sessions(root_dir, file_pattern="trial_info")
    file_name = behv_plot_dir / "Fig1K_accuracy_by_session.pdf"
    plot_reward_by_session(file_name, df)

    combined_df = combine_dfs_from_sessions(root_dir)
    combined_df = add_performance_level(combined_df)
    plot_file_name = (
        behv_plot_dir / "FigS5D_combined_switches_early_late_diffchoice.pdf"
    )
    plot_data_early_and_late(plot_file_name, combined_df, "diffchoice")

    # New calls for block length histograms
    df_block = combine_dfs_from_sessions(root_dir, file_pattern="trial_info")
    plot_file_name = behv_plot_dir / "FigS2F_combined_block_length.pdf"
    plot_block_length_histogram(plot_file_name, df_block)
    plot_file_name = behv_plot_dir / "FigS2D_combined_block_length_obj.pdf"
    plot_block_length_histogram(plot_file_name, df_block, "pos_in_block_obj")

    df_outcome = combine_dfs_from_sessions(root_dir)
    plot_file_name = behv_plot_dir / "FigS2B_combined_outcome_hist.pdf"
    plot_outcome_histogram(plot_file_name, df_outcome)

    # fig 2 switch plots:
    combined_df = combine_dfs_from_sessions(root_dir)
    combined_df = add_performance_level(combined_df)
    plot_file_name = behv_plot_dir / "Fig2B_combined_switches.pdf"
    plot_data_outcome(plot_file_name, combined_df)
    plot_file_name = (
        behv_plot_dir / "Fig2K_combined_switches_actor_observer.pdf"
    )
    plot_data_condition(plot_file_name, combined_df)
    plot_file_name = behv_plot_dir / "Fig2E_combined_switches_early_late.pdf"
    plot_data_early_and_late(plot_file_name, combined_df, "Switches")
    plot_file_name = behv_plot_dir / "Fig2H_combined_switches_tokens.pdf"
    plot_data_by_tokens(plot_file_name, combined_df)
    for subject in ["O", "L"]:
        combined_df = combine_dfs_from_sessions(root_dir, subject=subject)
        combined_df = add_performance_level(combined_df)
        plot_switch_by_congruency(
            behv_plot_dir / f"FigS5EF_switch_by_congruency_{subject}.pdf",
            combined_df,
        )
        plot_file_name = (
            behv_plot_dir
            / f"FigS4CD_combined_switches_{subject}_early_late.pdf"
        )
        plot_data_early_and_late(plot_file_name, combined_df)
        plot_file_name = (
            behv_plot_dir / f"FigS4AB_combined_switches_{subject}_outcome.pdf"
        )
        plot_data_outcome(plot_file_name, combined_df)
        plot_file_name = (
            behv_plot_dir / f"FigS4EF_combined_switches_{subject}_tokens.pdf"
        )
        plot_data_by_tokens(plot_file_name, combined_df, late_trials_only=False)

    df_accuracy = combine_dfs_from_sessions(root_dir, "*accuracy")
    plot_file_name = behv_plot_dir / "Fig1K_combined_accuracy_animals.pdf"
    plot_accuracy_by_switch(plot_file_name, df_accuracy, option="animals")


if __name__ == "__main__":
    main()
