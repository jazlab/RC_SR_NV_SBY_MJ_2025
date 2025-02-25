"""
This script makes behavior plots for the paper, for the two-player human task.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import seaborn as sns
from copy import deepcopy
from utils.plots import beutify
from behavior.plot_switch_social import (
    plot_data_outcome,
    plot_data_condition,
    plot_data_early_and_late,
    plot_data_by_tokens,
)
from behavior.trial_info_utils import (
    preprocess_for_regression,
)
from behavior.congruency_plots import (
    plot_switch_by_congruency,
    plot_confidence_by_congruency,
)
from behavior.behavior_models import (
    model_switches_mixedlogit,
    model_confidence_mixedlinear,
)
from behavior.plot_stat_monkey_labeled import (
    plot_performance_by_difficulty,
    add_performance_level,
    plot_outcome_histogram,
)
from behavior.plot_slopes import plot_slopes_by_session
import warnings
from ast import literal_eval

warnings.filterwarnings(action="ignore", category=FutureWarning)
# Suppress the SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'


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
    # plot accuracy over trial after switch measure. make a mean plot
    # from all sessions in black, then individual session in gray.

    fig = plt.figure(figsize=(2, 2))
    if option == "combined":
        ax = sns.lineplot(
            x="trial_after",
            y="accuracy",
            data=df,
            color="black",
            estimator="mean",
            ci=None,
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
    # add plot for each player, all with the same color gray and linestyle
    unique_players = df["player"].unique()
    for player in unique_players:
        sns.lineplot(
            x="trial_after",
            y="accuracy",
            data=df[df["player"] == player],
            color="gray",
            estimator="mean",
            linestyle="-",
            linewidth=0.2,
        )
    sns.set_theme(font_scale=1.4)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # set x lim
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
    if field == "pos_in_block":
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_xlim([1, 50])
    elif field == "pos_in_block_obj":
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlim([1, 32])
    ax.set_xlim([1, 30])
    plt.xticks([1, 10, 20, 30])
    beutify(ax)
    plt.xlabel("Block length")
    plt.ylabel("N(trials)")
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()

    # Print some statistics about block lengths:
    # range, median, 25th, and 75th percentiles
    # ALso number of block lengths greater than 30
    block_lengths = np.array(block_lengths)
    n_greater_than_30 = np.sum(block_lengths > 30)
    print(f" stats for {field} ")
    print(
        f"Range (min to max): {np.min(block_lengths)} to {np.max(block_lengths)}"
    )
    print(f"Median block length: {np.median(block_lengths)}")
    print(f"25th percentile: {np.percentile(block_lengths, 25)}")
    print(f"75th percentile: {np.percentile(block_lengths, 75)}")
    print(f"Number of block lengths greater than 30: {n_greater_than_30}")


def swap_player(df):
    """
    Swaps player-specific data fields for subject comparisons.

    Args:
        df (pd.DataFrame): DataFrame with player-based data stored in lists.

    Returns:
        None (modifies the DataFrame in place)
    """
    for col in df.columns:
        if df[col].apply(type).eq(str).all():
            # check if the first character is [
            if df[col].str[0][0] == "[":
                df[col] = df[col].apply(lambda x: literal_eval(x))
    # for each column, if the value in that column is a list, swap the first and second element
    for col in df.columns:
        if df[col].apply(type).eq(list).all():
            df[col] = df[col].apply(lambda x: [x[1], x[0]])


def combine_dfs_from_sessions_human(
    behv_data_dir, file_pattern="*switches.csv"
):
    """
    Combines session-level data from multiple human experiment files into a single DataFrame.

    Args:
        behv_data_dir (Path): Directory containing behavioral data files.
        file_pattern (str, optional): File name pattern to filter data. Defaults to "*switches.csv".

    Returns:
        pd.DataFrame: Combined DataFrame with processed session data.
    """
    pairs = [d for d in behv_data_dir.iterdir() if d.is_dir()]
    dfs = []
    session_numbers = {}
    for pair in pairs:
        pair_ski_resort_csv = pair / "ski-resort_csv"
        csv_files = list(pair_ski_resort_csv.glob(file_pattern))
        # sort files alphanumerically
        csv_files = sorted(csv_files, key=lambda x: x.stem)
        for csv_file in csv_files:
            # ignore files that start with .
            if csv_file.stem.startswith("."):
                continue
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding="ISO-8859-1")
            # add variables to the dataframe
            year, month, day = csv_file.stem.split("_")[:3]
            subject_1, subject_2 = pair.stem.split("-")
            if file_pattern == "*switches.csv":
                subject = df["subject"][0]
                df["date"] = f"{year}_{month}_{day}_{subject}"
                dfs.append(df)
            elif file_pattern == "*info.csv":
                # add a session number term that starts at 1 for each subject
                if pair not in session_numbers:
                    session_numbers[pair] = 0
                else:
                    session_numbers[pair] += 1
                session_number = session_numbers[pair]
                df["session_number"] = session_number
                df_1 = df
                df_1["date"] = f"{year}_{month}_{day}_{subject_1}"
                df_1["subject"] = subject_1

                df_2 = deepcopy(df)
                df_2["date"] = f"{year}_{month}_{day}_{subject_2}"
                df_2["subject"] = subject_2
                # for df_2, conditions are swapped
                swap_player(df_2)
                # add subject field based on the parent directory name
                dfs.append(df_1)
                dfs.append(df_2)
            elif file_pattern == "*accuracy.csv":
                dfs.append(df)
    # concatenate list of dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["diffchoice"] = 0
    # if incongruent is in condition, then set diffchoice to 1
    if file_pattern == "*info.csv":
        combined_df["diffchoice"] = combined_df["agent_choices"].apply(
            lambda x: 1 if x[0] != x[1] else 0
        )
        combined_df["Pos_in_block"] = combined_df["pos_in_block"]
    elif file_pattern == "*switches.csv":
        combined_df.loc[
            combined_df["Condition"].str.contains("incongruent"), "diffchoice"
        ] = 1
    return combined_df


def load_human_multi(root_dir):
    """
    Loads and combines multi-session human experiment data.

    Args:
        root_dir (Path): Root directory containing human multi-session data.

    Returns:
        pd.DataFrame: Combined DataFrame with all session data.
    """
    human_multi_dir = root_dir / "human_multi"
    csv_files = list(human_multi_dir.glob("*.csv"))
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding="ISO-8859-1")
        dfs.append(df)
    # concatenate list of dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["history"] = combined_df["Outcome"]
    combined_df["difficulty"] = combined_df["Difficulty"]
    combined_df["touched_polls"] = combined_df["Performance"]
    return combined_df


def plot_accuracy_by_session(plot_file_name, df):
    """
    Plots the probability of correct responses over sessions.

    Args:
        plot_file_name (str or Path): Path to save the plot.
        df (pd.DataFrame): DataFrame containing session-wise performance data.

    Returns:
        None
    """

    fig = plt.figure(figsize=(9, 9), facecolor="white")

    ax = sns.lineplot(
        x="session_number",
        y="correct_a0",
        hue="subject",
        data=df,
        color="gray",
        estimator="mean",
        err_style="band",
        linestyle="-",
    )

    # print the mean and standard error of the mean of correct_a0 and a1
    mean_corr_a0 = df.groupby("session_number")["correct_a0"].mean().mean()
    se_corr_a0 = df.groupby("session_number")["correct_a0"].mean().std() / 6
    mean_corr_a1 = df.groupby("session_number")["correct_a1"].mean().mean()
    se_corr_a1 = df.groupby("session_number")["correct_a1"].mean().std() / 6
    print(f"Mean correct_a0: {mean_corr_a0} +/- {se_corr_a0}")
    print(f"Mean correct_a1: {mean_corr_a1} +/- {se_corr_a1}")

    ax.tick_params(axis="y", colors="black")

    ax.set_xticks([0, 9])
    ax.set_xticklabels([1, 10])
    # Remove x and y axis lines
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Change x and y labels
    ax.set_xlabel("Session number")
    ax.set_ylabel("Probability")

    # remove legend
    ax.get_legend().remove()

    # Add horizontal grid lines
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ylim = [0.45, 1.05]
    ax.set_ylim(ylim)
    plt.grid(axis="y")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


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

    root_dir = Path(findrootdir())
    behv_plot_dir = root_dir / "plots_paper/"
    stats_dir = root_dir / "stats_paper/"
    # make the directory if it does not exist
    if not behv_plot_dir.exists():
        behv_plot_dir.mkdir(parents=True)
    behv_data_dir = root_dir / "human_data" / "human_social"

    # statistics for S2
    df_block = combine_dfs_from_sessions_human(
        behv_data_dir, file_pattern="*info.csv"
    )
    plot_file_name = behv_plot_dir / "FigS2E_combined_block_length.pdf"
    plot_block_length_histogram(plot_file_name, df_block)
    plot_file_name = behv_plot_dir / "FigS2C_combined_block_length_obj.pdf"
    plot_block_length_histogram(plot_file_name, df_block, "pos_in_block_obj")
    df_outcome = combine_dfs_from_sessions_human(behv_data_dir)
    plot_file_name = behv_plot_dir / "FigS2A_combined_outcome_hist.pdf"
    plot_outcome_histogram(plot_file_name, df_outcome)

    # make plot of accuracy by session
    df = combine_dfs_from_sessions_human(
        behv_data_dir, file_pattern="*info.csv"
    )
    file_name = behv_plot_dir / "Fig1I_accuracy_by_session.pdf"
    plot_accuracy_by_session(file_name, df)
    combined_df = combine_dfs_from_sessions_human(behv_data_dir)
    subject = "humans_all"
    plot_confidence_by_congruency(
        behv_plot_dir / f"FigS5C_confidence_by_congruency_{subject}.pdf",
        combined_df,
    )
    df_accuracy = combine_dfs_from_sessions_human(
        behv_data_dir, "*accuracy.csv"
    )
    plot_file_name = behv_plot_dir / "Fig1I_combined_accuracy.pdf"
    plot_accuracy_by_switch(plot_file_name, df_accuracy)

    df_correct = combine_dfs_from_sessions_human(behv_data_dir, "*info.csv")
    plot_file_name = behv_plot_dir / "Fig1I_accuracy_by_session.pdf"
    plot_accuracy_by_session(plot_file_name, df_correct)

    combined_df = combine_dfs_from_sessions_human(behv_data_dir)
    combined_df = add_performance_level(combined_df)
    combined_df = preprocess_for_regression(combined_df, subject=None)
    fig, model = model_switches_mixedlogit(combined_df, human=True)
    fig_file = behv_plot_dir / "FigS3A_switches_regression_mixedlogit.pdf"
    fig.savefig(fig_file, bbox_inches="tight", dpi=300)
    plt.close()
    # save model coefficients to a csv
    model_file = stats_dir / "switches_mixedlogit_model.csv"
    model.to_csv(model_file)
    # plot confidence regression
    fig, model = model_confidence_mixedlinear(combined_df)
    fig_file = behv_plot_dir / "FigS3A_confidence_regression_mixedlinear.pdf"
    fig.savefig(fig_file, bbox_inches="tight", dpi=300)
    plt.close()
    # save model coefficients to a csv
    model_file = stats_dir / "confidence_mixedlinear_model.csv"
    model.to_csv(model_file)

    df_multi = load_human_multi(root_dir)
    pfn = behv_plot_dir / "FigS2G_performance_by_difficulty_humans.pdf"
    plot_performance_by_difficulty(pfn, df_multi)
    combined_df = combine_dfs_from_sessions_human(behv_data_dir)
    combined_df = add_performance_level(combined_df)
    subject = "humans_all"
    plot_slopes_by_session(root_dir, combined_df, subject)
    plot_file_name = behv_plot_dir / "Fig2A_switches_outcome_human.pdf"
    plot_data_outcome(plot_file_name, combined_df, human=True)
    plot_file_name = behv_plot_dir / "Fig2J_switches_actor_observer_human.pdf"
    plot_data_condition(plot_file_name, combined_df, human=True)
    plot_file_name = behv_plot_dir / "Fig2D_combined_switches_early_late.pdf"
    plot_data_early_and_late(plot_file_name, combined_df, "Switches")
    plot_file_name = behv_plot_dir / "Fig2G_combined_switches_gates.pdf"
    plot_data_by_tokens(plot_file_name, combined_df, human=True)

    plot_file_name = behv_plot_dir / "FigS5A_human_early_late_diffchoice.pdf"
    plot_data_early_and_late(plot_file_name, combined_df, "diffchoice")

    subject = "humans_all"
    plot_switch_by_congruency(
        behv_plot_dir / f"FigS5B_switch_by_congruency_{subject}.pdf",
        combined_df,
    )


if __name__ == "__main__":
    main()
