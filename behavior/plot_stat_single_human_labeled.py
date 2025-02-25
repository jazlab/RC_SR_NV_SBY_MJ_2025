"""
This script makes behavior plots for the paper, for the single player human task.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from utils.plots import beutify
from utils.get_session_info import findrootdir


def plot_accuracy_by_session(plot_file_name, df):
    """
    Plots the probability of correct responses over sessions.

    Args:
        plot_file_name (str or Path): Path to save the plot.
        df (pd.DataFrame): DataFrame containing session-wise performance data.

    Returns:
        None
    """

    # map value of correct_a0 and correct_a1 from 1,-1 to 1,0
    df["correct_a0"] = df["correct_a0"].map({1: 1, -1: 0, 0: 0})
    df["correct_a1"] = df["correct_a1"].map({1: 1, -1: 0, 0: 0})
    fig = plt.figure(figsize=(9, 9), facecolor="white")

    ax = sns.lineplot(
        x="session_number",
        y="correct_a0",
        hue="subject",
        data=df,
        estimator="mean",
        err_style="band",
        linestyle="-",
    )
    # remove the legend
    ax.get_legend().remove()

    # print the mean and standard error of the mean of correct_a0 and a1
    mean_corr_a0 = df.groupby("session_number")["correct_a0"].mean().mean()
    se_corr_a0 = df.groupby("session_number")["correct_a0"].mean().std() / 6
    mean_corr_a1 = df.groupby("session_number")["correct_a1"].mean().mean()
    se_corr_a1 = df.groupby("session_number")["correct_a1"].mean().std() / 6
    print(f"Mean correct_a0: {mean_corr_a0} +/- {se_corr_a0}")
    print(f"Mean correct_a1: {mean_corr_a1} +/- {se_corr_a1}")

    ax.tick_params(axis="y", colors="black")

    # add a legend of 'Accuracy' in black and 'Reward' in green
    ax.set_xticks([0, 4])
    ax.set_xticklabels([1, 5])
    # Remove x and y axis lines
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Change x and y labels
    ax.set_xlabel("Session number")
    ax.set_ylabel("Probability")

    # Add horizontal grid lines
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ylim = [0.5, 0.9]
    ax.set_ylim(ylim)
    plt.grid(axis="y")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


def combine_dfs_from_sessions_single_human(
    behv_data_dir, file_pattern="*switches.csv", sub=None
):
    """
    Combines session-level data from multiple human experiment files into a single DataFrame.

    Args:
        behv_data_dir (Path): Directory containing behavioral data files.
        file_pattern (str, optional): File name pattern to filter data. Defaults to "*switches.csv".

    Returns:
        pd.DataFrame: Combined DataFrame with processed session data.
    """
    subjects = [d for d in behv_data_dir.iterdir() if d.is_dir()]
    dfs = []
    session_numbers = {}
    for subject in subjects:
        if sub is not None:
            if subject.stem != sub:
                continue
        subj_csv_path = subject / "ski-resort_csv"
        csv_files = list(subj_csv_path.glob(file_pattern))
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
            if file_pattern == "*switches.csv":
                subject = df["subject"][0]
                df["date"] = f"{year}_{month}_{day}_{subject}"
                dfs.append(df)
            elif file_pattern == "*info.csv":
                # add a session number term that starts at 1 for each subject
                if subject not in session_numbers:
                    session_numbers[subject] = 0
                else:
                    session_numbers[subject] += 1
                session_number = session_numbers[subject]
                df["session_number"] = session_number
                df["date"] = f"{year}_{month}_{day}_{subject}"
                df["subject"] = subject
                df["player"] = subject
                # add subject field based on the parent directory name
                dfs.append(df)
            elif file_pattern == "*accuracy.csv":
                dfs.append(df)
    # concatenate list of dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    # if incongruent is in condition, then set diffchoice to 1
    # if file_pattern == "*info.csv":
    #     combined_df["Pos_in_block"] = combined_df["pos_in_block"]
    return combined_df


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

    # print the mean and std of accuracy overall for each "player"
    print(df.groupby("player")["accuracy"].mean())

    fig = plt.figure(figsize=(6, 6))
    ax = sns.lineplot(
        x="trial_after",
        y="accuracy",
        data=df,
        color="black",
        estimator="mean",
        marker="o",
        markersize=5,
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
    root_dir = findrootdir()
    root_dir = Path(root_dir)
    behv_plot_dir = root_dir / "plots_paper"
    # creat plot dir if not present
    if not behv_plot_dir.exists():
        behv_plot_dir.mkdir(parents=True, exist_ok=True)
    behv_data_dir = root_dir / "human_data" / "human_single"
    # update_and_write_switches(behv_data_dir)
    df_accuracy = combine_dfs_from_sessions_single_human(
        behv_data_dir, file_pattern="*accuracy.csv"
    )
    plot_file_name = f"{behv_plot_dir}/Fig1H_combined_accuracy_single.pdf"
    plot_accuracy_by_switch(plot_file_name, df_accuracy, option="combined")

    df_correct = combine_dfs_from_sessions_single_human(
        behv_data_dir, file_pattern="*info.csv"
    )
    df_correct["correct_a0"] = df_correct["correct_a0"].map({1: 1, -1: 0, 0: 0})
    mean_corr = df_correct.groupby("date")["correct_a0"].mean().mean()
    sd_corr = df_correct.groupby("date")["correct_a0"].mean().std()
    n_sessions = len(df_correct["date"].unique())
    print(
        f"L Mean correct from {n_sessions} sessions: {mean_corr} +/- {sd_corr}"
    )
    df_last_session = df_correct[df_correct["session_number"] == 4]
    mean_corr_last = df_last_session["correct_a0"].mean()
    sd_corr_last = df_last_session["correct_a0"].std()
    print(f"Last session correct: {mean_corr_last} +/- {sd_corr_last}")
    file_name = f"{behv_plot_dir}/Fig1H_accuracy_by_session_single.pdf"
    plot_accuracy_by_session(file_name, df_correct)


if __name__ == "__main__":
    main()
