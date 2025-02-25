"""
Generate observer-ignoring behavior from single player dataset and
compare with social condition data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plots import beutify
from behavior.plot_stat_single_human_labeled import (
    combine_dfs_from_sessions_single_human,
)
from behavior.plot_stat_human_labeled import combine_dfs_from_sessions_human
from behavior.model_ignoring_agent_labeled import compute_simulated_accuracy
from utils.LoadSession import findrootdir


def plot_accuracy_by_switch(plot_file_name, df, option="combined"):
    """
    Plots accuracy over trials after a switch.

    Generates a plot showing accuracy as a function of trials after a switch,
    including a mean plot across all sessions.
    Args:
        plot_file_name (str): The file path to save the plot.
        df (pandas.DataFrame): DataFrame containing 'trial_after' and 'accuracy' columns,
                                and optionally 'player' for individual session data.
        option (str, optional): Plotting option. Defaults to "combined".
    """
    fig = plt.figure(figsize=(6, 6))
    ax = sns.lineplot(
        x="trial_after",
        y="accuracy",
        data=df,
        color="black",
        estimator="mean",
        linestyle="--",
        marker="o",
        markersize=5,
    )
    sns.set_theme(font_scale=1.4)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Change x and y labels
    ax.set_xlabel("N (trials after switch)")
    ax.set_ylabel("Choice accuracy")
    plt.legend(frameon=False)  # Remove the box around the legend
    # Add horizontal grid lines
    ax.set_xticks([-4, 0, 4, 8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.grid(visible=True, which="major", axis="y", color="0.9", linestyle="-")
    # Customize the figure and axes to have a transparent background
    fig.patch.set_facecolor("none")
    ax.patch.set_facecolor("none")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


def compute_model_ignoring_agent(basedir, subject="Lalo"):
    """
    Computes the simulated accuracy for an agent ignoring the other player's actions.

    This function calculates the accuracy based on the agent's own history,
    simulating a scenario where the agent disregards the other player's behavior.

    Args:
        basedir (str): Base directory containing session data.
        subject (str, optional): Subject identifier. Defaults to "Lalo".

    Returns:
        pandas.DataFrame: DataFrame containing simulated accuracy data.
    """
    df_data = combine_dfs_from_sessions_single_human(
        basedir,
        file_pattern="*accuracy.csv",
        sub=subject,
    )
    df_all = []
    for date in df_data["date"].unique():
        df_date = df_data[df_data["date"] == date]
        # make sure df is sorted by trial_after
        df_date = df_date.sort_values("trial_after")
        df_simulated = compute_simulated_accuracy(df_date)
        df_simulated["date"] = date
        df_all.append(df_simulated)
    df_all = pd.concat(df_all, ignore_index=True)
    return df_all


def main():
    root_dir = findrootdir()
    root_dir = Path(root_dir)
    behv_data_dir = root_dir / "human_data" / "human_single"
    behv_plot_dir = root_dir / "plots_paper"
    subjects = [d.stem for d in behv_data_dir.iterdir() if d.is_dir()]
    df_sim_all = []
    for subject in subjects:
        df_simulated = compute_model_ignoring_agent(
            behv_data_dir, subject=subject
        )
        df_simulated["player"] = subject
        df = combine_dfs_from_sessions_single_human(
            behv_data_dir,
            file_pattern="*accuracy.csv",
            sub=subject,
        )
        df["player"] = subject
        df_sim_all.append(df_simulated)
    df_simulated = pd.concat(df_sim_all, ignore_index=True)

    plot_file_name = behv_plot_dir / "Fig1I_model_ignoring_agent.pdf"
    plot_accuracy_by_switch(plot_file_name, df_simulated, option="combined")
    behv_data_dir_social = root_dir / "human_data" / "human_social"
    df_social = combine_dfs_from_sessions_human(
        behv_data_dir_social, "*accuracy.csv"
    )
    df_simulated = df_simulated[df_simulated["trial_after"] >= 0]
    df_social = df_social[df_social["trial_after"] >= 0]
    from scipy.stats import ttest_ind

    mean_sim = df_simulated.groupby("date")["accuracy"].mean()
    print(f"Simulated human Mean: {mean_sim.mean()}, SD: {mean_sim.std()}")
    mean_sub = df_social.groupby("date")["accuracy"].mean()
    print(ttest_ind(mean_sim, mean_sub))


if __name__ == "__main__":
    main()
