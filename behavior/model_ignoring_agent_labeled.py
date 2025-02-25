"""
Generate observer-ignoring behavior from single player dataset and
compare with social condition data.
"""

# import sys
# import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plots import beutify
from behavior.plot_stat_single_labeled import (
    combine_dfs_from_sessions_single,
)
from behavior.trial_info_utils import combine_dfs_from_sessions
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
            markersize=4,
            linewidth=0.5,
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


def compute_simulated_accuracy(df):
    """
    Computes the simulated accuracy for each trial_after value in the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'accuracy' (float) and 'trial_after' (int).

    Returns:
        pd.DataFrame: A new DataFrame with columns:
                      - 'trials_after': the trial_after value (n)
                      - 'original_accuracy': the accuracy corresponding to trial_after n
                      - 'simulated_accuracy': computed as
                        sum_{k=0}^{n} [ binom(n, k) * (0.5)^n * accuracy(trial_after==k) ]
    """
    # Create a dictionary mapping trial_after to accuracy for quick lookup.
    # This assumes each trial_after appears only once.
    accuracy_dict = df.set_index("trial_after")["accuracy"].to_dict()

    results = []

    # Iterate through the sorted trial_after values (i.e., n = 0, 1, ..., 10)
    for n in range(0, 11):
        sim_acc = 0.0
        # For each k from 0 to n, compute the weighted accuracy
        for k in range(0, n + 1):
            # Ensure we have an accuracy value for trial_after==k
            if k in accuracy_dict:
                weight = math.comb(n, k) * (0.5**n)
                sim_acc += weight * accuracy_dict[k]
            else:
                raise ValueError(
                    f"Missing accuracy value for trial_after = {k}"
                )
        # Get the original accuracy for this n (trial_after==n)
        original_acc = accuracy_dict[n]
        results.append(
            {
                "trial_after": n,
                "original_accuracy": original_acc,
                "accuracy": sim_acc,
            }
        )

    # Convert the list of dictionaries into a DataFrame
    return pd.DataFrame(results)


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
    df_data = combine_dfs_from_sessions_single(
        basedir,
        file_pattern="*accuracy",
        subject=subject,
    )
    df_all = []  # store simulation of each date
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
    rootdir = findrootdir()
    df_L = combine_dfs_from_sessions_single(
        rootdir,
        file_pattern="*accuracy",
        subject="Lalo",
    )
    df_O = combine_dfs_from_sessions_single(
        rootdir,
        file_pattern="*accuracy",
        subject="Offenbach",
    )
    df_simulated_L = compute_model_ignoring_agent(rootdir, subject="Lalo")
    df_simulated_L["player"] = "Lalo"
    df_simulated_O = compute_model_ignoring_agent(rootdir, subject="Offenbach")
    df_simulated_O["player"] = "Offenbach"
    df_simulated = pd.concat(
        [df_simulated_O, df_simulated_L], ignore_index=True
    ).reset_index(drop=True)
    plot_file_name = f"{rootdir}/plots_paper/Fig1K_model_ignoring_agent.pdf"
    plot_accuracy_by_switch(plot_file_name, df_simulated, option="animals")

    # compare against social data
    df_accuracy_social = combine_dfs_from_sessions(rootdir, "*accuracy")
    df_L = df_accuracy_social[df_accuracy_social["player"] == "L"]
    df_O = df_accuracy_social[df_accuracy_social["player"] == "O"]
    df_L = df_L[df_L["trial_after"] >= 0]
    df_O = df_O[df_O["trial_after"] >= 0]
    print(
        f"Lalo Sim: {df_simulated_L.groupby('date')['accuracy'].mean().mean()} +- {df_simulated_L.groupby('date')['accuracy'].mean().std()}"
    )
    print(
        f"Offenbach Sim: {df_simulated_O.groupby('date')['accuracy'].mean().mean()} +- {df_simulated_O.groupby('date')['accuracy'].mean().std()}"
    )
    # perform t-test on the mean accuracy grouped by date for each animal
    mean_sL = df_simulated_L.groupby("date")["accuracy"].mean()
    mean_sO = df_simulated_O.groupby("date")["accuracy"].mean()
    mean_L = df_L.groupby("date")["accuracy"].mean()
    mean_O = df_O.groupby("date")["accuracy"].mean()
    from scipy.stats import ttest_ind

    print(ttest_ind(mean_sL, mean_L))
    print(ttest_ind(mean_sO, mean_O))


if __name__ == "__main__":
    main()
