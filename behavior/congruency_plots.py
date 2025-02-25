"""
This script is used to make plots for the confidence and switch data as a
function of congruency. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
from utils.plots import beutify


def plot_confidence_by_congruency(plot_file_name, df, condition="Actor"):
    """
    Plots confidence levels as a function of congruency.

    Generates a plot showing how confidence varies depending on whether the
    choices made by the agents are the same or different, specifically for
    non-rewarded trials.

    Args:
        plot_file_name (str): The file path to save the plot.
        df (pandas.DataFrame): DataFrame containing the data, including
                                'Condition', 'Outcome', 'Switches', 'diffchoice',
                                and 'Confidence' columns.
        condition (str, optional): The condition to filter the data by
                                   (e.g., 'Actor'). Defaults to "Actor".
    """
    fig = plt.figure(figsize=(2, 2))
    ax = plt.subplot(111)
    # get the data for the plot
    df = df[
        (df["Condition"] == condition)
        | (df["Condition"] == f"{condition}-incongruent")
    ]
    df_noreward = df[df["Outcome"] != "1R"]
    df_noswitch = df_noreward[df_noreward["Switches"] == 0]

    # plot the mean and std of confidence for each condition
    # stat for Actor:
    mean_stat = df_noreward.groupby("diffchoice")["Confidence"].mean()
    std_stat = df_noreward.groupby("diffchoice")["Confidence"].std()
    print(mean_stat)
    print(std_stat)

    sns.set_theme(font_scale=1)
    ax = sns.lineplot(
        x="diffchoice",
        y="Confidence",
        data=df_noswitch,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linewidth=0.5,
        marker="o",
        markersize=4,
        color="black",
        linestyle="-",
        legend=False,
    )

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    # save the plot
    plt.grid(False)
    plt.xticks([0, 1])
    ax.set_xticklabels(["Same", "Different"])
    plt.ylabel("Confidence")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)


def plot_switch_by_congruency(plot_file_name, df, condition="Actor"):
    """
    Plots the probability of switching as a function of congruency.

    Generates a plot showing how the probability of switching varies depending
    on whether the choices made by the agents are the same or different,
    specifically for non-rewarded trials.  Also plots objective switches.

    Args:
        plot_file_name (str): The file path to save the plot.
        df (pandas.DataFrame): DataFrame containing the data, including
                                'Condition', 'Outcome', 'Switches', 'diffchoice',
                                and 'Objective_switches' columns.
        condition (str, optional): The condition to filter the data by
                                   (e.g., 'Actor'). Defaults to "Actor".
    """
    fig = plt.figure(figsize=(2, 2))
    ax = plt.subplot(111)
    # get the data for the plot
    df = df[
        (df["Condition"] == condition)
        | (df["Condition"] == f"{condition}-incongruent")
    ]
    df_noreward = df[df["Outcome"] != "1R"]

    sns.set_theme(font_scale=1)
    ax = sns.lineplot(
        x="diffchoice",
        y="Switches",
        data=df_noreward,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linewidth=0.5,
        marker="o",
        markersize=4,
        color="black",
        linestyle="-",
        legend=False,
    )
    ax = sns.lineplot(
        x="diffchoice",
        y="Objective_switches",
        data=df_noreward,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linewidth=0.5,
        marker="*",
        markersize=4,
        color="black",
        linestyle="-",
        legend=False,
    )

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    # save the plot
    plt.grid(False)
    plt.xticks([0, 1])
    plt.yticks([0, 0.5, 1])
    plt.ylim(0, 1)
    ax.set_xticklabels(["Same", "Different"])
    plt.ylabel("P(switch)")
    beutify(ax)
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
