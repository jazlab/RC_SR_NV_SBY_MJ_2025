"""
Plot the probability of switching as a function of the outcome and 
player identity for one session, based on trial_info.csv
"""

from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plots import beutify

from copy import deepcopy


def common_plot_function(
    file_name,
    data,
    x,
    y,
    hue,
    hue_order,
    style_order,
    xlabel,
    ylabel,
    ylim,
    show_objective=True,
    data_transformation_function=None,
    additional_plot_params=None,
):
    # Set up the figure
    figsize = (2, 2)
    plt.figure(figsize=figsize, facecolor="white")

    # Apply any data transformations if provided
    if data_transformation_function:
        data = data_transformation_function(data)

    # Plotting: separate into two plots based on hue
    if hue is None:
        data_actor = data[data["Condition"].str.contains("Actor")]
        data_observer = data[data["Condition"].str.contains("Observer")]
        # unless it's the condition plot
        if x == "Condition":
            data_actor = data
    else:
        data_actor = data[data[hue] == hue_order[0]]
        data_observer = data[data[hue] == hue_order[1]]
    # print the mean and std of plot_field for each condition
    print(file_name)
    for condition, group in data.groupby("Condition"):
        for value in group[x].unique():
            date_means = group[group[x] == value].groupby("date")[y].mean()
            print(
                f"{condition} - {x}={value}: {date_means.mean()} +/- {date_means.std()}"
            )

    if hue is not None and len(hue_order) > 2:
        data_actor_incongruent = data[data[hue] == hue_order[2]]
    # first plot: actor (just the error bar)
    ax = sns.lineplot(
        x=x,
        y=y,
        data=data_actor,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linewidth=0.5,
        color="black",
        linestyle="-",  # Keep the linestyle as simple lines
        legend=False,
        **(additional_plot_params if additional_plot_params else {}),
    )
    # Calculate the mean of y for each x value
    grouped_actor = data_actor.groupby(x)[y].mean().reset_index()
    grouped_observer = data_observer.groupby(x)[y].mean().reset_index()

    # compare the mean of each group of x (within actor) with t-test
    for i in range(len(grouped_actor)):
        for j in range(i + 1, len(grouped_actor)):
            if grouped_actor[x][i] == grouped_actor[x][j]:
                continue
            t, p = ttest_ind(
                data_actor[data_actor[x] == grouped_actor[x][i]][y],
                data_actor[data_actor[x] == grouped_actor[x][j]][y],
            )
            print(
                f"Actor: {grouped_actor[x][i]} vs {grouped_actor[x][j]}: t={t}, p={p}"
            )

    # Plot the means as filled and open circles
    plt.plot(
        grouped_actor[x], grouped_actor[y], "o", markersize=4, color="black"
    )
    plt.plot(
        grouped_observer[x],
        grouped_observer[y],
        "o",
        markersize=4,
        color="black",
        markerfacecolor="none",
    )
    # second plot: observer
    ax = sns.lineplot(
        x=x,
        y=y,
        data=data_observer,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linewidth=0.5,
        markers=True,
        markersize=4,
        fillstyle="none",
        color="black",
        linestyle="--",  # Keep the linestyle as simple lines
        legend=False,
        dashes=False,
        **(additional_plot_params if additional_plot_params else {}),
    )
    # compare the mean of each group of x (within observer) with t-test
    for i in range(len(grouped_observer)):
        for j in range(i + 1, len(grouped_observer)):
            if grouped_observer[x][i] == grouped_observer[x][j]:
                continue
            t, p = ttest_ind(
                data_observer[data_observer[x] == grouped_observer[x][i]][y],
                data_observer[data_observer[x] == grouped_observer[x][j]][y],
            )
            print(
                f"Observer: {grouped_observer[x][i]} vs {grouped_observer[x][j]}: t={t}, p={p}"
            )

    if hue is not None and len(hue_order) > 2:
        # add a plot for actor incongruent
        ax = sns.lineplot(
            x=x,
            y=y,
            hue=hue,
            data=data_actor_incongruent,
            estimator="mean",
            errorbar=("ci", 95),
            err_style="bars",
            linewidth=0.5,
            markers=True,
            markersize=4,
            palette=["red"],
            linestyle="-",  # Keep the linestyle as simple lines
            legend=False,
            dashes=False,
            **(additional_plot_params if additional_plot_params else {}),
        )
    if show_objective:
        # add a plot for objective_switch
        ax = sns.lineplot(
            x=x,
            y="Objective_switches",
            data=data_actor,
            estimator="mean",
            errorbar=("ci", 95),
            err_style="bars",
            marker="*",
            linewidth=0.5,
            markersize=4,
            legend=False,
            color="black",
            linestyle="-",
            **(additional_plot_params if additional_plot_params else {}),
        )
        ax = sns.lineplot(
            x=x,
            y="Objective_switches",
            data=data_observer,
            estimator="mean",
            errorbar=("ci", 95),
            err_style="bars",
            linewidth=0.5,
            marker="*",
            markersize=4,
            legend=False,
            color="black",
            linestyle="--",
            **(additional_plot_params if additional_plot_params else {}),
        )

    # add control attention to the plot
    if y == "Attention":
        ax = sns.lineplot(
            x=x,
            y="Attention_control",
            data=data_actor,
            estimator="mean",
            errorbar=("ci", 95),
            err_style="bars",
            linewidth=0.5,
            markers=True,
            markersize=4,
            color="red",
            linestyle="-",  # Keep the linestyle as simple lines
            legend=False,
            **(additional_plot_params if additional_plot_params else {}),
        )
        ax = sns.lineplot(
            x=x,
            y="Attention_control",
            data=data_observer,
            estimator="mean",
            errorbar=("ci", 95),
            err_style="bars",
            linewidth=0.5,
            markers=True,
            markersize=4,
            color="red",
            linestyle="--",  # Keep the linestyle as simple lines
            legend=False,
            **(additional_plot_params if additional_plot_params else {}),
        )

    # Set axis labels and limits
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    ax.set_yticks(np.arange(0, ylim[1], 0.1))

    # Customize the aesthetics of the plot
    sns.set_theme(font_scale=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_facecolor("white")
    ax.grid(visible=True, which="major", axis="y", color="0.9", linestyle="-")
    beutify(ax)
    # Save the plot
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_distribution_attention(file_name, P_switch_df, option="congruent"):
    if option == "congruent":
        conditions = ["Actor", "Observer"]
    elif option == "incongruent":
        conditions = ["Actor-incongruent", "Observer-incongruent"]
    df = deepcopy(P_switch_df)
    df = df[df["Outcome"] != "1R"]
    df = df[df["Condition"].isin(conditions)]
    bins = np.linspace(0, 1, 11)
    # histogram of attention from each session
    df_histogram = pd.DataFrame(
        columns=[
            "date",
            "Condition",
            "edge",
            "probability",
            "probability_control",
        ]
    )
    for i, date in enumerate(df["date"].unique()):
        df_date = df[df["date"] == date]
        for condition in df_date["Condition"].unique():
            df_condition = df_date[df_date["Condition"] == condition]
            hist, _ = np.histogram(df_condition["Attention"], bins=bins)
            hist = hist / np.sum(hist)
            hist_control, _ = np.histogram(
                df_condition["Attention_control"], bins=bins
            )
            hist_control = hist_control / np.sum(hist_control)
            df_hist = pd.DataFrame(
                {
                    "date": date,
                    "Condition": condition,
                    "edge": bins[:-1],
                    "probability": hist,
                    "probability_control": hist_control,
                }
            )
            df_histogram = pd.concat([df_histogram, df_hist], ignore_index=True)
    # plot the mean and std of the histograms
    fig = plt.figure(figsize=(2, 2))
    ax = sns.lineplot(
        x="edge",
        y="probability",
        data=df_histogram,
        hue="Condition",
        hue_order=conditions,
        palette=["black", "black"],
        style="Condition",
        style_order=conditions,
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
    )
    # remove background
    ax.set_facecolor("white")
    # set x ticks to 0.1
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    # set y ticks to 0.1
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    beutify(ax)

    # Add labels, title, and legend
    plt.xlabel("Value")
    plt.ylabel("Probability")
    # save the plot
    plt.savefig(file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_data_by_attention_lvl(
    file_name,
    P_switch_df,
    plot_field="Switches",
    pib="all",
    option="congruent",
    levels=2,
):
    def data_transformation_function(data):
        data = data.copy()
        data.loc[:, "attention_level"] = "Low"
        data.loc[data["Attention"] > 0.9, "attention_level"] = "High"
        if plot_field == "Corrects":
            data["Corrects"] = [1 if x == 1 else 0 for x in data["Corrects"]]
        data["attention"] = pd.Categorical(
            data["attention_level"], categories=["Low", "High"], ordered=True
        )
        if levels > 2:
            # set attention to n levels from 0.1 to 1.0
            data["attention"] = pd.cut(
                data["Attention"],
                bins=np.linspace(0, 1, levels + 1),
                labels=[f"{i/10:.1f}" for i in range(1, levels + 1)],
            )
        data = data[data["Outcome"] != "1R"]
        if pib == "late":
            data = data[data["Pos_in_block"] >= 10]
        if pib == "mid":
            data = data[
                (data["Pos_in_block"] > 5) & (data["Pos_in_block"] < 10)
            ]
        if pib == "early":
            data = data[data["Pos_in_block"] <= 5]
        return data

    conditions = ["Actor", "Observer"]
    if option == "incongruent":
        conditions = ["Actor-incongruent", "Observer-incongruent"]
    common_plot_function(
        file_name=file_name,
        data=P_switch_df,
        x="attention",
        y=plot_field,
        hue="Condition",
        hue_order=conditions,
        style_order=conditions,
        xlabel="Attention",
        ylabel=plot_field,
        ylim=[-0.1, 1.1],
        data_transformation_function=data_transformation_function,
        show_objective=False,
    )


def plot_data_early_and_late(
    file_name, P_switch_df, plot_field="Switches", separate=False
):
    def data_transformation_function(data):
        data = data.copy()
        data.loc[:, "Early_or_late"] = "Early"
        data.loc[data["Pos_in_block"] > 5, "Early_or_late"] = "Mid"
        data.loc[data["Pos_in_block"] > 10, "Early_or_late"] = "Late"
        data["Early_or_late"] = pd.Categorical(
            data["Early_or_late"],
            categories=["Early", "Mid", "Late"],
            ordered=True,
        )
        return data[data["Outcome"] != "1R"]

    show_objective = True
    hue = "Condition"
    hue_order = ["Actor", "Observer"]
    style_order = ["Actor", "Observer"]
    ylabel = "P(switch)"
    ylim = [-0.1, 1.1]
    if separate:
        hue = "Outcome"
        hue_order = ["1NR", "2NR"]
        style_order = ["1NR", "2NR"]
    else:
        hue = None
        hue_order = None
        style_order = None
    if plot_field == "diffchoice":
        show_objective = False
        ylabel = "P(different)"
        ylim = [-0.05, 0.55]
    if plot_field == "Attention":
        show_objective = False
        ylabel = "Attention"
        ylim = [0, 1]
    common_plot_function(
        file_name=file_name,
        data=P_switch_df,
        x="Early_or_late",
        y=plot_field,
        hue=hue,
        hue_order=hue_order,
        style_order=style_order,
        xlabel="Position in block",
        ylabel=ylabel,
        ylim=ylim,
        show_objective=show_objective,
        data_transformation_function=data_transformation_function,
    )


def plot_data_by_tokens(
    file_name, P_switch_df, late_trials_only=False, human=False
):
    def data_transformation_function(data):
        data = data.copy()
        if late_trials_only:
            data = data[data["Pos_in_block"] > 5]
        data["performance_level"] = pd.Categorical(
            data["performance_level"],
            categories=["Low", "Mid", "High"],
            ordered=True,
        )
        return data[data["Outcome"] != "1R"]

    common_plot_function(
        file_name=file_name,
        data=P_switch_df,
        x="performance_level",
        y="Switches",
        hue="Condition",
        hue_order=["Actor", "Observer"],
        style_order=["Actor", "Observer"],
        xlabel="N(captured)",
        ylabel="P(switch)",
        ylim=[-0.1, 1.1],
        data_transformation_function=data_transformation_function,
    )


def plot_data_outcome(
    file_name, P_switch_df, late_trials_only=False, human=False
):
    def data_transformation_function(data):
        if late_trials_only:
            data = data[data["Pos_in_block"] > 5]
        if human:
            return data[data["Outcome"].isin(["1R", "1NR", "2NR", "3NR"])]
        else:
            return data[data["Outcome"].isin(["1R", "1NR", "2NR"])]

    common_plot_function(
        file_name=file_name,
        data=P_switch_df,
        x="Outcome",
        y="Switches",
        hue="Condition",
        hue_order=[
            "Actor",
            "Observer",
        ],
        style_order=[
            "Actor",
            "Observer",
        ],
        xlabel="Outcome",
        ylabel="P(switch)",
        ylim=[-0.1, 1.1],
        data_transformation_function=data_transformation_function,
    )


def plot_data_condition(
    file_name,
    P_switch_df,
    plot_field="Switches",
    late_trials_only=False,
    human=False,
    option="congruent",
):
    conditions = ["Actor", "Observer"]
    if option == "incongruent":
        conditions = ["Actor-incongruent", "Observer-incongruent"]

    def data_transformation_function(data):
        data = data.dropna(subset=[plot_field])
        if late_trials_only:
            data = data[data["Pos_in_block"] > 5]
        data = data[data["Condition"].isin(conditions)]
        return data[data["Outcome"] != "1R"]

    if plot_field == "Switches":
        show_objective = True
    else:
        show_objective = False

    # remove nan values for the plot_field; first, make a copy
    P_switch_df = deepcopy(P_switch_df)
    P_switch_df = P_switch_df.dropna(subset=[plot_field])
    P_switch_df = P_switch_df[P_switch_df["Outcome"] != "1R"]
    P_switch_df = P_switch_df[P_switch_df["Condition"].isin(conditions)]
    # t-test on the mean of each condition
    grouped = P_switch_df.groupby("Condition")[plot_field].mean().reset_index()
    std_grouped = (
        P_switch_df.groupby("Condition")[plot_field].std().reset_index()
    )
    # print mean and std for each condition
    print(grouped)
    print(std_grouped)
    for i in range(len(grouped)):
        for j in range(i + 1, len(grouped)):
            t, p = ttest_ind(
                P_switch_df[
                    P_switch_df["Condition"] == grouped["Condition"][i]
                ][plot_field],
                P_switch_df[
                    P_switch_df["Condition"] == grouped["Condition"][j]
                ][plot_field],
            )
            print(
                f"{grouped['Condition'][i]} vs {grouped['Condition'][j]}: t={t}, p={p}"
            )

    common_plot_function(
        file_name=file_name,
        data=P_switch_df,
        x="Condition",
        y=plot_field,
        hue=None,
        hue_order=None,
        style_order=None,
        xlabel="Condition",
        ylabel=plot_field,
        ylim=[0.0, 0.6],
        data_transformation_function=data_transformation_function,
        show_objective=show_objective,
    )


from scipy.stats import bootstrap


def bootstrap_ci(data, n_resamples=1000, ci=0.95):
    res = bootstrap(
        (data,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=ci,
        random_state=42,
    )
    return res.confidence_interval.low, res.confidence_interval.high


def plot_attention_stacked(file_name, P_switch_df, option="congruent"):
    # plot attention and attention_control stacked bar plot for each condition
    conditions = ["Actor", "Observer"]
    if option == "incongruent":
        conditions = ["Actor-incongruent", "Observer-incongruent"]
    df = deepcopy(P_switch_df)
    df = df[df["Outcome"] != "1R"]
    df = df[df["Condition"].isin(conditions)]
    df = df.dropna(subset=["Attention", "Attention_control"])
    df["total_attention"] = df["Attention"] + df["Attention_control"]
    group_stats = df.groupby("Condition")[["Attention", "total_attention"]].agg(
        ["mean", "std"]
    )
    means = group_stats.xs("mean", level=1, axis=1)
    stds = group_stats.xs("std", level=1, axis=1)
    # print mean and std for each condition
    print(file_name)
    print(means)
    print(stds)
    # bootstrap plot
    bootstrap_means = {"Attention": [], "sum": []}
    bootstrap_cis = {"Attention": [], "sum": []}
    for condition, group in df.groupby("Condition"):
        # Attention
        low, high = bootstrap_ci(group["Attention"].values)
        bootstrap_means["Attention"].append(group["Attention"].mean())
        bootstrap_cis["Attention"].append((low, high))

        # Sum
        low, high = bootstrap_ci(group["total_attention"].values)
        bootstrap_means["sum"].append(group["total_attention"].mean())
        bootstrap_cis["sum"].append((low, high))
    # Extract means and error bars from bootstrap results
    conditions = df["Condition"].unique()
    attention_means = bootstrap_means["Attention"]
    attention_err = [
        (mean - low, high - mean)
        for mean, (low, high) in zip(
            attention_means, bootstrap_cis["Attention"]
        )
    ]

    sum_means = bootstrap_means["sum"]
    sum_err = [
        (mean - low, high - mean)
        for mean, (low, high) in zip(sum_means, bootstrap_cis["sum"])
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(2, 2))
    bar_width = 0.4
    x = np.arange(len(conditions))
    # Bars for the sum with bootstrap CI
    sum_err_low, sum_err_high = zip(*sum_err)
    ax.bar(
        x,
        sum_means,
        yerr=[sum_err_low, sum_err_high],
        capsize=5,
        width=bar_width,
        edgecolor="black",
        color="#BCBEC0",
    )
    # Bars for attention with bootstrap CI
    attention_err_low, attention_err_high = zip(*attention_err)
    ax.bar(
        x,
        attention_means,
        yerr=[attention_err_low, attention_err_high],
        capsize=5,
        width=bar_width,
        edgecolor="black",
        color="#808285",
    )

    # Customizing the plot
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.05)  # Slightly above 1 to accommodate error bars
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    beutify(ax)

    plt.savefig(file_name, bbox_inches="tight", dpi=300)
