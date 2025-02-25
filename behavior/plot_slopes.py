"""
make plots of slopes for different factors and regression coefficients
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from utils.plots import beutify
from behavior.trial_info_utils import calculate_regression_coefficients


def plot_slopes_by_session(root_dir, df, subject):
    # for each df (session) calculate the slope of regression for switches

    unique_sessions = df["date"].unique()
    slopes_actor = []
    slopes_observer = []
    slopes_actor_pos = []
    slopes_observer_pos = []
    slopes_actor_ngates = []
    slopes_observer_ngates = []
    slopes_actor_diffchoice = []
    slopes_condition = []
    for session in unique_sessions:
        df_session = df[df["date"] == session]
        df_session.loc[:, "Early_or_late"] = 0
        df_session.loc[df_session["Pos_in_block"] > 5, "Early_or_late"] = 1
        df_session.loc[df_session["Pos_in_block"] > 10, "Early_or_late"] = 2
        df_session.loc[:, "Less_or_more"] = 1
        df_session.loc[
            df_session["Performance_numerical"] < 9, "Less_or_more"
        ] = 0
        df_session.loc[
            df_session["Performance_numerical"] > 9, "Less_or_more"
        ] = 2
        for condition in ["Actor", "Observer", "Both", "Actor-diffchoice"]:
            if condition == "Actor-diffchoice":
                data_condition = df_session[
                    (df_session["Condition"] == "Actor")
                    | (df_session["Condition"] == "Actor-incongruent")
                ]
            elif condition == "Actor":
                data_condition = df_session[
                    df_session["Condition"] == condition
                ]
            elif condition == "Observer":
                data_condition = df_session[
                    df_session["Condition"] == condition
                ]
            elif condition == "Both":
                data_condition = df_session[
                    (df_session["Condition"] == "Actor")
                    | (df_session["Condition"] == "Observer")
                ]
            # Slope for Outcome
            x_values = data_condition["Outcome"]
            # map x_values to integers
            x_values = x_values.map(
                {"1R": 0, "1NR": 1, "2NR": 2, "3NR": 3, "4NR": 4}
            )
            y_values = data_condition["Switches"].values
            idx_nan = np.isnan(x_values)
            x_values = x_values[~idx_nan].values
            y_values = y_values[~idx_nan]
            # use logistic regression to find the slope for outcome
            x = np.vstack([x_values, np.ones(len(x_values))]).T
            model = LogisticRegression()
            model.fit(x, y_values)
            slope = model.coef_[0][0]
            idx_nr = data_condition["Outcome"] != "1R"

            # select nonreward trials only for other variables
            data_condition = data_condition[idx_nr]
            # Slope for position
            y_values = data_condition["Switches"].values
            x = np.vstack(
                [
                    data_condition["Pos_in_block"].values,
                    np.ones(len(data_condition["Pos_in_block"])),
                ]
            ).T
            model = LogisticRegression()
            model.fit(x, y_values)
            slope_pos = model.coef_[0][0]

            # Slope for ngates
            x = np.vstack(
                [
                    data_condition["Performance_numerical"],
                    np.ones(len(data_condition["Performance_numerical"])),
                ]
            ).T
            y_values = data_condition["Switches"].values
            model = LogisticRegression()
            model.fit(x, y_values)
            slope_ngates = model.coef_[0][0]

            # Slope for diffchoice
            if condition == "Actor-diffchoice":
                x = np.vstack(
                    [
                        data_condition["diffchoice"],
                        np.ones(len(data_condition["diffchoice"])),
                    ]
                ).T
                y_values = data_condition["Switches"].values
                model = LogisticRegression()
                model.fit(x, y_values)
                slope_diffchoice = model.coef_[0][0]

            # Slope for condition (actor or observer)
            if condition == "Both":
                x = np.vstack(
                    [
                        data_condition["Condition"].map(
                            {"Actor": 0, "Observer": 1}
                        ),
                        np.ones(len(data_condition["Condition"])),
                    ]
                ).T
                y_values = data_condition["Switches"].values
                model = LogisticRegression()
                model.fit(x, y_values)
                slope_condition = model.coef_[0][0]
                slopes_condition.append(slope_condition)

            if condition == "Actor":
                slopes_actor.append(slope)
                slopes_actor_pos.append(slope_pos)
                slopes_actor_ngates.append(slope_ngates)
            elif condition == "Observer":
                slopes_observer.append(slope)
                slopes_observer_pos.append(slope_pos)
                slopes_observer_ngates.append(slope_ngates)
            elif condition == "Actor-diffchoice":
                slopes_actor_diffchoice.append(slope_diffchoice)
    # build regression model for the animal separately for actor and observer
    coefs_actor, coefs_observer, coefs_both, coefs_actor_diffchoice = (
        calculate_regression_coefficients(subject, df)
    )

    # make plot for individual animals only
    if subject == "Both":
        return

    # Call the function for each factor
    plot_slopes(
        root_dir,
        subject,
        slopes_actor,
        slopes_observer,
        coefs_actor,
        coefs_observer,
        "Outcome",
    )
    plot_slopes(
        root_dir,
        subject,
        slopes_actor_pos,
        slopes_observer_pos,
        coefs_actor,
        coefs_observer,
        "Pos_in_block",
    )
    plot_slopes(
        root_dir,
        subject,
        slopes_actor_ngates,
        slopes_observer_ngates,
        coefs_actor,
        coefs_observer,
        "Performance_numerical",
    )
    plot_slopes(
        root_dir,
        subject,
        slopes_condition,
        None,
        coefs_both,
        None,
        "Condition",
    )
    plot_slopes(
        root_dir,
        subject,
        slopes_actor_diffchoice,
        None,
        coefs_actor_diffchoice,
        None,
        "diffchoice",
    )


def plot_slopes_only(
    root_dir,
    subject,
    slopes_actor,
    slopes_observer,
    factor_name,
):
    ylims = {
        "EvDir_fdbk": [-0.5, 3],
    }
    # get p value for slopes
    _, p_actor = stats.ttest_1samp(slopes_actor, 0)
    if slopes_observer is not None:
        _, p_observer = stats.ttest_1samp(slopes_observer, 0)
        # paired t-test between slopes_actor and slopes_observer
        _, p_paired = stats.ttest_rel(slopes_actor, slopes_observer)
    if subject == "O":
        c = "green"
    elif subject == "L":
        c = "purple"
    else:
        c = "black"
    fig = plt.figure(figsize=(5, 5), facecolor="white")
    ax = plt.subplot(111)
    n_sessions = len(slopes_actor)
    rng = np.random.default_rng(67890)  # reproducible
    random_shift = rng.uniform(-0.05, 0.05, n_sessions)
    ax.scatter(
        [1] * n_sessions + random_shift,
        slopes_actor,
        color=c,
        label="Actor",
    )
    # add stars for p value
    ypos = ylims[factor_name][0] + 0.95 * (
        ylims[factor_name][1] - ylims[factor_name][0]
    )
    significance_label = (
        "***"
        if p_actor < 0.001
        else "**" if p_actor < 0.01 else "*" if p_actor < 0.05 else ""
    )
    if p_actor < 0.05:
        ax.text(1, ypos, significance_label, fontsize=15, color="black")

    if slopes_observer is not None:
        ax.scatter(
            [1.2] * n_sessions + random_shift,
            slopes_observer,
            color=c,
            facecolors="none",
            edgecolors=c,
            label="Observer",
        )
        # add stars for p value
        ypos = ylims[factor_name][0] + 0.95 * (
            ylims[factor_name][1] - ylims[factor_name][0]
        )
        significance_label = (
            "***"
            if p_observer < 0.001
            else (
                "**" if p_observer < 0.01 else "*" if p_observer < 0.05 else ""
            )
        )
        if p_observer < 0.05:
            ax.text(
                1.2,
                ypos,
                significance_label,
                fontsize=15,
                color="black",
            )
        significance_label_paired = (
            "***"
            if p_paired < 0.001
            else ("**" if p_paired < 0.01 else "*" if p_paired < 0.05 else "")
        )
        if p_paired < 0.05:
            ax.text(
                1.1,
                ypos + 0.1,
                significance_label_paired,
                fontsize=15,
                color="black",
            )

    ax.set_ylim(ylims[factor_name])
    plt.grid(False)
    plt.xticks([1, 1.2], ["Actor", "Observer"])
    beutify(ax)
    plt.xlabel("Condition")
    plt.ylabel("Slope")
    plt.xlim(0.8, 1.4)
    figname = {
        "both": "Fig4D",
        "O": "FigS7E",
        "L": "FigS7F",
    }
    plot_file_name = (
        root_dir
        / "plots_paper"
        / f"{figname[subject]}_{subject}_slopes_{factor_name}.pdf"
    )
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()


def plot_slopes(
    root_dir,
    subject,
    slopes_actor,
    slopes_observer,
    coefs_actor,
    coefs_observer,
    factor_name,
):
    ylims = {
        "Performance_numerical": [-0.4, 1.2],
        "Outcome": [0, 5],
        "Pos_in_block": [-0.1, 0.6],
        "diffchoice": [-1, 4],
        "EvDir_fdbk": [-0.5, 2.5],
        "EvDir_prechoice": [-0.5, 2.5],
        "Condition": [-2, 1],
    }
    ylims2 = {
        "Performance_numerical": [-0.15, 0.45],
        "Outcome": [0, 2],
        "Pos_in_block": [0.08, 0.22],
        "diffchoice": [-0.2, 2.5],
        "EvDir_fdbk": [-0.5, 2.5],
        "EvDir_prechoice": [-0.5, 2.5],
        "Condition": [-2, 1],
    }
    # print value of coef for actor and observer
    beta_actor = coefs_actor[factor_name][0]
    ci_actor = coefs_actor[factor_name + "_ci"][0]
    print(
        f"{factor_name} beta value for actor: {beta_actor} with CI: {ci_actor}"
    )
    if coefs_observer is not None:
        beta_observer = coefs_observer[factor_name][0]
        ci_observer = coefs_observer[factor_name + "_ci"][0]
        print(
            f"{factor_name} beta value for observer: {beta_observer} with CI: {ci_observer}"
        )
    # get p value for slopes
    _, p_actor = stats.ttest_1samp(slopes_actor, 0)
    if slopes_observer is not None:
        _, p_observer = stats.ttest_1samp(slopes_observer, 0)
    if subject == "Offenbach":
        c = "green"
    elif subject == "Lalo":
        c = "purple"
    else:
        c = "black"
    fig = plt.figure(figsize=(5, 5), facecolor="white")
    ax = plt.subplot(111)
    n_sessions = len(slopes_actor)
    rng = np.random.default_rng(12345)  # for reproducibility
    random_shift = rng.uniform(-0.05, 0.05, n_sessions)
    ax.scatter(
        [1] * n_sessions + random_shift,
        slopes_actor,
        color=c,
        label="Actor",
    )
    # add stars for p value
    ypos = ylims[factor_name][0] + 0.95 * (
        ylims[factor_name][1] - ylims[factor_name][0]
    )
    significance_label = (
        "***"
        if p_actor < 0.001
        else "**" if p_actor < 0.01 else "*" if p_actor < 0.05 else ""
    )
    if p_actor < 0.05:
        ax.text(1, ypos, significance_label, fontsize=15, color="black")

    if slopes_observer is not None:
        ax.scatter(
            [1.2] * n_sessions + random_shift,
            slopes_observer,
            color=c,
            facecolors="none",
            edgecolors=c,
            label="Observer",
        )
        # add stars for p value
        ypos = ylims[factor_name][0] + 0.95 * (
            ylims[factor_name][1] - ylims[factor_name][0]
        )
        significance_label = (
            "***"
            if p_observer < 0.001
            else (
                "**" if p_observer < 0.01 else "*" if p_observer < 0.05 else ""
            )
        )
        if p_observer < 0.05:
            ax.text(
                1.2,
                ypos,
                significance_label,
                fontsize=15,
                color="black",
            )
    ax2 = ax.twinx()
    ax2.spines["right"].set_color("black")
    ax2.yaxis.label.set_color("black")
    ax2.tick_params(axis="y", colors="black")
    if coefs_actor is not None:
        (plotline, _, _) = ax2.errorbar(
            2,
            coefs_actor[factor_name][0],
            yerr=(
                coefs_actor[factor_name + "_ci"][0][1]
                - coefs_actor[factor_name + "_ci"][0][0]
            )
            / 2,
            fmt="s",
            color=c,
            capsize=5,
        )
        ypos = ylims2[factor_name][0] + 0.95 * (
            ylims2[factor_name][1] - ylims2[factor_name][0]
        )
        significance_label = (
            "***"
            if coefs_actor[factor_name + "_pval"] < 0.001
            else (
                "**"
                if coefs_actor[factor_name + "_pval"] < 0.01
                else ("*" if coefs_actor[factor_name + "_pval"] < 0.05 else "")
            )
        )
        if coefs_actor[factor_name + "_pval"] < 0.05:
            ax2.text(2, ypos, significance_label, fontsize=15, color="black")
    if coefs_observer is not None:
        (plotline, _, _) = ax2.errorbar(
            2.2,
            coefs_observer[factor_name][0],
            yerr=(
                coefs_observer[factor_name + "_ci"][0][1]
                - coefs_observer[factor_name + "_ci"][0][0]
            )
            / 2,
            fmt="s",
            color=c,
            capsize=5,
        )
        ypos = ylims2[factor_name][0] + 0.95 * (
            ylims2[factor_name][1] - ylims2[factor_name][0]
        )
        significance_label = (
            "***"
            if coefs_observer[factor_name + "_pval"] < 0.001
            else (
                "**"
                if coefs_observer[factor_name + "_pval"] < 0.01
                else (
                    "*" if coefs_observer[factor_name + "_pval"] < 0.05 else ""
                )
            )
        )
        if coefs_observer[factor_name + "_pval"] < 0.05:
            ax2.text(
                2.2,
                ypos,
                significance_label,
                fontsize=15,
                color="black",
            )
        plotline.set_markerfacecolor("none")

    # check if any slope values are outside ylims (in either direction) and report the values
    def check_slopes_outside_limits(slopes, label):
        if any(np.array(slopes) > ylims[factor_name][1]):
            print(
                f"{label} slopes above ylims for {factor_name}: {np.array(slopes)[np.array(slopes) > ylims[factor_name][1]]}"
            )
        if any(np.array(slopes) < ylims[factor_name][0]):
            print(
                f"{label} slopes below ylims for {factor_name}: {np.array(slopes)[np.array(slopes) < ylims[factor_name][0]]}"
            )

    check_slopes_outside_limits(slopes_actor, "actor")
    if slopes_observer is not None:
        check_slopes_outside_limits(slopes_observer, "observer")

    ax.set_ylim(ylims[factor_name])
    ax2.set_ylim(ylims2[factor_name])
    plt.grid(False)
    plt.xticks([1, 2], ["Slopes", "Beta"])
    beutify(ax)
    beutify(ax2, "right")
    plt.xlabel("Condition")
    plt.ylabel("Slope")
    plt.xlim(0.5, 2.5)
    # plt.show()
    factor_to_fig_name = {
        "Outcome": "Fig2C",
        "Pos_in_block": "Fig2F",
        "Performance_numerical": "Fig2I",
        "Condition": "Fig2L",
        "diffchoice": "FigS5G",
        "EvDir_fdbk": "EvDir_fdbk",
        "EvDir_prechoice": "EvDir_prechoice",
    }
    plot_file_name = (
        root_dir
        / "plots_paper"
        / f"{factor_to_fig_name[factor_name]}_{subject}_slopes_{factor_name}.pdf"
    )
    fig.savefig(plot_file_name, bbox_inches="tight", dpi=300)
    plt.close()
