"""
This script makes a plot of the rate differences between 1R, 1NR, and 2NR
conditions for Actor and Observer trials.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
from utils.LoadSession import findrootdir
from pyTdr.tdrLoadAccData import tdrLoadAccData
from pyTdr.tdrSelectByCondition import tdrSelectByCondition
from pyTdr.tdrNormalize import smooth_and_normalize
from pyTdr.tdrAverageCondition import tdrAverageCondition
from pyTdr.tdrUtils import select_time


def select_condition(data, conditions, crossvalidate=None, min_trials=5):
    task_index = {key: np.array(value) - 1 for key, value in conditions.items()}
    return tdrSelectByCondition(
        data, task_index, crossvalidate=crossvalidate, min_trials=min_trials
    )


def average_conditions(data, conditions):
    task_index = {key: np.array(value) - 1 for key, value in conditions.items()}
    return tdrAverageCondition(data, task_index)


def filter_by_unit(dataT, unit_idx):
    if "unit" in dataT:
        dataT["unit"] = [
            unit
            for unit in dataT["unit"]
            if unit["unit_idx_master"] in unit_idx
        ]
    return dataT


def calc_mean(data):
    return data.mean()


def print_confint(data):
    bootstraps = sns.algorithms.bootstrap(
        data, func=calc_mean, n_boot=1000, seed=0
    )

    # Calculate confidence interval (e.g., 95%)
    confidence_interval = np.percentile(bootstraps, [2.5, 97.5])

    # Print the exact confidence interval
    print("Exact Confidence Interval:", confidence_interval)
    return f"Exact Confidence Interval: {confidence_interval}\n"


def plot_rate_difference(root_dir, df_ratediff, subject="both"):
    fig, ax = plt.subplots(figsize=(6, 6))
    df_ratediff = df_ratediff[
        (df_ratediff["agent"] == "self") | (df_ratediff["agent"] == "other")
    ]
    df_1nr = df_ratediff[df_ratediff["history"] == 1]
    df_2nr = df_ratediff[df_ratediff["history"] == 2]
    df_1nr["history"] = "1NR"
    df_2nr["history"] = "2NR"
    df_1r = pd.DataFrame(
        {
            "history": ["1R", "1R"],
            "agent": ["self", "other"],
            "rate_diff": [0, 0],
        }
    )
    df_combined = pd.concat([df_1r, df_1nr, df_2nr])
    ax = sns.lineplot(
        x="history",
        y="rate_diff",
        data=df_combined,
        hue="agent",
        palette=["black", "black"],
        style="agent",
        dashes=[(None, None), (5, 5)],
        estimator="mean",
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        linewidth=2,
        markers=True,
        markersize=8,
    )

    ax.set_xlabel("Self/Other")
    ax.set_ylabel("Rate difference")
    # ax.set_ylim(0.1, 0.6)
    ax.tick_params(
        direction="out",
        length=6,
        width=2,
        colors="black",
        grid_color="black",
        grid_alpha=0.5,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_position(("outward", 5))
        # Turning off the top and right spines to match the example style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.set_ylim([0, 0.7])
    # save figure
    fignum = {
        "L": "S6I",
        "O": "S6H",
        "both": "3I",
    }
    fig.savefig(
        f"{root_dir}/plots_paper/Fig{fignum[subject]}_rate_diff_{subject}.pdf",
        dpi=300,
    )

    # print rate differences mean and 2.5% and 97.5% percentiles
    ratediff_self_1nr = df_1nr[df_1nr["agent"] == "self"]["rate_diff"].values
    ratediff_other_1nr = df_1nr[df_1nr["agent"] == "other"]["rate_diff"].values
    ratediff_self_2nr = df_2nr[df_2nr["agent"] == "self"]["rate_diff"].values
    ratediff_other_2nr = df_2nr[df_2nr["agent"] == "other"]["rate_diff"].values

    # paired t-test for rate differences between 1nr-self and 1nr-other
    _, p_paired_1nr = stats.ttest_rel(ratediff_self_1nr, ratediff_other_1nr)
    _, p_paired_2nr = stats.ttest_rel(ratediff_self_2nr, ratediff_other_2nr)

    print(f"\nSubject: {subject}")
    print(f"1NR self: {ratediff_self_1nr.mean()}")
    _ = print_confint(ratediff_self_1nr)
    print(f"2NR self: {ratediff_self_2nr.mean()}")
    _ = print_confint(ratediff_self_2nr)
    print(f"1NR other: {ratediff_other_1nr.mean()}")
    _ = print_confint(ratediff_other_1nr)
    print(f"2NR other: {ratediff_other_2nr.mean()}")
    _ = print_confint(ratediff_other_2nr)

    output_file_path = f"{root_dir}/stats_paper/rate_diff_{subject}.txt"
    # Open the file in write mode
    with open(output_file_path, "w") as file:
        # Redirect the print statements to the file
        file.write(f"\nSubject: {subject}\n")
        file.write(f"1NR self: {ratediff_self_1nr.mean()}\n")
        file.write(print_confint(ratediff_self_1nr))
        file.write(f"1NR other: {ratediff_other_1nr.mean()}\n")
        file.write(print_confint(ratediff_other_1nr))
        file.write(f"p_val 1NR: {p_paired_1nr}\n")
        file.write(f"2NR self: {ratediff_self_2nr.mean()}\n")
        file.write(print_confint(ratediff_self_2nr))
        file.write(f"2NR other: {ratediff_other_2nr.mean()}\n")
        file.write(print_confint(ratediff_other_2nr))
        file.write(f"p_val 2NR: {p_paired_2nr}\n")


def main():
    root_dir = findrootdir()
    event = "fdbk"
    rate_diffs = {}
    # initialize dataframe
    df_ratediff = pd.DataFrame(
        columns=["subject", "history", "agent", "rate_diff"]
    )
    df_rate = pd.DataFrame(columns=["subject", "history", "agent", "rate"])
    # 1. read data from tdrLoadAccData
    for subject in ["L", "O"]:
        dataT, metadata = tdrLoadAccData(root_dir, subject, event)
        dataT = smooth_and_normalize(dataT)
        rate_diffs[subject] = {}
        # 2. read significant units from csv files
        option = "integration_sig_all_both"
        rate_diffs[subject][option] = {}
        df_sig_units = pd.read_csv(f"{root_dir}/stats_paper/{option}.csv")
        sig_units = df_sig_units[df_sig_units["subject"] == subject][
            "unit_index"
        ].values
        # equalize number of self vs. other rew significant units
        df_sig_units_self = pd.read_csv(
            f"{root_dir}/stats_paper/integration_sig_self_both.csv"
        )
        sig_units_self = df_sig_units_self[
            df_sig_units_self["subject"] == subject
        ]["unit_index"].values
        df_sig_units_other = pd.read_csv(
            f"{root_dir}/stats_paper/integration_sig_other_both.csv"
        )
        sig_units_other = df_sig_units_other[
            df_sig_units_other["subject"] == subject
        ]["unit_index"].values
        # find the intersection of significant units between sig_units and sig_units_self
        sig_units_self = np.intersect1d(sig_units, sig_units_self)
        sig_units_other = np.intersect1d(sig_units, sig_units_other)
        n_units_self = len(sig_units_self)
        n_units_other = len(sig_units_other)
        # as a control equalize number of units from actor and observer selective units
        test_bias = False
        if test_bias and (n_units_self > n_units_other):
            n_units_diff = n_units_self - n_units_other
            # select n_units_diff from sig_units_self that are not in sig_units_other
            sig_units_self_only = np.setdiff1d(sig_units_self, sig_units_other)
            units_to_remove = np.random.choice(
                sig_units_self_only, n_units_diff, replace=False
            )
            # remove thse from sig_units
            sig_units = np.setdiff1d(sig_units, units_to_remove)

        for agent in ["self", "other"]:
            # 3. filter data by significant units
            dataT_ = deepcopy(dataT)
            dataT_ = filter_by_unit(dataT_, sig_units)
            conditions = {
                "history": [1, 2, 3],
            }
            if agent == "self":
                conditions["actor"] = [1, 1, 1]
            elif agent == "other":
                conditions["actor"] = [2, 2, 2]
            dataT_ = select_condition(dataT_, conditions)
            # 5. average conditions
            conditions = {
                "history": [1, 2, 3],
            }
            dataT_, _ = average_conditions(dataT_, conditions)
            dataT_ = select_time(dataT_, tmin=0, tmax=0.6, combinebins=True)
            # 7. calculate rate difference
            rate_diff_1 = (
                dataT_["response"][:, 0, 1] - dataT_["response"][:, 0, 0]
            )
            rate_diff_2 = (
                dataT_["response"][:, 0, 2] - dataT_["response"][:, 0, 0]
            )
            # If rate diff 1 is negative, invert both rate diff 1 and 2
            for i in range(len(rate_diff_1)):
                if rate_diff_1[i] < 0:
                    rate_diff_1[i] = -rate_diff_1[i]
                    rate_diff_2[i] = -rate_diff_2[i]
            rate_diffs[subject][option][agent] = np.vstack(
                [rate_diff_1, rate_diff_2]
            )
            n_units = rate_diff_1.shape[0]
            for i in range(n_units):
                df_ratediff = pd.concat(
                    [
                        df_ratediff,
                        pd.DataFrame(
                            {
                                "subject": [subject],
                                "history": [1],
                                "agent": [agent],
                                "rate_diff": [rate_diff_1[i]],
                            }
                        ),
                        pd.DataFrame(
                            {
                                "subject": [subject],
                                "history": [2],
                                "agent": [agent],
                                "rate_diff": [rate_diff_2[i]],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                for i_history in range(3):
                    df_rate = pd.concat(
                        [
                            df_rate,
                            pd.DataFrame(
                                {
                                    "subject": [subject],
                                    "history": [i_history + 1],
                                    "agent": [agent],
                                    "rate": [
                                        dataT_["response"][i, 0, i_history]
                                    ],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

    # 8. plot rate difference
    plot_rate_difference(root_dir, df_ratediff)
    for subject in ["L", "O"]:
        df_subject = df_ratediff[df_ratediff["subject"] == subject]
        plot_rate_difference(root_dir, df_subject, subject)


if __name__ == "__main__":
    main()
