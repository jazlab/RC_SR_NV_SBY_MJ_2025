"""
This file contains helper functions for behavior plots.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from utils.get_session_info import find_behavior_sessions
from copy import deepcopy
import pandas as pd
import numpy as np
from ast import literal_eval
from utils.LoadSession import findrootdir
from pymer4.models import Lmer


def plot_coefficients(param_names, coefficients, pvalues, conf_intervals):
    """
    Plots regression coefficients with confidence intervals and significance levels.

    Args:
        param_names (list of str): List of parameter names corresponding to the coefficients.
        coefficients (numpy.ndarray): Array of coefficient values.
        pvalues (numpy.ndarray): Array of p-values corresponding to the coefficients.
        conf_intervals (numpy.ndarray): Array of confidence intervals for each coefficient.
                                        Should be of shape (n_coefficients, 2).

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the plot.
    """
    significance_labels = [
        "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        for p in pvalues
    ]
    # Plotting
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111)
    # Plot each coefficient along with its confidence interval
    for i in range(len(param_names)):
        # Plot the confidence intervals
        ax.plot(
            conf_intervals[i],
            [i, i],
            "k-",
            marker="_",
            markersize=5,
            linewidth=1,
        )
        # Plot the point estimate with a filled square
        ax.plot(
            coefficients[i], i, "ks", markersize=8
        )  # 's' specifies a square marker

        # Annotate the coefficient and p-value stars
        # Coefficient with two digits of precision
        ax.text(
            coefficients[i],
            i + 0.2,
            f"{coefficients[i]:.2f}",
            ha="center",
            va="center",
            fontsize=10,
        )
        # Significance stars next to the coefficient
        star = significance_labels[i]
        if star:
            ax.text(
                coefficients[i] + 0.15,
                i + 0.2,
                f"{star}",
                ha="left",
                va="center",
                fontsize=10,
            )

    # Set the y-axis to show the names of the variables, instead of default numeric ticks
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names)

    # Set the y-axis range to be one more than the number of coefficients
    ax.set_ylim(-1, len(param_names))

    ax.set_xlim(-1, 2)
    if max(coefficients) < 0.5:
        ax.set_xlim(-0.5, 0.5)

    # Set the labels and title of the plot
    ax.set_xlabel("Estimates")

    # Show a grid
    ax.grid(True)
    # remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def remove_params(
    parames_to_remove,
    param_names,
    coefficients,
    pvalues,
    conf_intervals,
    std_errors,
):
    """
    Removes specified parameters from the regression results.

    Args:
        parames_to_remove (list of str): List of parameter names to remove.
        param_names (list of str): List of all parameter names.
        coefficients (numpy.ndarray): Array of coefficient values.
        pvalues (numpy.ndarray): Array of p-values.
        conf_intervals (numpy.ndarray): Array of confidence intervals.
        std_errors (numpy.ndarray): Array of standard errors.

    Returns:
        tuple: A tuple containing the updated lists/arrays:
            - param_names (list of str)
            - coefficients (numpy.ndarray)
            - pvalues (numpy.ndarray)
            - conf_intervals (numpy.ndarray)
            - std_errors (numpy.ndarray)
    """
    idx_skip = [
        i
        for i, param_name in enumerate(param_names)
        if param_name in parames_to_remove
    ]
    param_names = [
        param_name
        for i, param_name in enumerate(param_names)
        if i not in idx_skip
    ]
    coefficients = np.delete(coefficients, idx_skip)
    pvalues = np.delete(pvalues, idx_skip)
    conf_intervals = np.delete(conf_intervals, idx_skip, axis=0)
    std_errors = np.delete(std_errors, idx_skip)
    return param_names, coefficients, pvalues, conf_intervals, std_errors


def combine_dfs_from_sessions(
    basedir, file_pattern="*_switches", bhv_dirs=None, subject=None
):
    """
    Combines dataframes from multiple sessions into a single dataframe.

    Args:
        basedir (str): Base directory containing the session directories.
        file_pattern (str, optional):  Pattern to match the CSV files to load. Defaults to "*_switches".
        bhv_dirs (list of pathlib.Path, optional): List of behavioral session directories.
            If None, it will be discovered using `find_behavior_sessions(basedir)`. Defaults to None.
        subject (str, optional): Subject identifier to filter the dataframes. Defaults to None. If "Both", no filtering is applied.

    Returns:
        pandas.DataFrame: A combined dataframe containing data from all sessions.
    """
    if bhv_dirs is None:
        bhv_dirs = find_behavior_sessions(basedir)
    dfs = []
    for bhv_dir in bhv_dirs:
        # check number of sessions from trial_info.csv
        csv_files = list(bhv_dir.glob(f"trial_info.csv"))
        if len(csv_files) == 0:
            continue
        df = pd.read_csv(csv_files[0])

        n_trials = len(df)
        df["date"] = df["date_time"].apply(lambda x: "-".join(x.split("_")[:3]))
        if df["date"][0] < "2022-12-20":
            continue
        # for per-session measures, require at least 500 trials
        # if file_pattern=='*accuracy':
        if n_trials < 500:
            continue
        # load .csv to a dataframe and add to list
        csv_files = list(bhv_dir.glob(f"{file_pattern}.csv"))
        if len(csv_files) == 0:
            print(f"No switch files found in {bhv_dir}")
            continue
        if subject is not None:
            if subject != "Both":
                csv_files = [f for f in csv_files if subject in f.name]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
    # concatenate list of dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    # add a field in dataframe to indicate whether choices are the same
    combined_df["diffchoice"] = 0
    # if incongruent is in condition, then set diffchoice to 1
    if file_pattern == "trial_info":
        combined_df["agent_choices"] = combined_df["agent_choices"].apply(
            literal_eval
        )
        combined_df["diffchoice"] = combined_df["agent_choices"].apply(
            lambda x: 1 if x[0] != x[1] else 0
        )
        combined_df["Pos_in_block"] = combined_df["pos_in_block"]
    elif file_pattern == "*_switches":
        combined_df.loc[
            combined_df["Condition"].str.contains("incongruent"), "diffchoice"
        ] = 1
    return combined_df


def preprocess_for_regression(df, subject=None, include_incongruent=False):
    """
    Preprocesses the dataframe for regression analysis.

    Args:
        df (pandas.DataFrame): Input dataframe.
        subject (str, optional): Subject identifier to filter the data. Defaults to None.
        include_incongruent (bool, optional): Whether to include incongruent trials. Defaults to False.

    Returns:
        pandas.DataFrame: Preprocessed dataframe.
    """
    df = deepcopy(df)
    if subject is not None:
        df = df[df["subject"] == subject]
        df.drop(columns=["subject"])
    else:
        # convert subject by mapping each subject to a number
        subjects_all = df["subject"].unique()
        subjects = {subject: i for i, subject in enumerate(subjects_all)}
        df["subject"] = df["subject"].map(subjects)
    df["Outcome"] = df["Outcome"].map({"1NR": 1, "2NR": 2, "3NR": 3, "4NR": 4})
    df = df.dropna(subset=["Outcome"])
    # drop Performance column
    df = df.drop(columns=["Performance"])
    if include_incongruent:
        df = df[
            df["Condition"].isin(["Actor", "Observer", "Actor-incongruent"])
        ]
        df["Condition"] = df["Condition"].map(
            {
                "Observer": 1,
                "Actor": 0,
                "Actor-incongruent": 0,
            }
        )
    else:
        df = df[df["Condition"].isin(["Actor", "Observer"])]
        df["Condition"] = df["Condition"].map({"Observer": 1, "Actor": 0})

    return df


def calculate_regression_coefficients(animal, df):
    """
    Calculates regression coefficients using logistic regression and mixed effects logistic regression.

    Args:
        animal (str): Animal identifier. If "humans_all" or "Both", subject is set to None.
        df (pandas.DataFrame): Input dataframe.

    Returns:
        tuple: A tuple containing four dictionaries, each holding regression coefficients,
               confidence intervals, and p-values for different conditions:
            - coefs_actor (dict): Coefficients for the 'Actor' condition.
            - coefs_observer (dict): Coefficients for the 'Observer' condition.
            - coefs_both (dict): Coefficients for the 'Both' condition.
            - coefs_actor_diffchoice (dict): Coefficients for the 'Actor_diffchoice' condition.
    """
    root_dir = findrootdir()
    coefs_actor = {
        "Condition": [],
        "Condition_ci": [],
        "Condition_pval": 1,
        "Outcome": [],
        "Outcome_ci": [],
        "Outcome_pval": 1,
        "Performance_numerical": [],
        "Performance_numerical_ci": [],
        "Performance_numerical_pval": 1,
        "Pos_in_block": [],
        "Pos_in_block_ci": [],
        "Pos_in_block_pval": 1,
        "diffchoice": [],
        "diffchoice_ci": [],
        "diffchoice_pval": 1,
        "Condition:Performance_numerical": [],
        "Condition:Performance_numerical_ci": [],
        "Condition:Performance_numerical_pval": 1,
        "(Intercept)": [],
        "(Intercept)_ci": [],
        "(Intercept)_pval": 1,
    }
    coefs_observer = deepcopy(coefs_actor)
    coefs_actor_diffchoice = deepcopy(coefs_actor)
    coefs_both = deepcopy(coefs_actor)
    for condition in ["Both", "Actor", "Observer", "Actor_diffchoice"]:
        df_cond = deepcopy(df)
        if condition == "Actor":
            df_cond = df_cond[(df_cond["Condition"] == "Actor")]
        elif condition == "Actor_diffchoice":
            df_cond = df_cond[
                (df_cond["Condition"] == "Actor")
                | (df_cond["Condition"] == "Actor-incongruent")
            ]
        elif condition == "Observer":
            df_cond = df_cond[df_cond["Condition"] == "Observer"]
        elif condition == "Both":
            df_cond = df_cond[
                (df_cond["Condition"] == "Actor")
                | (df_cond["Condition"] == "Observer")
            ]
        # if animal is "humans_all" then set subject to None
        if animal == "humans_all" or animal == "Both":
            animal = None
        if condition == "Both":  # Actor and Observer combined
            df_cond = preprocess_for_regression(
                df_cond, subject=animal, include_incongruent=False
            )
            model_formula = "Switches ~ Condition + Outcome + Performance_numerical + Pos_in_block"
        elif (
            condition == "Actor_diffchoice"
        ):  # Actor including incongruent trials
            df_cond = preprocess_for_regression(
                df_cond, subject=animal, include_incongruent=True
            )
            model_formula = "Switches ~ Outcome + Performance_numerical + Pos_in_block + diffchoice"
        else:  # Actor or Observer only
            df_cond = preprocess_for_regression(
                df_cond, subject=animal, include_incongruent=False
            )
            model_formula = (
                "Switches ~ Outcome + Performance_numerical + Pos_in_block"
            )
        model = smf.logit(model_formula, df_cond).fit()
        param_names = model.params.index
        coefficients = model.params.values
        std_errors = model.bse.values
        pvalues = model.pvalues.values
        conf_intervals = model.conf_int().values
        param_names, coefficients, pvalues, conf_intervals, std_errors = (
            remove_params(
                ["Intercept"],
                param_names,
                coefficients,
                pvalues,
                conf_intervals,
                std_errors,
            )
        )
        # save parameters to a csv
        label = animal
        n_subjects = len(df_cond["subject"].unique())
        print(f"Number of subjects: {n_subjects}")
        if n_subjects == 2 and animal is None and condition == "Both":
            label = "FigS3B"
            df_params = pd.DataFrame(
                {
                    "param_names": param_names,
                    "coefficients": coefficients,
                    "std_errors": std_errors,
                    "pvalues": pvalues,
                }
            )
            df_params.to_csv(
                f"{root_dir}/stats_paper/{label}_{condition}_switches_logit.csv"
            )
            fig = plot_coefficients(
                param_names, coefficients, pvalues, conf_intervals
            )
            # save the plot
            fig.savefig(
                f"{root_dir}/plots_paper/{label}_{condition}_switches_logit.pdf",
                bbox_inches="tight",
                dpi=300,
            )

        # Second model: Mixed effects logistic regression
        if n_subjects > 2:  # this would be human data
            label = "FigS3A"
            mixed_model_formula = f"{model_formula}+ (1|subject)"
            mixed_model = Lmer(
                mixed_model_formula, data=df_cond, family="binomial"
            )
            result = mixed_model.fit()
            # save model results to a csv
            result.to_csv(
                f"{root_dir}/stats_paper/{label}_{condition}_switches_mixedlogit.csv"
            )
            # Extract parameter names
            param_names = result.index.values

            # Extract coefficients (estimates)
            coefficients = result["Estimate"].values

            # Extract p-values
            pvalues = result["P-val"].values

            # Check if confidence intervals are available in the result
            if "2.5_ci" in result.columns and "97.5_ci" in result.columns:
                # Extract confidence intervals directly
                conf_intervals = result[["2.5_ci", "97.5_ci"]].values
            else:
                # Compute confidence intervals manually using standard errors
                se = result["SE"].values  # Standard errors
                z_critical = 1.96  # For a 95% confidence interval
                conf_intervals = np.column_stack(
                    (
                        coefficients - z_critical * se,
                        coefficients + z_critical * se,
                    )
                )

        for i, param_name in enumerate(param_names):
            if condition == "Actor":
                coefs_actor[param_name].append(coefficients[i])
                coefs_actor[f"{param_name}_ci"].append(conf_intervals[i])
                coefs_actor[f"{param_name}_pval"] = pvalues[i]
            elif condition == "Observer":
                coefs_observer[param_name].append(coefficients[i])
                coefs_observer[f"{param_name}_ci"].append(conf_intervals[i])
                coefs_observer[f"{param_name}_pval"] = pvalues[i]
            elif condition == "Both":
                coefs_both[param_name].append(coefficients[i])
                coefs_both[f"{param_name}_ci"].append(conf_intervals[i])
                coefs_both[f"{param_name}_pval"] = pvalues[i]
            elif condition == "Actor_diffchoice":
                coefs_actor_diffchoice[param_name].append(coefficients[i])
                coefs_actor_diffchoice[f"{param_name}_ci"].append(
                    conf_intervals[i]
                )
                coefs_actor_diffchoice[f"{param_name}_pval"] = pvalues[i]
    return coefs_actor, coefs_observer, coefs_both, coefs_actor_diffchoice
