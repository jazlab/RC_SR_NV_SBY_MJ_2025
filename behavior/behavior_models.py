"""
This module contains functions to fit behavior models and plot the results.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from behavior.trial_info_utils import remove_params, plot_coefficients
import statsmodels.formula.api as smf
from pymer4.models import Lmer


def model_confidence_mixedlinear(df):
    """
    Fits a mixed-effects linear regression model for confidence.

    The model predicts confidence based on condition, outcome, performance,
    and position in block, with a random effect for subject.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        tuple: A tuple containing the plot and the regression result.
    """
    model_formula = "Confidence ~ Condition + Outcome + Performance_numerical + Pos_in_block + (1|subject)"
    mixed_model = Lmer(model_formula, data=df)
    result = mixed_model.fit()

    return plot_mixed_regression(result)


def model_switches_mixedlogit(df, human=False):
    """
    Fits a mixed-effects logistic regression model for switches.

    The model predicts the probability of a switch based on condition, outcome,
    performance, and position in block, with an optional random effect for subject
    (used for human data).

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        human (bool, optional): Whether the data is from humans. Defaults to False.

    Returns:
        tuple: A tuple containing the plot and the regression result.
    """
    model_formula = (
        "Switches ~ Condition + Outcome + Performance_numerical + Pos_in_block"
    )
    # if data is from human add subject as random effect
    if human:
        model_formula = f"{model_formula}+ (1|subject)"
    mixed_model = Lmer(model_formula, data=df, family="binomial")
    result = mixed_model.fit()

    return plot_mixed_regression(result)


def plot_mixed_regression(result):
    """
    Plots the coefficients from a mixed-effects regression result.

    Generates a plot showing the estimated coefficients with confidence intervals
    and significance levels.

    Args:
        result (pymer4.models.Lmer): The result object from a mixed-effects regression.

    Returns:
        tuple: A tuple containing the matplotlib Figure and the result object.
    """
    param_names = result.index.values
    coefficients = result["Estimate"].values
    pvalues = result["P-val"].values
    conf_intervals = result[["2.5_ci", "97.5_ci"]].values
    ses = result["SE"].values

    # Remove the intercept for plotting
    param_names, coefficients, pvalues, conf_intervals, ses = remove_params(
        ["Intercept", "(Intercept)"],
        param_names,
        coefficients,
        pvalues,
        conf_intervals,
        ses,
    )

    # Plot the coefficients
    fig = plot_coefficients(param_names, coefficients, pvalues, conf_intervals)

    return fig, result


def model_switches_logistic(df):
    """
    Fits a logistic regression model for switches.

    The model predicts the probability of a switch based on condition, outcome,
    performance, and position in block.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.

    Returns:
        tuple: A tuple containing the matplotlib Figure and the model object.
    """
    model_formula = (
        "Switches ~ Condition + Outcome + Performance_numerical + Pos_in_block"
    )
    model = smf.logit(model_formula, df).fit()

    # Extract the parameters, the coefficients, confidence intervals, and the p-values
    param_names = model.params.index
    coefficients = model.params.values
    pvalues = model.pvalues.values
    conf_intervals = model.conf_int().values
    # skip the intercept and group variable for plotting
    param_names, coefficients, pvalues, conf_intervals = remove_params(
        ["Intercept"],
        param_names,
        coefficients,
        pvalues,
        conf_intervals,
    )
    fig = plot_coefficients(param_names, coefficients, pvalues, conf_intervals)
    return fig, model
