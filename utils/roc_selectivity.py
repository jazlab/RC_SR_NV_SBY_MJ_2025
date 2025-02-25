"""
This module contains functions for calculating selectivity and related metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_rate_diff(condition1, condition2, n_reps=1000):
    """
    This function performs a bootstrap test to assess the significance of the difference
    in firing rates between two conditions.

    Args:
        condition1 (np.ndarray): Firing rates for the first condition.
        condition2 (np.ndarray): Firing rates for the second condition.
        n_reps (int, optional): Number of bootstrap replicates. Defaults to 1000.

    Returns:
        tuple: A tuple containing the following elements:
            - (float): The mean difference in firing rates between the two conditions.
            - (float): The 95% confidence interval for the mean difference.
            - (float): The p-value of the bootstrap test.
    """
    # Calculate observed difference in firing rates
    rate_diff = np.mean(condition1) - np.mean(condition2)

    # Combined pool for resampling
    all_data = np.concatenate((condition1, condition2))

    # For reproducibility, set rng
    rng = np.random.default_rng(123)

    # Bootstrap replicates
    replicate_diffs = []
    for _ in range(n_reps):
        # Resample from the combined dataset, maintaining group sizes
        cond1_resample = rng.choice(
            all_data, size=len(condition1), replace=True
        )
        cond2_resample = rng.choice(
            all_data, size=len(condition2), replace=True
        )

        replicate_diffs.append(
            np.mean(cond1_resample) - np.mean(cond2_resample)
        )

    # Confidence interval and p-value calculation (same as before)
    ci_lower = np.percentile(replicate_diffs, 2.5)
    ci_upper = np.percentile(replicate_diffs, 97.5)
    p_val = np.sum(np.abs(replicate_diffs) >= np.abs(rate_diff)) / n_reps

    return rate_diff, (ci_lower, ci_upper), p_val


def cross_validate(condition1, condition2, n_reps=1000):
    rate_diff = np.mean(condition1) - np.mean(condition2)
    sign_diff = np.sign(rate_diff)
    replicate_diffs = []
    # For reproducibility, set rng
    rng = np.random.default_rng(456)
    for _ in range(n_reps):
        # Resample from the combined dataset, maintaining group sizes
        cond1_resample = rng.choice(
            condition1, size=len(condition1), replace=True
        )
        cond2_resample = rng.choice(
            condition2, size=len(condition2), replace=True
        )
        replicate_diffs.append(
            np.mean(cond1_resample) - np.mean(cond2_resample)
        )
    p_val = np.sum(np.sign(replicate_diffs) == sign_diff) / n_reps
    return 1 - p_val


def calculate_selectivity(spike_counts, trial_outcomes):
    """
    Calculate the selectivity of a neuron based on spike counts and trial outcomes.

    :param spike_counts: A list or numpy array of spike counts for each trial.
    :param trial_outcomes: A list or numpy array of trial outcomes (1 for reward, 0 for non-reward).
    :return: Selectivity of the neuron ranging from -0.5 to 0.5.
    """
    # Ensure input arrays are numpy arrays
    spike_counts = np.array(spike_counts)
    trial_outcomes = np.array(trial_outcomes)

    # if there are no spikes, return N/A
    if np.sum(spike_counts) == 0:
        return np.nan, np.nan

    # if there is only one outcome, return N/A
    if len(np.unique(trial_outcomes)) == 1:
        return np.nan, np.nan

    # Calculate ROC AUC
    roc_auc = roc_auc_score(trial_outcomes, spike_counts)
    positive_scores = spike_counts[trial_outcomes == 1]
    negative_scores = spike_counts[trial_outcomes == 0]

    # compute p-value using bootstrap
    rate_diff, ci, p_val_bootstrap = bootstrap_rate_diff(
        positive_scores, negative_scores
    )

    # Convert ROC AUC to selectivity score (-0.5 to 0.5)
    selectivity = roc_auc - 0.5

    # limit to two significant digits
    selectivity = np.round(selectivity, 4)
    p_val_bootstrap = np.round(p_val_bootstrap, 4)
    return selectivity, p_val_bootstrap


def calculate_selectivity_from_triginfo(tf1, tf2, window=[0, 0.6]):
    n_trials_1 = len(tf1["trigStarts"])
    n_trials_2 = len(tf2["trigStarts"])
    labels_1 = np.ones(n_trials_1)
    # first tf is coded as 1 so that selectivity is positive for higher rate
    # in the 1st tf.
    labels_2 = np.zeros(n_trials_2)
    spike_counts_1 = np.zeros(n_trials_1)
    spike_counts_2 = np.zeros(n_trials_2)
    for i_trial in range(n_trials_1):
        spike_counts_1[i_trial] = np.sum(
            (tf1["events"][i_trial] > window[0])
            * (tf1["events"][i_trial] < window[1])
        )
    for i_trial in range(n_trials_2):
        spike_counts_2[i_trial] = np.sum(
            (tf2["events"][i_trial] > window[0])
            * (tf2["events"][i_trial] < window[1])
        )
    spike_counts = np.concatenate((spike_counts_1, spike_counts_2))
    trial_outcomes = np.concatenate((labels_1, labels_2))
    return calculate_selectivity(spike_counts, trial_outcomes)
