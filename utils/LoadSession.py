"""
This module contains functions for loading data for a session.
"""

import os
from ast import literal_eval
import json
import pandas as pd
import numpy as np
import hashlib


def findrootdir():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = root_dir.replace("code", "data") 
    if not os.path.exists(root_dir):
        ValueError("Root directory not found")
    return root_dir


def apply_literal_eval(df, columns):
    """
    Apply literal_eval to specified columns of a DataFrame after filling NA values with "[0,0]".

    :param df: The DataFrame to process.
    :param columns: List of column names to apply literal_eval after filling NAs with "[0,0]".
    """
    fill_value = "[0,0]"
    for column in columns:
        df[column] = df[column].fillna(fill_value).apply(literal_eval)


def load_data_for_session(spikes_dir, config):
    """Load the necessary data files.
    :param spikes_dir: Path to the spikes directory for the session.
    :param config: A dictionary of configuration parameters.
    :return: A tuple containing the loaded data.
    """
    base_dir = os.path.dirname(os.path.dirname(spikes_dir))

    trials_path = os.path.join(
        base_dir, "open_ephys_events", "open_ephys_trials.json"
    )
    behavior_path = os.path.join(base_dir, "moog_events", "trial_info.csv")
    spike_times_per_cluster_path = os.path.join(
        spikes_dir, "spike_times_per_cluster.json"
    )
    valid_units_path = os.path.join(spikes_dir, "valid_units.json")
    if not os.path.exists(valid_units_path):
        return None
    cluster_labels_path = os.path.join(spikes_dir, "cluster_labels.json")

    trials = json.load(open(trials_path, "r"))
    behavioral_df = pd.read_csv(behavior_path)
    apply_literal_eval(
        behavioral_df,
        columns=[
            "switched_actor",
            "n_pre_switch_actor",
            "n_pre_switch_self",
            "reaction_time",
            "agent_choices",
        ],
    )
    sessionid = "_".join(spikes_dir.split("/")[-4:])
    seed = int(hashlib.sha256(sessionid.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    behavioral_df["randint"] = np.random.randint(2, size=len(behavioral_df))
    # create a new column with random number either 0 or 1 for control
    spike_times_per_cluster = json.load(open(spike_times_per_cluster_path, "r"))
    valid_units = json.load(open(valid_units_path, "r"))
    cluster_labels = json.load(open(cluster_labels_path, "r"))

    valid_units = [
        unit
        for unit in valid_units
        if cluster_labels[unit] != "noise" and valid_units[unit]
    ]

    return (
        spike_times_per_cluster,
        trials,
        behavioral_df,
        valid_units,
    )
