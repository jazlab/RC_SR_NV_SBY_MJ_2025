"""
This script is used to modify the task variable of the data.
"""

import pathlib
import pandas as pd
from utils.LoadSession import findrootdir
from ast import literal_eval
import numpy as np


def load_bhv(date):
    root_dir = findrootdir()
    bhv_path = (
        pathlib.Path(root_dir)
        / "social_O_L"
        / date
        / "results"
        / "moog_events"
        / "trial_info.csv"
    )
    bhv_df = pd.read_csv(bhv_path)
    bhv_df["n_pre_switch_actor"] = bhv_df["n_pre_switch_actor"].apply(
        literal_eval
    )
    bhv_df["reaction_time"] = bhv_df["reaction_time"].apply(literal_eval)
    return bhv_df


def remove_first_trial(data):
    for unit in data["unit"]:
        if unit["task_variable"]["trial_num"][0] == 1:
            unit["response"] = unit["response"][1:]
            unit["task_variable"] = {
                key: value[1:] for key, value in unit["task_variable"].items()
            }
            unit["trial_ids"] = unit["trial_ids"][1:]
    return data


def use_prev_switch(data, metadata):
    if "unit" in data:
        data = remove_first_trial(data)
        for i, unit_dict in enumerate(data["unit"]):
            date = metadata["unit"]["date"][i]
            subject = metadata["unit"]["subject"][i]
            bhv_df = load_bhv(date)
            use_prev_switch_single(unit_dict, bhv_df, subject)
    else:
        date = metadata["date"]
        subject = metadata["subject"]
        bhv_df = load_bhv(date)
        use_prev_switch_single(data, bhv_df, subject)
    return data


def use_prev_switch_single(data, bhv_df, subject):
    subject_idx = 0 if subject == "O" else 1
    task_var = data["task_variable"]
    unit_trial_numbers = task_var["trial_num"]
    idx_remove = []
    for i_data, trial_num in enumerate(unit_trial_numbers):
        idx_trial = np.where(bhv_df["trial_num"] == trial_num)[0][0]
        # if reaction time is greater than 1.5 s, remove this trial
        if bhv_df.iloc[idx_trial]["reaction_time"][subject_idx] > 1.5:
            idx_remove.append(i_data)
            continue
        n_pre_switch = bhv_df.loc[idx_trial - 1, "n_pre_switch_actor"][
            subject_idx
        ]
        actor = bhv_df.loc[idx_trial - 1, "player"]
        reward = bhv_df.loc[idx_trial - 1, "reward"]
        # map reward from [0,1] to [-1,1]
        reward = 2 * reward - 1
        history = bhv_df.loc[idx_trial - 1, "history"]
        total_captured = bhv_df.loc[idx_trial - 1, "touched_polls"]
        # if history is not one of '1R', '1NR', '2NR', remove this trial
        if history not in ["1R", "1NR", "2NR"]:
            # remove trial from data
            idx_remove.append(i_data)
            continue
        # map history from 1R, 1NR, 2NR to -1, 0, 1
        history = (
            history.replace("1R", "-1").replace("1NR", "0").replace("2NR", "1")
        )
        # for Lalo, we need to flip the actor variable
        if subject == "L":
            actor = 1 - actor
        if actor == 0:
            actor = -1
        data["task_variable"]["actor"][i_data] = actor
        data["task_variable"]["n_pre_switch"][i_data] = n_pre_switch
        pos_in_block = bhv_df.loc[idx_trial - 1, "pos_in_block"]
        n_pre_switch_lvl = n_pre_switch if n_pre_switch < 3 else 3
        # map n_pre_switch_lvl from [1,2,3] to [1,0,-1]
        n_pre_switch_lvl = -1 * n_pre_switch_lvl + 2
        data["task_variable"]["n_pre_switch_lvl"][i_data] = n_pre_switch_lvl
        data["task_variable"]["pos_in_block"][i_data] = pos_in_block
        data["task_variable"]["reward"][i_data] = reward
        data["task_variable"]["history"][i_data] = history
        data["task_variable"]["total_captured"][i_data] = total_captured
    if idx_remove:
        # in single unit case, response shape is (n_trials, n_bins)
        # but in simultaneous case, response shape is (n_units, n_bins, n_trials)
        # so we need to check
        if len(data["response"].shape) == 3:
            data["response"] = np.delete(data["response"], idx_remove, axis=2)
        else:
            data["response"] = np.delete(data["response"], idx_remove, axis=0)
        data["task_variable"] = {
            key: np.delete(value, idx_remove)
            for key, value in data["task_variable"].items()
        }
        data["trial_ids"] = np.delete(data["trial_ids"], idx_remove)
