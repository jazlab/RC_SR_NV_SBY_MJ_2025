"""
This module contains functions to get task variables from behavior data.
"""

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from pyTdr.tdrUpdateBehavior import add_block_length, add_switch_block_length
from utils.get_session_info import get_unit_spike_dirs, get_behavior_df


def add_task_vars(task_var, bhv_df_unit, subject_name):
    task_var_base = get_task_vars(bhv_df_unit, subject_name, False)
    # match trial num
    for key in task_var_base.keys():
        if key == "trial_num":
            continue
        task_var[key] = np.zeros(len(task_var["trial_num"]))
    for i, trial_num in enumerate(task_var["trial_num"]):
        if key == "trial_num":
            continue
        idx = np.where(task_var_base["trial_num"] == trial_num)[0]
        for key in task_var_base.keys():
            if key == "trial_num":
                continue
            task_var[key][i] = task_var_base[key][idx]
    return task_var


def add_new_task_var(data, metadata):
    spike_dirs = get_unit_spike_dirs(data, metadata)
    bhv_df = get_behavior_df(spike_dirs[0])
    bhv_df = add_block_length(bhv_df)
    agent_idx = 0 if metadata["unit"]["subject"][0] == "O" else 1
    bhv_df = add_switch_block_length(bhv_df, agent_idx)
    for i_unit, unit_dict in enumerate(data["unit"]):
        spike_dir = spike_dirs[i_unit]
        if i_unit > 0 and (not spike_dir == spike_dirs[i_unit - 1]):
            bhv_df = get_behavior_df(spike_dir)
            bhv_df = add_block_length(bhv_df)
            bhv_df = add_switch_block_length(bhv_df, agent_idx)
        task_var = unit_dict["task_variable"]
        trials_unit = task_var["trial_num"]
        block_length = []
        switch_block_length = []
        actor_1back = []
        for trial in trials_unit:
            i_trial = np.where(bhv_df["trial_num"] == trial)[0][0]
            if i_trial == 0:
                actor_1back.append(0)
            else:
                actor_1back.append(bhv_df["player"].values[i_trial - 1])
            block_length.append(
                bhv_df[bhv_df["trial_num"] == trial]["block_length"].values[0]
            )
            switch_block_length.append(
                bhv_df[bhv_df["trial_num"] == trial][
                    "switch_block_length"
                ].values[0]
            )
        # change actor_1back to 0 if it's self, 1 if it's other
        if metadata["unit"]["subject"][0] == "L":
            actor_1back = [-actor + 1 for actor in actor_1back]
        task_var["block_length"] = np.array(block_length)
        task_var["switch_block_length"] = np.array(switch_block_length)
        task_var["actor_1back"] = np.array(actor_1back)
        unit_dict["task_variable"] = task_var
    return data


def get_task_vars(bhv_df_unit, subject_name, load_rew_switch=False):
    task_var = {}
    subject_idx = 0 if subject_name == "O" else 1
    task_var["actor"] = bhv_df_unit["player"].values
    task_var["trial_num"] = bhv_df_unit["trial_num"].values
    # note here actor is 0 if it's self, and 1 if it's other
    if subject_name == "L":
        # flip actor labels scuh that 0 is self and 1 is other
        task_var["actor"] = -task_var["actor"] + 1
    task_var["reward"] = bhv_df_unit["reward"].values
    task_var["choice"] = bhv_df_unit["player_choice"].values
    task_var["choice_a0"] = bhv_df_unit["choice_a0"].values
    task_var["choice_a1"] = bhv_df_unit["choice_a1"].values
    task_var["history"] = bhv_df_unit["history"].values
    task_var["pos_in_block"] = bhv_df_unit["pos_in_block"].values
    task_var["trial_duration"] = bhv_df_unit["trial_duration"].values
    task_var["n_tokens"] = bhv_df_unit["total_polls"].values
    task_var["total_captured"] = bhv_df_unit["touched_polls"].values
    task_var["block_length"] = bhv_df_unit["block_length"].values
    task_var["switch_block_length"] = bhv_df_unit["switch_block_length"].values

    # after removing trials with n_pre_switch = -1,
    # recompute n_pre_switch
    n_pre_switch = [
        val[subject_idx] for val in bhv_df_unit["n_pre_switch_actor"].values
    ]
    n_pre_switch = np.array(n_pre_switch)
    task_var["n_pre_switch"] = n_pre_switch
    # add n_pre_switch_lvl based on n_pre_switch
    n_pre_switch_lvl = n_pre_switch.copy()  # 1,2,3
    n_pre_switch_lvl[n_pre_switch_lvl >= 3] = 3

    task_var["n_pre_switch_lvl"] = n_pre_switch_lvl

    rew_switch = np.zeros(len(bhv_df_unit))
    rew_switch[(bhv_df_unit["history"] == "1R")] = -1
    switched = bhv_df_unit["switched_actor"].values
    bstayed = np.array([switch[subject_idx] == 0 for switch in switched])
    bswitched = np.array([switch[subject_idx] == 1 for switch in switched])
    nonreward = np.array((bhv_df_unit["history"] != "1R"))
    rew_switch[nonreward & bstayed] = 0
    rew_switch[nonreward & bswitched] = 1
    # for each of task variable normalize values to [-1,1]
    for key in task_var.keys():
        var = task_var[key]
        var_encoder = LabelEncoder()
        var_encoded = var_encoder.fit_transform(var)
        if (
            key == "trial_num"
            or key == "trial_duration"
            or key == "total_captured"
            or key == "pos_in_block"
            or key == "choice_a0"
            or key == "choice_a1"
            or key == "n_pre_switch"
            or key == "block_length"
            or key == "switch_block_length"
        ):
            continue
        if key == "history":
            sorting_order = {"1R": -1, "1NR": 0, "2NR": 1}
            # for trials with history other than 1R, 1NR, 2NR, use a dummy value
            # in analyses that require history we will exclude it by selecting
            # the first three values of history.
            var_scaled_values = np.array([sorting_order.get(v, 2) for v in var])
        elif key == "n_pre_switch_lvl":
            sorting_order = {1: 1, 2: 0, 3: -1}
            var_scaled_values = np.array([sorting_order[v] for v in var])
        else:
            var_scaled = MinMaxScaler(feature_range=(-1, 1))
            var_scaled_values = var_scaled.fit_transform(
                var_encoded.reshape(-1, 1)
            ).flatten()
        task_var[key] = var_scaled_values
    if load_rew_switch:
        task_var["rew_switch"] = rew_switch
    return task_var
