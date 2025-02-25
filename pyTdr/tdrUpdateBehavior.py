"""
This script updates the behavior dataframe to include new columns
"""

import json
from ast import literal_eval
import numpy as np
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute path to the config.json file
config_path = os.path.join(script_dir, "config", "config.json")
config = json.load(open(config_path))


def safe_literal_eval(val):
    try:
        return literal_eval(val)
    except ValueError:
        return val


def add_block_length(df):
    """
    add a column to df that indicates the length of each block
    based on the pos_in_block column which is the largest number in each block
    for example, if pos_in_block = [0,1,2,3,4,0,1,2,3,4,5,6,7,8,9]
    then block_length = [5,5,5,5,5,10,10,10,10,10,10,10,10,10,10]
    calculate the block_length variable from the end backwards sequencially
    resetting at every 0.
    """
    pos_in_block = df["pos_in_block"].values
    block_length = np.zeros(len(pos_in_block))
    block_length[-1] = pos_in_block[-1] + 1
    for i in range(len(pos_in_block) - 2, -1, -1):
        if pos_in_block[i + 1] == 0:
            block_length[i] = pos_in_block[i] + 1
        else:
            block_length[i] = block_length[i + 1]
    df["block_length"] = block_length
    return df


def add_switch_block_length(df, agent_idx):
    """
        simialr to block length, but for number of trials to next switch
    for example, if n_pre_switch = [4,3,2,1,1,3,2,1]
    then switch_block_length = [4,4,4,4,1,3,3,3]
    calculate the switch_block_length variable by starting from beginning,
    resetting at 1
    """
    df["n_pre_switch_actor"] = df["n_pre_switch_actor"].apply(literal_eval)
    n_pre_switch = [val[agent_idx] for val in df["n_pre_switch_actor"].values]
    switch_block_length = np.zeros(len(n_pre_switch))
    switch_block_length[0] = n_pre_switch[0]
    for i in range(1, len(n_pre_switch)):
        if n_pre_switch[i - 1] == 1:
            switch_block_length[i] = n_pre_switch[i]
        else:
            switch_block_length[i] = switch_block_length[i - 1]
    df["switch_block_length"] = switch_block_length
    return df


def update_behavior(subject_idx, df):
    # add new columns to df
    df = add_block_length(df)
    df = add_switch_block_length(df, subject_idx)
    load_prev_history = config["load_prev_history"]
    cols_to_drop = df.filter(like="Unnamed").columns
    df = df.drop(columns=cols_to_drop)
    df["trial_num"] = df["trial_idx"] + 1
    df["switched_actor"] = df["switched_actor"].fillna("[0,0]")
    df["switched_actor"] = df["switched_actor"].apply(literal_eval)
    df["agent_choices"] = df["agent_choices"].apply(literal_eval)
    df["n_pre_switch_self"] = df["n_pre_switch_self"].apply(literal_eval)

    # Use previous trial's history if desired
    if load_prev_history:
        df["history"] = df["history"].shift(1)
        df["player"] = df["player"].shift(1)
        df["switched_actor"] = df["switched_actor"].shift(1)
        df["player_choice"] = df["player_choice"].shift(1)
        df["n_pre_switch_self"] = df["n_pre_switch_self"].shift(1)
        df["n_pre_switch_actor"] = df["n_pre_switch_actor"].shift(1)
        df["agent_choices"] = df["agent_choices"].shift(1)
        # remove first entry in behaviroal_df
        df = df.iloc[1:]

    n_pre_switch = np.array(
        [val[subject_idx] for val in df["n_pre_switch_actor"].values]
    )
    idx_pre_switch = n_pre_switch > 0  # remove trials without future switch
    rew_switch = np.zeros(len(df))
    rew_switch[(df["history"] == "1R")] = -1
    switched = df["switched_actor"].values
    bstayed = [switch[subject_idx] == 0 for switch in switched]
    bswitched = [switch[subject_idx] == 1 for switch in switched]
    rew_switch[(df["history"] != "1R") & bstayed] = 0
    rew_switch[(df["history"] != "1R") & bswitched] = 1
    df["rew_switch"] = rew_switch
    # Filter df
    df = df[idx_pre_switch]
    return df
