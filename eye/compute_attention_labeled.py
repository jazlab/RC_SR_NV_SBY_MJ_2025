"""
This script contains code for computing the "attention", which is the same as
 P(gaze) reported in the paper.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from copy import deepcopy
from utils.LoadSession import findrootdir


def find_closest_indices(t_box, t):
    """
    For each time in t, find the index of the closest time in t_box.
    Both t_box and t should be sorted in ascending order.
    """
    # Search for insertion positions.
    idx_array = np.searchsorted(t_box, t, side="left")

    # Clip to valid range (1 ... len(t_box)-1) for neighbor checks
    idx_array = np.clip(idx_array, 1, len(t_box) - 1)

    # Compare distance to left vs. right neighbors
    left_indices = idx_array - 1
    right_indices = idx_array
    left_dist = np.abs(t_box[left_indices] - t)
    right_dist = np.abs(t_box[right_indices] - t)

    # Pick whichever is closer
    closest_idx = np.where(left_dist < right_dist, left_indices, right_indices)
    return closest_idx


def get_attention(eye_pos, box_attention, sample_length, control=False):
    """
    Compute fraction of eye samples falling within bounding boxes
    (defined by box_attention), after matching each eye time to the
    closest bounding-box time via find_closest_indices.
    If control=True, x-bounds are flipped horizontally around x=0.5.
    """
    x, y, t = eye_pos
    if len(x) == 0:
        return np.nan

    # Convert bounding box lists to NumPy arrays
    t_box = np.array(box_attention["relative_time"])
    min_x_arr = np.array(box_attention["min_x"])
    max_x_arr = np.array(box_attention["max_x"])
    min_y_arr = np.array(box_attention["min_y"])
    max_y_arr = np.array(box_attention["max_y"])

    # 1) Find closest box index for each eye time
    closest_ids = find_closest_indices(t_box, t)

    # 2) Extract bounding box limits
    cmin_x = min_x_arr[closest_ids]
    cmax_x = max_x_arr[closest_ids]
    cmin_y = min_y_arr[closest_ids]
    cmax_y = max_y_arr[closest_ids]

    # 3) If control condition: flip bounding box horizontally around x=0.5
    if control:
        widths = cmax_x - cmin_x
        cmin_x_new = 1.0 - cmax_x
        cmax_x_new = cmin_x_new + widths
        # Make sure to keep min_x <= max_x
        cmin_x = np.minimum(cmin_x_new, cmax_x_new)
        cmax_x = np.maximum(cmin_x_new, cmax_x_new)

    # 4) Check which eye points fall inside bounding boxes
    in_box = (x >= cmin_x) & (x <= cmax_x) & (y >= cmin_y) & (y <= cmax_y)
    return np.sum(in_box) / sample_length


def compute_trial_attention_per_subject(
    moog_df,
    box_attentions,
    eye_t,
    eyex_v,
    eyey_v,
    offset_0,
    offset_1,
    tolerance,
    alignment="navoff",
):
    """
    Given MOOG df (length N), MWorks df (<= N or >= N), bounding boxes,
    and eye data, compute the attention metrics trial-by-trial.

    The moog_df is the 'left' dataset with full set of trials we want
    to maintain. The mworks_df is merged onto it. Any row lacking navon/navoff
    will get NaNs in all attention columns.
    """
    n_moog = len(moog_df)
    # Arrays to store results (same length as moog_df)
    attention_narrow = [np.nan] * n_moog
    attention_wide = [np.nan] * n_moog
    attention_full = [np.nan] * n_moog
    attention_control_narrow = [np.nan] * n_moog
    attention_control_wide = [np.nan] * n_moog
    attention_control_full = [np.nan] * n_moog

    for i in range(n_moog):
        # If navoff or navon are NaN, skip (no data)
        if pd.isna(moog_df.loc[i, "navoff"]) or pd.isna(
            moog_df.loc[i, "navon"]
        ):
            continue

        # Get the eye data mask
        nav_on = moog_df.loc[i, "navon"]
        nav_off = moog_df.loc[i, "navoff"]
        t_onset = moog_df.loc[i, alignment]

        mask = (eye_t >= t_onset + offset_0) & (eye_t <= t_onset + offset_1)
        duration = offset_1 - offset_0
        # use the larger of the two sample lengths
        if offset_1 == 0:
            # Special case: if offset_1 is 0, include entire navigation phase
            mask = (eye_t >= nav_on) & (eye_t <= nav_off)
            duration = nav_off - nav_on
        if np.sum(mask) == 0:  # eyelink offline for this trial
            continue
        sample_length = max(duration * 1000, sum(mask))
        # Eye data within the relevant time window
        eye_time_trial = eye_t[mask] - nav_on
        x_trial = eyex_v[mask]
        y_trial = eyey_v[mask]

        # 1) Narrow bounding boxes
        att_narrow = get_attention(
            [x_trial, y_trial, eye_time_trial],
            box_attentions[i],
            sample_length,
            control=False,
        )
        att_control_narrow = get_attention(
            [x_trial, y_trial, eye_time_trial],
            box_attentions[i],
            sample_length,
            control=True,
        )
        attention_narrow[i] = att_narrow
        attention_control_narrow[i] = att_control_narrow

        # 2) Wide bounding boxes
        wide_box = deepcopy(box_attentions[i])
        # Decide if the mean bounding box is left or right
        if np.mean(wide_box["max_x"]) < 0.5:
            wide_box["min_x"] = [-tolerance] * len(wide_box["min_x"])
            wide_box["max_x"] = [0.5 + tolerance] * len(wide_box["max_x"])
        else:
            wide_box["min_x"] = [0.5 - tolerance] * len(wide_box["min_x"])
            wide_box["max_x"] = [1.0 + tolerance] * len(wide_box["max_x"])

        att_wide = get_attention(
            [x_trial, y_trial, eye_time_trial],
            wide_box,
            sample_length,
            control=False,
        )
        att_control_wide = get_attention(
            [x_trial, y_trial, eye_time_trial],
            wide_box,
            sample_length,
            control=True,
        )
        attention_wide[i] = att_wide
        attention_control_wide[i] = att_control_wide

        # 3) Full bounding boxes
        full_box = deepcopy(wide_box)
        full_box["min_y"] = [0.0 - tolerance] * len(full_box["min_y"])
        full_box["max_y"] = [1.0 + tolerance] * len(full_box["max_y"])

        att_full = get_attention(
            [x_trial, y_trial, eye_time_trial],
            full_box,
            sample_length,
            control=False,
        )
        att_control_full = get_attention(
            [x_trial, y_trial, eye_time_trial],
            full_box,
            sample_length,
            control=True,
        )

        attention_full[i] = att_full
        attention_control_full[i] = att_control_full

    # Convert to DataFrame
    df_attention = pd.DataFrame(
        {
            "trial_num": moog_df["trial_num"],
            "attention_narrow": attention_narrow,
            "attention_wide": attention_wide,
            "attention_full": attention_full,
            "attention_control_narrow": attention_control_narrow,
            "attention_control_wide": attention_control_wide,
            "attention_control_full": attention_control_full,
        }
    )
    return df_attention


def compute_attention(date):
    tolerance = 0.075  # how much to extend the bounding box
    offset_0 = 0  # start offset from alignment
    offset_1 = 0.5  # end offset from alignment
    alignment = "navoff"
    root_dir = findrootdir()

    # ------------------------------------------------------------------------
    # 1) Load MOOG trial info & navigation info
    # ------------------------------------------------------------------------
    moog_dir = f"{root_dir}/social_O_L/{date}/results/moog_events"
    if not os.path.exists(moog_dir):
        print(f"Moog events folder does not exist for {date}")
        return

    trial_info_moog = pd.read_csv(f"{moog_dir}/trial_info.csv")
    nav_info = pd.read_csv(f"{moog_dir}/nav_info.csv")

    n_trials_moog = len(trial_info_moog)

    # ------------------------------------------------------------------------
    # 2) Build bounding boxes over time for each trial (box_attentions)
    #    (One dictionary per trial index in MOOG order)
    # ------------------------------------------------------------------------
    box_attentions = []
    box_attention_blank = {
        "trial_num": [],
        "relative_time": [],
        "min_x": [],
        "max_x": [],
        "min_y": [],
        "max_y": [],
    }

    # We'll store box_attentions in the order of moog trials:
    #   box_attentions[i] will correspond to moog trial i
    # So first build an array of empty dicts for each trial
    for _ in range(n_trials_moog):
        box_attentions.append(deepcopy(box_attention_blank))

    # We need a quick index map from trial_num -> moog row
    moog_trialnum_to_index = {
        t: i for i, t in enumerate(trial_info_moog["trial_num"].values)
    }

    for _, row in nav_info.iterrows():
        trial_num = row["trial_idx"] + 1
        if trial_num not in moog_trialnum_to_index:
            # If nav_info has a trial that isn't in moog, skip
            continue

        i_moog = moog_trialnum_to_index[trial_num]
        # We can track the number of nav_info rows for each trial i_moog
        # For each row in that trial, we do i_nav = len() of "relative_time"
        i_nav_for_trial = len(box_attentions[i_moog]["relative_time"])
        t_nav = i_nav_for_trial / 60.0

        box_attentions[i_moog]["trial_num"].append(trial_num)
        box_attentions[i_moog]["relative_time"].append(t_nav)

        agent_x = row["agent_pos_x"]
        agent_y = row["agent_pos_y"]
        target_x = row["target_pos_x"]
        target_y = row["target_pos_y"]

        if target_x != 0:
            min_x = min(agent_x, target_x) - tolerance
            max_x = max(agent_x, target_x) + tolerance
            min_y = min(agent_y, target_y) - tolerance
            max_y = max(agent_y, target_y) + tolerance
        else:
            min_x = agent_x - tolerance
            max_x = agent_x + tolerance
            min_y = agent_y - tolerance
            max_y = agent_y + tolerance

        box_attentions[i_moog]["min_x"].append(min_x)
        box_attentions[i_moog]["max_x"].append(max_x)
        box_attentions[i_moog]["min_y"].append(min_y)
        box_attentions[i_moog]["max_y"].append(max_y)

    # Vectorized rounding to 3 decimals
    for i_moog in range(n_trials_moog):
        ba = box_attentions[i_moog]
        for field in ["relative_time", "min_x", "max_x", "min_y", "max_y"]:
            arr = np.round(np.array(ba[field]), 3)
            ba[field] = arr.tolist()

    # Optionally save as JSON for reference (can be skipped if not needed)
    box_json_path = f"{moog_dir}/box_attention.json"
    with open(box_json_path, "w") as f:
        json.dump(box_attentions, f)

    # ------------------------------------------------------------------------
    # 3) Subject O: Merge MWorks data onto MOOG by trial_num
    # ------------------------------------------------------------------------
    mwkdir_O = "mworks_events"
    mworks_dir_O = f"{root_dir}/social_O_L/{date}/results/{mwkdir_O}"

    with open(os.path.join(mworks_dir_O, "trial_info.json"), "r") as jf:
        trial_info_mworks_O = json.load(jf)

    # Convert MWorks data to a DataFrame
    df_mworks_O = pd.DataFrame(trial_info_mworks_O)

    # Merge: keep all moog trials, add columns navon, navoff from MWorks if available
    merged_moog_O = pd.merge(
        trial_info_moog,
        df_mworks_O[["trial_num", "navon", "navoff"]],
        on="trial_num",
        how="left",  # left-join: keep all MOOG rows
        suffixes=("", "_mwk"),
    )

    # Load eye data
    eye_t_O = np.load(os.path.join(mworks_dir_O, "eyex_t.npy"))
    eyex_v_O = np.load(os.path.join(mworks_dir_O, "eyex_v.npy"))
    eyey_v_O = np.load(os.path.join(mworks_dir_O, "eyey_v.npy"))

    # Transform eye positions from [-20, 20] to [0, 1]
    eyex_v_O = (eyex_v_O + 20.0) / 40.0
    eyey_v_O = (eyey_v_O + 20.0) / 40.0

    # ------------------------------------------------------------------------
    # 4) Compute attention for subject O (aligned w/ MOOG rows)
    # ------------------------------------------------------------------------
    df_attention_O = compute_trial_attention_per_subject(
        merged_moog_O,
        box_attentions,  # same order as moog trials
        eye_t_O,
        eyex_v_O,
        eyey_v_O,
        offset_0,
        offset_1,
        tolerance,
        alignment=alignment,
    )

    # Add columns for subject O to MOOG DataFrame
    trial_info_moog["attention"] = df_attention_O["attention_narrow"]
    trial_info_moog["attention_control"] = df_attention_O[
        "attention_control_narrow"
    ]
    trial_info_moog["attention_wide"] = df_attention_O["attention_wide"]
    trial_info_moog["attention_control_wide"] = df_attention_O[
        "attention_control_wide"
    ]
    trial_info_moog["attention_full"] = df_attention_O["attention_full"]
    trial_info_moog["attention_control_full"] = df_attention_O[
        "attention_control_full"
    ]

    # Save CSV of attentions for subject O (same length as MOOG)
    df_attention_O.to_csv(
        os.path.join(mworks_dir_O, f"attention_per_trial.csv"),
        index=False,
    )

    # ------------------------------------------------------------------------
    # 5) Subject L: same procedure
    # ------------------------------------------------------------------------
    mwkdir_L = "mworks_events_2nd_eyelink"
    mworks_dir_L = f"{root_dir}/social_O_L/{date}/results/{mwkdir_L}"

    with open(os.path.join(mworks_dir_L, "trial_info.json"), "r") as jf:
        trial_info_mworks_L = json.load(jf)

    df_mworks_L = pd.DataFrame(trial_info_mworks_L)
    if "trial_num" not in df_mworks_L.columns:
        df_mworks_L["trial_num"] = np.arange(len(df_mworks_L))

    merged_moog_L = pd.merge(
        trial_info_moog,
        df_mworks_L[["trial_num", "navon", "navoff"]],
        on="trial_num",
        how="left",
        suffixes=("", "_mwk"),
    )

    # Load eye data for L
    eye_t_L = np.load(os.path.join(mworks_dir_L, "eyex_t.npy"))
    eyex_v_L = np.load(os.path.join(mworks_dir_L, "eyex_v.npy"))
    eyey_v_L = np.load(os.path.join(mworks_dir_L, "eyey_v.npy"))

    # Transform eye positions
    eyex_v_L = (eyex_v_L + 20.0) / 40.0
    eyey_v_L = (eyey_v_L + 20.0) / 40.0

    # Compute attention for subject L
    df_attention_L = compute_trial_attention_per_subject(
        merged_moog_L,
        box_attentions,
        eye_t_L,
        eyex_v_L,
        eyey_v_L,
        offset_0,
        offset_1,
        tolerance,
        alignment=alignment,
    )

    # Add columns for subject L
    trial_info_moog["attention_L"] = df_attention_L["attention_narrow"]
    trial_info_moog["attention_control_L"] = df_attention_L[
        "attention_control_narrow"
    ]
    trial_info_moog["attention_wide_L"] = df_attention_L["attention_wide"]
    trial_info_moog["attention_control_wide_L"] = df_attention_L[
        "attention_control_wide"
    ]
    trial_info_moog["attention_full_L"] = df_attention_L["attention_full"]
    trial_info_moog["attention_control_full_L"] = df_attention_L[
        "attention_control_full"
    ]

    # Save CSV of attentions for subject L (same length as MOOG)
    df_attention_L.to_csv(
        os.path.join(mworks_dir_L, f"attention_per_trial.csv"),
        index=False,
    )

    # ------------------------------------------------------------------------
    # 6) Save the final updated trial_info_moog to CSV
    # ------------------------------------------------------------------------
    trial_info_moog.to_csv(f"{moog_dir}/trial_info.csv", index=False)


def main():
    date = sys.argv[1]
    compute_attention(date)


if __name__ == "__main__":
    main()
