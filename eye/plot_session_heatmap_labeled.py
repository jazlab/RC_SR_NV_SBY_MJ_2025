"""
This code plots heatmaps of eye gaze position on the screen.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

from utils.LoadSession import findrootdir
from attention_plots import plot_heatmap_from_trials


def collect_eye_data_for_date(date, subject="O", congruent=True):
    """
    Loads the data for a single date, organizes it by outcome type, and returns
    all relevant arrays in a dictionary. This can then be used to plot or combine
    with data from other dates.
    """
    root_dir = findrootdir()
    moog_dir = f"{root_dir}/social_O_L/{date}/results/moog_events"
    mwk_dname = (
        "mworks_events" if subject == "O" else "mworks_events_2nd_eyelink"
    )
    json_path = os.path.join(
        root_dir, "social_O_L", date, "results", mwk_dname, "trial_info.json"
    )

    # check if the date is valid
    if not os.path.exists(moog_dir) or not os.path.exists(json_path):
        return None

    # Load moog info
    trial_info_moog = pd.read_csv(f"{moog_dir}/trial_info.csv")

    # Load mworks info
    with open(json_path, "r") as jf:
        trial_info_mworks = json.load(jf)

    nav_on = [trial["navon"] for trial in trial_info_mworks]
    nav_off = [trial["navoff"] for trial in trial_info_mworks]
    trial_num_mworks = [trial["trial_num"] for trial in trial_info_mworks]
    trial_num_moog = trial_info_moog["trial_num"].to_numpy()

    alignment = nav_off
    offset_0 = 0.0
    offset_1 = 0.5

    # Check that moog and mworks have at least same number
    assert len(trial_info_moog) >= len(
        trial_info_mworks
    ), f"Moog trials ({len(trial_info_moog)}) < MWorks trials ({len(trial_info_mworks)}) for date {date}."

    # Gather moog columns
    history = trial_info_moog["history"].to_numpy()
    player = trial_info_moog["player"].to_numpy()
    choice_a0 = trial_info_moog["choice_a0"].to_numpy()
    choice_a1 = trial_info_moog["choice_a1"].to_numpy()

    # Load eye data
    eyex_t = np.load(
        f"{root_dir}/social_O_L/{date}/results/{mwk_dname}/eyex_t.npy"
    )
    eyex_v = np.load(
        f"{root_dir}/social_O_L/{date}/results/{mwk_dname}/eyex_v.npy"
    )
    eyey_t = np.load(
        f"{root_dir}/social_O_L/{date}/results/{mwk_dname}/eyey_t.npy"
    )
    eyey_v = np.load(
        f"{root_dir}/social_O_L/{date}/results/{mwk_dname}/eyey_v.npy"
    )

    # Transform eyex and eyey from [-20,20] to [0,1]
    eyex_v = (eyex_v + 20) / 40.0
    eyey_v = (eyey_v + 20) / 40.0

    # Identity: 0 if subject="O", else 1
    identity = 0 if subject == "O" else 1

    # Build containers for outcome-based data
    data = {
        "eyex_outcome_actor_left": [],
        "eyex_outcome_actor_right": [],
        "eyex_outcome_observer_left": [],
        "eyex_outcome_observer_right": [],
        "eyey_outcome_actor_left": [],
        "eyey_outcome_actor_right": [],
        "eyey_outcome_observer_left": [],
        "eyey_outcome_observer_right": [],
        "time_eye_outcome_actor_left": [],
        "time_eye_outcome_actor_right": [],
        "time_eye_outcome_observer_left": [],
        "time_eye_outcome_observer_right": [],
    }

    # Iterate over trials
    for i in range(1, len(nav_on)):
        trial_num = trial_num_mworks[i]
        # Find matching moog trial
        idx_moog_arr = np.where(trial_num_moog == trial_num)[0]
        if len(idx_moog_arr) == 0:
            # If there's no corresponding moog trial, skip
            continue
        idx_moog = idx_moog_arr[0]

        play = player[idx_moog]
        action_self = choice_a0[idx_moog]
        action_other = choice_a1[idx_moog]
        if subject == "L":
            # For subject = L, swap the meaning of self vs other
            action_self = choice_a1[idx_moog]
            action_other = choice_a0[idx_moog]

        if congruent:
            # Skip incongruent trials
            if action_self != action_other:
                continue
        else:
            # Skip congruent trials
            if action_self == action_other:
                continue

        # Skip rewarded trials
        if history[idx_moog] == "1R":
            continue

        # window for plot - change here for different window
        t_onset = alignment[i]
        mask = (eyex_t >= t_onset + offset_0) & (eyex_t <= t_onset + offset_1)
        if (offset_0 == 0) and (offset_1 == 0):
            # Special case: if offset_1 is 0, include entire navigation phase
            mask = (eyex_t >= nav_on[i]) & (eyex_t <= nav_off[i])
        # mask = (eyex_t >= alignment[i] + offset_on) & (
        #     eyex_t <= navigation_off[i]
        # )

        if play != identity and action_other == -1.0:
            data["eyex_outcome_observer_left"].append(eyex_v[mask])
            data["eyey_outcome_observer_left"].append(eyey_v[mask])
            data["time_eye_outcome_observer_left"].append(
                eyex_t[mask] - alignment[i]
            )

        if play != identity and action_other == 1.0:
            data["eyex_outcome_observer_right"].append(eyex_v[mask])
            data["eyey_outcome_observer_right"].append(eyey_v[mask])
            data["time_eye_outcome_observer_right"].append(
                eyex_t[mask] - alignment[i]
            )

        if play == identity and action_self == -1.0:
            data["eyex_outcome_actor_left"].append(eyex_v[mask])
            data["eyey_outcome_actor_left"].append(eyey_v[mask])
            data["time_eye_outcome_actor_left"].append(
                eyex_t[mask] - alignment[i]
            )

        if play == identity and action_self == 1.0:
            data["eyex_outcome_actor_right"].append(eyex_v[mask])
            data["eyey_outcome_actor_right"].append(eyey_v[mask])
            data["time_eye_outcome_actor_right"].append(
                eyex_t[mask] - alignment[i]
            )

    return data


def plot_and_save_heatmaps(day_data, save_dir, prefix=""):
    """
    Given the dictionary of day_data (i.e. outcome-based eyex/eyey arrays),
    create the heatmaps and save them into `save_dir`. The `prefix` can be
    used to identify per-day or combined plots.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # You may want a different naming scheme, or multiple calls with different kwargs
    # e.g. you can pass an output filename into plot_heatmap_from_trials if it supports that
    fig, ax = plot_heatmap_from_trials(
        day_data["eyex_outcome_actor_right"],
        day_data["eyey_outcome_actor_right"],
    )
    ax.set_title(f"Actor Right")
    fig.savefig(f"{save_dir}/{prefix}_actor_right.pdf", dpi=300, format="pdf")
    plt.close(fig)  # close to free memory, if desired

    fig, ax = plot_heatmap_from_trials(
        day_data["eyex_outcome_actor_left"], day_data["eyey_outcome_actor_left"]
    )
    ax.set_title(f"Actor Left")
    fig.savefig(f"{save_dir}/{prefix}_actor_left.pdf", dpi=300, format="pdf")
    plt.close(fig)

    fig, ax = plot_heatmap_from_trials(
        day_data["eyex_outcome_observer_right"],
        day_data["eyey_outcome_observer_right"],
    )
    ax.set_title(f"Observer Right")
    fig.savefig(
        f"{save_dir}/{prefix}_observer_right.pdf", dpi=300, format="pdf"
    )
    plt.close(fig)

    fig, ax = plot_heatmap_from_trials(
        day_data["eyex_outcome_observer_left"],
        day_data["eyey_outcome_observer_left"],
    )
    ax.set_title(f"Observer Left")
    fig.savefig(f"{save_dir}/{prefix}_observer_left.pdf", dpi=300, format="pdf")
    plt.close(fig)


def combine_data(data_list):
    """
    Given a list of dictionaries (each from collect_eye_data_for_date),
    combine them into a single dictionary with appended arrays.
    """
    combined = {
        "eyex_outcome_actor_left": [],
        "eyex_outcome_actor_right": [],
        "eyex_outcome_observer_left": [],
        "eyex_outcome_observer_right": [],
        "eyey_outcome_actor_left": [],
        "eyey_outcome_actor_right": [],
        "eyey_outcome_observer_left": [],
        "eyey_outcome_observer_right": [],
        "time_eye_outcome_actor_left": [],
        "time_eye_outcome_actor_right": [],
        "time_eye_outcome_observer_left": [],
        "time_eye_outcome_observer_right": [],
    }

    for d in data_list:
        for k in combined.keys():
            combined[k].extend(d[k])

    return combined


def main1():
    """
    Usage:
    python this_script.py 20230101 20230102 20230216 ...
    or you can define the list of dates inside code.
    """
    # If passing in via command line, a list of dates might be sys.argv[1:]
    # Example:
    #   python script.py 20230216 20230217 ...
    congruent = True
    congruent_label = "congruent" if congruent else "incongruent"
    dates = sys.argv[1:]
    if not dates:
        # fallback if no arguments are given
        dates = (
            pd.date_range(start="2022-12-23", end="2023-04-01")
            .strftime("%Y%m%d")
            .tolist()
        )

    all_data_both = []
    root_dir = findrootdir()

    def process_date(date, subject):
        day_data = collect_eye_data_for_date(
            date, subject=subject, congruent=congruent
        )
        if day_data is None:
            return None, None

        mwk_dname = (
            "mworks_events" if subject == "O" else "mworks_events_2nd_eyelink"
        )
        save_dir = os.path.join(
            root_dir, "social_O_L", date, "results", mwk_dname
        )
        # Plot & save per-day heatmaps
        # plot_and_save_heatmaps(day_data, save_dir, prefix=f"{date}_{contruent_label}")

        return day_data, save_dir

    for subject in ["O", "L"]:
        all_day_data = []

        # Use ThreadPoolExecutor to process dates in parallel
        with ThreadPoolExecutor() as executor:
            future_to_date = {
                executor.submit(process_date, date, subject): date
                for date in dates
            }
            for future in as_completed(future_to_date):
                try:
                    day_data, save_dir = future.result()
                    if day_data:
                        all_day_data.append(day_data)
                        all_data_both.append(day_data)
                except Exception as e:
                    print(f"Error processing {future_to_date[future]}: {e}")

        # Now combine all data into one structure
        combined_data = combine_data(all_day_data)

        # Optionally, plot and save a combined heatmap
        combined_save_dir = os.path.join(root_dir, "plots_paper")
        plot_and_save_heatmaps(
            combined_data,
            combined_save_dir,
            prefix=f"FigS1A_D_{subject}_{congruent_label}",
        )


if __name__ == "__main__":
    main1()
