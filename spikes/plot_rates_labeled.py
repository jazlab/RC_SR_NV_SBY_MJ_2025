"""Create rasters and firing rate plots."""

import os
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib as mpl
import pandas as pd
import json
from ast import literal_eval
from spikes.spike_utils import get_condition_spike_times
from spikes.spike_utils import get_firing_rates_per_condition
from spikes.plot_conditional_firing_rates import (
    plot_conditional_firing_rates_overlap,
    plot_history_firing_rates,
)
from utils.get_session_info import load_subject_names
from utils.LoadSession import findrootdir


def make_plot_for_unit(session, unit_id, option="outcome"):
    """Make raster and histogram plots for each cluster."""

    ############################################################################
    ####  LOAD DATA
    ############################################################################
    root_dir = findrootdir()
    save_fig_dir = os.path.join(root_dir, "plots_paper")

    mpl.rcParams["pdf.fonttype"] = 42
    print("LOADING DATA")
    subject_names = load_subject_names()
    spikes_dir = f"{root_dir}/social_O_L/{session}/spikes"
    print(f"spikes_dir: {spikes_dir}")
    date = spikes_dir.split("/")[-4]
    v_probe = spikes_dir.split("/")[-2]
    if v_probe == "v_probe_1":
        subject = subject_names.loc[
            subject_names["date"] == int(date), "subject1"
        ].values[0]
    elif v_probe == "v_probe_2":
        subject = subject_names.loc[
            subject_names["date"] == int(date), "subject2"
        ].values[0]
    if subject == "O":
        agent_idx = 0
    elif subject == "L":
        agent_idx = 1
    data_dir = os.path.dirname(os.path.dirname(spikes_dir))
    data_dir = data_dir.replace("phys_raw", "results")
    print(f"data_dir: {data_dir}")
    trials_path = os.path.join(
        data_dir, "open_ephys_events/open_ephys_trials.json"
    )
    print(f"trials_dir: {trials_path}")
    behavior_path = os.path.join(data_dir, "moog_events/trial_info.csv")
    behavioral_df = pd.read_csv(behavior_path)
    behavioral_df["agent_choices"] = behavioral_df["agent_choices"].apply(
        literal_eval
    )
    behavioral_df["reaction_time"] = behavioral_df["reaction_time"].apply(
        literal_eval
    )
    behavioral_df["switched_actor"] = behavioral_df["switched_actor"].apply(
        literal_eval
    )

    print(f"Number of trials in behavioral data = {len(behavioral_df)}")

    spike_times_per_trial_path = os.path.join(
        spikes_dir, "spike_times_per_trial.json"
    )
    spike_times_per_trial = json.load(open(spike_times_per_trial_path, "r"))

    # filter spike_times_per_trial to only include unit_id
    if unit_id is not None:
        spike_times_per_trial = {
            k: v for k, v in spike_times_per_trial.items() if k == unit_id
        }
    for event in ["fdbk", "choice"]:
        condition_spike_times = get_condition_spike_times(
            spike_times_per_trial, behavioral_df, agent_idx, event=event
        )
        windowstart = -0.1
        windowend = 1
        if event == "choice":
            windowstart = -1
            windowend = 0.1
        time_bins, firing_rates = get_firing_rates_per_condition(
            condition_spike_times, windowstart, windowend
        )

        if option == "outcome":
            filenam_prefix = f"Fig3F_{date}_{v_probe}"
            plot_conditional_firing_rates_overlap(
                firing_rates, time_bins, save_fig_dir, filenam_prefix, event
            )
        elif option == "history":
            if event == "fdbk":
                filenam_prefix = f"Fig3H_{date}_{v_probe}"
                plot_history_firing_rates(
                    firing_rates, time_bins, save_fig_dir, filenam_prefix, event
                )

    return


def main():
    # units presented in paper: 20230215_v_probe_1_16, 20230214_v_probe_2_279
    # 20230216_v_probe_2_118, 20230123_v_probe_1_11
    session_unit_id_pairs = [
        ("20230216/results/v_probe_2", "118"),
        ("20230123/results/v_probe_1", "11"),
    ]
    for session, unit_id in session_unit_id_pairs:
        make_plot_for_unit(session, unit_id, option="outcome")
    session_unit_id_pairs = [
        ("20230214/results/v_probe_2", "279"),
        ("20230215/results/v_probe_1", "16"),
    ]
    for session, unit_id in session_unit_id_pairs:
        make_plot_for_unit(session, unit_id, option="history")


if __name__ == "__main__":
    main()
