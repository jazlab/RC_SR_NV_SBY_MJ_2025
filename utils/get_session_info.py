"""
This module contains functions for finding sessions and spike directories.
"""

import os
import pandas as pd
from pathlib import Path
from utils.LoadSession import findrootdir


def load_subject_names():
    rootdir = findrootdir()
    subject_names_path = f"{rootdir}/subject_names.csv"
    return pd.read_csv(subject_names_path)


def find_unit_idx(date, v_probe, unit_id):
    """
    Find the index of a unit in the master list.
    """
    root_dir = findrootdir()
    master_list_L = pd.read_csv(f"{root_dir}/master_list_L_0.csv")
    master_list_O = pd.read_csv(f"{root_dir}/master_list_O_0.csv")
    master_list = pd.concat([master_list_L, master_list_O])
    unit_idx = master_list[
        (master_list["date"] == int(date))
        & (master_list["v_probe"] == v_probe)
        & (master_list["unit_number"] == int(unit_id))
    ]["unit_index"].values[0]
    return unit_idx


def get_spk_metadata(spike_dir):
    vprobe_dir = os.path.dirname(spike_dir)
    vprobe = os.path.basename(vprobe_dir)
    session_dir = os.path.dirname(os.path.dirname(vprobe_dir))
    subject_names = load_subject_names()
    date = session_dir.split("/")[-1]
    subject_name = subject_names.loc[
        subject_names["date"] == int(date), f"subject{vprobe[-1]}"
    ].values[0]
    coords = subject_names.loc[
        subject_names["date"] == int(date), f"{subject_name}_coords"
    ].values[0]
    ks_dir = os.path.join(session_dir, "results", vprobe, "ks_output")
    return {
        "subject_name": subject_name,
        "date": date,
        "session_dir": session_dir,
        "vprobe": vprobe,
        "spike_dir": spike_dir,
        "coords": coords,
        "manual_label_file": os.path.join(ks_dir, "cluster_info.tsv"),
        "unit_trials_valid_file": os.path.join(
            spike_dir, "unit_trials_valid.csv"
        ),
    }


def get_behavior_df(spike_dir):
    behavior_dir = os.path.join(
        os.path.dirname(os.path.dirname(spike_dir)), "moog_events"
    )
    behavior_df = pd.read_csv(os.path.join(behavior_dir, "trial_info.csv"))
    return behavior_df


def get_unit_spike_dirs(dataT, metadata):
    # given a tdr data dictionary, return a dictionary of [unit_id]: spike_dir
    unit_spike_dirs = []
    from utils.LoadSession import findrootdir

    root_dir = findrootdir()
    data_dir = root_dir + "/social_O_L"
    subject_names = load_subject_names()
    for i_unit, unit_dict in enumerate(dataT["unit"]):
        session_id = unit_dict["session_id"]
        date = subject_names.loc[
            subject_names["session #"] == session_id, "date"
        ].values[0]
        v_probe = metadata["unit"]["probe"][i_unit]
        spike_dir = os.path.join(
            data_dir, f"{date}", "results", f"{v_probe}", "spikes"
        )
        unit_spike_dirs.append(spike_dir)
    return unit_spike_dirs


def filter_spike_dirs(spike_dirs, subject_name):
    """Filter spike_dirs to only those for a specific subject."""
    filtered_spike_dirs = []
    if subject_name == "any":
        return spike_dirs
    for spike_dir in spike_dirs:
        spk_metadata = get_spk_metadata(spike_dir)
        if spk_metadata["subject_name"] == subject_name:
            filtered_spike_dirs.append(spike_dir)
    return filtered_spike_dirs


def find_sessions_nested(base_dir, subject_name=None):
    """Recursively find all sessions in the directory for a subject."""
    spike_dirs = []
    probe_dirs = ["v_probe_1", "v_probe_2"]
    if subject_name == "any":
        probe_dirs = ["v_probe_1"]
    base_dir = os.path.join(base_dir, "social_O_L")
    for date in os.listdir(base_dir):
        date_dir = os.path.join(base_dir, date)
        if os.path.isdir(date_dir):
            results_dir = os.path.join(date_dir, "results")
            if os.path.isdir(results_dir):
                for v_probe in probe_dirs:
                    v_probe_dir = os.path.join(results_dir, v_probe)
                    if os.path.isdir(v_probe_dir):
                        spikes_dir = os.path.join(v_probe_dir, "spikes")
                        if os.path.isdir(spikes_dir):
                            spike_dirs.append(spikes_dir)
    if subject_name is not None:
        spike_dirs = filter_spike_dirs(spike_dirs, subject_name)
    return spike_dirs


def find_behavior_sessions(base_dir, experiment="social_O_L"):
    """Recursively find all sessions in the directory."""
    bhv_dirs = []
    base_dir = Path(base_dir) / experiment
    bhv_dirs = []
    for date_dir in base_dir.iterdir():
        if date_dir.is_dir():
            results_dir = date_dir / "results"
            if results_dir.is_dir():
                moog_dir = results_dir / "moog_events"
                if moog_dir.is_dir():
                    bhv_dirs.append(moog_dir)
                    continue
        print(f"Skipping {date_dir} for lack of moog_events")
    return bhv_dirs
