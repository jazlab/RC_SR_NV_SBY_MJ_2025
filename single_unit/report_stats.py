"""
This script performs the ROC analysis for the single-unit data.
"""

import os
import sys
import numpy as np
import json
import pandas as pd
from utils.get_session_info import load_subject_names, find_sessions_nested
from utils.stat_roc_window import roc_analysis
from utils.LoadSession import load_data_for_session
from multiprocessing import Pool
from tqdm import tqdm


# Define analyze_factor_phase as a top-level function
def analyze_factor_phase(args):
    (
        spike_times,
        trials,
        behavioral_df,
        valid_trials,
        agent_id,
        config,
        factor,
        phase,
    ) = args
    config["factor"] = factor
    config["phase"] = phase
    return roc_analysis(
        spike_times, trials, behavioral_df, valid_trials, agent_id, config
    )


def analyze_neuron(
    spike_times, trials, behavioral_df, valid_trials, subject, config
):
    factors = [config["factor"]]
    phases = [
        "choice",
        "fdbk",
    ]

    agent_id = 0 if subject == "O" else 1
    neuron_stat = {
        factor: {phase: {} for phase in phases} for factor in factors
    }

    # Prepare arguments for each factor-phase combination
    args = [
        (
            spike_times,
            trials,
            behavioral_df,
            valid_trials,
            agent_id,
            dict(config),
            factor,
            phase,
        )
        for factor in factors
        for phase in phases
    ]

    # Using multiprocessing to parallelize the analysis
    # n_processes = min(len(args), os.cpu_count())
    run_serial = False
    if run_serial:
        results = [analyze_factor_phase(arg) for arg in args]
    else:
        n_processes = 20
        print(f"Using {n_processes} processes to analyze the data")
        with Pool(n_processes) as pool:
            results = pool.map(analyze_factor_phase, args)

    # Organizing the results back into the neuron_stat dictionary
    for i, (factor, phase) in enumerate(
        [(factor, phase) for factor in factors for phase in phases]
    ):
        neuron_stat[factor][phase] = results[i]

    return neuron_stat


def analyze_session(session_dir, subject, config):
    spikes_dir = os.path.join(session_dir, "spikes")
    session_data = load_data_for_session(spikes_dir, config)
    if session_data is None:
        return None
    if not os.path.exists(os.path.join(spikes_dir, "unit_trials_valid.csv")):
        return None
    unit_trials_valid = np.loadtxt(
        os.path.join(spikes_dir, "unit_trials_valid.csv"), delimiter=","
    )
    if session_data is None:
        return None  # Skip this session if data loading failed
    (
        spike_times_per_cluster,
        trials,
        behavioral_df,
        valid_units,
    ) = session_data
    cluster_ids = list(spike_times_per_cluster.keys())
    # figure out master list index
    root_dir = findrootdir()
    master_list_path = os.path.join(
        root_dir, "master_list_{}.csv".format(subject)
    )
    master_list = pd.read_csv(master_list_path)
    # filter dataframe to only include the current session
    master_list = master_list[
        master_list["date"] == int(session_dir.split("/")[-3])
    ]
    # if master list is empty, return None
    if master_list.empty:
        return None

    session_stats = {}
    for neuron_id in valid_units:
        spike_times = spike_times_per_cluster[str(neuron_id)]
        unit_idx = cluster_ids.index(neuron_id)
        valid_trials = np.where(unit_trials_valid[unit_idx, :] == 1)[0]
        if len(spike_times) == 0:
            continue
        # find number of 2NR trials
        if valid_trials.max() >= len(behavioral_df):
            continue
        unit_df = behavioral_df.loc[valid_trials]
        n_2NR = len(unit_df[unit_df["history"] == "2NR"])
        if n_2NR < 20:
            continue
        results = analyze_neuron(
            spike_times,
            trials,
            behavioral_df,
            valid_trials,
            subject,
            config,
        )
        unit_idx_master = master_list[
            (master_list["unit_number"] == int(neuron_id))
        ]["unit_index"].values[0]
        unit_idx_master = int(unit_idx_master)
        session_stats[unit_idx_master] = results
    # Save the results
    stats_dir = spikes_dir.replace("spikes", "stats")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    report_path = os.path.join(stats_dir, f"roc_stats_{config['factor']}.json")
    # delete existing file
    if os.path.exists(report_path):
        os.remove(report_path)
    for key, value in session_stats.items():
        if isinstance(value, np.ndarray):
            session_stats[key] = value.tolist()
    with open(report_path, "w") as f:
        json.dump(session_stats, f, indent=4)


def main(session_dir):
    # Split the path into components
    config = {
        "binsize": 0.1,
        "stepsize": 0.05,
        "xmin": -0.6,
        "xmax": 0.6,
        "xstep": 0.6,  # this is for the window sweep
        "factor": "reward",
    }
    path_components = os.path.normpath(session_dir).split(os.sep)
    # Get the desired components
    session_date = path_components[-3]
    v_probe = path_components[-1]

    subject_names = load_subject_names()
    subject = subject_names.loc[
        subject_names["date"] == int(session_date), f"subject{v_probe[-1]}"
    ].values[0]
    analyze_session(session_dir, subject, config)


if __name__ == "__main__":
    # check if input argument is given
    if len(sys.argv) > 1:
        session_dir = sys.argv[1]
        main(session_dir)
        sys.exit(0)
    from utils.LoadSession import findrootdir

    root_dir = findrootdir()
    # find all session directories
    spike_dirs = find_sessions_nested(root_dir)
    for spike_dir in tqdm(
        spike_dirs,
        desc="sessions",
        leave=False,
    ):
        session_dir = os.path.dirname(spike_dir)
        main(session_dir)
