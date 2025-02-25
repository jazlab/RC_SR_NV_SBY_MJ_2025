"""
This module contain functions to load session data for the TDR analysis.
"""

import json
import numpy as np
import os
import pandas as pd
from pyTdr.tdrUpdateBehavior import update_behavior
from pyTdr.tdrGetTaskVars import add_task_vars
from pyTdr.tdrUtils import load_config
from utils.get_session_info import load_subject_names
from utils.get_session_info import get_spk_metadata
from ast import literal_eval
import multiprocessing
import logging

config = load_config()
logging.basicConfig(level=logging.ERROR, filename="process_unit_errors.log")


def get_gate_rates_per_unit(
    spike_times, trials, valid_trials, behavioral_df, window, bin_size
):
    timebins = np.arange(
        window[0],
        window[1] + bin_size,
        bin_size,
    )
    n_trials = len(trials)
    rate_matrix = np.zeros([n_trials * 15, len(timebins) - 1])
    task_variable = {
        "trial_num": [],
        "gate_captures": [],
        "gate_number": [],
    }
    i_row = 0
    for trial in trials:
        trial_num = trial["trial_num"]
        nav_on = trial["relative_phase_times"][1]
        if trial_num not in valid_trials:
            continue
        if trial_num not in behavioral_df["trial_num"].values:
            continue
        gate_touch_times = np.array(trial["t_gate_touches"])
        for i_gate, gate_time in enumerate(gate_touch_times):
            # Get the spike times for this gate touch
            t_gate = gate_time - nav_on
            # calculate the number of gate irrespective of capture
            gate_number = t_gate // 0.35  # 1-15 chronological order of gate
            # first gate can happen within 0.35s
            if t_gate < 0.35:
                gate_number = 1
            if not (gate_number >= 1 and gate_number < 16):
                print(
                    f"gate number out of range on trial {trial_num} for gate {i_gate}"
                )
                print(f"gate time: {gate_time}, nav_on: {nav_on}")
                continue
            gate_touch_spikes = spike_times - (gate_time + trial["t_start"])
            gate_touch_spikes = gate_touch_spikes[
                (gate_touch_spikes >= window[0])
                & (gate_touch_spikes <= window[1] + bin_size)
            ]
            # Bin the spike times
            binned_rates, _ = np.histogram(
                gate_touch_spikes,
                bins=timebins,
            )
            if np.any(np.isnan(binned_rates)):
                print("Binned rates contain NaNs")
                raise ValueError("Binned rates contain NaNs")
            # put the binned rates to the matrix
            rate_matrix[i_row, :] = binned_rates
            # add the task variables
            task_variable["trial_num"].append(trial_num)
            task_variable["gate_captures"].append(i_gate + 1)
            task_variable["gate_number"].append(gate_number)
            # also add the temporal postion of the gate touch by checking
            # time from navigation start of the trial

            i_row += 1
    for key in task_variable.keys():
        task_variable[key] = np.array(task_variable[key])
    # truncate matrx to i_row
    rate_matrix = rate_matrix[:i_row, :]
    if np.any(np.isnan(rate_matrix)):
        print("Binned rates contain NaNs")
        raise ValueError("Binned rates contain NaNs")
    return {
        "response": rate_matrix,
        "task_variable": task_variable,
    }


def get_rates_per_unit(
    spike_times,
    trials,
    valid_trials,
    behavioral_df,
    event,
    window,
    bin_size,
    subject_name,
):
    timebins = np.arange(
        window[0],
        window[1] + bin_size,
        bin_size,
    )
    agent_id = 0 if subject_name == "O" else 1
    n_trials = len(trials)
    rate_matrix = np.zeros([n_trials, len(timebins) - 1])
    task_variable = {"trial_num": []}
    i_row = 0
    reaction_times = behavioral_df["reaction_time"].values.tolist()
    reaction_times = np.array([literal_eval(x) for x in reaction_times])
    for trial in trials:
        trial_num = trial["trial_num"]
        if trial_num not in valid_trials:
            continue
        if trial_num not in behavioral_df["trial_num"].values:
            continue
        trial_idx = np.where(behavioral_df["trial_num"] == trial_num)[0][0]
        event_times = {
            "start": trial["t_start"],
            "navon": trial["relative_phase_times"][1] + trial["t_start"],
            "fdbk": trial["relative_phase_times"][2] + trial["t_start"],
            "choice": reaction_times[trial_idx][agent_id] + trial["t_start"],
        }
        event_time = event_times[event]
        event_spike_times = np.array(spike_times) - event_time
        event_spike_times = event_spike_times[
            (event_spike_times >= window[0])
            & (event_spike_times <= window[1] + bin_size)
        ]
        # Bin the spike times
        binned_rates, _ = np.histogram(
            event_spike_times,
            bins=timebins,
        )
        if np.any(np.isnan(binned_rates)):
            raise ValueError("Binned rates contain NaNs")
        # put the binned rates to the matrix
        rate_matrix[i_row, :] = binned_rates
        i_row += 1
        task_variable["trial_num"].append(trial_num)
    task_variable["trial_num"] = np.array(task_variable["trial_num"])
    # truncate matrx to i_row
    rate_matrix = rate_matrix[:i_row, :]
    if np.any(np.isnan(rate_matrix)):
        raise ValueError("Binned rates contain NaNs")
    return {
        "response": rate_matrix,
        "task_variable": task_variable,
    }


def process_unit(
    unit,
    unit_idx_master,
    spike_times_per_cluster,
    oe_trials,
    valid_trials,
    behavioral_df,
    event,
    window,
    bin_size,
    session_id,
    subject_name,
):
    try:
        print("Processing unit {}".format(unit))
        spike_times = spike_times_per_cluster[unit]
        if len(valid_trials) == 0:
            return None
        if event == "gates":
            unit_dict = get_gate_rates_per_unit(
                spike_times,
                oe_trials,
                valid_trials,
                behavioral_df,
                window,
                bin_size,
            )
            unit_dict["trial_ids"] = (
                session_id * 20000
                + unit_dict["task_variable"]["trial_num"] * 16
                + unit_dict["task_variable"]["gate_captures"]
            )
        else:
            unit_dict = get_rates_per_unit(
                spike_times,
                oe_trials,
                valid_trials,
                behavioral_df,
                event,
                window,
                bin_size,
                subject_name,
            )
            unit_dict["trial_ids"] = (
                session_id * 20000 + unit_dict["task_variable"]["trial_num"]
            )
        unit_dict["task_variable"] = add_task_vars(
            unit_dict["task_variable"], behavioral_df, subject_name
        )
        unit_dict["session_id"] = session_id
        unit_dict["unit_idx_master"] = unit_idx_master

        metadata_unit = {
            "subject": subject_name,
            "event": event,
            "unit_number": unit,
            "unit_idx_master": unit_idx_master,
        }
        return unit_dict, metadata_unit
    except Exception as e:
        logging.error(f"Error processing unit {unit}: {e}", exc_info=True)
        # Handle the exception or return None
        return None


def process_unit_wrapper(args):
    return process_unit(*args)


def load_session_data(spikes_dir, event, window=(-2, 0), bin_size=0.1):
    spk_metadata = get_spk_metadata(spikes_dir)
    subject_name = spk_metadata["subject_name"]
    session_dir = spk_metadata["session_dir"]
    root_dir = os.path.dirname(os.path.dirname(session_dir))
    master_list_path = os.path.join(
        root_dir, "master_list_{}.csv".format(subject_name)
    )
    master_list = pd.read_csv(master_list_path)
    # filter dataframe to only include the current session
    master_list = master_list[
        master_list["date"] == int(session_dir.split("/")[-1])
    ]
    vprobe = spk_metadata["vprobe"]
    subject_idx = 0 if subject_name == "O" else 1
    date = session_dir.split("/")[-1]
    subject_names = load_subject_names()
    session_id = subject_names.loc[
        subject_names["date"] == int(date), "session #"
    ].values[0]
    # check if unit_trials_valid exists
    if not os.path.exists(os.path.join(spikes_dir, "unit_trials_valid.csv")):
        print("File does not exist: {}".format(spikes_dir))
        return None
    unit_trials_valid = np.loadtxt(
        os.path.join(spikes_dir, "unit_trials_valid.csv"), delimiter=","
    )
    # Load trial data
    data_dir = os.path.dirname(os.path.dirname(spikes_dir))
    with open(
        os.path.join(data_dir, "open_ephys_events/open_ephys_trials.json"), "r"
    ) as f:
        oe_trials = json.load(f)

    # Load spike times per cluster
    with open(
        os.path.join(spikes_dir, "spike_times_per_cluster.json"), "r"
    ) as f:
        spike_times_per_cluster = json.load(f)

    # load and update behavior
    bhv_path = os.path.join(
        session_dir, "results", "moog_events", "trial_info.csv"
    )
    behavioral_df = pd.read_csv(bhv_path)

    behavioral_df = update_behavior(
        subject_idx,
        behavioral_df,
    )
    unit_dict_session = []
    metadata_dict_session = []
    timebins = np.arange(
        window[0],
        window[1] + bin_size,
        bin_size,
    )
    cluster_ids = list(spike_times_per_cluster.keys())
    total_units = len(cluster_ids)
    n_loaded = 0
    unit_dict_session = []
    metadata_dict_session = []
    ks_dir = os.path.join(session_dir, "results", vprobe, "ks_output")
    manual_label_path = os.path.join(ks_dir, "cluster_info.tsv")
    if not os.path.exists(manual_label_path):
        logging.info("File does not exist: {}".format(manual_label_path))
        return None
    manual_labels = pd.read_csv(manual_label_path, sep="\t")

    # Prepare arguments for each unit
    args = []
    for unit in cluster_ids:
        if int(unit) not in manual_labels["cluster_id"].values.tolist():
            continue
        i_label = manual_labels["cluster_id"].values.tolist().index(int(unit))
        unit_label = manual_labels["group"][i_label]
        if unit_label == "noise":
            continue
        unit_idx = cluster_ids.index(unit)
        valid_trials = np.where(unit_trials_valid[unit_idx, :] == 1)[0]
        # this will be used to compare with trial_num which is 1-based
        valid_trials += 1
        unit_idx_master = master_list[
            (master_list["unit_number"] == int(unit))
        ]["unit_index"].values[0]
        if len(valid_trials) > 0:
            args.append(
                (
                    unit,
                    unit_idx_master,
                    spike_times_per_cluster,
                    oe_trials,
                    valid_trials,
                    behavioral_df,
                    event,
                    window,
                    bin_size,
                    session_id,
                    subject_name,
                )
            )

    num_cores = 100
    num_cores = min(num_cores, len(args))

    # Parallel processing using multiprocessing.Pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_unit_wrapper, args)

    # for debugging only: run sequentially
    # results = [process_unit_wrapper(arg) for arg in args]

    for result in results:
        if result is not None:
            unit_dict, metadata_unit = result
            metadata_unit["date"] = date
            metadata_unit["probe"] = vprobe
            unit_dict_session.append(unit_dict)
            metadata_dict_session.append(metadata_unit)
    if len(unit_dict_session) == 0:
        return None
    # combine values in metadata
    combined_dict = {}
    for key in metadata_dict_session[0]:
        combined_dict[key] = []
        for d in metadata_dict_session:
            combined_dict[key].append(d[key])
    n_loaded = len(unit_dict_session)
    return {
        "unit": unit_dict_session,
        "metadata": combined_dict,
        "timepoints": timebins[:-1],
        "n_rejected": total_units - n_loaded,
    }
