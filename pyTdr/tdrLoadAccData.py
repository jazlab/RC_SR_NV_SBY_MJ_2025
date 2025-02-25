"""
Load ephys and behavior data in tdr format.

Usage:
dataT, metadata = tdrLoadAccData(root_dir, event='fdbk', animal = 'O', Fd = 20)

dataT: ['unit', 'time']

dataT['unit']: array of trials
dataT['unit'][0]: dict of ['response','task_variable','dimension']

dataT['unit'][0]['response']: 
[ntrials,ntimepoints] e.g. [2043,15] - firing rate from binned_rates

dataT['unit'][0]['task_variable']: dict ['actor', 'reward', 'choice', 'history']
     - from behavior_df

dataT['time']: time bins, same len as unit-response, ntimepoints.
"""

import numpy as np
import os
import json
from utils.get_session_info import find_sessions_nested
from pyTdr.tdrUtils import load_config, update_config
import logging
from pyTdr.tdrLoadSessionData import load_session_data
from pyTdr.tdrGetTaskVars import add_new_task_var

# Set up logging
logging.basicConfig(filename="file_check.log", level=logging.INFO)


def from_json(d):
    if isinstance(d, dict) and d.get("type") == "ndarray":
        return np.array(d["value"])
    return d


# Function to convert numpy arrays to lists
def default(o):
    if isinstance(o, np.ndarray):
        return {"type": "ndarray", "value": o.tolist()}
    elif isinstance(o, np.integer):
        return int(o)
    else:
        import pdb

        pdb.set_trace()
    raise TypeError


def tdrLoadAccData(root_dir, subject_name, event, time_window=[-3, 3]):
    config = load_config()
    max_units = config["max_units"]
    # Define the cache file path
    cache_file = os.path.join(
        root_dir,
        f"cache_{subject_name}_{event}_{time_window[0]}_{time_window[1]}.json",
    )
    # Check if cache exists
    if os.path.isfile(cache_file):
        # Load from cache
        with open(cache_file, "r") as f:
            dataT, metadata = json.load(f, object_hook=from_json)
        # add new task variable: just run once
        # dataT = add_new_task_var(dataT, metadata)
        # # Save to cache
        # with open(cache_file, "w") as f:
        #     json.dump((dataT, metadata), f, default=default)
        return dataT, metadata
    dataT = {"unit": []}
    metadata = {"unit": []}
    metadata_sessions = []
    timepoints = None

    n_sessions = 0
    n_units_loaded = 0
    n_rejected = 0

    sessions = find_sessions_nested(root_dir, subject_name)
    for spike_dir in sessions:
        print(f"Loading spikes from {spike_dir}")
        session_data = load_session_data(
            spike_dir,
            event=event,
            window=time_window,
        )
        if session_data is not None:
            print(len(session_data["unit"]))
            dataT["unit"].extend(session_data["unit"])
            metadata_sessions.append(session_data["metadata"])
            timepoints = session_data["timepoints"]
            n_units_loaded += len(session_data["unit"])
            n_rejected += session_data["n_rejected"]
        else:
            print(f"Session {spike_dir} failed to load")

        n_sessions += 1
        print(f"loaded {n_units_loaded} units from {n_sessions} sessions")

        if n_units_loaded >= max_units:
            break
    # combine dictionaries in metadata['unit']
    flattened_lists = {
        key: [
            item
            for sublist in [d[key] for d in metadata_sessions]
            for item in sublist
        ]
        for key in metadata_sessions[0]
    }
    metadata["unit"] = flattened_lists
    # assign dimension to all units
    for i in range(len(dataT["unit"])):
        dataT["unit"][i]["dimension"] = f"unit_{i+1:03}"

    print(
        f"loaded {n_units_loaded} units from {n_sessions} sessions, rejected {n_rejected} units"
    )

    dataT["time"] = timepoints

    # Save to cache
    with open(cache_file, "w") as f:
        json.dump((dataT, metadata), f, default=default)
    return dataT, metadata
