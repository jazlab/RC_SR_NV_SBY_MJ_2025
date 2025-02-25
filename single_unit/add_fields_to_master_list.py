"""
This script adds selectivity fields to the master list of units.
"""

import os
import pandas as pd
import numpy as np
from utils.get_session_info import find_sessions_nested
import json


def add_selectivity_field(root_dir, event, factor, subject_name):
    # load master list
    master_list_path = f"{root_dir}/master_list_{subject_name}.csv"
    master_list = pd.read_csv(master_list_path)

    # find all session directories
    sessions = find_sessions_nested(root_dir, subject_name)
    for spikes_dir in sessions:
        stats_dir = f"{os.path.dirname(spikes_dir)}/stats"
        if not os.path.exists(stats_dir):
            continue
        file_name = f"{stats_dir}/roc_stats_{factor}.json"
        if factor == "reward":
            file_name = f"{stats_dir}/roc_stats.json"
        if not os.path.exists(file_name):
            print(f"File not found: {file_name}")
            continue
        with open(file_name, "r") as f:
            results = json.load(f)

        for unit_index, stats_unit in results.items():
            idx = master_list[
                (master_list["unit_index"] == int(unit_index))
            ].index
            if len(idx) == 0:
                print(f"Unit not found in master list")
                continue
            idx = idx[0]
            windows = stats_unit[factor][event].keys()
            for field in [
                "selectivity_self",
                "selectivity_other",
                "n_self",
                "n_other",
                "p_self",
                "p_other",
                "p_both",
            ]:
                for window in windows:
                    field_name = f"{event}_{factor}_{window}_{field}"
                    if factor == "pos_in_block":
                        field_name = f"{event}_pib_{window}_{field}"
                    if field_name not in master_list.columns:
                        master_list[field_name] = np.nan
                    master_list.at[idx, field_name] = stats_unit[factor][event][
                        window
                    ][field]
    print(f"added {event}_{factor} columns to master_list")
    # save master list
    master_list.to_csv(master_list_path, index=False)


def main():
    from utils.LoadSession import findrootdir

    root_dir = findrootdir()
    # add fields indicating which units have data for each event
    for subject_name in ["O", "L"]:
        for factor in [
            "reward",
            "nback_all",
            "nback_AOOA",
        ]:
            for event in [
                "choice",
                "fdbk",
            ]:
                add_selectivity_field(root_dir, event, factor, subject_name)


if __name__ == "__main__":
    main()
