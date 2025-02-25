"""
construct master list (per subject) of units with fields:
unit_number date v_probe spikes_dir
with subsequent fields to be added by add_fields_to_master_list
"""

import os
import pandas as pd
import numpy as np
from utils.get_session_info import find_sessions_nested, get_spk_metadata
from ast import literal_eval


def create_master_list(subject_name, root_dir):
    df = pd.DataFrame(
        columns=[
            "unit_index",
            "date",
            "v_probe",
            "unit_number",
            "group",
            "coords",
            "n_trials",
        ]
    )
    sessions = find_sessions_nested(root_dir, subject_name)
    # make sure sessions is sorted alphabetically
    sessions.sort()
    n_units = 0
    for spikes_dir in sessions:
        print(f"Loading data from {spikes_dir}")
        # get date
        spk_metadata = get_spk_metadata(spikes_dir)
        coords = spk_metadata["coords"]
        # use literal eval to convert coords to list
        coords = literal_eval(coords)
        date = spikes_dir.split("/")[-4]
        if not os.path.exists(spk_metadata["unit_trials_valid_file"]):
            print("File does not exist: {}".format(spikes_dir))
            continue
        if not os.path.exists(spk_metadata["manual_label_file"]):
            print(
                "File does not exist: {}".format(
                    spk_metadata["manual_label_file"]
                )
            )
            continue
        unit_trials_valid = np.loadtxt(
            spk_metadata["unit_trials_valid_file"], delimiter=","
        )
        manual_labels = pd.read_csv(
            spk_metadata["manual_label_file"], delimiter="\t"
        )
        unit_numbers = manual_labels["cluster_id"].values
        labels = manual_labels["group"].values
        depths = manual_labels["depth"].values
        assert len(unit_numbers) == unit_trials_valid.shape[0]
        for unit in range(unit_trials_valid.shape[0]):
            n_trials = int(sum(unit_trials_valid[unit, :]))
            unit_number = unit_numbers[unit]
            coords_unit = coords[:2] + [coords[2] - 3.15 + depths[unit] / 1000]
            # limit to 3 decimal places
            coords_unit = [round(x, 3) for x in coords_unit]
            df.loc[len(df)] = {
                "unit_index": n_units,
                "date": date,
                "v_probe": spikes_dir.split("/")[-2],
                "unit_number": unit_number,
                "coords": coords_unit,
                "group": labels[unit],
                "n_trials": n_trials,
            }
            n_units += 1
    df.to_csv(f"{root_dir}/master_list_{subject_name}.csv", index=False)


def main():
    from utils.LoadSession import findrootdir

    root_dir = findrootdir()
    for subject_name in ["O", "L"]:
        create_master_list(subject_name, root_dir)


if __name__ == "__main__":
    main()
