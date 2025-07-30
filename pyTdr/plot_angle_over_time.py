"""
This script plots the angle between a switch evidence dimension computed at
outcome time and subsequent time points. The angle over time plot shows the
rotation dynamics of the switch evidence.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from utils.LoadSession import findrootdir
from utils.plots import beutify
from compute_angle import compute_angle


# --- Data Aggregation and Alignment (Unchanged) ---
def aggregate_and_align_data(file_pattern, output_file):
    """
    Reads vector data from multiple files, extracts the trial type (fdbk/choice),
    aligns them to common axes, and saves the aggregated data.
    """
    print(f"--- Aggregating and aligning data from '{file_pattern}' ---")
    filepaths = sorted(glob.glob(file_pattern))
    if not filepaths:
        print(f"Error: No files found matching the pattern '{file_pattern}'.")
        return
    print(f"Found {len(filepaths)} files:")
    for fp in filepaths:
        print(f"  - {os.path.basename(fp)}")
    all_data = {}
    all_unit_sets = []
    for fp in filepaths:
        with open(fp, "r") as f:
            data = json.load(f)
        match = re.search(
            r"_(fdbk|choice)_act_obs_dimensions_(-?\d+\.\d+)_(-?\d+\.\d+)\.json",
            fp,
        )
        if not match:
            print(
                f"Warning: Could not parse filename: {os.path.basename(fp)}. Skipping."
            )
            continue
        trial_type = match.group(1)
        time_key = f"{match.group(2)}_{match.group(3)}"
        all_data[time_key] = {
            "coef_switch": np.array(data["coef_switch"]),
            "unit_idx_master": data["unit_idx_master"],
            "type": trial_type,
        }
        all_unit_sets.append(set(data["unit_idx_master"]))
    if not all_unit_sets:
        print("Error: No valid data was loaded.")
        return
    common_units = set.intersection(*all_unit_sets)
    common_units_list = sorted(list(common_units))
    if not common_units_list:
        print(
            "Error: No common unit indices found across all files. Cannot proceed."
        )
        return
    print(f"\nFound {len(common_units_list)} common units across all files.")
    aligned_data = {"common_unit_idx": common_units_list, "timepoints": {}}
    for time_key, data in all_data.items():
        original_indices = data["unit_idx_master"]
        original_coefs = data["coef_switch"].T
        unit_to_col_idx = {
            unit_id: i for i, unit_id in enumerate(original_indices)
        }
        cols_to_keep = [
            unit_to_col_idx[unit_id] for unit_id in common_units_list
        ]
        aligned_coefs = original_coefs[:, cols_to_keep]
        aligned_data["timepoints"][time_key] = {
            "aligned_coef_switch": aligned_coefs.tolist(),
            "type": data["type"],
        }
    with open(output_file, "w") as f:
        json.dump(aligned_data, f, indent=4)
    print(f"\nSuccessfully aligned data and saved to '{output_file}'.\n")


def _plot_angle_curve(
    plot_data,
    title,
    xlabel,
    ylabel,
    filename_suffix,
    basename,
    xticks=None,
    xticklabels=None,
):
    """Generic helper function to create and save a plot."""
    plot_data.sort()
    plot_times = [item[0] for item in plot_data]
    plot_angles = [item[1] for item in plot_data]

    FONT_SIZE = 14
    plt.figure(figsize=(4, 4))
    plt.plot(plot_times, plot_angles, marker="o", linestyle="-", markersize=8)

    plt.title(title, fontsize=FONT_SIZE + 2, weight="bold")
    plt.xlabel(xlabel, fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)

    plt.ylim(0, 90)

    # Use custom ticks if provided, otherwise use default
    if xticks is not None and xticklabels is not None:
        plt.xticks(
            ticks=xticks,
            labels=xticklabels,
            fontsize=FONT_SIZE - 2,
            rotation=45,
        )
    else:
        plt.xticks(fontsize=FONT_SIZE - 2)

    plt.yticks(np.arange(0, 91, 10), fontsize=FONT_SIZE - 2)

    ax = plt.gca()
    beutify(ax)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.6)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    plot_filename = basename.replace(".json", f"_{filename_suffix}.pdf")
    plot_filename = plot_filename.replace("stats_paper/", "plots_paper/FigS15")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to '{plot_filename}'")
    plt.close()


def plot_angle_over_time(mean_vectors, basename):
    """Plot 1: Only feedback data, referenced to the first fdbk timepoint."""
    # This function remains unchanged.
    print("\n--- Generating Plot 1: Feedback Data Only ---")
    fdbk_vectors = {
        k: v for k, v in mean_vectors.items() if v["type"] == "fdbk"
    }
    if not fdbk_vectors:
        print("No feedback data found to plot.")
        return
    time_keys = sorted(
        fdbk_vectors.keys(), key=lambda t: float(t.split("_")[0])
    )
    initial_time_key = time_keys[0]
    initial_vector = fdbk_vectors[initial_time_key]["vector"]
    initial_start_time = float(initial_time_key.split("_")[0])
    plot_data = []
    for time_key in time_keys[1:]:
        time_offset = float(time_key.split("_")[0]) - initial_start_time
        angle = compute_angle(initial_vector, fdbk_vectors[time_key]["vector"])
        plot_data.append((time_offset, angle))
    _plot_angle_curve(
        plot_data,
        "Dimension Change During Feedback",
        "Time Offset from Feedback Start (s)",
        "Angle with Initial Vector [degrees]",
        "fdbk_only_plot",
        basename,
    )


def analyze_and_plot_angles(aggregated_file):
    """Controller function to load data and generate plot."""
    print(f"--- Analyzing and plotting angles from '{aggregated_file}' ---")
    with open(aggregated_file, "r") as f:
        data = json.load(f)
    mean_vectors = {}
    for time_key, values in data["timepoints"].items():
        aligned_coefs = np.array(values["aligned_coef_switch"])
        mean_vectors[time_key] = {
            "vector": np.mean(aligned_coefs, axis=0),
            "type": values["type"],
        }
    plot_angle_over_time(mean_vectors, aggregated_file)


if __name__ == "__main__":
    root_dir = findrootdir()
    for animal in ["O", "L"]:
        print(f"==========================================")
        print(f"Processing data for animal: {animal}")
        print(f"==========================================")
        INPUT_FILE_PATTERN = (
            f"{root_dir}/stats_paper/{animal}_*_act_obs_dimensions_*.json"
        )
        AGGREGATED_OUTPUT_FILE = (
            f"{root_dir}/stats_paper/{animal}_aggregated_aligned_data.json"
        )
        aggregate_and_align_data(INPUT_FILE_PATTERN, AGGREGATED_OUTPUT_FILE)
        analyze_and_plot_angles(AGGREGATED_OUTPUT_FILE)
    print("\n--- Finished processing all data. ---")
