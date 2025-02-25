"""
This module contains functions to plot angles in polar coordinates.
"""

import numpy as np
import json
from utils.LoadSession import findrootdir


def plot_angles(ax, theta, magnitude, color, alpha):
    # check if theta is a single number or a list
    if not isinstance(theta, list):
        theta = [theta]
        magnitude = [magnitude]
    for t, a in zip(theta, magnitude):
        ax.plot([0, np.deg2rad(t)], [0, a], color=color, alpha=alpha)
    ax.set_thetamin(0)
    ax.set_thetamax(180)


def make_combined_files(event, item):
    datadir = findrootdir()
    file_name_L = f"{datadir}/stats_paper/L_{event}_{item}.json"
    file_name_O = f"{datadir}/stats_paper/O_{event}_{item}.json"
    results_O = json.load(open(file_name_O, "r"))
    results_L = json.load(open(file_name_L, "r"))
    results = combine_results(results_L, results_O)
    file_name = f"{datadir}/stats_paper/both_{event}_{item}.json"
    save_results(file_name, results)


def combine_results(results1, results2):
    combined_results = {}
    for key in results1:
        if key == "unit_idx_master":
            # to distinguish between O and L, add 10000 to results2
            combined_results[key] = results1[key] + [
                i + 10000 for i in results2[key]
            ]
        combined_results[key] = results1[key] + results2[key]
    return combined_results


def save_results(file_name, results):
    with open(file_name, "w") as f:
        json.dump(results, f)
