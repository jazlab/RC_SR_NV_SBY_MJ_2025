#!/usr/bin/env python3
"""plot_loss.py

Aggregates TensorBoard scalar summaries (loss, mse_loss, theta_cosine,
theta_loss) across runs, splits them into two cohorts via regex, caches the
aggregated statistics *per tag* (e.g. cohorts_stats_loss.json) so you can
tweak plots without recomputing, and generates separate mean ± 95 % CI PDFs.

Usage
-----
    python plot_loss.py LOGDIR [--regex REGEX] [--cache_prefix PREFIX]
                               [--output_dir DIR] [--refresh]

Example
-------
    python plot_loss.py experiments/ --regex ortho
    # creates cohorts_stats_loss.json, cohorts_stats_mse_loss.json, ...
    # and loss_summary.pdf, mse_loss_summary.pdf, ...

"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import re
from glob import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from utils.LoadSession import findrootdir
from plot_functions import beutify

CI_Z = 1.96  # 95 % Gaussian two‑sided
TAGS = ["loss", "mse_loss", "theta_cosine", "theta_loss"]


# --------------------------------------------------------------------------- #
# Data loading and aggregation
# --------------------------------------------------------------------------- #


def load_scalar_from_event(event_path: str, tag: str):
    ea = event_accumulator.EventAccumulator(
        event_path, size_guidance={"scalars": 0}
    )
    try:
        ea.Reload()
    except Exception as err:
        print(f"[WARN] Skipping {event_path}: {err}")
        return None, None

    if tag not in ea.Tags().get("scalars", []):
        return None, None

    scalar_events = ea.Scalars(tag)
    steps = np.array([ev.step for ev in scalar_events], dtype=np.int64)
    vals = np.array([ev.value for ev in scalar_events], dtype=np.float32)
    return steps, vals


def aggregate_runs(root_dir: str, regex1: str, regex2: str, tag: str):
    pattern1 = re.compile(regex1)
    pattern2 = re.compile(regex2)
    cohorts = {"match": defaultdict(list), "rest": defaultdict(list)}

    event_files = list(
        glob(
            os.path.join(root_dir, "**", "events.out.tfevents.*"),
            recursive=True,
        )
    )
    if not event_files:
        raise FileNotFoundError(f"No event files found under {root_dir}")

    for ev_file in event_files:
        group = None
        if pattern1.search(ev_file):
            group = "match"
        elif pattern2.search(ev_file):
            group = "rest"
        else:
            continue

        steps, vals = load_scalar_from_event(ev_file, tag)
        if steps is None:
            continue

        for s, v in zip(steps, vals):
            cohorts[group][int(s)].append(float(v))  # JSON serialisable

    return cohorts


def compute_mean_ci(cohort_dict):
    if not cohort_dict:
        return [], [], []

    steps_sorted = sorted(cohort_dict.keys())
    means, cis = [], []
    for s in steps_sorted:
        arr = np.array(cohort_dict[s], dtype=np.float32)
        mean = float(arr.mean())
        ci = (
            float(CI_Z * arr.std(ddof=1) / np.sqrt(len(arr)))
            if len(arr) > 1
            else 0.0
        )
        means.append(mean)
        cis.append(ci)
    return steps_sorted, means, cis


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_single(tag, stats_tag, output_path, dpi=300):
    colors = {"match": "tab:blue", "rest": "tab:orange"}
    labels = {"match": "Group 1", "rest": "Group 2"}

    plt.figure(figsize=(4, 4))
    plotted = False
    for cohort in ["match", "rest"]:
        steps, means, cis = stats_tag.get(cohort, ([], [], []))
        if not steps:
            continue
        steps_np = np.asarray(steps)
        means_np = np.asarray(means)
        cis_np = np.asarray(cis)
        # we will limit the confint plot to be larger than 0 for loss (nonneg)
        if "loss" in output_path:
            confint_floor = np.maximum(means_np - cis_np, 0.0)
        else:
            confint_floor = means_np - cis_np

        plt.plot(steps_np, means_np, label=labels[cohort], color=colors[cohort])
        plt.fill_between(
            steps_np,
            confint_floor,
            means_np + cis_np,
            alpha=0.3,
            color=colors[cohort],
        )
        plotted = True

    if not plotted:
        print(f"[WARN] No data for tag '{tag}', skipping plot.")
        plt.close()
        return

    ax = plt.gca()
    beutify(ax)
    plt.xlabel("Training step")
    plt.ylabel(tag)
    plt.title(f"Mean {tag} ± 95 % CI")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"[INFO] Saved plot {output_path}")


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #


def load_json(path: Path):
    try:
        with open(path, "r") as fp:
            return json.load(fp)
    except Exception:
        return None


def dump_json(data, path: Path):
    with open(path, "w") as fp:
        json.dump(data, fp)
    print(f"[INFO] Wrote cache {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    job_id = "41838359"
    root_dir = findrootdir()
    output_dir = os.path.join(root_dir, "rnn_snapshots", job_id)
    cache_prefix = os.path.join(output_dir, "cohorts_stats")
    plot_dir = os.path.join(root_dir, "plots_paper")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare stats dict keyed by tag
    stats = {}
    regex1 = "ortho"
    regex2 = "paral"

    for tag in TAGS:
        cache_file = Path(f"{cache_prefix}_{tag}.json")
        tag_stats = None

        if cache_file.exists():
            print(f"[INFO] Loading {cache_file}")
            tag_stats = load_json(cache_file)

        if tag_stats is None:
            print(f"[INFO] Computing statistics for tag '{tag}'…")
            cohorts = aggregate_runs(output_dir, regex1, regex2, tag)
            tag_stats = {
                "match": compute_mean_ci(cohorts["match"]),
                "rest": compute_mean_ci(cohorts["rest"]),
            }
            dump_json(tag_stats, cache_file)

        stats[tag] = tag_stats

        # plot
        if tag == "mse_loss":
            plot_path = f"{plot_dir}/FigS13a_{tag}_summary.pdf"
            plot_single(tag, tag_stats, plot_path)


if __name__ == "__main__":
    main()
