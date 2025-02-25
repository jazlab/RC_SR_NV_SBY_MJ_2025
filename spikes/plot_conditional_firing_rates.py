"""
This module runs roc analysis on given windows of time.
"""

import matplotlib.pyplot as plt
import os
from utils.plots import beutify


def plot_trial_type(
    ax, time_bins, firing_rates, cluster, trial_type, color, line_style=None
):
    if line_style is None:
        if "actor" in trial_type:
            line_style = "solid"
        else:
            line_style = "dashed"
    ax.plot(
        1000 * time_bins,
        firing_rates[cluster][f"{trial_type}_trials_spikes"],
        color=color,
        label=trial_type,
        linestyle=line_style,
    )


def plot_history_firing_rates(
    firing_rates, time_bins, save_dir, filenam_prefix, event
):
    for cluster, _ in firing_rates.items():
        plt.figure(facecolor="white", figsize=(9, 9))
        plt.suptitle(f"Cluster {cluster}")
        ax = plt.subplot(111)
        factor1 = "1R_both"
        factor2 = "1NR_both"
        factor3 = "2NR_both"

        plot_trial_type(
            ax,
            time_bins,
            firing_rates,
            cluster,
            factor1,
            "darkgreen",
            line_style="solid",
        )
        plot_trial_type(
            ax,
            time_bins,
            firing_rates,
            cluster,
            factor2,
            "orangered",
            line_style="solid",
        )
        plot_trial_type(
            ax,
            time_bins,
            firing_rates,
            cluster,
            factor3,
            "darkred",
            line_style="solid",
        )
        ax.set_xlabel(f"Time from {event} (s)")
        ax.set_ylabel("Firing rate")

        # set the y-axis to be the same for all subplots
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max)
        ax.vlines(
            x=0,
            ymin=y_min,
            ymax=y_max,
            linestyle="dashed",
            color="k",
        )
        # set x tick labels to be -1 and 1 instead of 1000
        ax.set_xlim(-0.1 * 1000, 0.8 * 1000)
        ax.set_xticklabels(ax.get_xticks() / 1000)
        ax.set_xlabel(f"Time from {event} (s)")
        ax.set_ylabel("Firing rate (Hz)")
        beutify(ax)
        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.4)

        filename = f"{filenam_prefix}_{cluster}_{event}_rates.pdf"
        plt.savefig(os.path.join(save_dir, filename), dpi=200)
        plt.close()


def plot_conditional_firing_rates_overlap(
    firing_rates, time_bins, save_dir, filenam_prefix, event
):
    for cluster, _ in firing_rates.items():
        plt.figure(facecolor="white", figsize=(9, 9))
        plt.suptitle(f"Cluster {cluster}")

        # plot each of rew, history, early/late, less/more, switch/stay factors
        # in both, self, other columns in subplots of 5x3
        factor_1 = "rew"
        factor_2 = "no_rew"
        axs = []
        ax = plt.subplot(111)
        if factor_1 == "rew":
            color_1 = "darkgreen"
        else:
            color_1 = "orangered"
        role_1 = "actor"
        role_2 = "observer"
        plot_trial_type(
            ax,
            time_bins,
            firing_rates,
            cluster,
            f"{factor_1}_{role_1}",
            color_1,
        )
        plot_trial_type(
            ax,
            time_bins,
            firing_rates,
            cluster,
            f"{factor_2}_{role_1}",
            "darkred",
        )
        plot_trial_type(
            ax,
            time_bins,
            firing_rates,
            cluster,
            f"{factor_1}_{role_2}",
            color_1,
        )
        plot_trial_type(
            ax,
            time_bins,
            firing_rates,
            cluster,
            f"{factor_2}_{role_2}",
            "darkred",
        )
        ax.set_xlabel(f"Time from {event} (s)")
        ax.set_ylabel("Firing rate")
        axs.append(ax)

        # set the y-axis to be the same for all subplots
        y_max = max([ax.get_ylim()[1] for ax in axs])
        y_min = min([ax.get_ylim()[0] for ax in axs])
        for irow, ax in enumerate(axs):
            ax.set_ylim(y_min, y_max)
            ax.vlines(
                x=0,
                ymin=y_min,
                ymax=y_max,
                linestyle="dashed",
                color="k",
            )
            # set x tick labels to be -1 and 1 instead of 1000
            if event == "fdbk":
                ax.set_xlim(-0.1 * 1000, 0.8 * 1000)
            if event == "choice":
                ax.set_xlim(-0.8 * 1000, 0.1 * 1000)
            ax.set_xticklabels(ax.get_xticks() / 1000)
            ax.set_xlabel(f"Time from {event} (s)")
            ax.set_ylabel("Firing rate (Hz)")
            beutify(ax)

        filename = f"{filenam_prefix}_{cluster}_{event}.pdf"
        plt.savefig(os.path.join(save_dir, filename), dpi=200)
        plt.close()
