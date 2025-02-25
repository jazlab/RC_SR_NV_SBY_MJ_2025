"""
This script makes the raster and histogram plot of two simultaneously recorded 
units.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ast import literal_eval
from utils.makeTrigInfo import makeTrigInfo
from spikes.plot_utils import categorize_trials
from utils.LoadSession import findrootdir


def generate_dual_raster(
    spikes_O, spikes_info_O, spikes_L, spikes_info_L, oe_trials, bhv_df
):
    # preprocess data
    valid_trials_1 = np.array([si["valid"] for si in spikes_info_O])
    valid_trials_2 = np.array([si["valid"] for si in spikes_info_L])
    valid_trials = np.intersect1d(
        np.where(valid_trials_1)[0], np.where(valid_trials_2)[0]
    )
    # find longest continuous sequence of valid trials
    valid_trials_diff = np.diff(valid_trials)
    continuous_sequences = np.split(
        valid_trials, np.where(valid_trials_diff != 1)[0] + 1
    )
    longest_sequence = max(continuous_sequences, key=len)
    valid_trials = longest_sequence
    # Build triginfo for self rew, other rew, self nr, other nr, for each unit
    tfs_choice = {"O": [], "L": []}
    tfs_fdbk = {"O": [], "L": []}
    tfs_nav = {"O": [], "L": []}
    xmax = 6
    bin_size = 0.1
    for subject in ["O", "L"]:
        spikes = spikes_O if subject == "O" else spikes_L
        subject_idx = 0 if subject == "O" else 1
        conditions = categorize_trials(bhv_df, "reward_so", subject_idx)
        for i_condition, (condition, trials) in enumerate(conditions.items()):
            trigger_times_fdbk = []
            for i_trial, d in enumerate(oe_trials):
                trial_idx = d["trial_num"] - 1
                if trial_idx not in trials:
                    continue
                trigger_times_fdbk.append(
                    d["relative_phase_times"][2] + d["t_start"]
                )
            tfs_fdbk[subject].append(
                makeTrigInfo(
                    trigger_times_fdbk,
                    spikes,
                    xl=xmax + bin_size,
                    binsize=bin_size,
                )
            )
        condition_lr = categorize_trials(bhv_df, "choice_so", subject_idx)
        for i_condition, (condition, trials) in enumerate(condition_lr.items()):
            trigger_times_choice = []
            offset_times_choice = []
            for i_trial, d in enumerate(oe_trials):
                trial_idx = d["trial_num"] - 1
                reaction_time = bhv_df.iloc[trial_idx]["reaction_time"][
                    subject_idx
                ]
                if trial_idx not in trials:
                    continue
                trigger_times_choice.append(reaction_time + d["t_start"])
                # offset for the choice phase is nav start
                offset_times_choice.append(
                    d["relative_phase_times"][1] - reaction_time
                )
            tf_choice = makeTrigInfo(
                trigger=trigger_times_choice,
                events=spikes,
                # offset=offset_times_choice,
                xl=xmax + bin_size,
                binsize=bin_size,
            )
            # remove bins after 0.75 for choice phase because it ends at 0.75
            idx_keep = np.where(tf_choice["edges_step"] <= 0.75)[0]
            tf_choice["edges_step"] = tf_choice["edges_step"][idx_keep]
            tf_choice["rds_step"] = tf_choice["rds_step"][idx_keep]
            tfs_choice[subject].append(tf_choice)
        conditions_nav = categorize_trials(bhv_df, "nav_so", subject_idx)
        for i_condition, (condition, trials) in enumerate(
            conditions_nav.items()
        ):
            trigger_times_nav = []
            offset_times_nav = []
            for i_trial, d in enumerate(oe_trials):
                trial_idx = d["trial_num"] - 1
                if trial_idx not in trials:
                    continue
                trigger_times_nav.append(
                    d["relative_phase_times"][1] + d["t_start"]
                )
                # offset for the navigation phase is navigation end
                offset_times_nav.append(
                    d["relative_phase_times"][2] - d["relative_phase_times"][1]
                )
            tfs_nav[subject].append(
                makeTrigInfo(
                    trigger_times_nav,
                    spikes,
                    offset=offset_times_nav,
                    xl=xmax + bin_size,
                    binsize=bin_size,
                )
            )

    lw_0 = 0.25
    lw_marker = 1.5
    c0 = "k"
    c1 = "g"
    c2 = "m"
    # create plot
    # Define xlims
    xlim_choice = (-0.1, 1)
    xlim_nav = (-0.1, 1)
    xlim_fdbk = (-0.1, 1)

    # Calculate width ratios based on xlims
    width_choice = xlim_choice[1] - xlim_choice[0]
    width_nav = xlim_nav[1] - xlim_nav[0]
    width_fdbk = xlim_fdbk[1] - xlim_fdbk[0]

    w_ratios = [
        width_choice,
        width_choice,
        width_nav,
        width_nav,
        width_fdbk,
        width_fdbk,
    ]

    fig = plt.figure(figsize=(8, 3))
    axs = []

    # Create subplots with GridSpec for width ratios
    gs = fig.add_gridspec(
        2,
        6,
        width_ratios=w_ratios,
        height_ratios=[2, 1],
        hspace=0.2,
        wspace=0.3,
    )

    for i in range(12):
        axs.append(fig.add_subplot(gs[i // 6, i % 6]))

    # first subplot: rasters
    trials_to_plot = 25
    if len(valid_trials) > trials_to_plot + 25:
        valid_trials = valid_trials[25 : trials_to_plot + 25]
    for irow, itrial in enumerate(valid_trials):
        lw_1 = 0.25
        lw_2 = 0.25
        trial_idx = oe_trials[itrial]["trial_num"] - 1
        if bhv_df["player"][itrial] == 0:
            lw_1 = 0.5
        else:
            lw_2 = 0.5
        reward_color = "r"
        if bhv_df.iloc[trial_idx]["reward"] == 1:
            reward_color = "g"
        tstart = oe_trials[itrial]["t_start"]
        reaction_times = bhv_df.iloc[trial_idx]["reaction_time"]
        player = bhv_df.iloc[trial_idx]["player"]
        player_color = c1 if player == 0 else c2
        choice_O = bhv_df.iloc[trial_idx]["choice_a0"]
        choice_L = bhv_df.iloc[trial_idx]["choice_a1"]
        choice_actor = bhv_df.iloc[trial_idx]["player_choice"]

        # if reaction_times[0] > 1.5 or reaction_times[1] > 1.5:
        #     continue
        tchoice_O = tstart + reaction_times[0]
        tchoice_L = tstart + reaction_times[1]
        tnav = tstart + oe_trials[itrial]["relative_phase_times"][1]
        tfdbk_relative = oe_trials[itrial]["relative_phase_times"][2]
        tfdbk_from_nav = (
            tfdbk_relative - oe_trials[itrial]["relative_phase_times"][1]
        )
        tnav_from_choice_O = tnav - tchoice_O
        tnav_from_choice_L = tnav - tchoice_L
        tfdbk = tstart + tfdbk_relative
        spikes_O_trial = spikes_O[
            (spikes_O < tstart + oe_trials[itrial]["t_end"] + 5)
            & (spikes_O > tstart)
        ]
        spikes_L_trial = spikes_L[
            (spikes_L < tstart + oe_trials[itrial]["t_end"] + 5)
            & (spikes_L > tstart)
        ]
        spikes_choice_1 = spikes_O_trial - tchoice_O
        spikes_choice_1_beforenav = spikes_choice_1[
            spikes_choice_1 < tnav_from_choice_O
        ]
        spikes_choice_1_afternav = spikes_choice_1[
            spikes_choice_1 > tnav_from_choice_O
        ]
        spikes_fdbk_1 = spikes_O_trial - tfdbk
        spikes_nav_1 = spikes_O_trial - tnav
        spikes_choice_2 = spikes_L_trial - tchoice_L
        spikes_choice_2_beforenav = spikes_choice_2[
            spikes_choice_2 < tnav_from_choice_L
        ]
        spikes_choice_2_afternav = spikes_choice_2[
            spikes_choice_2 > tnav_from_choice_L
        ]
        spikes_fdbk_2 = spikes_L_trial - tfdbk
        spikes_nav_2 = spikes_L_trial - tnav

        axs[0].vlines(
            spikes_choice_1_beforenav,
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_0,
        )
        axs[0].vlines(
            spikes_choice_1_afternav,
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_1,
        )
        # plot ticks for choice
        color_choice = "c" if choice_O == -1 else "y"
        color_choice_actor = "c" if choice_actor == -1 else "y"
        linestyle = "-"
        axs[0].vlines(
            0,
            irow + 0.2,
            irow + 0.8,
            colors=color_choice,
            linewidths=1,
            linestyles=linestyle,
        )
        # add a choice marker outside the raster
        axs[0].plot(
            1.1,
            irow + 0.5,
            "o",
            color=color_choice,
            markersize=1,
            transform=axs[0].transData,
            clip_on=False,
        )
        axs[0].vlines(
            tnav_from_choice_O,
            irow + 0.2,
            irow + 0.8,
            colors=color_choice_actor,
            linewidths=lw_1,
        )
        axs[2].vlines(
            0,
            irow + 0.2,
            irow + 0.8,
            colors=color_choice_actor,
            linewidths=lw_1 * 2,
        )
        # if player is 0, add marker at 1.1
        if player == 0:
            axs[2].plot(
                1.1,
                irow + 0.5,
                "o",
                color=player_color,
                markersize=1,
                transform=axs[2].transData,
                clip_on=False,
            )
        else:
            axs[3].plot(
                -0.2,
                irow + 0.5,
                "o",
                color=player_color,
                markersize=1,
                transform=axs[3].transData,
                clip_on=False,
            )
        axs[2].vlines(
            spikes_nav_1[spikes_nav_1 <= 0],
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_0,
        )
        axs[2].vlines(
            spikes_nav_1[spikes_nav_1 > 0],
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_1,
        )
        axs[2].vlines(
            tfdbk_from_nav,
            irow + 0.2,
            irow + 0.8,
            colors=reward_color,
            linewidths=lw_1 * 2,
        )
        axs[4].vlines(
            spikes_fdbk_1,
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_1,
        )
        axs[4].vlines(
            0,
            irow + 0.2,
            irow + 0.8,
            colors=reward_color,
            linewidths=lw_1 * 2,
        )
        if player == 0:
            axs[4].plot(
                1.1,
                irow + 0.5,
                "o",
                color=reward_color,
                markersize=1,
                transform=axs[4].transData,
                clip_on=False,
            )
        else:
            axs[5].plot(
                -0.2,
                irow + 0.5,
                "o",
                color=reward_color,
                markersize=1,
                transform=axs[5].transData,
                clip_on=False,
            )
        # choice for L
        color_choice = "c" if choice_L == -1 else "y"
        axs[1].vlines(
            0,
            irow + 0.2,
            irow + 0.8,
            colors=color_choice,
            linewidths=1,
        )
        # add a choice marker outside the raster
        axs[1].plot(
            -0.2,
            irow + 0.5,
            "o",
            color=color_choice,
            markersize=1,
            transform=axs[1].transData,
            clip_on=False,
        )
        axs[1].vlines(
            spikes_choice_2_beforenav,
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_0,
        )
        axs[1].vlines(
            spikes_choice_2_afternav,
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_2,
        )
        axs[1].vlines(
            tnav_from_choice_L,
            irow + 0.2,
            irow + 0.8,
            colors=color_choice_actor,
            linewidths=lw_2,
        )
        axs[3].vlines(
            0,
            irow + 0.2,
            irow + 0.8,
            colors=color_choice_actor,
            linewidths=lw_2 * 2,
        )
        axs[3].vlines(
            spikes_nav_2[spikes_nav_2 <= 0],
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_0,
        )
        axs[3].vlines(
            spikes_nav_2[spikes_nav_2 > 0],
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_2,
        )
        axs[3].vlines(
            tfdbk_from_nav,
            irow + 0.2,
            irow + 0.8,
            colors=reward_color,
            linewidths=lw_2 * 2,
        )
        axs[5].vlines(
            spikes_fdbk_2,
            irow + 0.1,
            irow + 0.9,
            colors=c0,
            linewidths=lw_2,
        )
        axs[5].vlines(
            0,
            irow + 0.2,
            irow + 0.8,
            colors=reward_color,
            linewidths=lw_2 * 2,
        )
        # # Add markers
        # for i in [0, 3]:
        #     axs[i].plot(
        #         0.2,
        #         irow + 0.5,
        #         "o",
        #         color=player_color,
        #         markersize=2,
        #         transform=axs[i].transData,
        #         clip_on=False,
        #     )

    # plot histograms
    colors_choice = ["c", "y", "c", "y"]
    linestyle_choice = ["-", "-", ":", ":"]
    colors_rew = ["g", "r", "g", "r"]
    linestyles = ["-", "-", "--", "--"]
    for i in range(4):
        axs[6].plot(
            tfs_choice["O"][i]["edges_step"],
            tfs_choice["O"][i]["rds_step"],
            color=colors_choice[i],
            linestyle=linestyle_choice[i],
            linewidth=lw_0,
        )
        axs[8].plot(
            tfs_nav["O"][i]["edges_step"],
            tfs_nav["O"][i]["rds_step"],
            color=colors_choice[i],
            linestyle=linestyles[i],
            linewidth=lw_0,
        )
        axs[10].plot(
            tfs_fdbk["O"][i]["edges_step"],
            tfs_fdbk["O"][i]["rds_step"],
            color=colors_rew[i],
            linestyle=linestyles[i],
            linewidth=lw_0,
        )
        axs[7].plot(
            tfs_choice["L"][i]["edges_step"],
            tfs_choice["L"][i]["rds_step"],
            color=colors_choice[i],
            linestyle=linestyle_choice[i],
            linewidth=lw_0,
        )
        axs[9].plot(
            tfs_nav["L"][i]["edges_step"],
            tfs_nav["L"][i]["rds_step"],
            color=colors_choice[i],
            linestyle=linestyles[i],
            linewidth=lw_0,
        )
        axs[11].plot(
            tfs_fdbk["L"][i]["edges_step"],
            tfs_fdbk["L"][i]["rds_step"],
            color=colors_rew[i],
            linestyle=linestyles[i],
            linewidth=lw_0,
        )
    syncYAxis(axs[6:])
    # syncYAxis(axs[9:])

    # make surrounding box green or magenta
    for i in [0, 2, 4]:
        axs[i].spines["bottom"].set_color("g")
        axs[i].spines["top"].set_color("g")
        axs[i].spines["right"].set_color("g")
        axs[i].spines["left"].set_color("g")
    for i in [1, 3, 5]:
        axs[i].spines["bottom"].set_color("m")
        axs[i].spines["top"].set_color("m")
        axs[i].spines["right"].set_color("m")
        axs[i].spines["left"].set_color("m")
    for i in [6, 7, 8, 9, 10, 11]:
        axs[i].spines["right"].set_visible(False)
    # set appropriate xlims
    for i in [0, 1, 6, 7]:  # choice
        axs[i].set_xlim(xlim_choice)
    for i in [2, 3, 8, 9]:  # navigation
        axs[i].set_xlim(xlim_nav)
    for i in [4, 5, 10, 11]:  # reward
        axs[i].set_xlim(xlim_fdbk)
    # remove yticks for middle subplots
    for i in [1, 2, 3, 4]:
        axs[i].yaxis.set_visible(False)
    # remove xticks for top subplots
    for i in [0, 1, 2, 3, 4, 5]:
        axs[i].xaxis.set_visible(False)
    # Show yticks on the right spine for rightmost subplots
    for i in [5]:
        axs[i].yaxis.tick_right()
    # remove top line for bottom row
    for i in [6, 7, 8, 9, 10, 11]:
        axs[i].spines["top"].set_visible(False)
    # change ytick to [0,19] and labels to 1,20
    for i in [0, 5]:
        axs[i].set_yticks([0.5, 24.5])
        axs[i].set_yticklabels([1, 25])
    for i in [6, 7, 8, 9, 10, 11]:
        axs[i].set_yticks([0, 10, 20])
        axs[i].set_yticklabels([])
        axs[i].set_ylim([0, 25])
        # add dashed line at 0
        axs[i].axvline(0, linestyle="--", color="k", linewidth=0.5)
    for i in [6]:
        axs[i].set_yticks([0, 10, 20])
        axs[i].set_yticklabels([0, 10, 20])

    return fig


def syncYAxis(axes):
    ylims = [ax.get_ylim() for ax in axes]
    ymin = min([ylim[0] for ylim in ylims])
    ymax = max([ylim[1] for ylim in ylims])
    for ax in axes:
        ax.set_ylim([ymin, ymax])


def load_spikes(spikes_dir, unit_idx):
    """
    Generate raster plots for each unit, with subplots for different factors,
    and an aligned histogram for each condition.

    Parameters:
    spike_times_per_trial (dict): Dictionary containing spike times for each unit and trial.
    behavioral_df (pd.DataFrame): Behavioral data for categorizing trials.
    factors (list): List of factors ('reward', 'nback', 'switch') to create subplots for.
    bin_size (float): Size of the bins for the histogram in seconds.
    """
    spike_times_per_cluster_path = os.path.join(
        spikes_dir, "spike_times_per_cluster.json"
    )
    spike_times_per_trial_path = os.path.join(
        spikes_dir, "spike_times_per_trial.json"
    )
    data_dir = os.path.dirname(os.path.dirname(spikes_dir))
    data_dir = data_dir.replace("phys_raw", "results")
    trials_path = os.path.join(
        data_dir, "open_ephys_events/open_ephys_trials.json"
    )
    spike_times_per_cluster = json.load(open(spike_times_per_cluster_path, "r"))
    spike_times_per_trial = json.load(open(spike_times_per_trial_path, "r"))
    oe_trials = json.load(open(trials_path, "r"))

    spikes = np.array(spike_times_per_cluster[str(unit_idx)])
    spikes_info = spike_times_per_trial[str(unit_idx)]
    return spikes, spikes_info, oe_trials


def main():
    mpl.rcParams["pdf.fonttype"] = 42
    """Make raster and histogram plots for each cluster."""

    ############################################################################
    ####  LOAD DATA
    ############################################################################

    print("LOADING DATA - raster_per_unit.py")
    root_dir = findrootdir()
    date_unit_idx_list = [
        ["20230216", 2, 48],
    ]
    for date, unit_idx_L, unit_idx_O in date_unit_idx_list:
        make_raster_plot(root_dir, date, unit_idx_L, unit_idx_O)


def make_raster_plot(root_dir, date, unit_idx_L, unit_idx_O):
    session_dir = f"{root_dir}/social_O_L/{date}"
    spikes_dir = f"{session_dir}/results/v_probe_1/spikes"
    spikes_L, spikes_info_L, oe_trials = load_spikes(spikes_dir, unit_idx_L)
    spikes_dir = f"{session_dir}/results/v_probe_2/spikes"
    spikes_O, spikes_info_O, _ = load_spikes(spikes_dir, unit_idx_O)

    data_dir = os.path.dirname(os.path.dirname(spikes_dir))
    data_dir = data_dir.replace("phys_raw", "results")
    print(f"data_dir: {data_dir}")
    trials_path = os.path.join(
        data_dir, "open_ephys_events/open_ephys_trials.json"
    )
    print(f"trials_dir: {trials_path}")
    behavior_path = os.path.join(data_dir, "moog_events/trial_info.csv")
    behavioral_df = pd.read_csv(behavior_path)
    behavioral_df["agent_choices"] = behavioral_df["agent_choices"].apply(
        literal_eval
    )
    behavioral_df["reaction_time"] = behavioral_df["reaction_time"].apply(
        literal_eval
    )
    behavioral_df["switched_actor"] = behavioral_df["switched_actor"].apply(
        literal_eval
    )

    print(f"Number of trials in behavioral data = {len(behavioral_df)}")

    fig = generate_dual_raster(
        spikes_O,
        spikes_info_O,
        spikes_L,
        spikes_info_L,
        oe_trials,
        behavioral_df,
    )
    # save figure
    fig.savefig(
        f"{root_dir}/plots_paper/Fig3A_raster_dual_{date}_{unit_idx_O}_{unit_idx_L}.pdf",
        dpi=300,
    )
    print("finished plotting - raster_dual.py")


if __name__ == "__main__":
    main()
