"""
This script plots the results of the single-session predciton of switches from
neural data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
from scipy import stats
import json
import numpy as np
from pathlib import Path
from utils.LoadSession import findrootdir
from behavior.plot_slopes import plot_slopes_only
from utils.plots import beutify


def plot_pswitch_by_sessions_leave_one_out(
    stat_sessions, formatted_window, plot_actor_observer=False
):
    f, ax = plt.subplots(figsize=(6, 6))
    all_values = []
    pvals = []
    n_sessions = 0
    n_sig = 0
    for stat_session in stat_sessions.values():
        # check if stat_session is a dict
        if type(stat_session) != dict:
            continue
        for key, value in stat_session.items():
            if type(value) == list:
                stat_session[key] = np.array(value)
        pred_switches = np.array(stat_session["pred_switches"])
        actu_switches = np.array(stat_session["actu_switches"])
        pswitch_0 = actu_switches[pred_switches == 0]
        pswitch_1 = actu_switches[pred_switches == 1]
        all_values.extend(pswitch_0)
        all_values.extend(pswitch_1)
        # to assess significance of difference, we perform a rank sum test
        _, p = stats.ranksums(pswitch_0, pswitch_1)
        n_sessions += 1
        color = "gray"
        if p < 0.05 and np.mean(pswitch_0) < np.mean(pswitch_1):
            n_sig += 1
            color = "black"
        pvals.append(p)
        ax.scatter(
            np.mean(pswitch_0),
            np.mean(pswitch_1),
            s=5,
            facecolor=color,
            edgecolors="none",  # No edges
            zorder=2,
        )
        # add error bars using standard deviation
        ax.errorbar(
            np.mean(pswitch_0),
            np.mean(pswitch_1),
            xerr=np.std(pswitch_0) / np.sqrt(len(pswitch_0)),
            yerr=np.std(pswitch_1) / np.sqrt(len(pswitch_1)),
            fmt="o",
            color=color,
            elinewidth=0.5,
            zorder=1,
        )
    print(f"n_sig: {n_sig} / {n_sessions}")
    min_val = max(min(all_values) - 0.1, 0)
    max_val = min(max(all_values) + 0.1, 1)
    ax.plot(
        [min_val, max_val], [min_val, max_val], color="black"
    )  # Add unit line
    ax.set_xlim(0, 0.5)  # Set x axis limits
    ax.set_ylim(0, 0.5)  # Set y axis limits
    ax.set_xlabel("P(switch|low)")  # Change x axis label
    ax.set_ylabel("P(switch|high)")  # Change y axis label
    beutify(ax)
    return f


def get_slope_and_beta(stat_sessions):
    slopes_actor = []
    slopes_observer = []
    projections_all = []
    histories = []
    actors = []
    for stat_session in stat_sessions.values():
        # check if stat_session is a dict
        if type(stat_session) != dict:
            continue
        for key, value in stat_session.items():
            if type(value) == list:
                stat_session[key] = np.array(value)
        projections = stat_session["projection"]
        history = stat_session["history"]
        switch = stat_session["actu_switches"]
        actor = stat_session["actor"]
        pval = stat_session["pval_ranksum"]
        if pval > 0.05:
            continue
        projections_all.append(projections)
        histories.append(history)
        actors.append(actor)
        idx_actor = np.where((actor == -1) & (history < 2))
        idx_observer = np.where((actor == 1) & (history < 2))
        # perform regression of projection on history values, for each actor
        slope_actor, intercept, r_value, p_value, std_err = stats.linregress(
            history[idx_actor], projections[idx_actor]
        )
        slope_observer, intercept, r_value, p_value, std_err = stats.linregress(
            history[idx_observer], projections[idx_observer]
        )
        stat_session["slopes_actor"] = slope_actor
        stat_session["slopes_observer"] = slope_observer
        slopes_actor.append(slope_actor)
        slopes_observer.append(slope_observer)
    stat_sessions["slopes_actor"] = slopes_actor
    stat_sessions["slopes_observer"] = slopes_observer
    # get beta value from all sessions combined
    idx_actor = np.where(
        (np.concatenate(actors) == -1) & (np.concatenate(histories) < 2)
    )
    idx_observer = np.where(
        (np.concatenate(actors) == 1) & (np.concatenate(histories) < 2)
    )
    slope_actor, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate(histories)[idx_actor],
        np.concatenate(projections_all)[idx_actor],
    )
    slope_observer, intercept, r_value, p_value, std_err = stats.linregress(
        np.concatenate(histories)[idx_observer],
        np.concatenate(projections_all)[idx_observer],
    )
    stat_sessions["slope_actor_all"] = slope_actor
    stat_sessions["slope_observer_all"] = slope_observer
    return stat_sessions


def plot_slope_and_coef(stat_sessions, subject, event):
    # use behavior.plot_slopes.py
    root_dir = Path(findrootdir())
    stat_sessions = get_slope_and_beta(stat_sessions)
    slopes_actor = stat_sessions["slopes_actor"]
    slopes_observer = stat_sessions["slopes_observer"]
    factor_name = f"EvDir_{event}"
    plot_slopes_only(
        root_dir,
        subject,
        slopes_actor,
        slopes_observer,
        factor_name,
    )


def main():
    # load data
    root_dir = findrootdir()
    for event in ["fdbk"]:
        stat_sessions = {}
        # plot pswitch for each unit on a scatter plot
        if event == "fdbk":
            tstart = 0.0
            tend = 0.6
        elif event == "prechoice":
            tstart = -0.6
            tend = 0.0
        formatted_window = [round(tstart, 1), round(tend, 1)]
        formatted_window = str(formatted_window)
        for subject in ["O", "L"]:
            file_name = f"{root_dir}/stats_paper/{subject}_{event}_{formatted_window}_switch_dir_stats.json"
            stat_session = json.load(open(file_name, "r"))
            fig = plot_pswitch_by_sessions_leave_one_out(
                stat_session, formatted_window
            )
            fig_name_scatter = f"{root_dir}/plots_paper/FigS7CD_switch_by_sessions_{subject}.pdf"
            fig.savefig(fig_name_scatter, dpi=300)
            plt.close()
            # calculate slope and beta for each session
            stat_session = get_slope_and_beta(stat_session)
            plot_slope_and_coef(stat_session, subject, event)
            # append subject to each key of stat_session
            for key in list(stat_session.keys()):
                stat_session[f"{subject}_{key}"] = stat_session.pop(key)
            # combine stat_sessions
            stat_sessions.update(stat_session)
        # combine stat_sessions
        plot_slope_and_coef(stat_sessions, "both", event)
        # plot pswitch for each unit on a scatter plot
        fig = plot_pswitch_by_sessions_leave_one_out(
            stat_sessions, formatted_window
        )
        fig_name_scatter = (
            f"{root_dir}/plots_paper/Fig4C_switch_by_sessions_{event}.pdf"
        )
        fig.savefig(fig_name_scatter, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
