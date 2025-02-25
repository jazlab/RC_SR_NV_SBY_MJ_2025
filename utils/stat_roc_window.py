"""
This module runs roc analysis on given windows of time.
"""

import numpy as np
from utils.makeTrigInfo import makeTrigInfo
from utils.roc_selectivity import calculate_selectivity_from_triginfo


def roc_analysis(spikes, trials, behavioral_df, valid_trials, agent_id, config):
    trigger_1_self = []
    trigger_2_self = []
    trigger_1_other = []
    trigger_2_other = []
    other_id = 1 - agent_id

    events = spikes
    n_phases = 4
    binsize = config["binsize"]
    phase = config["phase"]
    factor = config["factor"]
    xmin = config["xmin"]  # this sets up the boundaries of the sweep
    xmax = config["xmax"]
    xstep = config["xstep"]
    count = 0  # number of trials
    assert len(trials) == len(behavioral_df)
    for i_trial, d in enumerate(trials):
        # check if the trial is valid
        if i_trial not in valid_trials:
            continue
        # skip first and last trial
        if i_trial < 2 or i_trial == len(trials) - 1:
            continue
        num_phases = len(d["relative_phase_times"])
        if num_phases != n_phases:
            # Incomplete trial
            continue
        count += 1
        trial_start_time = d["t_start"]
        t_nav_off = d["relative_phase_times"][2] + trial_start_time
        player = d["player"]
        reward = behavioral_df["reward"][i_trial]
        history = behavioral_df["history"][i_trial]
        rt = behavioral_df["reaction_time"][i_trial][agent_id]
        self_choice = behavioral_df["agent_choices"][i_trial][agent_id]
        other_choice = behavioral_df["agent_choices"][i_trial][other_id]

        if phase == "fdbk":
            trigger_time = t_nav_off
        elif phase == "choice":
            trigger_time = rt + trial_start_time
        elif phase == "prechoice":
            trigger_time = rt + trial_start_time - xmax

        if phase == "choice" or phase == "prechoice":
            player = behavioral_df["player"][i_trial - 1]
            reward = behavioral_df["reward"][i_trial - 1]
            history = behavioral_df["history"][i_trial - 1]

        if (phase == "fdbk") and ("nback" in factor):
            if self_choice != other_choice:
                continue

        if factor == "reward":
            if reward == 1:
                if player == agent_id:
                    trigger_1_self.append(trigger_time)
                else:
                    trigger_1_other.append(trigger_time)
            elif reward == 0:
                if player == agent_id:
                    trigger_2_self.append(trigger_time)
                else:
                    trigger_2_other.append(trigger_time)

        if factor == "nback_all":  # 1back A vs 2back A or 1back O vs 2back O
            if history == "1NR":
                if player == agent_id:
                    trigger_1_self.append(trigger_time)
                else:
                    trigger_1_other.append(trigger_time)
            elif history == "2NR":
                if player == agent_id:
                    trigger_2_self.append(trigger_time)
                elif player != agent_id:
                    trigger_2_other.append(trigger_time)

        if factor == "nback_AOOA":  # 2 back of S vs O (other) or O vs S (self)
            if history == "1NR":
                if player != agent_id:
                    trigger_1_self.append(trigger_time)
                else:
                    trigger_1_other.append(trigger_time)
            elif history == "2NR":
                if player == agent_id:
                    trigger_2_self.append(trigger_time)
                else:
                    trigger_2_other.append(trigger_time)

    trigger_1 = trigger_1_self + trigger_1_other
    trigger_2 = trigger_2_self + trigger_2_other

    tf_1_self = makeTrigInfo(
        trigger_1_self, events, xl=xmax + binsize, binsize=binsize
    )
    tf_2_self = makeTrigInfo(
        trigger_2_self, events, xl=xmax + binsize, binsize=binsize
    )
    tf_1_other = makeTrigInfo(
        trigger_1_other, events, xl=xmax + binsize, binsize=binsize
    )
    tf_2_other = makeTrigInfo(
        trigger_2_other, events, xl=xmax + binsize, binsize=binsize
    )
    tf_1 = makeTrigInfo(trigger_1, events, xl=xmax + binsize, binsize=binsize)
    tf_2 = makeTrigInfo(trigger_2, events, xl=xmax + binsize, binsize=binsize)
    # roc score and p value for each window in the sweep
    rocInfo = {}
    for tstart in np.arange(xmin, xmax + xstep, xstep):
        for tend in np.arange(tstart + xstep, tstart + xstep + 1, xstep):
            # check if tstart is -0.
            tstart = np.around(tstart, decimals=1)
            tend = np.around(tend, decimals=1)
            if tstart == -0.0:
                tstart = 0.0
            if tend == -0.0:
                tend = 0.0
            window = [tstart, tend]
            selctivity_self, pv_self = calculate_selectivity_from_triginfo(
                tf_1_self, tf_2_self, window
            )
            selctivity_other, pv_other = calculate_selectivity_from_triginfo(
                tf_1_other, tf_2_other, window
            )
            selectivity_both, pv_both = calculate_selectivity_from_triginfo(
                tf_1, tf_2, window
            )
            s_self = selctivity_self
            s_other = selctivity_other
            s_both = selectivity_both
            p_self = pv_self
            p_other = pv_other
            p_both = pv_both
            formatted_window = [float(round(tstart, 1)), float(round(tend, 1))]
            # convert formatted_window to string
            formatted_window = str(formatted_window)
            rocInfo[formatted_window] = {
                "selectivity_self": s_self,
                "selectivity_other": s_other,
                "selectivity_both": s_both,
                "p_self": p_self,
                "p_other": p_other,
                "p_both": p_both,
                "n_self": len(trigger_1_self) + len(trigger_2_self),
                "n_other": len(trigger_1_other) + len(trigger_2_other),
                "n_both": len(trigger_1) + len(trigger_2),
            }

    return rocInfo
