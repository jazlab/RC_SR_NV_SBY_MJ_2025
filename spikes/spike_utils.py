import numpy as np
from scipy.stats import sem, t


def initialize_dictionary():
    # Define components of the keys
    history = ["1R", "1NR", "2NR"]
    roles = [
        "actor",
        "observer",
        "both",
        "actor-actor",
        "actor-observer",
        "observer-actor",
        "observer-observer",
        "AOOA",
        "AAOO",
        "hasobs",
        "noobs",
    ]
    gates = ["less_gates", "more_gates"]
    phases = ["early", "late"]
    reward_conditions = [
        "low_rew",
        "high_rew",
        "low_no_rew",
        "high_no_rew",
        "rew",
        "no_rew",
    ]
    actions = ["Stay", "Switch"]

    # Initialize the dictionary
    spike_times = {}

    # Generate keys
    for role in roles:
        for condition in reward_conditions:
            spike_times[f"{condition}_{role}_trials_spikes"] = []
        # Generate history and role based keys
        for h in history:
            spike_times[f"{h}_{role}_trials_spikes"] = []
        # Generate gates and role based keys
        for gate in gates:
            spike_times[f"{gate}_{role}_trials_spikes"] = []
        # Generate phase and role based keys
        for phase in phases:
            spike_times[f"{phase}_{role}_trials_spikes"] = []
        # Generate action and role based keys
        for action in actions:
            spike_times[f"{action}_{role}_trials_spikes"] = []
    # Special case for "1NRO_trials_spikes"
    spike_times["1NRO_trials_spikes"] = []
    return spike_times


def get_condition_spike_times(spike_dict, bhv_df, agent_idx, event="fdbk"):
    # Function to calculate median performance
    def median_performance(reward):
        return np.median(bhv_df[bhv_df["reward"] == reward]["touched_polls"])

    # Print median performances
    rew_median_performance = median_performance(1)
    no_rew_median_performance = median_performance(0)
    print(f"Median performance in rewarded trials = {rew_median_performance}")
    print(
        f"Median performance in not-rewarded trials = {no_rew_median_performance}"
    )

    # Function to process spike times for each cluster
    def process_cluster(cluster_data):

        spike_times = initialize_dictionary()
        trial_counts = {
            key.replace("_trials_spikes", "_trials_count"): 0
            for key in spike_times
        }

        for trial, trial_data in enumerate(cluster_data):
            if trial_data["valid"] == 0:
                continue
            start_time = trial_data["relative_phase_times"][0]
            reaction_time = bhv_df.iloc[trial]["reaction_time"][agent_idx]
            if event == "fdbk":
                event_time = trial_data["relative_phase_times"][2]
            elif event == "choice":
                event_time = start_time + reaction_time
            elif event == "prechoice":
                event_time = start_time + reaction_time - 0.6
            nav_end_spike_times = (
                np.array(trial_data["relative_spike_times"]) - event_time
            )

            # Process reward-based self and other conditions
            reward = bhv_df.iloc[trial]["reward"]
            player = bhv_df.iloc[trial]["player"]
            player_prev = bhv_df.iloc[trial - 1]["player"]
            # for start and choice, use the reward and player from last trial
            if event == "choice" or event == "prechoice":
                if trial < 2:
                    # skip the first two trials for these events
                    continue
                reward = bhv_df.iloc[trial - 1]["reward"]
                player = bhv_df.iloc[trial - 1]["player"]
                player_prev = bhv_df.iloc[trial - 2]["player"]
            actor_observer = "actor" if player == agent_idx else "observer"
            ao_prev = "actor" if player_prev == agent_idx else "observer"
            twoback_with_obs = "hasobs"
            if player == agent_idx and player_prev == agent_idx:
                twoback_with_obs = "noobs"
            actor_observer_2back = ao_prev + "-" + actor_observer
            if actor_observer == ao_prev:
                actor_observer_2back_set = "AAOO"
            else:
                actor_observer_2back_set = "AOOA"
            if reward == 1:
                condition = f"rew_{actor_observer}"
                spike_times[f"rew_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts[f"rew_both_trials_count"] += 1
            else:
                condition = f"no_rew_{actor_observer}"
                spike_times[f"no_rew_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts[f"no_rew_both_trials_count"] += 1
            spike_times[f"{condition}_trials_spikes"].append(
                nav_end_spike_times.tolist()
            )
            trial_counts[f"{condition}_trials_count"] += 1

            # Process performance-based conditions
            touched_polls = bhv_df.iloc[trial]["touched_polls"]
            # for start and choice, use the touched_polls from last trial
            if event == "choice" or event == "prechoice":
                touched_polls = bhv_df.iloc[trial - 1]["touched_polls"]
            if reward == 1:
                condition = (
                    "low_rew"
                    if touched_polls < rew_median_performance
                    else "high_rew"
                )
            else:
                condition = (
                    "low_no_rew"
                    if touched_polls < no_rew_median_performance
                    else "high_no_rew"
                )
            spike_times[f"{condition}_both_trials_spikes"].append(
                nav_end_spike_times.tolist()
            )
            trial_counts[f"{condition}_both_trials_count"] += 1

            # Process n_gates_touched conditions
            n_gates_touched = bhv_df.iloc[trial]["touched_polls"]
            if event == "choice" or event == "prechoice":
                n_gates_touched = bhv_df.iloc[trial - 1]["touched_polls"]
            if reward == 0 and n_gates_touched < 11:
                spike_times["less_gates_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts["less_gates_both_trials_count"] += 1
                spike_times[
                    f"less_gates_{actor_observer}_trials_spikes"
                ].append(nav_end_spike_times.tolist())
                trial_counts[f"less_gates_{actor_observer}_trials_count"] += 1
            elif reward == 0 and n_gates_touched >= 11:
                spike_times["more_gates_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts["more_gates_both_trials_count"] += 1
                spike_times[
                    f"more_gates_{actor_observer}_trials_spikes"
                ].append(nav_end_spike_times.tolist())
                trial_counts[f"more_gates_{actor_observer}_trials_count"] += 1

            # Process early vs late conditions
            if reward == 0 and bhv_df.iloc[trial]["pos_in_block"] < 5:
                spike_times["early_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts["early_both_trials_count"] += 1

                spike_times[f"early_{actor_observer}_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts[f"early_{actor_observer}_trials_count"] += 1
            elif reward == 0 and bhv_df.iloc[trial]["pos_in_block"] >= 5:
                spike_times["late_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts["late_both_trials_count"] += 1
                spike_times[f"late_{actor_observer}_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts[f"late_{actor_observer}_trials_count"] += 1

            # Process history-based conditions
            history = bhv_df.iloc[trial]["history"]
            if event == "choice" or event == "prechoice":
                history = bhv_df.iloc[trial - 1]["history"]
            if history in ["1R", "1NR", "2NR"]:
                spike_times[f"{history}_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts[f"{history}_both_trials_count"] += 1
                spike_times[f"{history}_{actor_observer}_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts[f"{history}_{actor_observer}_trials_count"] += 1
            if history == "2NR":
                spike_times[
                    f"{history}_{actor_observer_2back}_trials_spikes"
                ].append(nav_end_spike_times.tolist())
                trial_counts[
                    f"{history}_{actor_observer_2back}_trials_count"
                ] += 1
                spike_times[
                    f"{history}_{actor_observer_2back_set}_trials_spikes"
                ].append(nav_end_spike_times.tolist())
                trial_counts[
                    f"{history}_{actor_observer_2back_set}_trials_count"
                ] += 1
                spike_times[
                    f"{history}_{twoback_with_obs}_trials_spikes"
                ].append(nav_end_spike_times.tolist())
                trial_counts[f"{history}_{twoback_with_obs}_trials_count"] += 1

            # Process switch condition
            if reward == 0:
                switch_condition = (
                    "Stay"
                    if bhv_df.iloc[trial]["switched_actor"][agent_idx] == 0
                    else "Switch"
                )
                if event == "choice" or event == "prechoice":
                    switch_condition = (
                        "Stay"
                        if bhv_df.iloc[trial - 1]["switched_actor"][agent_idx]
                        == 0
                        else "Switch"
                    )
                spike_times[f"{switch_condition}_both_trials_spikes"].append(
                    nav_end_spike_times.tolist()
                )
                trial_counts[f"{switch_condition}_both_trials_count"] += 1
                spike_times[
                    f"{switch_condition}_{actor_observer}_trials_spikes"
                ].append(nav_end_spike_times.tolist())
                trial_counts[
                    f"{switch_condition}_{actor_observer}_trials_count"
                ] += 1

        return {**spike_times, **trial_counts}

    # Process each cluster
    return {
        cluster: process_cluster(cluster_data)
        for cluster, cluster_data in spike_dict.items()
        if cluster_data is not None
    }


def find_firing_rate(spike_times, trial_count, window_start, window_end):
    # find the firing rate with a sliding window of 100 ms with the starting bin centered on window_start to the last bin centered closest to window_end
    time_bins = []
    firing_rate = []
    confidence_intervals = []
    bin_center = window_start - 0.150 / 2
    while bin_center <= window_end + 0.150 / 2:
        time_bins.append(bin_center)
        rates = []
        for trial in spike_times:
            trial = np.array(trial)
            rate = np.sum(
                (trial >= (bin_center - 0.150 / 2))
                & (trial < (bin_center + 0.150 / 2))
            ) * (1000 / 150)
            rates.append(rate)
        mean_rate = np.mean(rates)
        firing_rate.append(mean_rate)
        # calculate 95% confidence interval
        ci = sem(rates) * 1.96
        confidence_intervals.append((mean_rate - ci, mean_rate + ci))
        bin_center = bin_center + 0.010
    # smooth firing rate with a rolling window of 3
    firing_rate = np.convolve(firing_rate, np.ones(3) / 3, mode="same")
    # cut firing rate to window_start and window_end
    start_idx = int((window_start - time_bins[0]) / 0.010)
    end_idx = int((window_end - time_bins[0]) / 0.010) + 1
    time_bins = time_bins[start_idx:end_idx]
    firing_rate = firing_rate[start_idx:end_idx]
    confidence_intervals = confidence_intervals[start_idx:end_idx]

    return np.array(time_bins), firing_rate, confidence_intervals


def get_firing_rates_per_condition(
    condition_spike_times, window_start, window_end
):
    firing_rates = {cluster: {} for cluster in condition_spike_times}

    for cluster, cluster_spike_times in condition_spike_times.items():
        for condition, spikes in cluster_spike_times.items():
            if condition.endswith("_spikes"):  # Process only spike data
                count_key = condition.replace("_spikes", "_count")
                if count_key in cluster_spike_times:
                    (
                        time_bins,
                        firing_rates[cluster][condition],
                        firing_rates[cluster][
                            condition.replace("_spikes", "_ci")
                        ],
                    ) = find_firing_rate(
                        spikes,
                        cluster_spike_times[count_key],
                        window_start,
                        window_end,
                    )

    return time_bins, firing_rates
