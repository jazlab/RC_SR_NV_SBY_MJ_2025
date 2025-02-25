import numpy as np


def categorize_trials(behavioral_df, factor, agent_idx=0):
    """
    Categorize trials based on the specified factor.

    Parameters:
    behavioral_df (pd.DataFrame): Behavioral data for categorizing trials.
    factor (str): The factor by which to categorize trials ('reward', 'nback', 'switch').

    Returns:
    dict: A dictionary with conditions as keys and lists of trial numbers as values.
    """
    conditions = {}

    if factor == "reward":
        # Categorize based on the 'reward' column
        rewarded_trials = behavioral_df[
            behavioral_df["reward"] == 1
        ].index.tolist()
        nonrewarded_trials = behavioral_df[
            behavioral_df["reward"] == 0
        ].index.tolist()
        conditions = {
            "rewarded": rewarded_trials,
            "nonrewarded": nonrewarded_trials,
        }
    elif factor == "choice_so":
        # Categorize based on the 'choice' column
        o_left_trials = behavioral_df[
            behavioral_df["choice_a0"] == -1
        ].index.tolist()
        o_right_trials = behavioral_df[
            behavioral_df["choice_a0"] == 1
        ].index.tolist()
        l_left_trials = behavioral_df[
            behavioral_df["choice_a1"] == -1
        ].index.tolist()
        l_right_trials = behavioral_df[
            behavioral_df["choice_a1"] == 1
        ].index.tolist()
        if agent_idx == 0:
            self_left_trials = o_left_trials
            self_right_trials = o_right_trials
            other_left_trials = l_left_trials
            other_right_trials = l_right_trials
        else:
            self_left_trials = l_left_trials
            self_right_trials = l_right_trials
            other_left_trials = o_left_trials
            other_right_trials = o_right_trials
        conditions = {
            "sLoL": [
                idx for idx in self_left_trials if idx in other_left_trials
            ],
            "sRoR": [
                idx for idx in self_right_trials if idx in other_right_trials
            ],
            "sLoR": [
                idx for idx in self_left_trials if idx in other_right_trials
            ],
            "sRoL": [
                idx for idx in self_right_trials if idx in other_left_trials
            ],
        }
    elif factor == "nav_so":
        # Categorize based on the 'player_choice' column
        left_trials = behavioral_df[
            behavioral_df["player_choice"] == -1
        ].index.tolist()
        right_trials = behavioral_df[
            behavioral_df["player_choice"] == 1
        ].index.tolist()
        self_trials = behavioral_df[
            behavioral_df["player"] == agent_idx
        ].index.tolist()
        other_trials = behavioral_df[
            behavioral_df["player"] != agent_idx
        ].index.tolist()
        conditions = {
            "self_left": [idx for idx in self_trials if idx in left_trials],
            "self_right": [idx for idx in self_trials if idx in right_trials],
            "other_left": [idx for idx in other_trials if idx in left_trials],
            "other_right": [idx for idx in other_trials if idx in right_trials],
        }

    elif factor == "reward_so":
        # Categorize based on the 'reward' column
        rewarded_trials = behavioral_df[
            behavioral_df["reward"] == 1
        ].index.tolist()
        nonrewarded_trials = behavioral_df[
            behavioral_df["reward"] == 0
        ].index.tolist()
        # Select only trials where the agent was the actor in the trial
        conditions = {
            "rewarded_self": [
                idx
                for idx in rewarded_trials
                if behavioral_df["player"][idx] == agent_idx
            ],
            "nonrewarded_self": [
                idx
                for idx in nonrewarded_trials
                if behavioral_df["player"][idx] == agent_idx
            ],
            "rewarded_other": [
                idx
                for idx in rewarded_trials
                if behavioral_df["player"][idx] != agent_idx
            ],
            "nonrewarded_other": [
                idx
                for idx in nonrewarded_trials
                if behavioral_df["player"][idx] != agent_idx
            ],
        }

    elif factor == "nback":
        # Categorize based on the 'history' column
        conditions = {
            "1R": behavioral_df[
                behavioral_df["history"] == "1R"
            ].index.tolist(),
            "1NR": behavioral_df[
                behavioral_df["history"] == "1NR"
            ].index.tolist(),
            "2NR": behavioral_df[
                behavioral_df["history"] == "2NR"
            ].index.tolist(),
        }
        conditions["1NR"] = [
            idx
            for idx in conditions["1NR"]
            if idx in np.array(conditions["2NR"]) - 1
        ]
        conditions["1R"] = [
            idx
            for idx in conditions["1R"]
            if idx in np.array(conditions["1NR"]) - 1
        ]

    elif factor == "nback_self":
        # Categorize based on the 'history' column
        conditions = {
            "1R": behavioral_df[
                behavioral_df["history"] == "1R"
            ].index.tolist(),
            "1NR": behavioral_df[
                behavioral_df["history"] == "1NR"
            ].index.tolist(),
            "2NR": behavioral_df[
                behavioral_df["history"] == "2NR"
            ].index.tolist(),
        }
        conditions["1NR"] = [
            idx
            for idx in conditions["1NR"]
            if idx in np.array(conditions["2NR"]) - 1
        ]
        conditions["1R"] = [
            idx
            for idx in conditions["1R"]
            if idx in np.array(conditions["1NR"]) - 1
        ]
        # Select only trials where the agent was the actor in the trial
        conditions = {
            key: [
                idx
                for idx in value
                if behavioral_df["player"][idx] == agent_idx
            ]
            for key, value in conditions.items()
        }
        # For 2NR, select only trials where the agent was the actor in the previous trial
        conditions["2NR"] = [
            idx
            for idx in conditions["2NR"]
            if behavioral_df["player"][idx - 1] == agent_idx
        ]

    elif factor == "switch":
        # Categorize based on the 'switched_actor' column
        switch_trials = behavioral_df[
            behavioral_df["switched_actor"].apply(lambda x: x[agent_idx] == 1)
        ].index.tolist()
        nonswitch_trials = behavioral_df[
            behavioral_df["switched_actor"].apply(lambda x: x[agent_idx] == 0)
        ].index.tolist()
        conditions = {"switch": switch_trials, "nonswitch": nonswitch_trials}

    return conditions
