"""
This module selects data based on task_variable and implements cross-validation.
"""

import numpy as np
import pandas as pd


# select data based on task_index
def tdrSelectByCondition(
    data,
    task_index,
    crossvalidate="none",
    min_trials=10,
    p_test=0.5,
    random_seed=0,  # random
):
    rng = np.random.default_rng(seed=random_seed)
    variable_name = list(task_index.keys())
    nvr = len(variable_name)
    if "unit" in data:
        trial_ids_all = np.unique(
            np.concatenate([unit["trial_ids"] for unit in data["unit"]])
        )
        if crossvalidate == "leave_1_out":
            test_trial = rng.choice(trial_ids_all, 1)
        variable_name_all = list(data["unit"][0]["task_variable"].keys())
        selected_data = {
            "unit": [],
            "time": data["time"],
            "unit_idx_master": [],
        }
        for unit in data["unit"]:
            # Number of trials
            ntr = len(unit["response"])
            ncd = len(task_index[variable_name[0]])
            trial_ids_unit = unit["trial_ids"]

            # Initialize trials to select
            condition_masks = np.zeros((ncd, ntr), dtype=bool)
            test_condition_masks = np.zeros((ncd, ntr), dtype=bool)

            # Unique variable names
            variable_unique = [
                np.sort(pd.unique(unit["task_variable"][var]))
                for var in variable_name
            ]
            selected_unit = {
                "response": [],
                "response_test": [],
                "task_variable": {key: [] for key in variable_name_all},
                "task_variable_test": {key: [] for key in variable_name_all},
                "trial_ids": [],
                "trial_ids_test": [],
                "unit_idx_master": [],
            }
            n_condition_met = 0
            # Loop over conditions
            for icd in range(ncd):
                # Trials to select
                variable_masks = np.zeros((nvr, ntr), dtype=bool)

                # Loop over variables
                for ivr in range(nvr):
                    if task_index[variable_name[ivr]][icd] == -1:
                        # Take all trials
                        variable_masks[ivr, :] = np.ones(ntr, dtype=bool)
                    else:
                        # Find matching trials
                        aa = unit["task_variable"][variable_name[ivr]]
                        cc = task_index[variable_name[ivr]][icd]
                        if len(variable_unique[ivr]) <= cc:
                            # print(
                            #     f'rejecting unit {unit["dimension"]} because of'
                            #     f" {variable_name[ivr]} is not sufficient"
                            # )
                            continue
                        bb = variable_unique[ivr][cc]
                        variable_masks[ivr, :] = aa == bb

                # Fulfill constraints on all indices
                condition_masks[icd, :] = np.all(variable_masks, axis=0)
                test_condition_masks[icd, :] = False
                if crossvalidate == "random":
                    # randomly select half of the trials in condition_masks
                    # return two datasets (train and test)
                    # Find the indices of the True values
                    true_indices = np.where(condition_masks[icd, :])[0]
                    # Determine the number of True values to change
                    if len(true_indices) >= 2:
                        num_to_change = int(len(true_indices) * p_test)
                        # Randomly select a subset of the True indices
                        indices_to_change = rng.choice(
                            true_indices, size=num_to_change, replace=False
                        )
                        # Set the selected indices to False
                        condition_masks[icd, indices_to_change] = False
                        # Set the selected indices to True in the test data
                        test_condition_masks[icd, indices_to_change] = True
                elif crossvalidate == "leave_one_out":
                    # leave one out cross validation
                    # select one trial in condition_masks
                    # return two datasets (train and test)
                    # Find the indices of the True values
                    true_indices = np.where(condition_masks[icd, :])[0]
                    # Determine the number of True values to change
                    if len(true_indices) >= 1:
                        # Randomly select a subset of the True indices
                        indices_to_change = rng.choice(
                            true_indices, size=1, replace=False
                        )
                        # Set the selected indices to False
                        condition_masks[icd, indices_to_change] = False
                        # Set the selected indices to True in the test data
                        test_condition_masks[icd, indices_to_change] = True
                # if any value in condition_masks is true, select those values.
                elif crossvalidate == "leave_1_out":
                    # leave out specific trial
                    if test_trial in trial_ids_unit:
                        indices_to_change = np.where(
                            trial_ids_unit == test_trial
                        )[0]
                        condition_masks[icd, indices_to_change] = False
                if sum(condition_masks[icd, :]) >= min_trials:
                    mask = np.where(condition_masks[icd, :])[0]
                    mask_test = np.where(test_condition_masks[icd, :])[0]
                else:
                    continue
                n_condition_met += 1
            # check if any condition is empty
            if n_condition_met < ncd:
                continue

            for icd in range(ncd):
                mask = np.where(condition_masks[icd, :])[0]
                mask_test = np.where(test_condition_masks[icd, :])[0]
                if len(selected_unit["response"]) == 0:
                    selected_unit["response"] = unit["response"][mask]
                    selected_unit["response_test"] = unit["response"][mask_test]
                    selected_unit["trial_ids"] = unit["trial_ids"][mask]
                    selected_unit["trial_ids_test"] = unit["trial_ids"][
                        mask_test
                    ]
                else:
                    selected_unit["response"] = np.append(
                        selected_unit["response"],
                        unit["response"][mask],
                        axis=0,
                    )
                    selected_unit["response_test"] = np.append(
                        selected_unit["response_test"],
                        unit["response"][mask_test],
                        axis=0,
                    )
                    selected_unit["trial_ids"] = np.append(
                        selected_unit["trial_ids"],
                        unit["trial_ids"][mask],
                        axis=0,
                    )
                    selected_unit["trial_ids_test"] = np.append(
                        selected_unit["trial_ids_test"],
                        unit["trial_ids"][mask_test],
                        axis=0,
                    )

                for key in variable_name_all:
                    selected_unit["task_variable"][key] = np.append(
                        selected_unit["task_variable"][key],
                        unit["task_variable"][key][mask],
                        axis=0,
                    )
                    selected_unit["task_variable_test"][key] = np.append(
                        selected_unit["task_variable_test"][key],
                        unit["task_variable"][key][mask_test],
                        axis=0,
                    )

            if crossvalidate == "leave_1_out":
                # add test trial to response_test and task_variable_test
                idx_trial = np.where(unit["trial_ids"] == test_trial)[0]
                selected_unit["response_test"] = unit["response"][idx_trial]
                selected_unit["trial_ids_test"] = unit["trial_ids"][idx_trial]
                for key in variable_name_all:
                    selected_unit["task_variable_test"][key] = unit[
                        "task_variable"
                    ][key][idx_trial]
            selected_unit["dimension"] = unit["dimension"]
            selected_unit["unit_idx_master"] = unit["unit_idx_master"]
            selected_unit["session_id"] = unit["session_id"]
            selected_data["unit"].append(selected_unit)
            selected_data["unit_idx_master"].append(unit["unit_idx_master"])
        # single session case: all units in one matrix; select trials based on
        # task variable
        return selected_data
    else:
        return tdrSelectByCondition_single_session(
            data,
            task_index,
            crossvalidate,
            min_trials=min_trials,
            random_seed=random_seed,
        )


def tdrSelectByCondition_single_session(
    data,
    task_index,
    crossvalidate="none",
    min_trials=10,
    p_test=0.5,
    random_seed=0,  # random
):
    rng = np.random.default_rng(seed=random_seed)
    variable_name = list(task_index.keys())
    variable_name_all = list(data["task_variable"].keys())
    nvr = len(variable_name)
    trial_ids_all = data["trial_ids"]
    trial_ids_train = []
    trial_ids_test = []
    # Initialize the selected data dictionary
    selected_data = {
        "response": [],
        "response_test": [],
        "task_variable": {key: [] for key in variable_name_all},
        "task_variable_test": {key: [] for key in variable_name_all},
        "trial_ids": [],
        "trial_ids_test": [],
        "time": data["time"],
    }
    response = data["response"]
    task_variable = data["task_variable"]
    ntr = response.shape[2]
    ncd = len(task_index[variable_name[0]])
    # Initialize trials to select
    condition_masks = np.zeros((ncd, ntr), dtype=bool)
    test_condition_masks = np.zeros((ncd, ntr), dtype=bool)
    # Unique variable names
    variable_unique = [
        np.sort(pd.unique(task_variable[var])) for var in variable_name
    ]
    n_condition_met = 0
    # Loop over conditions
    for icd in range(ncd):
        # Trials to select
        variable_masks = np.zeros((nvr, ntr), dtype=bool)
        # Loop over variables
        for ivr in range(nvr):
            if task_index[variable_name[ivr]][icd] == -1:
                # Take all trials
                variable_masks[ivr, :] = np.ones(ntr, dtype=bool)
            else:
                # Find matching trials
                aa = task_variable[variable_name[ivr]]
                cc = task_index[variable_name[ivr]][icd]
                if len(variable_unique[ivr]) <= cc:
                    continue
                bb = variable_unique[ivr][cc]
                variable_masks[ivr, :] = aa == bb

        # Fulfill constraints on all indices
        condition_masks[icd, :] = np.all(variable_masks, axis=0)
        test_condition_masks[icd, :] = False
        if crossvalidate == "random":
            # randomly select half of the trials in condition_masks
            # return two datasets (train and test)
            # Find the indices of the True values
            true_indices = np.where(condition_masks[icd, :])[0]
            # Determine the number of True values to change
            if len(true_indices) >= 0:
                num_to_change = int(len(true_indices) * p_test)
                # Randomly select a subset of the True indices
                indices_to_change = rng.choice(
                    true_indices, size=num_to_change, replace=False
                )
                # Set the selected indices to False
                condition_masks[icd, indices_to_change] = False
                # Set the selected indices to True in the test data
                test_condition_masks[icd, indices_to_change] = True
        # if there are more than 10 trials in the condition, select those values.
        if sum(condition_masks[icd, :]) >= min_trials:
            mask = np.where(condition_masks[icd, :])[0]
            mask_test = np.where(test_condition_masks[icd, :])[0]
        else:
            continue
        trial_ids_train.append(data["trial_ids"][mask])
        trial_ids_test.append(data["trial_ids"][mask_test])
        # If the combination exists in task_index, select the data
        if len(selected_data["response"]) == 0:
            selected_data["response"] = data["response"][:, :, mask]
            selected_data["response_test"] = data["response"][:, :, mask_test]
        else:
            selected_data["response"] = np.concatenate(
                (selected_data["response"], data["response"][:, :, mask]),
                axis=2,
            )
            selected_data["response_test"] = np.concatenate(
                (
                    selected_data["response_test"],
                    data["response"][:, :, mask_test],
                ),
                axis=2,
            )
        for key in variable_name_all:
            selected_data["task_variable"][key] = np.append(
                selected_data["task_variable"][key],
                data["task_variable"][key][mask],
                axis=0,
            )
            selected_data["task_variable_test"][key] = np.append(
                selected_data["task_variable_test"][key],
                data["task_variable"][key][mask_test],
                axis=0,
            )
        n_condition_met += 1
    # check if any condition is empty
    if n_condition_met < ncd:
        return None
    selected_data["dimension"] = data["dimension"]
    selected_data["unit_idx_master"] = data["unit_idx_master"]
    selected_data["trial_ids"] = np.concatenate(trial_ids_train)
    selected_data["trial_ids_test"] = np.concatenate(trial_ids_test)
    # leave_one_out condition: randomly select one trial
    if crossvalidate == "leave_one_out":
        n_trials = selected_data["response"].shape[2]
        test_trial = rng.choice(n_trials, 1, replace=False)
        trial_ids_test = selected_data["trial_ids"][test_trial]
        trial_ids_train = np.delete(selected_data["trial_ids"], test_trial)
        selected_data["trial_ids"] = trial_ids_train
        selected_data["trial_ids_test"] = trial_ids_test
        selected_data["response_test"] = selected_data["response"][
            :, :, test_trial
        ]
        selected_data["response"] = np.delete(
            selected_data["response"], test_trial, axis=2
        )
        for key in variable_name_all:
            selected_data["task_variable_test"][key] = selected_data[
                "task_variable"
            ][key][test_trial]
            selected_data["task_variable"][key] = np.delete(
                selected_data["task_variable"][key], test_trial, axis=0
            )
    return selected_data
