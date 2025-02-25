"""
This module contains utility functions for the TDR analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from pyTdr.tdrRegression import perform_regression
from pyTdr.tdrSelectByCondition import tdrSelectByCondition
from utils.find_subset_with_all_ones import find_subset_with_all_ones
from utils.LoadSession import findrootdir
from copy import deepcopy


def load_config():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the absolute path to the config.json file
    config_path = os.path.join(script_dir, "config", "config.json")
    config = json.load(open(config_path))
    return config


def save_config(config):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the absolute path to the config.json file
    config_path = os.path.join(script_dir, "config", "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def update_config(field, value):
    # load config.json file
    config = load_config()
    config[field] = value
    save_config(config)


def prepare_test_data(data):
    data_test = deepcopy(data)
    if "unit" in data_test:
        for unit in data_test["unit"]:
            unit["response"] = unit["response_test"]
            unit["task_variable"] = unit["task_variable_test"]
            unit["trial_ids"] = unit["trial_ids_test"]
    else:
        data_test["response"] = data_test["response_test"]
        data_test["task_variable"] = data_test["task_variable_test"]
        data_test["trial_ids"] = data_test["trial_ids_test"]
    return data_test


def remove_low_rate(data, metadata, min_rate=0.5):
    if "unit" in data:
        idx_to_remove = []
        for i_unit, unit in enumerate(data["unit"]):
            rate = np.mean(unit["response"])
            if rate < min_rate:
                idx_to_remove.append(i_unit)
        # remove units with low firing rate
        for i in sorted(idx_to_remove, reverse=True):
            data["unit"].pop(i)
            for key in metadata["unit"]:
                metadata["unit"][key].pop(i)
    else:
        n_units = data["response"].shape[0]
        # Create a list to store indices of units to remove
        units_to_remove = []

        for i in range(n_units):
            rate = np.mean(data["response"][i])
            if rate < min_rate:
                # Add index to the list of units to remove
                units_to_remove.append(i)

        # Remove units with low firing rate
        data["response"] = np.delete(data["response"], units_to_remove, axis=0)
        data["dimension"] = np.delete(data["dimension"], units_to_remove)
        data["unit_idx_master"] = np.delete(
            data["unit_idx_master"], units_to_remove
        )
    return data, metadata


def select_time(dataT, tmin=-3, tmax=3, combinebins=False, min_rate=-np.inf):
    time = dataT["time"]
    i_tmin = np.argmin(np.abs(np.array(time) - tmin))
    i_tmax = np.argmin(np.abs(np.array(time) - tmax)) + 1
    tmin_task = -1
    tmax_task = 0
    i_min_task = np.argmin(np.abs(np.array(time) - tmin_task))
    i_max_task = np.argmin(np.abs(np.array(time) - tmax_task)) + 1
    dataT["time"] = time[i_tmin:i_tmax]
    if combinebins:
        dataT["time"] = time[i_tmin].reshape(-1, 1)
    if "unit" in dataT:
        idx_to_remove = []
        for i_unit, unit in enumerate(dataT["unit"]):
            rate = np.mean(unit["response"][i_min_task:i_max_task])
            if rate < min_rate:
                idx_to_remove.append(i_unit)
                continue
            if combinebins:
                unit["response"] = (
                    unit["response"][:, i_tmin:i_tmax]
                    .mean(axis=1)
                    .reshape(-1, 1)
                )
            else:
                unit["response"] = unit["response"][:, i_tmin:i_tmax]
        for i in sorted(idx_to_remove, reverse=True):
            dataT["unit"].pop(i)
    else:
        n_units = dataT["response"].shape[0]
        units_to_remove = []
        for i in range(n_units):
            rate = np.mean(dataT["response"][i])
            if rate < min_rate:
                # Add index to the list of units to remove
                units_to_remove.append(i)
        # Remove units with low firing rate
        dataT["response"] = np.delete(
            dataT["response"], units_to_remove, axis=0
        )
        dataT["dimension"] = np.delete(dataT["dimension"], units_to_remove)
        dataT["unit_idx_master"] = np.delete(
            dataT["unit_idx_master"], units_to_remove
        )
        if combinebins:
            ntr = dataT["response"].shape[2]
            dataT["response"] = (
                dataT["response"][:, i_tmin:i_tmax, :]
                .mean(axis=1)
                .reshape(-1, 1, ntr)
            )
        else:
            dataT["response"] = dataT["response"][:, i_tmin:i_tmax, :]
    return dataT


def remove_rewarded_trials(dataT):
    if "unit" in dataT:
        for unit in dataT["unit"]:
            idx_reward = np.where(unit["task_variable"]["reward"] == 1)[0]
            unit["response"] = np.delete(unit["response"], idx_reward, 0)
            for task_var in unit["task_variable"]:
                unit["task_variable"][task_var] = np.delete(
                    unit["task_variable"][task_var], idx_reward, 0
                )
    else:
        idx_reward = np.where(dataT["task_variable"]["reward"] == 1)[0]
        dataT["response"] = np.delete(dataT["response"], idx_reward, 2)
        for task_var in dataT["task_variable"]:
            dataT["task_variable"][task_var] = np.delete(
                dataT["task_variable"][task_var], idx_reward, 0
            )
    return dataT


def select_congruent_trials(dataT):
    if "unit" in dataT:
        for unit in dataT["unit"]:
            idx_congruent = np.where(
                unit["task_variable"]["choice_a0"]
                == unit["task_variable"]["choice_a1"]
            )[0]
            unit["response"] = unit["response"][idx_congruent, :]
            for task_var in unit["task_variable"]:
                unit["task_variable"][task_var] = unit["task_variable"][
                    task_var
                ][idx_congruent]
    else:
        idx_congruent = np.where(
            dataT["task_variable"]["choice_a0"]
            == dataT["task_variable"]["choice_a1"]
        )[0]
        dataT["response"] = dataT["response"][:, :, idx_congruent]
        for task_var in dataT["task_variable"]:
            dataT["task_variable"][task_var] = dataT["task_variable"][task_var][
                idx_congruent
            ]
    return dataT


def select_unequal_trials(task_variable, lvl=0):
    # get the number of trials for actor and observer
    idx_trials_actor = np.where(
        (task_variable["n_pre_switch_lvl"] == lvl)
        & (task_variable["actor"] == -1)
    )[0]
    idx_trials_observer = np.where(
        (task_variable["n_pre_switch_lvl"] == lvl)
        & (task_variable["actor"] == 1)
    )[0]
    # if they are not equal, delete trials from the one with more trials
    if len(idx_trials_actor) > len(idx_trials_observer):
        return np.random.choice(
            idx_trials_actor,
            len(idx_trials_actor) - len(idx_trials_observer),
            replace=False,
        )
    elif len(idx_trials_observer) > len(idx_trials_actor):
        return np.random.choice(
            idx_trials_observer,
            len(idx_trials_observer) - len(idx_trials_actor),
            replace=False,
        )
    else:
        return []


def tdrEqualizeActorObserverSwitchTrials(dataT):
    # for each unit, equalize the number of trials where preswitch is -1, 0, 1
    # for actor and observer
    if "unit" in dataT:
        units = dataT["unit"]
        for unit in units:
            task_variable = unit["task_variable"]
            idx_trials_to_delete = []
            idx_trials_to_delete.extend(
                select_unequal_trials(task_variable, lvl=-1)
            )
            idx_trials_to_delete.extend(
                select_unequal_trials(task_variable, lvl=0)
            )
            idx_trials_to_delete.extend(
                select_unequal_trials(task_variable, lvl=1)
            )
            if len(idx_trials_to_delete) > 0:
                unit["response"] = np.delete(
                    unit["response"], idx_trials_to_delete, axis=0
                )
                for key in unit["task_variable"]:
                    unit["task_variable"][key] = np.delete(
                        unit["task_variable"][key], idx_trials_to_delete, axis=0
                    )
    return dataT


def select_condition(
    data,
    conditions,
    crossvalidate=None,
    min_trials=10,
    p_test=0.5,
    random_seed=0,
):
    task_index = {key: np.array(value) - 1 for key, value in conditions.items()}
    return tdrSelectByCondition(
        data,
        task_index,
        crossvalidate=crossvalidate,
        min_trials=min_trials,
        p_test=p_test,
        random_seed=random_seed,
    )


def project_activity(data, coefficients):
    npt, ntr = data.shape[1:]
    projected_activity = np.einsum("ij,ijk->jk", coefficients, data)
    return projected_activity.reshape(npt, ntr)


def process_data_and_regression(
    data, conditions, regression_vars, crossvalidate="none", min_trials=10
):
    data_regr = select_condition(
        data, conditions, crossvalidate=crossvalidate, min_trials=min_trials
    )
    if data_regr is None:
        return None, None, None
    if "unit" in data_regr:
        selected_dimensions = [unit["dimension"] for unit in data_regr["unit"]]
    else:
        selected_dimensions = data_regr["dimension"]
    return (
        selected_dimensions,
        perform_regression(data_regr, regression_vars),
        data_regr,
    )


def reverse_vars(data, var_name):
    if "unit" in data:
        for unit_dict in data["unit"]:
            var_values = unit_dict["task_variable"][var_name]
            # reverse var_values by taking the negative of values
            var_values = -var_values
            unit_dict["task_variable"][var_name] = var_values
    else:
        var_values = data["task_variable"][var_name]
        # reverse var_values by taking the negative of values
        var_values = -var_values
        data["task_variable"][var_name] = var_values
    return data


def select_significant_units(data, metadata, event, direction="positive"):
    # load master list csv
    root_dir = findrootdir()
    subject = metadata["unit"]["subject"][0]
    master_list_path = f"{root_dir}/master_list_{subject}.csv"
    master_list = pd.read_csv(master_list_path)
    idx_to_remove = []
    if direction == "any":
        return data
    # select units based on unit_idx in master list and dataT
    if event == "prechoice":
        window = "[-0.6, 0.0]"
        epoch = "choice"
    elif event == "fdbk":
        epoch = "fdbk"
        window = "[0.0, 0.6]"
    field_rew_p_self = f"{epoch}_reward_{window}_p_self"
    field_rew_p_other = f"{epoch}_reward_{window}_p_other"

    for i, unit in enumerate(data["unit"]):
        unit_idx = unit["unit_idx_master"]
        idx = master_list[(master_list["unit_index"] == unit_idx)].index
        if len(idx) == 0:
            print(f"Unit not found in master list")
            continue
        idx = idx[0]
        if (
            master_list.at[idx, field_rew_p_self] < 0.05
            or master_list.at[idx, field_rew_p_other] < 0.05
        ):
            # for positive direction, require selectivity to be positively correlated
            correlation = (
                master_list.at[idx, f"{epoch}_reward_{window}_selectivity_self"]
                * master_list.at[
                    idx, f"{epoch}_reward_{window}_selectivity_other"
                ]
            )
            if direction == "positive":
                if correlation > 0:
                    continue
            elif direction == "negative":
                if correlation < 0:
                    continue
            elif direction == "both":
                continue
        idx_to_remove.append(i)
    for i in sorted(idx_to_remove, reverse=True):
        data["unit"].pop(i)
        for key in metadata["unit"]:
            metadata["unit"][key].pop(i)
    return data, metadata


def select_integration_units_both(data, metadata, event):
    # load master list csv
    root_dir = findrootdir()
    subject = metadata["unit"]["subject"][0]
    master_list_path = f"{root_dir}/master_list_{subject}.csv"
    master_list = pd.read_csv(master_list_path)
    session_stat_file_name = f"{root_dir}/stats_paper/{subject}_{event}_[0.0, 0.6]_switch_dir_stats.json"
    stat_sessions = json.load(open(session_stat_file_name, "r"))
    dates_sig = []
    for stat_session in stat_sessions.values():
        pred_switches = np.array(stat_session["pred_switches"])
        actu_switches = np.array(stat_session["actu_switches"])
        pswitch_0 = actu_switches[pred_switches == 0]
        pswitch_1 = actu_switches[pred_switches == 1]
        # to assess significance of difference, we perform a rank sum test
        _, p = stats.ranksums(pswitch_0, pswitch_1)
        if p < 0.05 and np.mean(pswitch_1) > np.mean(pswitch_0):
            dates_sig.append(stat_session["date"])
    idx_to_remove = []
    # select units based on unit_idx in master list and dataT
    if event == "prechoice":
        window = "[-0.6, 0.0]"
        epoch = "choice"
    elif event == "fdbk":
        epoch = "fdbk"
        window = "[0.0, 0.6]"
    field_rew_p_self = f"{epoch}_reward_{window}_p_self"
    field_rew_p_other = f"{epoch}_reward_{window}_p_other"
    rew_self_sel = f"{event}_reward_{window}_selectivity_self"
    rew_other_sel = f"{event}_reward_{window}_selectivity_other"
    sig_rew_self = np.where((master_list[field_rew_p_self] < 0.05))[0]
    sig_rew_other = np.where((master_list[field_rew_p_other] < 0.05))[0]

    factor = "nback_AOOA"
    nr1_nr2_OA_sel = f"{event}_{factor}_{window}_selectivity_self"
    nr1_nr2_AO_sel = f"{event}_{factor}_{window}_selectivity_other"
    nr1_nr2_OA_p = f"{event}_{factor}_{window}_p_self"
    nr1_nr2_AO_p = f"{event}_{factor}_{window}_p_other"
    factor = "nback_all"
    nr1_nr2_AA_sel = f"{event}_{factor}_{window}_selectivity_self"
    nr1_nr2_AA_p = f"{event}_{factor}_{window}_p_self"
    nr1_nr2_OO_sel = f"{event}_{factor}_{window}_selectivity_other"
    nr1_nr2_OO_p = f"{event}_{factor}_{window}_p_other"
    sig_AA = np.where(
        (
            (master_list[field_rew_p_self] < 0.05)
            | (master_list[field_rew_p_other] < 0.05)
        )
        & (master_list[rew_self_sel] * master_list[rew_other_sel] > 0)
        & (master_list[nr1_nr2_AA_p] < 0.05)
        & (master_list[rew_self_sel] * master_list[nr1_nr2_AA_sel] > 0)
    )[0]
    sig_OA = np.where(
        (
            (master_list[field_rew_p_self] < 0.05)
            | (master_list[field_rew_p_other] < 0.05)
        )
        & (master_list[rew_self_sel] * master_list[rew_other_sel] > 0)
        & (master_list[nr1_nr2_OA_p] < 0.05)
        & (master_list[rew_other_sel] * master_list[nr1_nr2_OA_sel] > 0)
    )[0]
    sig_AO = np.where(
        (
            (master_list[field_rew_p_self] < 0.05)
            | (master_list[field_rew_p_other] < 0.05)
        )
        & (master_list[rew_self_sel] * master_list[rew_other_sel] > 0)
        & (master_list[nr1_nr2_AO_p] < 0.05)
        & (master_list[rew_self_sel] * master_list[nr1_nr2_AO_sel] > 0)
    )[0]
    sig_OO = np.where(
        (
            (master_list[field_rew_p_self] < 0.05)
            | (master_list[field_rew_p_other] < 0.05)
        )
        & (master_list[rew_self_sel] * master_list[rew_other_sel] > 0)
        & (master_list[nr1_nr2_OO_p] < 0.05)
        & (master_list[rew_other_sel] * master_list[nr1_nr2_OO_sel] > 0)
    )[0]
    all_sig = np.union1d(sig_AA, sig_OO)
    all_sig = np.union1d(all_sig, sig_OA)
    all_sig = np.union1d(all_sig, sig_AO)

    for i, unit in enumerate(data["unit"]):
        unit_idx = unit["unit_idx_master"]
        idx = master_list[(master_list["unit_index"] == unit_idx)].index
        if len(idx) == 0:
            print(f"Unit not found in master list")
            continue
        idx = idx[0]
        date = metadata["unit"]["date"][i]  # 2024-08-29: filter by date
        # if date in dates_sig:
        # if idx in all_sig:
        if (date in dates_sig) and (idx in all_sig):
            continue
        idx_to_remove.append(i)
    for i in sorted(idx_to_remove, reverse=True):
        data["unit"].pop(i)
        for key in metadata["unit"]:
            metadata["unit"][key].pop(i)
    return data, metadata


def find_common_dimensions(*dimension_lists):
    common_dims = set(dimension_lists[0])
    for dims in dimension_lists[1:]:
        common_dims.intersection_update(dims)
    return np.array(list(common_dims))


def filter_responses_by_common_dims(data, common_dimensions):
    idx_common_dims = np.where(
        np.isin(data["unit_idx_master"], common_dimensions)
    )[0]
    if "unit" in data:
        data["unit"] = [data["unit"][i] for i in idx_common_dims]
    else:
        if len(data["response"].shape) == 2:
            data["response"] = data["response"][idx_common_dims, :]
        elif len(data["response"].shape) == 3:
            data["response"] = data["response"][idx_common_dims, :, :]
    data["unit_idx_master"] = [
        data["unit_idx_master"][i] for i in idx_common_dims
    ]
    return data


def filter_by_common_dimensions(*data):
    common_elements = find_common_dimensions(
        *[d["unit_idx_master"] for d in data]
    )
    for d in data:
        d = filter_responses_by_common_dims(d, common_elements)


def filter_simultaneous_units(dataT, metadata, date=None):
    # filter data to include simultaneously recorded units.
    # output: dataT with response matrix of shape (n_units, n_timepoints, n_trials)
    dataT_simultaneous = {
        "time": dataT["time"],
        "dimension": [],
        "unit_idx_master": [],
    }
    units = []
    if date is None:
        date = metadata["unit"]["date"][0]
    metadata_simultaneous = {"date": date, "subject": []}
    # remove units that are not recorded on the same date
    for i_unit, unit in enumerate(dataT["unit"]):
        if metadata["unit"]["date"][i_unit] != date:
            continue
        units.append(unit)
        metadata_simultaneous["subject"] = metadata["unit"]["subject"][i_unit]
    n_units = len(units)
    unit_trials_valid = np.zeros([n_units, 2000 * 15])
    for i_unit, unit_dict in enumerate(units):
        valid_trials = unit_dict["trial_ids"] - unit_dict["session_id"] * 20000
        unit_trials_valid[i_unit, valid_trials] = 1
    # find common trial indices
    # Get cluster ids from the keys of spike_times_per_trial
    # FIND LARGE SUBSET OF VALID UNIT TRIALS
    ind_units, trial_nums = find_subset_with_all_ones(unit_trials_valid)
    if ind_units is None:
        return None
    n_units_keep = len(ind_units)
    response_matrix = np.zeros(
        [n_units_keep, len(dataT["time"]), len(trial_nums)]
    )
    task_var_names = list(dataT["unit"][0]["task_variable"].keys())
    task_variable = {name: [] for name in task_var_names}
    for i_unit, idx_unit in enumerate(ind_units):
        # get response from this unit
        unit_dict = units[idx_unit]
        idx_trials = np.intersect1d(
            unit_dict["trial_ids"] - unit_dict["session_id"] * 20000,
            trial_nums,
            return_indices=True,
        )[1]
        response = unit_dict["response"][idx_trials, :]
        # add spike counts to dataT_simultaneous
        response_matrix[i_unit, :, :] = response.T
        dataT_simultaneous["dimension"].append(unit_dict["dimension"])
        dataT_simultaneous["unit_idx_master"].append(
            unit_dict["unit_idx_master"]
        )
    # add task variables to dataT_simultaneous using the last unit
    task_variable_unit = unit_dict["task_variable"]
    for name in task_var_names:
        task_variable[name] = task_variable_unit[name][idx_trials]
    dataT_simultaneous["response"] = response_matrix
    dataT_simultaneous["task_variable"] = task_variable
    dataT_simultaneous["trial_ids"] = np.array(trial_nums)

    return dataT_simultaneous, metadata_simultaneous
