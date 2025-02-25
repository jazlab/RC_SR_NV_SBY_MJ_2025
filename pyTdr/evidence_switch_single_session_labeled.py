"""
This script computes the predicted switch based on neural activity for each
session.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import os
import json
from copy import deepcopy
from pyTdr.tdrLoadAccData import tdrLoadAccData
from pyTdr.tdrAverageCondition import average_conditions
from pyTdr.tdrNormalize import smooth_and_normalize
from pyTdr.tdrRegression import perform_regression
from pyTdr.tdrVectorTimeAverage import tdrVectorTimeAverage
from pyTdr.tdrVectorOrthogonalize import tdrVectorOrthogonalize
from pyTdr.tdrModifyTaskVar import use_prev_switch
from pyTdr.tdrUtils import (
    findrootdir,
    select_condition,
    filter_simultaneous_units,
    select_time,
    remove_rewarded_trials,
    remove_low_rate,
)
from scipy import stats


def remove_first_trial(data):
    if data["task_variable"]["trial_num"][0] == 1:
        data["response"] = data["response"][:, :, 1:]
        data["task_variable"] = {
            key: value[1:] for key, value in data["task_variable"].items()
        }
        data["trial_ids"] = data["trial_ids"][1:]
    return data


def average_normalize_data(data, conditions):
    averaged_data, _ = average_conditions(data, conditions)
    normalized_data = smooth_and_normalize(averaged_data)
    return normalized_data


def compute_pswitch_quantiles(
    projected_activity, bswitch, n_quantiles, train_data=None
):
    ntrials = projected_activity.shape[0]
    pswitch_quantiles = np.zeros([n_quantiles])
    idx_sort = np.argsort(projected_activity)
    quantile_size = ntrials // n_quantiles
    if quantile_size == 0:
        print("Insufficient trials for quantile analysis")
        return pswitch_quantiles
    if train_data is not None:
        # use train data to determine threshold for each quantile
        # this is a more strict test as it uses the same threshold for
        # both train and test
        train_data = np.sort(train_data)
        quantile_size_train = train_data.shape[0] // n_quantiles
        # get threshold for each quantile from sorted training data
        threshold_quantiles = np.zeros([n_quantiles + 1])
        for i in range(n_quantiles - 1):
            threshold_quantiles[i + 1] = train_data[
                quantile_size_train * (i + 1)
            ]
        threshold_quantiles[0] = -np.inf
        threshold_quantiles[-1] = np.inf
        # compute pswitch for each quantile using threshold from training data
        for i in range(n_quantiles):
            idx_quantile = np.where(
                (projected_activity >= threshold_quantiles[i])
                & (projected_activity < threshold_quantiles[i + 1])
            )[0]
            if len(idx_quantile) == 0:
                pswitch_quantiles[i] = 0
            else:
                pswitch_quantiles[i] = sum(bswitch[idx_quantile]) / len(
                    idx_quantile
                )
    else:
        for i in range(n_quantiles):
            idx_quantile = idx_sort[i * quantile_size : (i + 1) * quantile_size]
            pswitch_quantiles[i] = sum(bswitch[idx_quantile]) / quantile_size
    return pswitch_quantiles


def remove_trial(data, trial_ids):
    idx_remove = np.isin(data["trial_ids"], trial_ids)
    data["response"] = data["response"][:, :, ~idx_remove]
    data["task_variable"] = {
        key: value[~idx_remove] for key, value in data["task_variable"].items()
    }
    data["trial_ids"] = data["trial_ids"][~idx_remove]
    return data


def perform_regression_and_compute_projection(
    dataT,
    date,
    min_trials,
    random_seed=0,
):
    # select random trials for cross-validation
    # first select units with enough trials
    conditions = {
        "n_pre_switch_lvl": [1, 2, 3],
    }
    dataT = select_condition(
        dataT, conditions, crossvalidate="none", min_trials=10
    )
    # then take one trial out for testing
    dataT_test = deepcopy(dataT)
    conditions = {
        "reward": [0],
    }
    dataT_test = select_condition(
        dataT_test,
        conditions,
        crossvalidate="leave_one_out",
        min_trials=min_trials,
        random_seed=random_seed,
    )
    dataT_test["response"] = dataT_test["response_test"]
    dataT_test["task_variable"] = dataT_test["task_variable_test"]
    dataT_test["trial_ids"] = dataT_test["trial_ids_test"]
    dataT_test = select_condition(
        dataT_test, conditions, crossvalidate="none", min_trials=1
    )
    # remove test trial from training data
    dataT = remove_trial(dataT, dataT_test["trial_ids"])
    if dataT is None:
        print(f"Insufficient trials for {animal} {date}")
        return None
    # Linear regression
    ### Switch ###
    # Regression parameters
    regressors = ["b0", "n_pre_switch_lvl", "choice"]
    # Linear regression
    coef_fulUN_switch = perform_regression(dataT, regressors)
    # orthogonalize regressors
    vecpars = {}
    vecpars["n_pre_switch_lvl"] = {}
    vecpars["n_pre_switch_lvl"]["time_win"] = [-2, 2]  # select all
    vecpars["choice"] = {}
    vecpars["choice"]["time_win"] = [-2, 2]  # select all
    plotflag = False

    # Compute regression axes
    vBeta_switch, _ = tdrVectorTimeAverage(coef_fulUN_switch, vecpars, plotflag)

    # Define task-related axes (orthogonalize regression vectors)
    # Regression axes parameters
    ortpars = {}
    ortpars["name"] = ["n_pre_switch_lvl", "choice"]

    # Orthogonalize regression vectors
    vAxes_switch, lowUN_lowTA = tdrVectorOrthogonalize(vBeta_switch, ortpars)

    # Define regression vectors using orthogonalized axes
    reg_vec = vAxes_switch["response"][:, :, 0]

    # # behavioral readout: switch or not
    # compute p_switch by quantiles
    dataT = remove_rewarded_trials(dataT)
    switch = dataT_test["task_variable"]["n_pre_switch_lvl"]
    bswitch = switch == 1
    activity_matrix = dataT_test["response"]
    activity_train = dataT["response"]
    projection = np.einsum("ij, ijk -> k", reg_vec, activity_matrix)
    projection_train = np.einsum("ij, ijk -> k", reg_vec, activity_train)
    projection_all = np.concatenate([projection_train, projection])
    threshold = np.median(projection_all)
    if projection > threshold:
        predicted_switch = 1
    else:
        predicted_switch = 0

    return {
        "predicted_switch": predicted_switch,
        "actual_switch": bswitch,
        "projection": projection,
        "actor": dataT_test["task_variable"]["actor"],
        "history": dataT_test["task_variable"]["history"],
    }


def analysis_session(
    dataT, metadata, animal, tmin, tmax, event="fdbk", date="20230216"
):
    # filter data to include simultaneously recorded units.
    # select time window for analysis
    min_trials = 10
    min_rate = 0.5
    if dataT is None:
        print(f"No data for {animal} {date}")
        return
    # do not use the dataT from input because it will be used for other windows
    dataT_ = deepcopy(dataT)
    dataT_ = select_time(dataT_, tmin=tmin, tmax=tmax, combinebins=True)
    if event in ["prechoice"]:
        dataT_ = remove_first_trial(dataT_)
        dataT_ = use_prev_switch(dataT_, metadata)
    n_units = dataT_["response"].shape[0]
    dataT_, _ = remove_low_rate(dataT_, None, min_rate)
    n_units_after = dataT_["response"].shape[0]
    print(
        f"Removed {n_units - n_units_after}/{n_units} units with low firing rate"
    )
    if n_units_after < 2:
        print(f"Insufficient units for {animal} {date}")
        return
    dataT_ = smooth_and_normalize(dataT_)

    # dataT_select is only used to check if dataT_ has enough trials
    conditions = {
        "n_pre_switch_lvl": [1, 2, 3] * 2,
        "choice": [1, 1, 1, 2, 2, 2],
    }
    dataT_select = select_condition(
        dataT_, conditions, crossvalidate="random", min_trials=min_trials
    )
    if dataT_select is None:
        print(f"Insufficient trials for {animal} {date}")
        return

    # perform n shuffles
    n_shuffles = 1000
    pred_switches = np.zeros([n_shuffles])
    actu_switches = np.zeros([n_shuffles])
    projections = np.zeros([n_shuffles])
    actors = np.zeros([n_shuffles])
    histories = np.zeros([n_shuffles])

    for i in range(n_shuffles):
        dataT__ = deepcopy(dataT_)
        results = perform_regression_and_compute_projection(
            dataT__,
            date,
            min_trials,
            random_seed=i,
        )
        pred_switches[i] = results["predicted_switch"]
        actu_switches[i] = results["actual_switch"]
        projections[i] = results["projection"]
        actors[i] = results["actor"]
        histories[i] = results["history"]
    pswitch_0 = actu_switches[pred_switches == 0]
    pswitch_1 = actu_switches[pred_switches == 1]
    _, p = stats.ranksums(pswitch_0, pswitch_1)
    return {
        "animal": animal,
        "date": date,
        "event": event,
        "n_units": n_units,
        "n_trials": dataT_["response"].shape[2],
        "unit_idx_master": dataT_["unit_idx_master"],
        "pswitch_lo": np.mean(actu_switches[pred_switches == 0]),
        "pswitch_hi": np.mean(actu_switches[pred_switches == 1]),
        "pval_ranksum": p,
        "pred_switches": pred_switches,
        "actu_switches": actu_switches,
        "projection": projections,
        "actor": actors,
        "history": histories,
    }


def main(animal, event):
    time_window = [-3, 3]
    datadir = findrootdir()
    savedir = f"{datadir}/stats_paper"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    load_event = event
    if event == "prechoice":
        load_event = "choice"
    dataT, metadata = tdrLoadAccData(datadir, animal, load_event, time_window)
    print(f"loaded data for {animal} {event}")
    dates = np.unique(metadata["unit"]["date"])
    stat_sessions = {}
    for date in dates:
        dataT_session, metadata_session = filter_simultaneous_units(
            dataT, metadata, date
        )
        if event == "fdbk":
            tstart = 0.0
            tend = 0.6
        elif event == "prechoice":
            tstart = -0.6
            tend = 0.0
        stats_session = analysis_session(
            dataT_session,
            metadata_session,
            animal,
            tstart,
            tend,
            event,
            date,
        )
        if stats_session is not None:
            for key, value in stats_session.items():
                if isinstance(value, np.ndarray):
                    stats_session[key] = value.tolist()
            formatted_window = [round(tstart, 1), round(tend, 1)]
            formatted_window = str(formatted_window)
            if formatted_window not in stat_sessions:
                stat_sessions[formatted_window] = {}
            stat_sessions[formatted_window][date] = stats_session
    for window, stats in stat_sessions.items():
        filename = f"{savedir}/{animal}_{event}_{window}_switch_dir_stats.json"
        with open(filename, "w") as f:
            json.dump(stats, f)
        print(f"Saved {filename}")


if __name__ == "__main__":
    # set numpy random seed
    np.random.seed(100)
    for event in ["fdbk", "prechoice"]:
        for animal in ["O", "L"]:
            main(animal, event)
