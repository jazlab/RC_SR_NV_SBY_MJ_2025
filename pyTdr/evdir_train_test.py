"""
This script computes the cross-validation results for the switch evidence dim
using leave-one-out cross validation and an additional null model where the
n_pre_switch_lvl is replaced with consecutive sequences of 3 trials unrelated
to switches.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import os
import json
from pyTdr.tdrLoadAccData import tdrLoadAccData
from pyTdr.tdrMeanAndStd import tdrMeanAndStd
from pyTdr.tdrNormalize import tdrNormalize
from pyTdr.tdrSelectByCondition import tdrSelectByCondition
from pyTdr.tdrRegression import tdrRegression
from pyTdr.tdrAverageCondition import tdrAverageCondition
from pyTdr.tdrVectorTimeAverage import tdrVectorTimeAverage
from pyTdr.tdrVectorOrthogonalize import tdrVectorOrthogonalize
from pyTdr.tdrUtils import (
    project_activity,
    select_time,
    filter_by_common_dimensions,
    remove_low_rate,
    prepare_test_data,
)
from utils.LoadSession import findrootdir
from copy import deepcopy
import multiprocessing


def select_condition(
    data, conditions, crossvalidate=None, min_trials=10, random_seed=0
):
    task_index = {key: np.array(value) - 1 for key, value in conditions.items()}
    return tdrSelectByCondition(
        data,
        task_index,
        crossvalidate=crossvalidate,
        min_trials=min_trials,
        random_seed=random_seed,
    )


def perform_regression(data, regressors, normalization="max_abs"):
    regpars = {
        "regressor": regressors,
        "regressor_normalization": normalization,
    }
    return tdrRegression(data, regpars, 0)


def average_conditions(data, conditions):
    task_index = {key: np.array(value) - 1 for key, value in conditions.items()}
    return tdrAverageCondition(data, task_index)


def fakePreSwitch(data):
    data_fake = deepcopy(data)
    if "unit" in data_fake:
        for unit in data_fake["unit"]:
            taskvar = unit["task_variable"]
            n_trials = len(taskvar["n_pre_switch_lvl"])
            # create array of [-1, 0, 1] repeated
            n_pre_switch_lvl_fake = np.tile([-1, 0, 1], n_trials // 3)
            # edge cases for when n_trials is not a multiple of 3
            n_pre_switch_lvl_fake = np.append(
                n_pre_switch_lvl_fake, [-1, 0, 1][: n_trials % 3]
            )
            taskvar["n_pre_switch_lvl"] = n_pre_switch_lvl_fake
    return data_fake


def perform_shuffles_helper(args):
    i_shuff, dataT_nrml = args
    # Copy with shifted n_pre_switch_lvl for null of conseq. seq. of 3 trials
    dataT_nrml_random = fakePreSwitch(
        dataT_nrml
    )  # randomize n_pre_switch_lvl only
    # Select conditions for regression
    conditions = {
        "n_pre_switch_lvl": [1, 2, 3, 1, 2, 3],
        "choice": [1, 1, 1, 2, 2, 2],
    }
    dataT_nrml = select_condition(
        dataT_nrml,
        conditions,
        min_trials=5,
    )
    # Select conditions for cross-validation
    conditions = {"n_pre_switch_lvl": [1, 2, 3]}
    dataT_nrml_regr = select_condition(
        dataT_nrml,
        conditions,
        crossvalidate="leave_one_out",
        min_trials=5,
        random_seed=i_shuff,
    )

    dataC, _ = average_conditions(dataT_nrml_regr, conditions)
    dataT_test = prepare_test_data(dataT_nrml_regr)
    dataC_test, _ = average_conditions(dataT_test, conditions)

    dataT_nrml_random_select = select_condition(
        dataT_nrml_random,
        conditions,
        crossvalidate="leave_one_out",
        min_trials=1,
        random_seed=i_shuff,
    )
    dataT_random_test = prepare_test_data(dataT_nrml_random_select)
    dataC_test_random, _ = average_conditions(dataT_random_test, conditions)

    # find the intersec of dimensions between dataC_act, dataC_obs and
    # regression coefficients
    filter_by_common_dimensions(
        dataT_nrml_regr,
        dataC,
        dataC_test,
        dataC_test_random,
    )

    # Perform linear regression on switch
    coef_switch = perform_regression(
        dataT_nrml_regr, ["b0", "n_pre_switch_lvl", "choice"]
    )

    # Define regression vectors
    # Regression vector parameters
    vecpars = {}
    vecpars["n_pre_switch_lvl"] = {}
    vecpars["n_pre_switch_lvl"]["time_win"] = [-2, 2]  # select all
    vecpars["choice"] = {}
    vecpars["choice"]["time_win"] = [-2, 2]  # select all
    plotflag = False

    vBeta_switch, _ = tdrVectorTimeAverage(coef_switch, vecpars, plotflag)

    # Define task-related axes (orthogonalize regression vectors)
    # Regression axes parameters
    ortpars = {}
    ortpars["name"] = ["n_pre_switch_lvl", "choice"]

    # Compute regression axes
    vAxes_switch, lowUN_lowTA = tdrVectorOrthogonalize(vBeta_switch, ortpars)

    # project train data
    proj_act_shuff_train = project_activity(
        dataC["response"], vAxes_switch["response"][:, :, 0]
    ).mean(axis=0)

    # project test data
    proj_act_shuff_test = project_activity(
        dataC_test["response"], vAxes_switch["response"][:, :, 0]
    ).mean(axis=0)

    # project random test data (random seq of 3 trials)
    proj_act_shuff_random_test = project_activity(
        dataC_test_random["response"], vAxes_switch["response"][:, :, 0]
    ).mean(axis=0)

    return {
        "proj_act_shuff_train": proj_act_shuff_train,
        "proj_act_shuff_test": proj_act_shuff_test,
        "proj_act_shuff_random_test": proj_act_shuff_random_test,
    }


def perform_shuffles(n_shuffles, dataT_nrml):
    def perform_shuffles_parallel(n_shuffles, dataT_nrml):
        # run one shuffle in seriel for debugging
        # results = perform_shuffles_helper((0, dataT_nrml))
        proj_act_shuff_train = np.zeros([n_shuffles, 3])
        proj_act_shuff_test = np.zeros([n_shuffles, 3])
        proj_act_shuff_random_test = np.zeros([n_shuffles, 3])
        run_serially = True
        if run_serially:
            results = []
            for i in range(n_shuffles):
                dataT_nrml_ = deepcopy(dataT_nrml)
                result = perform_shuffles_helper((i, dataT_nrml_))
                results.append(result)
        else:
            n_cores = min(multiprocessing.cpu_count(), 20)
            print(f"Using {n_cores} cores")
            with multiprocessing.Pool(processes=n_cores) as pool:
                results = pool.map(
                    perform_shuffles_helper,
                    [(i, dataT_nrml) for i in range(n_shuffles)],
                )
        for i, result in enumerate(results):
            proj_act_shuff_train[i, :] = result["proj_act_shuff_train"]
            proj_act_shuff_test[i, :] = result["proj_act_shuff_test"]
            proj_act_shuff_random_test[i, :] = result[
                "proj_act_shuff_random_test"
            ]

        return {
            "proj_act_shuff_train": proj_act_shuff_train,
            "proj_act_shuff_test": proj_act_shuff_test,
            "proj_act_shuff_random_test": proj_act_shuff_random_test,
        }

    results = perform_shuffles_parallel(n_shuffles, dataT_nrml)
    return results


def main(args):
    subject, event = args
    datadir = findrootdir()
    time_window = [-3, 3]
    load_event = event
    dataT, metadata = tdrLoadAccData(datadir, subject, load_event, time_window)
    # select time for analysis
    tmin = 0
    tmax = 0.6
    dataT = select_time(dataT, tmin=tmin, tmax=tmax)
    # filter data to exclude units that have low firing rate in this epoch
    dataT, metadata = remove_low_rate(dataT, metadata, min_rate=0.5)

    # Average data: get mean and std first
    avg_params = {"trial": [], "time": []}
    meanT, stdT = tdrMeanAndStd(dataT, avg_params)

    # Normalization parameters
    norm_params = {"ravg": meanT, "rstd": stdT}
    dataT_nrml = tdrNormalize(dataT, norm_params)

    n_shuffles = 100
    dataT_nrml["animal"] = subject
    dataT_nrml["event"] = event
    # results here are from all units across sessions
    results = perform_shuffles(n_shuffles, dataT_nrml)

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
    if not os.path.exists(f"{datadir}/stats_paper"):
        os.makedirs(f"{datadir}/stats_paper")
    # save result to json file
    with open(
        f"{datadir}/stats_paper/{subject}_{event}_swe_cross_validation.json",
        "w",
    ) as f:
        json.dump(results, f)


if __name__ == "__main__":
    for animal in ["O", "L"]:
        for event in ["fdbk"]:
            main((animal, event))
