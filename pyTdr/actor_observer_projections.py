"""
This script computes the projections of neural activity onto the switch evidence
dimension for actor and observer trials.

"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import json
from pyTdr.tdrLoadAccData import tdrLoadAccData
from pyTdr.tdrMeanAndStd import tdrMeanAndStd
from pyTdr.tdrNormalize import tdrNormalize
from pyTdr.tdrSelectByCondition import tdrSelectByCondition
from pyTdr.tdrRegression import tdrRegression
from pyTdr.tdrAverageCondition import tdrAverageCondition
from pyTdr.tdrTemporalSmoothing import tdrTemporalSmoothing
from pyTdr.tdrVectorTimeAverage import tdrVectorTimeAverage
from pyTdr.tdrVectorOrthogonalize import tdrVectorOrthogonalize
from pyTdr.tdrUtils import (
    project_activity,
    select_time,
    filter_by_common_dimensions,
    select_integration_units_both,
    select_congruent_trials,
    remove_low_rate,
    prepare_test_data,
    tdrEqualizeActorObserverSwitchTrials,
    equalize_actor_observer_unit,
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


def smooth_and_normalize(data, filter_type="gauss", width=0.04):
    smthpars = {"filter": filter_type, "width": width}
    data_smth = tdrTemporalSmoothing(data, smthpars)
    avgpars = {"trial": [], "time": []}
    mean, std = tdrMeanAndStd(data_smth, avgpars)
    nrmlpars = {"ravg": mean, "rstd": std}
    return tdrNormalize(data_smth, nrmlpars)


def average_conditions(data, conditions):
    task_index = {key: np.array(value) - 1 for key, value in conditions.items()}
    return tdrAverageCondition(data, task_index)


def perform_shuffles_helper(args):
    i_shuff, dataT_nrml = args
    # Select conditions for regression
    conditions = {
        "n_pre_switch_lvl": [1, 2, 3, 1, 2, 3],
        "choice": [1, 1, 1, 2, 2, 2],
    }
    dataT_nrml_regr = select_condition(
        dataT_nrml,
        conditions,
        crossvalidate="none",
        min_trials=10,
        random_seed=i_shuff,
    )
    conditions = {
        "history": [1, 2, 3, 1, 2, 3],
        "actor": [1, 1, 1, 2, 2, 2],
    }
    dataT_nrml_regr = select_condition(
        dataT_nrml_regr,
        conditions,
        crossvalidate="leave_one_out",
        min_trials=1,
        random_seed=i_shuff,
    )
    dataT_test = prepare_test_data(dataT_nrml_regr)

    conditions = {
        "history": [1, 2, 3],
        "actor": [1, 1, 1],
    }
    dataT_act = select_condition(
        dataT_test, conditions, crossvalidate="none", min_trials=1
    )
    dataC_act, _ = average_conditions(dataT_act, conditions)
    conditions = {
        "history": [1, 2, 3],
        "actor": [2, 2, 2],
    }
    dataT_obs = select_condition(
        dataT_test, conditions, crossvalidate="none", min_trials=1
    )
    conditions = {
        "history": [1, 2, 3],
    }
    dataC_obs, _ = average_conditions(dataT_obs, conditions)
    # find the intersec of dimensions between dataC_act, dataC_obs and
    # regression coefficients
    filter_by_common_dimensions(
        dataT_nrml_regr,
        dataC_act,
        dataC_obs,
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

    # Project activity
    # first take mean over axis 1 of responses
    dataC_act["response"] = np.expand_dims(
        np.mean(dataC_act["response"], axis=1), axis=1
    )
    dataC_obs["response"] = np.expand_dims(
        np.mean(dataC_obs["response"], axis=1), axis=1
    )

    proj_act_shuff_actor = project_activity(
        dataC_act["response"], vAxes_switch["response"][:, :, 0]
    )
    proj_act_shuff_observer = project_activity(
        dataC_obs["response"], vAxes_switch["response"][:, :, 0]
    )

    unit_idx_master = [
        unit["unit_idx_master"] for unit in dataT_nrml_regr["unit"]
    ]

    return {
        "proj_act_shuff_actor": proj_act_shuff_actor,
        "proj_act_shuff_observer": proj_act_shuff_observer,
        "dimensions": coef_switch["dimension"],
        "unit_idx_master": unit_idx_master,
    }


def perform_shuffles(n_shuffles, dataT_nrml):
    def perform_shuffles_parallel(n_shuffles, dataT_nrml):
        # run one shuffle in seriel for debugging
        results = perform_shuffles_helper((0, dataT_nrml))
        npt = len(dataT_nrml["time"])
        proj_act_shuff_actor = np.zeros([n_shuffles, npt, 3])
        proj_act_shuff_observer = np.zeros([n_shuffles, npt, 3])
        dimensions = []
        unit_idx_master = []
        run_serially = False
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
            proj_act_shuff_actor[i, :, :] = result["proj_act_shuff_actor"]
            proj_act_shuff_observer[i, :, :] = result["proj_act_shuff_observer"]
            dimensions.append(result["dimensions"])
            unit_idx_master.append(result["unit_idx_master"])
        unit_idx_master = np.array(unit_idx_master[0])

        return {
            "proj_act_shuff_actor": proj_act_shuff_actor.reshape(
                n_shuffles * npt, 3
            ),
            "proj_act_shuff_observer": proj_act_shuff_observer.reshape(
                n_shuffles * npt, 3
            ),
            "unit_idx_master": unit_idx_master,
        }

    results = perform_shuffles_parallel(n_shuffles, dataT_nrml)
    return results


def main(args):
    subject, event = args
    datadir = findrootdir()
    time_window = [-3, 3]
    dataT, metadata = tdrLoadAccData(datadir, subject, event, time_window)
    # select time for analysis
    tmin = 0
    tmax = 0.6
    dataT = select_congruent_trials(dataT)
    dataT = select_time(dataT, tmin=tmin, tmax=tmax, combinebins=True)
    # filter data to exclude units that have low firing rate in this epoch
    dataT, metadata = remove_low_rate(dataT, metadata, min_rate=0.5)
    # select units with significant selectivity using master list
    dataT, metadata = select_integration_units_both(dataT, metadata, "fdbk")

    # Average data: get mean and std first
    avg_params = {"trial": [], "time": []}
    meanT, stdT = tdrMeanAndStd(dataT, avg_params)

    # Normalization parameters
    norm_params = {"ravg": meanT, "rstd": stdT}
    dataT_nrml = tdrNormalize(dataT, norm_params)

    control_n_trials = True
    # Optional control: equalize the number of Act/Obs switch trials
    if control_n_trials:
        dataT_nrml = select_congruent_trials(dataT_nrml)
        dataT_nrml = tdrEqualizeActorObserverSwitchTrials(dataT_nrml)
    # Optional control: equalize the number of Act/Obs selective neurons
    control_n_neurons = True
    if control_n_neurons:
        dataT_nrml, _ = equalize_actor_observer_unit(dataT_nrml, metadata)

    n_shuffles = 100
    dataT_nrml["animal"] = subject
    dataT_nrml["event"] = event
    # results here are from all units across sessions
    results = perform_shuffles(n_shuffles, dataT_nrml)

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
    # save result to json file
    result_name = f"{subject}_{event}_act_obs_projections"
    if control_n_trials:
        result_name += "_equalNSwitch"
    if control_n_neurons:
        result_name += "_equalNNeurons"
    with open(
        f"{datadir}/stats_paper/{result_name}.json",
        "w",
    ) as f:
        json.dump(results, f)


if __name__ == "__main__":
    np.random.seed(100)
    for animal in ["O", "L"]:
        for event in ["fdbk"]:
            main((animal, event))
