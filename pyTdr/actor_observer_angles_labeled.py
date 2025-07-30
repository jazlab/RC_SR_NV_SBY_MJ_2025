"""
This script computes three dimensions from neural activity:
switch evidence, actor outcome, and observer outcome, using TDR
Based on these dimensions angles between these are computed.
Each calculation is repeated resulting in a distribution of angles.
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
from pyTdr.tdrTemporalSmoothing import tdrTemporalSmoothing
from pyTdr.tdrVectorTimeAverage import tdrVectorTimeAverage
from pyTdr.tdrVectorOrthogonalize import tdrVectorOrthogonalize
from pyTdr.tdrUtils import (
    project_activity,
    select_time,
    find_common_dimensions,
    filter_by_common_dimensions,
    select_significant_units,
    select_congruent_trials,
    tdrEqualizeActorObserverSwitchTrials,
    equalize_actor_observer_unit,
    remove_low_rate,
)
from compute_angle import compute_angle
from utils.LoadSession import findrootdir
from utils.get_session_info import load_subject_names
from pyTdr.tdrModifyTaskVar import use_prev_switch
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
    # The rest of your function goes here
    # Select conditions for regression
    conditions = {
        "n_pre_switch_lvl": [1, 2, 3, 1, 2, 3],
        "choice": [1, 1, 1, 2, 2, 2],
    }
    dataT_nrml_regr = select_condition(
        dataT_nrml,
        conditions,
        crossvalidate="random",
        min_trials=10,
        random_seed=i_shuff,
    )

    if not dataT_nrml_regr["unit"]:
        return None

    conditions = {
        "reward": [1, 1, 2, 2],
        "actor": [1, 1, 1, 1],
        "choice": [1, 2, 1, 2],
    }
    dataT_act = select_condition(
        dataT_nrml_regr, conditions, crossvalidate="none", min_trials=5
    )
    conditions = {
        "history": [1, 2, 3],
        "actor": [1, 1, 1],
    }
    dataC_act, _ = average_conditions(dataT_act, conditions)
    conditions = {
        "reward": [1, 1, 2, 2],
        "actor": [2, 2, 2, 2],
        "choice": [1, 2, 1, 2],
    }
    dataT_obs = select_condition(
        dataT_nrml_regr, conditions, crossvalidate="none", min_trials=5
    )
    conditions = {
        "history": [1, 2, 3],
    }
    dataC_obs, _ = average_conditions(dataT_obs, conditions)
    # find the intersec of dimensions between dataC_act, dataC_obs and
    # regression coefficients
    filter_by_common_dimensions(
        dataT_nrml_regr,
        dataT_act,
        dataT_obs,
        dataC_act,
        dataC_obs,
    )

    if len(dataT_nrml_regr["unit"]) < 2:
        print(
            f"Not enough units for regression in session {dataT_nrml_regr['unit'][0]['session_id']}"
        )
        return None

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

    # Compute conditional averages for actor and observer in test set
    conditions = {
        "history": [1, 2, 3, 1, 2, 3],
        "actor": [1, 1, 1, 2, 2, 2],
    }
    dataC, _ = average_conditions(dataT_nrml, conditions)

    # perform linear regression on actor history
    coef_actor = perform_regression(
        dataT_act, ["b0", "reward", "choice"], normalization="max_abs"
    )
    vecpars = {}
    vecpars["reward"] = {}
    vecpars["reward"]["time_win"] = [-2, 2]  # select all
    vecpars["choice"] = {}
    vecpars["choice"]["time_win"] = [-2, 2]  # select all
    vBeta_actor, _ = tdrVectorTimeAverage(coef_actor, vecpars, plotflag)
    ortpars["name"] = ["reward", "choice"]
    vAxes_actor, lowUN_lowTA = tdrVectorOrthogonalize(vBeta_actor, ortpars)

    # perform linear regression on observer history
    coef_observer = perform_regression(
        dataT_obs, ["b0", "reward", "choice"], normalization="max_abs"
    )
    vBeta_observer, _ = tdrVectorTimeAverage(coef_observer, vecpars, plotflag)
    vAxes_observer, lowUN_lowTA = tdrVectorOrthogonalize(
        vBeta_observer, ortpars
    )

    # Smooth and normalize data for actor and observer
    dataC_nrml = smooth_and_normalize(dataC)
    dataC_nrml_act = smooth_and_normalize(dataC_act)
    dataC_nrml_obs = smooth_and_normalize(dataC_obs)

    # flip the sign of coefficients for reward
    coef_actor["response"][:, :, 1] = -coef_actor["response"][:, :, 1]
    coef_observer["response"][:, :, 1] = -coef_observer["response"][:, :, 1]
    vAxes_actor["response"][:, :, 0] = -vAxes_actor["response"][:, :, 0]
    vAxes_observer["response"][:, :, 0] = -vAxes_observer["response"][:, :, 0]

    # NEW: compute angle using the average coefficient over time
    theta_switch_actor_avg = compute_angle(
        vAxes_switch["response"][:, :, 0].mean(axis=1),
        vAxes_actor["response"][:, :, 0].mean(axis=1),
    )
    theta_switch_observer_avg = compute_angle(
        vAxes_switch["response"][:, :, 0].mean(axis=1),
        vAxes_observer["response"][:, :, 0].mean(axis=1),
    )
    theta_actor_observer_avg = compute_angle(
        vAxes_actor["response"][:, :, 0].mean(axis=1),
        vAxes_observer["response"][:, :, 0].mean(axis=1),
    )

    # Project activity
    # first take mean over axis 1 of responses
    dataC_nrml_act["response"] = np.expand_dims(
        np.mean(dataC_nrml_act["response"], axis=1), axis=1
    )
    dataC_nrml_obs["response"] = np.expand_dims(
        np.mean(dataC_nrml_obs["response"], axis=1), axis=1
    )

    proj_act_shuff_actor = project_activity(
        dataC_nrml_act["response"], vAxes_switch["response"][:, :, 0]
    )
    proj_act_shuff_observer = project_activity(
        dataC_nrml_obs["response"], vAxes_switch["response"][:, :, 0]
    )

    unit_idx_master = [
        unit["unit_idx_master"] for unit in dataT_nrml_regr["unit"]
    ]

    return {
        "proj_act_shuff_actor": proj_act_shuff_actor,
        "proj_act_shuff_observer": proj_act_shuff_observer,
        "theta_switch_actor": theta_switch_actor_avg,
        "theta_switch_observer": theta_switch_observer_avg,
        "theta_actor_observer": theta_actor_observer_avg,
        "coef_switch": vAxes_switch["response"][:, :, 0].mean(axis=1),
        "coef_actor": vAxes_actor["response"][:, :, 0].mean(axis=1),
        "coef_observer": vAxes_observer["response"][:, :, 0].mean(axis=1),
        "dimensions": coef_switch["dimension"],
        "unit_idx_master": unit_idx_master,
    }


def tdrSelectByID(data, session_id):
    print(f"Selecting data for session {session_id}")
    data_session = {
        "unit": [
            unit for unit in data["unit"] if unit["session_id"] == session_id
        ],
        "time": data["time"],
    }
    print(
        f"Number of units in session {session_id}: {len(data_session['unit'])}"
    )

    return data_session


def tdrSelectByTrial(data, trial_id_start, trial_id_end):
    print(f"Selecting data for trials {trial_id_start} to {trial_id_end}")
    data_trial = deepcopy(data)
    for unit in data_trial["unit"]:
        idx_trials = np.array(
            [
                trial_id_start <= trial_id < trial_id_end
                for trial_id in unit["task_variable"]["trial_num"]
            ]
        )
        unit["response"] = unit["response"][idx_trials, :]
        for task_var in unit["task_variable"]:
            unit["task_variable"][task_var] = unit["task_variable"][task_var][
                idx_trials
            ]
    print(f"Number of units in selected trials: {len(data_trial['unit'])}")

    return data_trial


def perform_shuffles(n_shuffles, dataT_nrml):
    def perform_shuffles_parallel(n_shuffles, dataT_nrml):
        # run one shuffle in seriel for debugging
        # results = perform_shuffles_helper((0, dataT_nrml))
        npt = len(dataT_nrml["time"])
        proj_act_shuff_actor = np.zeros([n_shuffles, npt, 3])
        proj_act_shuff_observer = np.zeros([n_shuffles, npt, 3])
        theta_switch_actor = np.zeros([n_shuffles, npt])
        theta_switch_observer = np.zeros([n_shuffles, npt])
        theta_actor_observer = np.zeros([n_shuffles, npt])
        coef_switches = []
        coef_self = []
        coef_other = []
        dimensions = []
        unit_idx_master = []
        run_serially = False
        if run_serially:
            results = []
            for i in range(n_shuffles):
                dataT_nrml_ = deepcopy(dataT_nrml)
                result = perform_shuffles_helper((i, dataT_nrml_))
                if result:
                    results.append(result)
        else:
            n_cores = min(multiprocessing.cpu_count(), 20)
            print(f"Using {n_cores} cores")
            with multiprocessing.Pool(processes=n_cores) as pool:
                results = pool.map(
                    perform_shuffles_helper,
                    [(i, dataT_nrml) for i in range(n_shuffles)],
                )
        if not results:
            return None
        for i, result in enumerate(results):
            proj_act_shuff_actor[i, :, :] = result["proj_act_shuff_actor"]
            proj_act_shuff_observer[i, :, :] = result["proj_act_shuff_observer"]
            theta_switch_actor[i, :] = result["theta_switch_actor"]
            theta_switch_observer[i, :] = result["theta_switch_observer"]
            theta_actor_observer[i, :] = result["theta_actor_observer"]
            coef_switches.append(result["coef_switch"])
            coef_self.append(result["coef_actor"])
            coef_other.append(result["coef_observer"])
            dimensions.append(result["dimensions"])
            unit_idx_master.append(result["unit_idx_master"])
        # use common dimensions from all shuffles
        common_idx = find_common_dimensions(*unit_idx_master)
        for i, result in enumerate(results):
            idx_common = np.isin(unit_idx_master[i], common_idx)
            coef_switches[i] = result["coef_switch"][idx_common]
            coef_self[i] = result["coef_actor"][idx_common]
            coef_other[i] = result["coef_observer"][idx_common]
            unit_idx_master[i] = np.array(unit_idx_master[i])[idx_common]
        unit_idx_master = np.array(unit_idx_master[0])
        coef_self = np.vstack(coef_self).T
        coef_other = np.vstack(coef_other).T
        coef_switches = np.vstack(coef_switches).T

        theta_actor_observer = [
            compute_angle(coef_self[:, i], coef_other[:, i])
            for i in range(n_shuffles)
        ]

        def mag(x, y):
            return np.abs(np.dot(x, y))

        magnitude_actor_observer = [
            mag(coef_self[:, i], coef_other[:, i]) for i in range(n_shuffles)
        ]
        magnitude_switch_actor = [
            mag(coef_switches[:, i], coef_self[:, i]) for i in range(n_shuffles)
        ]
        magnitude_switch_observer = [
            mag(coef_switches[:, i], coef_other[:, i])
            for i in range(n_shuffles)
        ]
        theta_switch_actor = [
            compute_angle(coef_switches[:, i], coef_self[:, i])
            for i in range(n_shuffles)
        ]
        theta_switch_observer = [
            compute_angle(coef_switches[:, i], coef_other[:, i])
            for i in range(n_shuffles)
        ]
        return {
            "proj_act_shuff_actor": proj_act_shuff_actor.reshape(
                n_shuffles * npt, 3
            ),
            "proj_act_shuff_observer": proj_act_shuff_observer.reshape(
                n_shuffles * npt, 3
            ),
            "theta_switch_actor": theta_switch_actor,
            "magnitude_actor_observer": magnitude_actor_observer,
            "magnitude_switch_actor": magnitude_switch_actor,
            "magnitude_switch_observer": magnitude_switch_observer,
            "theta_switch_observer": theta_switch_observer,
            "theta_actor_observer": theta_actor_observer,
            "coef_self": coef_self,
            "coef_other": coef_other,
            "coef_switch": coef_switches,
            "unit_idx_master": unit_idx_master,
        }

    results = perform_shuffles_parallel(n_shuffles, dataT_nrml)
    return results


def main(args):
    subject, event = args
    datadir = findrootdir()
    time_window = [-3, 3]
    load_event = event
    if load_event == "prechoice":
        load_event = "choice"
    dataT, metadata = tdrLoadAccData(datadir, subject, load_event, time_window)
    if event in ["choice", "prechoice"]:
        dataT = use_prev_switch(dataT, metadata)
    # select time for analysis
    tmin = 0
    tmax = 0.6
    if event == "prechoice":
        tmin = -0.6
        tmax = 0.0

    dataT = select_time(dataT, tmin=tmin, tmax=tmax)
    # filter data to exclude units that have low firing rate in this epoch
    dataT, metadata = remove_low_rate(dataT, metadata, min_rate=0.5)

    # select units with significant selectivity using master list
    dataT, metadata = select_significant_units(
        dataT, metadata, "fdbk", direction="both"
    )

    # Average data: get mean and std first
    avg_params = {"trial": [], "time": []}
    meanT, stdT = tdrMeanAndStd(dataT, avg_params)

    # Normalization parameters
    norm_params = {"ravg": meanT, "rstd": stdT}
    dataT_nrml = tdrNormalize(dataT, norm_params)

    # NEW: perform analysis on each session separately
    # get the list of sessions
    subject_names = load_subject_names()
    session_ids = np.unique([unit["session_id"] for unit in dataT_nrml["unit"]])
    compute_per_session = False
    if compute_per_session:
        for session_id in session_ids:
            # find the corresponding date based on session_id
            date = subject_names.loc[
                (
                    (subject_names["subject1"] == subject)
                    | (subject_names["subject2"] == subject)
                )
                & (subject_names["session #"] == session_id),
                "date",
            ].values[0]

            # select data for this session
            dataT_session = tdrSelectByID(dataT_nrml, session_id)

            if len(dataT_session["unit"]) < 2:
                print(
                    f"Not enough units for session {session_id} in {subject} {event}"
                )
                continue

            # perform shuffles for this session
            n_shuffles = 100
            dataT_session["animal"] = subject
            dataT_session["event"] = event
            results = perform_shuffles(n_shuffles, dataT_session)
            if results is None:
                print(
                    f"No results for session {session_id} in {subject} {event}"
                )
                continue

            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results[key] = value.tolist()
            if not os.path.exists(f"{datadir}/stats_paper"):
                os.makedirs(f"{datadir}/stats_paper")
            # save result to json file
            with open(
                f"{datadir}/stats_paper/{subject}_{event}_act_obs_dimensions_{date}.json",
                "w",
            ) as f:
                json.dump(results, f)
    control_n_neurons = False
    control_n_trials = False
    # Optional control: equalize the number of Act/Obs switch trials
    if control_n_trials:
        dataT_nrml = select_congruent_trials(dataT_nrml)
        dataT_nrml = tdrEqualizeActorObserverSwitchTrials(dataT_nrml)
    # Optional control: equalize the number of Act/Obs selective neurons
    if control_n_neurons:
        dataT_nrml = equalize_actor_observer_unit(dataT_nrml)

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
    result_name = f"{subject}_{event}_act_obs_dimensions"
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
    for event in ["fdbk", "prechoice"]:
        for animal in ["O", "L"]:
            main((animal, event))
