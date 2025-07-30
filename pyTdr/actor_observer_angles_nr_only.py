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
    select_time,
    filter_by_common_dimensions,
    select_significant_units,
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
    # Select conditions for regression
    conditions = {
        "n_pre_switch_lvl": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "choice": [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
        "actor": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    }
    dataT_nrml_regr = select_condition(
        dataT_nrml,
        conditions,
        crossvalidate="random",
        min_trials=5,
        random_seed=i_shuff,
    )

    # actor 1NR vs all 2NR. requiring actor 2NR resulted in too few trials
    conditions = {
        "history": [2, 2, 3, 3],
        "actor": [1, 1, 1, 1],
        "choice": [1, 2, 1, 2],
    }
    dataT_act = select_condition(
        dataT_nrml_regr, conditions, crossvalidate="none", min_trials=1
    )
    conditions = {
        "history": [1, 2],
        "actor": [1, 1],
    }
    dataC_act, _ = average_conditions(dataT_act, conditions)
    conditions = {
        "history": [2, 2, 3, 3],
        "actor": [2, 2, 2, 2],
        "choice": [1, 2, 1, 2],
    }
    dataT_obs = select_condition(
        dataT_nrml_regr, conditions, crossvalidate="none", min_trials=1
    )
    conditions = {
        "history": [1, 2],
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

    # Define regression vectors
    plotflag = False
    vecpars = {}
    vecpars["history"] = {}
    vecpars["history"]["time_win"] = [-2, 2]  # select all
    vecpars["choice"] = {}
    vecpars["choice"]["time_win"] = [-2, 2]  # select all

    # Define task-related axes (orthogonalize regression vectors)
    ortpars = {}
    ortpars["name"] = ["history", "choice"]

    # perform linear regression on actor history
    coef_actor = perform_regression(
        dataT_act, ["b0", "history", "choice"], normalization="max_abs"
    )
    vBeta_actor, _ = tdrVectorTimeAverage(coef_actor, vecpars, plotflag)
    vAxes_actor, _ = tdrVectorOrthogonalize(vBeta_actor, ortpars)

    # perform linear regression on observer history
    coef_observer = perform_regression(
        dataT_obs, ["b0", "history", "choice"], normalization="max_abs"
    )
    vBeta_observer, _ = tdrVectorTimeAverage(coef_observer, vecpars, plotflag)
    vAxes_observer, _ = tdrVectorOrthogonalize(vBeta_observer, ortpars)

    # flip the sign of coefficients for history
    vAxes_actor["response"][:, :, 0] = -vAxes_actor["response"][:, :, 0]
    vAxes_observer["response"][:, :, 0] = -vAxes_observer["response"][:, :, 0]

    # NEW: compute angle using the average coefficient over time
    theta_actor_observer_avg = compute_angle(
        vAxes_actor["response"][:, :, 0].mean(axis=1),
        vAxes_observer["response"][:, :, 0].mean(axis=1),
    )

    return {
        "theta_actor_observer": theta_actor_observer_avg,
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


def perform_shuffles(n_shuffles, dataT_nrml):
    def perform_shuffles_parallel(n_shuffles, dataT_nrml):
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

        # Get theta_actor_observer directly from the helper function results
        theta_actor_observer = [
            result["theta_actor_observer"] for result in results
        ]

        return {
            "theta_actor_observer": theta_actor_observer,
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
    run_per_session = False
    if run_per_session:
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

            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results[key] = value.tolist()
            if not os.path.exists(f"{datadir}/stats_paper"):
                os.makedirs(f"{datadir}/stats_paper")
            # save result to json file
            with open(
                f"{datadir}/stats_paper/{subject}_{event}_act_obs_dimensions_NR_{date}.json",
                "w",
            ) as f:
                json.dump(results, f)
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
        f"{datadir}/stats_paper/{subject}_{event}_act_obs_dimensions_NR.json",
        "w",
    ) as f:
        json.dump(results, f)


if __name__ == "__main__":
    for event in ["fdbk"]:
        for animal in ["O", "L"]:
            main((animal, event))
