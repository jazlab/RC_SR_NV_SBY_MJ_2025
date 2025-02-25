"""
This is ported from Matlab code from the original TDR codebase.
"""

import numpy as np
import pandas as pd
import copy


def average_conditions(data, conditions):
    task_index = {}
    for key, value in conditions.items():
        unique_values, indices = np.unique(value, return_inverse=True)
        task_index[key] = indices
    return tdrAverageCondition(data, task_index)


def tdrAverageCondition(data, task_index, use_test=False):
    """
    tdrAverageCondition compute condition averaged responses

    Inputs:
    data: population response
    task_index: indices of task variables to average.

    Outputs:
    data_avg: condition averages. Same format as simultaneous trial-by-trial
    responses.
    data_sub: condition-average-subtracted trial-by-trial responses. Here the
    responses in data_avg are subtracted from corresponding trials in data.
    Only meaningful if each trial contributes to at most 1 condition.
    """

    resp_field = "response"
    taskvar_field = "task_variable"
    if use_test:
        resp_field = "response_test"
        taskvar_field = "task_variable_test"

    # All the task indices
    variable_name = list(task_index.keys())
    nvr = len(variable_name)

    # Number of conditions
    ncd = len(task_index[variable_name[0]])

    # Number of time samples
    npt = len(data["time"])
    # Check if data is sequentially or simultaneously recorded
    if "unit" in data and data["unit"] is not None:
        # --- Sequential recordings ---
        # Initialize
        data_sub = copy.deepcopy(data)
        # Number of units
        nun = len(data["unit"])

        # Initialize
        response_avg = np.full((nun, npt, ncd), np.nan)
        n_trial = np.zeros((nun, ncd))
        variable_condition_value = np.full((nvr, ncd, nun), np.nan)
        dimension = []
        unit_idx_master = []

        for iun in range(nun):
            # Number of trials
            ntr = data["unit"][iun][resp_field].shape[0]

            # Initialize trials to average
            jj_cnd = np.zeros((ncd, ntr), dtype=bool)

            # Unique variable names
            variable_unique = [
                np.sort(pd.unique(data["unit"][iun][taskvar_field][var]))
                for var in variable_name
            ]
            # Loop over conditions
            for icd in range(ncd):
                # Trials to average
                jj_var = np.zeros((nvr, ntr), dtype=bool)

                # Loop over variables
                for ivr in range(nvr):
                    if task_index[variable_name[ivr]][icd] == -1:
                        # Take all trials
                        jj_var[ivr, :] = np.ones(ntr, dtype=bool)
                    else:
                        if task_index[variable_name[ivr]][icd] + 1 > len(
                            variable_unique[ivr]
                        ):
                            continue
                        # Find matching trials
                        jj_var[ivr, :] = (
                            data["unit"][iun][taskvar_field][variable_name[ivr]]
                            == variable_unique[ivr][
                                task_index[variable_name[ivr]][icd]
                            ]
                        )

                        # Keep variable value
                        variable_condition_value[ivr, icd, iun] = (
                            variable_unique[ivr][
                                task_index[variable_name[ivr]][icd]
                            ]
                        )

                # Fulfill constraints on all indices
                jj_cnd[icd, :] = np.all(jj_var, axis=0)

                # Number of trials for this condition
                n_trial[iun, icd] = np.sum(jj_cnd[icd, :])
                # Condition average
                if n_trial[iun, icd] > 0:
                    response_avg[iun, :, icd] = np.mean(
                        data["unit"][iun][resp_field][jj_cnd[icd, :], :], axis=0
                    )
                    # Average-subtracted responses
                    data_sub["unit"][iun][resp_field][jj_cnd[icd, :], :] = (
                        data["unit"][iun][resp_field][jj_cnd[icd, :], :]
                        - response_avg[iun, :, icd]
                    )
                else:
                    # a condition is missing, remove the unit
                    break
            if n_trial[iun, icd] == 0:
                response_avg[iun, :, :] = np.nan
                variable_condition_value[:, :, iun] = np.nan
                n_trial[iun, :] = 0
            else:
                dimension.append(data["unit"][iun]["dimension"])
                unit_idx_master.append(data["unit"][iun]["unit_idx_master"])
        # remove units with nan values
        jj = np.isnan(response_avg[:, 0, 0])
        response_avg = np.delete(response_avg, jj, 0)
        variable_condition_value = np.delete(variable_condition_value, jj, 2)
        n_trial = np.delete(n_trial, jj, 0)
        # Keep average values of the task variables
        task_variable = {
            var: np.nanmean(variable_condition_value[ivr, :, :], axis=1)
            for ivr, var in enumerate(variable_name)
        }

    else:
        # --- Simultaneous recordings ---
        # TODO function is not fully implemented for simultaneous recordings
        nun, npt, ntr = data[resp_field].shape

        # Initialize
        data_sub = data.copy()
        response_avg = np.full((nun, npt, ncd), np.nan)
        n_trial = np.zeros((1, ncd))
        task_variable = {var: np.full(ncd, np.nan) for var in variable_name}

        # Initialize trials to average
        jj_cnd = np.zeros((ncd, ntr), dtype=bool)

        # Unique variable names
        variable_unique = {
            var: pd.unique(data[taskvar_field][var]) for var in variable_name
        }

        # Loop over conditions
        for icd in range(ncd):
            # Trials to average
            jj_var = np.zeros((nvr, ntr), dtype=bool)

            # Loop over variables
            for ivr in range(nvr):
                if task_index[variable_name[ivr]][icd] == -1:
                    # Take all trials
                    jj_var[ivr, :] = np.ones(ntr, dtype=bool)
                else:
                    # Find matching trials
                    jj_var[ivr, :] = (
                        data[taskvar_field][variable_name[ivr]]
                        == variable_unique[variable_name[ivr]][
                            task_index[variable_name[ivr]][icd]
                        ]
                    )

                    # Keep variable value
                    task_variable[variable_name[ivr]][icd] = variable_unique[
                        variable_name[ivr]
                    ][task_index[variable_name[ivr]][icd]]

            # Fulfill constraints on all indices
            jj_cnd[icd, :] = np.all(jj_var, axis=0)

            # Number of trials for this condition
            n_trial[0, icd] = np.sum(jj_cnd[icd, :])

            # Condition average
            if n_trial[0, icd] > 0:
                response_avg[:, :, icd] = np.mean(
                    data[resp_field][:, :, jj_cnd[icd, :]], axis=2
                )

            # Average-subtracted response
            if n_trial[0, icd] > 0:
                data_sub[resp_field][:, :, jj_cnd[icd, :]] = data[resp_field][
                    :, :, jj_cnd[icd, :]
                ] - np.expand_dims(response_avg[:, :, icd], axis=2)
        dimension = data["dimension"]
        unit_idx_master = data["unit_idx_master"]
    # Keep what you need
    data_avg = {
        "response": response_avg,
        "task_variable": task_variable,
        "task_index": task_index,
        "n_trial": n_trial,
        "time": data["time"],
        "dimension": dimension,
        "unit_idx_master": unit_idx_master,
    }

    return data_avg, data_sub
