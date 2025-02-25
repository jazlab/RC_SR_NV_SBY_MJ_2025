"""
this script is used to plot the activation and angles of the neural network
it loads snapshots of RNNs for each variant, and makes plots of the activations 
and angles
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from models.multi_rnn import MultiRNN
from msi_task import MSI
import torch
from sklearn.linear_model import LinearRegression
import os
import re
import json
from plot_functions import (
    plot_sample_trial_output,
    plot_thetas_for_model_type,
    plot_ev_history,
    plot_polar_angles,
)
from msi_utils import (
    extract_paths,
    load_checkpoint,
    load_neural_results,
)
from utils.LoadSession import findrootdir

checkpoint_id = 100000


# For visualization of model activation, merge several sortings
def merge_arrays(*arrays):
    n = len(arrays)  # Number of arrays
    total_length = len(arrays[0])  # Assuming all arrays are of equal length
    min_per_array = total_length // n  # Minimum items per array
    extra_count = (
        total_length % n
    )  # Extra items to distribute among the first few arrays

    C = []

    def add_unique_items(source, target, count):
        added = 0
        for item in source:
            if item not in target:
                target.append(item)
                added += 1
                if added == count:
                    break

    for i, arr in enumerate(arrays):
        # Determine how many elements to take from the current array
        count = min_per_array + (1 if i < extra_count else 0)
        add_unique_items(arr, C, count)

    return C


# Compute the angle between two vectors
def compute_angle(v1, v2):
    """Compute the angle between two vectors.

    Parameters
    ----------
    v1 : array_like
        First vector.
    v2 : array_like
        Second vector.

    Returns
    -------
    angle : float
        Angle between the two vectors.

    """
    return np.degrees(
        np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    )


# Parse the parameter string to extract the model parameters
def parse_param_str(path):
    # Use regular expressions to find the model_type and hidden_size in the path
    model_type_match = re.search(r"model_type_([^_]+)", path)
    hidden_size_match = re.search(r"hidden_size_(\d+)", path)

    if model_type_match and hidden_size_match:
        model_type = model_type_match.group(1)
        n_units = int(hidden_size_match.group(1))
        return n_units, model_type
    else:
        return None, None


def get_history(trial_data, model_type):
    inputs = trial_data["inputs"]
    # get value and identity based on input
    if model_type == "paral":
        values = inputs[:, :, 0]
        identity = inputs[:, :, 1]
    elif model_type == "ortho":
        value_self = inputs[:, :, 0]
        value_other = inputs[:, :, 1]
        values = value_self + value_other
        identity = np.zeros_like(values)
        identity[value_self != 0] = 1
        identity[value_other != 0] = -1
    history = np.full_like(values, np.nan)
    # set history to -1 where value is positive  (-1: reward)
    history[values > 0] = -1
    # set history to 0 where value is negative   (0: nonreward)
    history[values < 0] = 0
    # if there are consecutive nonreward trials, increment history for each instance
    trialstarts = np.where(~np.isnan(history))[0]
    n_trials = len(trialstarts)
    for i in range(n_trials - 1):
        ts_prev = trialstarts[i]
        ts = trialstarts[i + 1]
        if history[ts_prev] > -1 and history[ts] > -1:  # consec nr trial
            history[ts] = history[ts_prev] + 1
    # fill nans with the previous non-nan value
    for i in range(1, len(history)):
        if np.isnan(history[i]):
            history[i] = history[i - 1]
    # make sure history has dim 1
    history = history.squeeze()
    return history


# Compute the angle between the vectors from the hidden units to the output unit
def compute_vector_angle_from_checkpoint(
    checkpoint_path,
    model_type,
    n_units,
    iti,
    mean_choices,
    integration_factor,
    random_seed=0,
):
    n_time = 5
    n_repeat = 1
    n_select = n_units
    theta_self_other = np.zeros([n_repeat, n_time])
    theta_self_out = np.zeros([n_repeat, n_time])
    theta_other_out = np.zeros([n_repeat, n_time])
    theta_self_out_ramp = np.zeros([n_repeat, n_time])
    theta_other_out_ramp = np.zeros([n_repeat, n_time])
    betas_self = np.zeros([n_repeat, n_select, n_time])
    betas_other = np.zeros([n_repeat, n_select, n_time])
    betas_outdir = np.zeros([n_repeat, n_select, n_time])  # for each tpt
    betas_outdir_fixed = np.zeros([n_repeat, n_select])  # fixed from all tpts
    betas_self_ramp = np.zeros(n_select)
    betas_other_ramp = np.zeros(n_select)
    sequence_length = 500
    # Test inputs: run x iterations to get mean activations
    for i_repeat in range(n_repeat):
        # Initialize the model and load the checkpoint
        model = MultiRNN(
            hidden_size=n_units,
            input_feature_len=3,
            activity_decay=0.2,
        )

        task = MSI(
            timesteps=sequence_length,
            model_type=model_type,
            integration_factor=integration_factor,
            mean_choices=mean_choices,
            value_std=0.1,
            iti=iti,
            rng=np.random.default_rng(random_seed * 100 + i_repeat),
        )

        load_checkpoint(model, checkpoint_path)
        # Test inputs
        trial_data = task()
        ramp_onsets = trial_data["rampstarts"]
        history = get_history(trial_data, model_type)
        identity = trial_data["identity"]
        # input occurs when the identity is nonzero
        idx_input = np.where(identity != 0)[0]
        # fill in identity values with previous value if 0
        for i in range(1, len(identity)):
            if identity[i] == 0:
                identity[i] = identity[i - 1]
        # convert trial data to batch first by swapping the first two dimensions
        trial_data["inputs"] = trial_data["inputs"].reshape(1, -1, 3)
        trial_data["labels"] = trial_data["labels"].reshape(1, -1, 1)
        test_inputs = trial_data["inputs"].squeeze()
        test_outputs = trial_data["labels"].squeeze()
        trial_data = {k: torch.from_numpy(v) for k, v in trial_data.items()}
        with torch.no_grad():
            out = model.forward(trial_data)
        output = out["outputs"]

        # select output and label from reporting period to save to results
        idx_report = np.where(test_inputs[:, 2] == 1)[0]
        idx_report_first = [idx_report[0]]
        idx_report_prev = idx_report[0]
        for idx in idx_report:
            if idx - idx_report_prev <= 1:
                idx_report_prev = idx
                continue
            else:
                idx_report_first.append(idx)
                idx_report_prev = idx
        idx_self = np.where(identity == 1)[0]
        idx_self_report = np.intersect1d(idx_self, idx_report)
        idx_other = np.where(identity == -1)[0]
        idx_other_report = np.intersect1d(idx_other, idx_report)
        target = test_outputs[idx_report]
        output_reporting = output.squeeze()[idx_report]
        history_reporting = history[idx_report]
        target_self = test_outputs[idx_self_report]
        output_self = output.squeeze()[idx_self_report]
        history_self = history[idx_self_report]
        target_other = test_outputs[idx_other_report]
        output_other = output.squeeze()[idx_other_report]
        history_other = history[idx_other_report]

        # convert a list of tensors to a numpy array
        activations_array = out["hiddens"].detach().numpy().squeeze().T
        # subsample activations to select a subset of units
        idx_select = np.random.choice(n_units, n_select, replace=False)
        activations_array = activations_array[idx_select, :]
        outputs_array = output.squeeze().numpy()  # N_time
        # use all timepoints to compute output direction
        x = activations_array
        y = outputs_array
        reg2 = LinearRegression(fit_intercept=True)
        reg2.fit(x.T, y)
        betas_outdir_fixed[i_repeat, :] = reg2.coef_
        for i_offset in range(n_time):
            # evidence dir: compute for each timepoint
            x = activations_array[:, idx_input + i_offset]
            y = outputs_array[
                idx_report_first
            ]  # regress against the output that matters

            for i_unit in range(n_select):
                reg = LinearRegression(fit_intercept=True)
                reg.fit(x[i_unit, :].reshape(-1, 1), y)
                betas_outdir[i_repeat, i_unit, i_offset] = reg.coef_[0]
            inputs_array = test_inputs
            self_trials = np.where(identity == 1)[0]
            self_trials = np.intersect1d(self_trials, idx_input)
            other_trials = np.where(identity == -1)[0]
            other_trials = np.intersect1d(other_trials, idx_input)
            # offseted index for self and other trials
            self_trials_offset = self_trials + i_offset
            other_trials_offset = other_trials + i_offset

            # self dir: fdbk (input) time
            x = activations_array[:, self_trials_offset]
            y = inputs_array[self_trials, 0]
            y[y > 0] = 1
            y[y < 0] = -1
            for i_unit in range(n_select):
                reg = LinearRegression(fit_intercept=True)
                reg.fit(x[i_unit, :].reshape(-1, 1), y)
                betas_self[i_repeat, i_unit, i_offset] = reg.coef_[0]
            # other dir: fdbk (input) time
            x = activations_array[:, other_trials_offset]
            y[y > 0] = 1
            y[y < 0] = -1
            if model_type == "ortho":
                y = inputs_array[other_trials, 1]
            else:
                y = inputs_array[other_trials, 0]
            for i_unit in range(n_select):
                reg = LinearRegression(fit_intercept=True)
                reg.fit(x[i_unit, :].reshape(-1, 1), y)
                betas_other[i_repeat, i_unit, i_offset] = reg.coef_[0]

            # self dir: ramp time
            self_trials = np.where(identity == 1)[0]
            self_ramp_onsets = np.intersect1d(self_trials, ramp_onsets)
            self_ramp_onsets_offset = self_ramp_onsets + i_offset
            x = activations_array[:, self_ramp_onsets_offset]
            self_trials_input = np.intersect1d(self_trials, idx_input)
            y = inputs_array[self_trials_input, 0]
            y[y > 0] = 1
            y[y < 0] = -1
            for i_unit in range(n_select):
                reg = LinearRegression(fit_intercept=True)
                reg.fit(x[i_unit, :].reshape(-1, 1), y)
                betas_self_ramp[i_unit] = reg.coef_[0]
            # other dir: ramp time
            other_trials = np.where(identity == -1)[0]
            other_ramp_onsets = np.intersect1d(other_trials, ramp_onsets)
            other_ramp_onsets_offset = other_ramp_onsets + i_offset
            x = activations_array[:, other_ramp_onsets_offset]
            other_trials_input = np.intersect1d(other_trials, idx_input)
            if model_type == "ortho":
                y = inputs_array[other_trials_input, 1]
            else:
                y = inputs_array[other_trials_input, 0]
            y[y > 0] = 1
            y[y < 0] = -1
            for i_unit in range(n_select):
                reg = LinearRegression(fit_intercept=True)
                reg.fit(x[i_unit, :].reshape(-1, 1), y)
                betas_other_ramp[i_unit] = reg.coef_[0]
            # theta between self-output and other-output
            theta_self_out_ramp[i_repeat, i_offset] = compute_angle(
                betas_self_ramp,
                betas_outdir[i_repeat, :, i_offset],
            )
            theta_other_out_ramp[i_repeat, i_offset] = compute_angle(
                betas_other_ramp,
                betas_outdir[i_repeat, :, i_offset],
            )

            theta_other_out[i_repeat, i_offset] = compute_angle(
                betas_other[i_repeat, :, i_offset],
                betas_outdir[i_repeat, :, i_offset],
            )
            theta_self_out[i_repeat, i_offset] = compute_angle(
                betas_self[i_repeat, :, i_offset],
                betas_outdir[i_repeat, :, i_offset],
            )
            theta_self_other[i_repeat, i_offset] = compute_angle(
                betas_self[i_repeat, :, i_offset],
                betas_other[i_repeat, :, i_offset],
            )

    theta_other_out = np.mean(theta_other_out, axis=0)  # average over repeats
    theta_self_out = np.mean(theta_self_out, axis=0)
    theta_self_other = np.mean(theta_self_other, axis=0)
    theta_self_out_ramp = np.mean(theta_self_out_ramp, axis=0)
    theta_other_out_ramp = np.mean(theta_other_out_ramp, axis=0)

    return {
        "target": target.tolist(),
        "output": output_reporting.tolist(),
        "history": history_reporting.tolist(),
        "target_self": target_self.tolist(),
        "output_self": output_self.tolist(),
        "history_self": history_self.tolist(),
        "target_other": target_other.tolist(),
        "output_other": output_other.tolist(),
        "history_other": history_other.tolist(),
        "theta_self_other": theta_self_other.tolist(),
        "theta_self_out": theta_self_out.tolist(),
        "theta_other_out": theta_other_out.tolist(),
        "theta_self_out_ramp": theta_self_out_ramp.tolist(),
        "theta_other_out_ramp": theta_other_out_ramp.tolist(),
        "coef_self": betas_self.tolist(),
        "coef_other": betas_other.tolist(),
        "coef_outdir": betas_outdir.tolist(),
    }


def save_and_close_fig(fig, save_path, dpi=300):
    """Helper function to save and close a matplotlib figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def process_and_plot_results(
    paths_with_ids,
    model_type,
    require_grad,
    iti,
    n_units,
):
    redo_analysis = False
    results = []
    n_nets = 0
    n_nets_max = 100
    for log_path, id in paths_with_ids:
        checkpoint_path = f"{log_path}/snapshots/{checkpoint_id}"
        # check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            continue
        result_path_one_net = f"{log_path}/results_with_coef_{model_type}_{require_grad}_{id}.json"
        if os.path.exists(result_path_one_net) and not redo_analysis:
            with open(result_path_one_net, "r") as file:
                result = json.load(file)
            results.append(result)
        else:
            result = compute_vector_angle_from_checkpoint(
                checkpoint_path,
                model_type,
                n_units,
                iti,
                [-0.5, 0.5],
                0.5,
                random_seed=n_nets,
            )
            # save results to file
            with open(result_path_one_net, "w") as file:
                json.dump(result, file)
            results.append(result)
        n_nets += 1
        if n_nets == n_nets_max:
            break

    return results


def main(model_type, require_grad, iti=5, n_units=200):
    root_dir = findrootdir()
    checkpoint_path = f"{root_dir}/rnn_snapshots/39553470"
    rcParams["pdf.fonttype"] = 42
    paths_with_ids = extract_paths(
        checkpoint_path,
        type=model_type,
        require_grad=require_grad,
    )
    postfix = f"{model_type}_{require_grad}"
    print(postfix)
    fig_dir = f"{root_dir}/plots_paper"

    # Ensure fig_dir exists
    os.makedirs(fig_dir, exist_ok=True)

    results_neural = load_neural_results(root_dir)
    # Process results
    results = process_and_plot_results(
        paths_with_ids,
        model_type,
        require_grad,
        iti,
        n_units,
    )

    # Generate and save concatenated results plots
    fignames_history = {
        "paral_False": "Fig4G",
        "ortho_False": "Fig4K",
    }
    if postfix in fignames_history:
        fig = plot_ev_history(results, model_type)
        save_and_close_fig(
            fig,
            f"{fig_dir}/{fignames_history[postfix]}_evidence_history_{postfix}.pdf",
        )

    fignames_angle = {
        "paral_False": "Fig4R",
        "ortho_False": "Fig4S",
        "paral_True": "FigS10D",
        "ortho_True": "FigS10B",
    }
    fig = plot_thetas_for_model_type(results, results_neural, model_type)
    save_and_close_fig(
        fig,
        f"{fig_dir}/{fignames_angle[postfix]}_angle_histogram_{postfix}.pdf",
    )

    fignames_angle = {
        "paral_False": "Fig4H",
        "ortho_False": "Fig4L",
        "paral_True": "FigS10D",
        "ortho_True": "FigS10B",
    }
    fig = plot_polar_angles(results, results_neural, model_type)
    save_and_close_fig(
        fig, f"{fig_dir}/{fignames_angle[postfix]}_polar_angles_{postfix}.pdf"
    )

    # Task and activation plots for latest checkpoint; only for fixed nets
    if require_grad == "True":
        return
    mean_choices = [-0.5, 0.5]
    fig = plot_sample_trial_output(
        f"{paths_with_ids[0][0]}/snapshots/{checkpoint_id}",
        model_type,
        mean_choices,
        n_units,
        0.5,
        test=True,
    )
    save_and_close_fig(
        fig,
        f"{fig_dir}/FigS8_task_and_output_{postfix}.pdf",
        dpi=600,
    )


if __name__ == "__main__":
    for model_type in ["ortho", "paral"]:
        for require_grad in ["True", "False"]:
            main(model_type, require_grad)
