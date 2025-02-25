"""
This module contains functions to plot the results of RNN simulations, including
activity, performance, and angle between evidence and input dimensions.
For comparison with neural data, the module also contains functions to plot
the angle between neural regression dimensions for evidence and outcome.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.multi_rnn import MultiRNN
from msi_task import MSI
import torch
from msi_utils import load_checkpoint


# Function to create histogram data and plot
def plot_histogram(ax, data, bin_size, color, label):
    bins = np.arange(-180, 180 + bin_size, bin_size)
    hist, _ = np.histogram(data, bins=bins)
    prob = hist / np.sum(hist)  # Convert to probability
    theta = np.deg2rad(
        bins[:-1] + bin_size / 2
    )  # Convert to radians and center bins
    bars = ax.bar(
        theta,
        prob,
        width=np.deg2rad(bin_size),
        bottom=0.0,
        color=color,
        alpha=0.5,
        label=label,
    )
    return prob


def plot_mean_and_sd(ax, data, color):
    mean_theta = np.mean(data)
    sem_theta = np.std(data)  # / np.sqrt(len(data))

    # Convert to radians
    mean_theta_rad = np.deg2rad(mean_theta)
    sem_theta_rad = np.deg2rad(sem_theta)

    # Plot mean as a vertical line from 0 to 1
    ax.vlines(mean_theta_rad, 0.01, 1, color=color, linewidth=0.5, zorder=3)

    # Plot SEM as a shaded area
    theta = np.linspace(
        mean_theta_rad - sem_theta_rad, mean_theta_rad + sem_theta_rad, 100
    )
    ax.fill_between(
        theta,
        np.ones_like(theta),
        y2=0.01,
        color=color,
        alpha=0.2,
        zorder=2,
    )

    return mean_theta, sem_theta


def beutify(ax, side="left"):
    # Make left and bottom spines more prominent by increasing their width
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Separate the x and y axis by setting the spine positions
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Adjust the ticks to be outside and include both major and minor ticks
    ax.tick_params(
        axis="both", direction="out", length=6, width=0.8, which="major"
    )
    ax.tick_params(
        axis="both", direction="out", length=4, width=0.8, which="minor"
    )

    if side == "left":
        # Set the ticks to only appear left side
        ax.yaxis.tick_left()
    elif side == "right":
        # Set the ticks to only appear right side
        ax.yaxis.tick_right()
    ax.xaxis.tick_bottom()

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


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


def plot_results(inputs, target, predicted):
    input_1, input_2 = inputs[:, 0].numpy(), inputs[:, 1].numpy()
    input_3 = inputs[:, 2].numpy()
    target = target.squeeze().numpy()
    target[input_3 == 0] = np.nan
    predicted = predicted.squeeze().detach().numpy()

    fig = plt.figure(figsize=(10, 8))
    timesteps = len(target)

    # Input 1
    plt.subplot(2, 1, 1)
    plt.step(range(timesteps), input_1, label="Input 1", where="mid")
    plt.step(range(timesteps), input_2, label="Input 2", where="mid", color="r")
    plt.title("Input 1")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    # Target Output
    plt.subplot(2, 1, 2)
    plt.plot(target, linestyle="--", label="Target Output")
    plt.plot(predicted, linestyle="-", label="Predicted Output")
    plt.title("Target Output")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    return fig


def plot_ev_history(results, model_type):
    # expects multiple results
    history_self = [np.array(result["history_self"]) for result in results]
    history_other = [np.array(result["history_other"]) for result in results]
    output_self = [np.array(result["output_self"]) for result in results]
    output_other = [np.array(result["output_other"]) for result in results]

    # flatten arrays
    history_self = np.concatenate(history_self)
    history_other = np.concatenate(history_other)
    output_self = np.concatenate(output_self)
    output_other = np.concatenate(output_other)
    idx_hisotry_self = np.where(history_self < 2)[0]
    history_self = history_self[idx_hisotry_self]
    output_self = output_self[idx_hisotry_self]
    idx_hisotry_other = np.where(history_other < 2)[0]
    history_other = history_other[idx_hisotry_other]
    output_other = output_other[idx_hisotry_other]
    # center output self and output other by subtracting the mean
    all_output = np.concatenate([output_self, output_other])
    mean_output = np.mean(all_output)
    output_self = output_self - mean_output
    output_other = output_other - mean_output
    # use sns.lineplot to plot output vs history for each of self/other
    fig = plt.figure(figsize=(4, 4))
    ax = sns.lineplot(
        x=history_self,
        y=-output_self,
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        label="Self",
        color="black",
        marker="o",
        linestyle="-",
    )
    sns.lineplot(
        x=history_other,
        y=-output_other,
        errorbar=("ci", 95),
        seed=0,
        n_boot=1000,
        err_style="bars",
        label="Other",
        color="black",
        marker="o",
        linestyle="-",
        markersize=4,
        markerfacecolor="none",
        markeredgewidth=1,
        markeredgecolor="black",
    )
    ax.set_xlim([-1.1, 1.1])
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim([-0.4, 0.8])
    ax.set_yticks([-0.4, 0, 0.4, 0.8])
    ax.set_xticklabels(["1R", "1NR", "2NR"])
    plt.xlabel("History")
    plt.ylabel("Evidence")
    # plt.legend()
    beutify(plt.gca())
    return fig


def plot_input_output(test_inputs, outputs_array, test_outputs):
    # Start plotting
    fig = plt.figure(figsize=(8, 4))

    # Create the normalization object
    test_inputs = test_inputs.squeeze()

    # remove output for when input_3 is 0
    input_3 = test_inputs[:, 2]
    test_outputs[input_3 < 1] = np.nan

    # Subsequent plots
    ax1 = fig.add_subplot(2, 1, 1)

    # Input[0] and Input[1] plot
    ax1.plot(
        test_inputs[:, 0], marker="o", linestyle="-", color="c", label="Input 1"
    )
    ax1.set_ylabel("Value")
    ax1.set_xlim([0, len(test_inputs) - 1])

    # Create a twinx axis for Input[1]
    ax1_twinx = ax1.twinx()
    ax1_twinx.plot(
        test_inputs[:, 1], marker="o", linestyle="-", color="r", label="Input 2"
    )
    ax1_twinx.set_xlim([0, len(test_inputs) - 1])

    ax1.set_ylabel("Input")
    ax1.set_xlim([0, len(test_inputs) - 1])

    # make sure 0 in ax1 and ax1_twinx are aligned
    max_abs_0 = max(abs(min(test_inputs[:, 0])), abs(max(test_inputs[:, 0])))
    max_abs_1 = max(abs(min(test_inputs[:, 1])), abs(max(test_inputs[:, 1])))
    ax1.set_ylim([-max_abs_0, max_abs_0])
    ax1_twinx.set_ylim([-max_abs_1, max_abs_1])
    # Combine handles and labels from both axes to create a unified legend
    handles, labels = [], []
    for ax in [ax1, ax1_twinx]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    ax2 = fig.add_subplot(2, 1, 2)
    # Output plot
    ax2.plot(-outputs_array, marker="o", linestyle="-", label="Model")
    ax2.plot(-test_outputs, marker="o", linestyle="--", label="Target")
    ax2.set_ylabel("-Output")
    ax2.set_xlim([0, len(test_inputs) - 1])
    ax2.set_xlabel("Trial")

    # Beautify the axes
    for ax in [ax1, ax2]:
        beutify(ax)
    beutify(ax1_twinx, side="right")

    plt.tight_layout()
    return fig


def plot_performance_for_model_type(results, model_type):
    output = [np.array(result["output"]) for result in results]
    target = [np.array(result["target"]) for result in results]
    # flatten output and target arrays
    output = np.concatenate(output)
    target = np.concatenate(target)
    # plot the output and target
    fig = plt.figure(figsize=(4, 4))
    ax = plt.scatter(target, output, color="k", alpha=0.5)
    # add identity line based on range of target and output
    plt.plot(
        [min(target), max(target)],
        [min(target), max(target)],
        color="black",
        linestyle="--",
    )
    # plt.plot([-1,0],[-1,0],color="black",linestyle="--")
    plt.xlabel("Target")
    plt.ylabel("Output")
    beutify(plt.gca())
    return fig


# Plot the task and activations
def plot_sample_trial_output(
    checkpoint_path,
    model_type,
    mean_choices,
    n_units,
    integration_factor,
    test=False,
):
    # Initialize the model and load the checkpoint
    model = MultiRNN(
        hidden_size=n_units,
        input_feature_len=3,
        activity_decay=0.2,
    )

    task = MSI(
        timesteps=200,
        model_type=model_type,
        integration_factor=integration_factor,
        mean_choices=mean_choices,
        value_std=0.1,
        test=test,
    )

    load_checkpoint(model, checkpoint_path)
    # Test inputs
    trial_data = task()
    # convert trial data to batch first by swapping the first two dimensions
    trial_data["inputs"] = trial_data["inputs"].reshape(1, -1, 3)
    trial_data["labels"] = trial_data["labels"].reshape(1, -1, 1)
    test_inputs = trial_data["inputs"].squeeze()
    test_outputs = trial_data["labels"].squeeze()
    trial_data = {k: torch.from_numpy(v) for k, v in trial_data.items()}
    with torch.no_grad():
        out = model.forward(trial_data)
    output = out["outputs"]
    outputs_array = output.squeeze().numpy()  # N_time

    fig = plot_input_output(
        test_inputs,
        outputs_array,
        test_outputs,
    )
    return fig


def plot_angles(ax, theta, magnitude, color, alpha):
    # check if theta is a single number or a list
    if not isinstance(theta, list) and not isinstance(theta, np.ndarray):
        theta = [theta]
        magnitude = [magnitude]
    for t, a in zip(theta, magnitude):
        ax.plot([0, np.deg2rad(t)], [0, a], color=color, alpha=alpha)
    ax.set_thetamin(0)
    ax.set_thetamax(180)


def extract_thetas_neural(results_neural):
    theta_self_other_neural_O = results_neural["O"]["theta_actor_observer"]
    theta_self_other_neural_L = results_neural["L"]["theta_actor_observer"]
    theta_self_switch_O = results_neural["O"]["theta_switch_actor"]
    theta_other_switch_O = results_neural["O"]["theta_switch_observer"]
    theta_self_switch_L = results_neural["L"]["theta_switch_actor"]
    theta_other_switch_L = results_neural["L"]["theta_switch_observer"]
    return (
        theta_self_other_neural_O,
        theta_self_other_neural_L,
        theta_self_switch_O,
        theta_other_switch_O,
        theta_self_switch_L,
        theta_other_switch_L,
    )


def extract_thetas_from_results(results_network, tstart=0, tend=1, ramp=5):
    theta_self_other = np.array(
        [result["theta_self_other"][tstart:tend] for result in results_network]
    )
    theta_self_out = np.array(
        [result["theta_self_out"][tstart:tend] for result in results_network]
    )
    theta_other_out = np.array(
        [result["theta_other_out"][tstart:tend] for result in results_network]
    )
    theta_self_out_ramp = np.array(
        [
            result["theta_self_out_ramp"][ramp - tend : ramp - tstart]
            for result in results_network
        ]
    )
    theta_other_out_ramp = np.array(
        [
            result["theta_other_out_ramp"][ramp - tend : ramp - tstart]
            for result in results_network
        ]
    )

    # # flatten theta arrays
    theta_self_other = np.mean(theta_self_other, axis=1)
    theta_self_out = np.mean(theta_self_out, axis=1)
    theta_other_out = np.mean(theta_other_out, axis=1)
    theta_self_out_ramp = np.mean(theta_self_out_ramp, axis=1)
    theta_other_out_ramp = np.mean(theta_other_out_ramp, axis=1)
    return (
        theta_self_other,
        theta_self_out,
        theta_other_out,
        theta_self_out_ramp,
        theta_other_out_ramp,
    )


def plot_polar_angle_magnitude(results_network, results_neural, model_type):
    # make polar plot
    fig = plt.figure(figsize=(4, 4), facecolor="white")
    ax = fig.add_subplot(111, polar=True)
    # Network data
    n_shuffles = len(results_network)
    for i in range(n_shuffles):
        coef_self = results_network[i]["coef_self"]
        coef_other = results_network[i]["coef_other"]
        angle = compute_angle(coef_self, coef_other)
        magnitude_network = np.abs(np.dot(coef_self, coef_other))
        plot_angles(ax, angle, magnitude_network, "k", 0.5)

    # Neural data for O
    coef_self_O = np.vstack(results_neural["O"]["coef_self"])
    coef_other_O = np.vstack(results_neural["O"]["coef_other"])
    n_shuffles = coef_self_O.shape[1]
    for i in range(n_shuffles):
        angle = compute_angle(coef_self_O[:, i], coef_other_O[:, i])
        magnitude_neural_O = np.abs(
            np.dot(coef_self_O[:, i], coef_other_O[:, i])
        )
        plot_angles(ax, angle, magnitude_neural_O, "g", 0.5)

    # Neural data for L
    coef_self_L = np.vstack(results_neural["L"]["coef_self"])
    coef_other_L = np.vstack(results_neural["L"]["coef_other"])
    n_shuffles = coef_self_L.shape[1]
    for i in range(n_shuffles):
        angle = compute_angle(coef_self_L[:, i], coef_other_L[:, i])
        magnitude_neural_L = np.abs(
            np.dot(coef_self_L[:, i], coef_other_L[:, i])
        )
        plot_angles(ax, angle, magnitude_neural_L, "m", 0.5)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8])
    ax.set_rlabel_position(22.5)
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    plt.legend()
    plt.title(f"Model type: {model_type}")
    return fig


def plot_polar_angles(results_network, results_neural, model_type):
    (
        theta_self_other,
        theta_self_out,
        theta_other_out,
        theta_self_out_ramp,
        theta_other_out_ramp,
    ) = extract_thetas_from_results(results_network)
    (theta_self_other_neural_O, theta_self_other_neural_L, _, _, _, _) = (
        extract_thetas_neural(results_neural)
    )

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection="polar"))

    # Plot histograms for each dataset
    prob1 = plot_histogram(
        ax, theta_self_other, bin_size=3, color="k", label="Network"
    )
    plot_mean_and_sd(ax, theta_self_other_neural_O, "g")
    plot_mean_and_sd(ax, theta_self_other_neural_L, "m")

    # Set the angle limits
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # set radial to log scale and set rlim to 0.01, 1
    ax.set_rscale("log")
    # # Set the radial ticks and labels to 0, 0.1, 1
    ax.set_rticks([0.01, 0.1, 1])
    ax.set_yticklabels([f"{tick:.2f}" for tick in [0.01, 0.1, 1]])

    # Add legend and title
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title(f"Model type: {model_type}")

    # Beautify the plot
    ax.grid(True)
    plt.tight_layout()

    return fig


def pairwise_sub(list1, list2):
    return [a - b for a, b in zip(list1, list2)]


def extract_thetas_switch_neural(results_neural):
    theta_self_switch_neural_O = results_neural["O"]["theta_switch_actor"]
    theta_other_switch_neural_O = results_neural["O"]["theta_switch_observer"]
    theta_self_switch_neural_L = results_neural["L"]["theta_switch_actor"]
    theta_other_switch_neural_L = results_neural["L"]["theta_switch_observer"]
    theta_diff_O = pairwise_sub(
        theta_other_switch_neural_O, theta_self_switch_neural_O
    )
    theta_diff_L = pairwise_sub(
        theta_other_switch_neural_L, theta_self_switch_neural_L
    )
    theta_diff_O = pairwise_sub(
        theta_other_switch_neural_O, theta_self_switch_neural_O
    )
    theta_diff_L = pairwise_sub(
        theta_other_switch_neural_L, theta_self_switch_neural_L
    )
    return theta_diff_O, theta_diff_L


def plot_theta_diff(results, results_neural, model_type):
    # plot the difference between self-output and other-output angles
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection="polar"))
    (
        theta_self_other,
        theta_self_out,
        theta_other_out,
        theta_self_out_ramp,
        theta_other_out_ramp,
    ) = extract_thetas_from_results(results)
    theta_diff = pairwise_sub(theta_other_out, theta_self_out)
    # neural data
    theta_diff_O, theta_diff_L = extract_thetas_switch_neural(results_neural)
    # plot the difference between self-output and other-output angles
    prob1 = plot_histogram(
        ax, theta_diff, bin_size=3, color="k", label="Network"
    )
    plot_mean_and_sd(ax, theta_diff_O, "g")
    plot_mean_and_sd(ax, theta_diff_L, "m")
    # set maximum of radial axis to be the maximum in the network data
    max_prob = max(prob1)
    ax.set_rlim(0, max_prob * 1.1)
    return fig


def plot_thetas_over_time(results):
    thetas_self_other = []
    thetas_self_out = []
    thetas_other_out = []
    n_tpts = 20
    for timestep in range(n_tpts):
        (
            theta_self_other,
            theta_self_out,
            theta_other_out,
            theta_self_out_ramp,
            theta_other_out_ramp,
        ) = extract_thetas_from_results(
            results, tstart=timestep, tend=timestep + 1
        )
        thetas_self_other.append(theta_self_other)
        thetas_self_out.append(theta_self_out)
        thetas_other_out.append(theta_other_out)
    # convert to mean
    thetas_self_out = [x.mean() for x in thetas_self_out]
    thetas_self_other = [x.mean() for x in thetas_self_other]
    thetas_other_out = [x.mean() for x in thetas_other_out]
    fig, ax = plt.subplots()
    x_time = np.arange(0, n_tpts, 1)
    # line plot
    ax = sns.lineplot(
        x=x_time,
        y=thetas_self_other,
        label="Self-other",
        color="black",
        marker="o",
        linestyle="-",
    )
    ax = sns.lineplot(
        x=x_time,
        y=thetas_self_out,
        label="Self-out",
        color="blue",
        marker="o",
        linestyle="-",
    )
    ax = sns.lineplot(
        x=x_time,
        y=thetas_other_out,
        label="other-out",
        color="red",
        marker="o",
        linestyle="-",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    beutify(ax)
    return fig


def beutify_polar(ax):
    # Set the angle limits
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # set radial to log scale and set rlim to 0.01, 1
    ax.set_rscale("log")
    # ax.set_rlim(0.01, 1)
    # # Set the radial ticks and labels to 0, 0.1, 1
    ax.set_rticks([0.01, 0.1, 1])
    # set the radial limit to [0,2.2]
    ax.set_rlim(0, 2.2)
    ax.set_yticklabels([])  # Remove radial labels
    # Remove theta labels
    ax.set_xticklabels([])
    # Remove specific guidelines at 30 and 60 degrees
    ax.set_thetagrids(
        [0, 90, 180]
    )  # Only show grid lines at 0, 90, and 180 degrees
    ax.grid(True)


def plot_error_bar_polar(ax, data, color_mean, color_sd):
    mean_theta = np.mean(data)
    sd_theta = np.std(data)  # / np.sqrt(len(data))

    r_min = 0.01
    r_max = 1.8
    r_dot = 2.2
    r_min_error = 1
    r_max_error = 1.8

    # Convert to radians
    mean_theta_rad = np.deg2rad(mean_theta)
    sd_theta_rad = np.deg2rad(sd_theta)
    theta1 = mean_theta_rad - sd_theta_rad
    theta2 = mean_theta_rad + sd_theta_rad

    # Plot mean as a vertical line
    ax.vlines(
        mean_theta_rad,
        r_min,
        r_max,
        color=color_mean,
        linewidth=0.5,
        linestyle="--",
        zorder=3,
    )

    # Plot sd as an arc using fill_between
    theta = np.linspace(theta1, theta2, 100)
    ax.fill_between(
        theta,
        r_min_error,
        r_max_error,
        color=color_mean,
        alpha=0.5,
        zorder=2,
        linewidth=0,
    )

    # plot a circle at the mean
    ax.plot(
        mean_theta_rad,
        r_dot,
        marker="o",
        color=color_sd,
        markersize=8,
        zorder=1,
    )


def plot_thetas_for_model_type(results_network, results_neural, model_type):
    (
        theta_self_other,
        theta_self_out,
        theta_other_out,
        theta_self_out_ramp,
        theta_other_out_ramp,
    ) = extract_thetas_from_results(results_network)
    (
        theta_self_other_neural_O,
        theta_self_other_neural_L,
        theta_self_switch_O,
        theta_other_switch_O,
        theta_self_switch_L,
        theta_other_switch_L,
    ) = extract_thetas_neural(results_neural)

    # print stats of thetas
    mean_self_other = np.mean(theta_self_other)
    std_self_other = np.std(theta_self_other)
    mean_self_out = np.mean(theta_self_out)
    std_self_out = np.std(theta_self_out)
    mean_other_out = np.mean(theta_other_out)
    std_other_out = np.std(theta_other_out)
    print(
        f"Self-other: {mean_self_other:.2f} ± {std_self_other:.2f}, "
        f"Self-out: {mean_self_out:.2f} ± {std_self_out:.2f}, "
        f"Other-out: {mean_other_out:.2f} ± {std_other_out:.2f}"
    )
    nbins = 18
    fig, ax = plt.subplots(
        figsize=(4.7, 4.7), subplot_kw=dict(projection="polar")
    )

    # plot histogram for self output angle
    # prob1 = plot_mean_and_sd(ax, theta_self_out, color="b")
    prob1 = plot_histogram(
        ax, theta_self_out, bin_size=3, color="b", label="Network"
    )
    # plot histogram for other output angle
    # prob2 = plot_mean_and_sd(ax, theta_other_out, color="r")
    prob2 = plot_histogram(
        ax, theta_other_out, bin_size=3, color="r", label="Network"
    )

    # plot mean as a circle at r=0.1
    ax.plot(np.deg2rad(mean_self_out), 0.1, marker="o", color="b", markersize=8)
    ax.plot(
        np.deg2rad(mean_other_out), 0.1, marker="o", color="r", markersize=8
    )

    # plot neural data
    plot_error_bar_polar(ax, theta_self_switch_O, color_mean="b", color_sd="g")
    plot_error_bar_polar(ax, theta_other_switch_O, color_mean="r", color_sd="g")
    plot_error_bar_polar(ax, theta_self_switch_L, color_mean="b", color_sd="m")
    plot_error_bar_polar(ax, theta_other_switch_L, color_mean="r", color_sd="m")
    beutify_polar(ax)

    plt.tight_layout()
    return fig
