"""
This script loads trained RNN snapshots, computes the Singular Value Decomposition (SVD)
of the recurrent weight matrix for each model, plots the cumulative variance explained,
and writes a .txt report with the range, mean, and 95% confidence interval of the
effective dimensionality.

It also includes a function to overlay the plots of two different model configurations
for direct comparison.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils.plots import beutify
from models.multi_rnn import MultiRNN
from msi_utils import extract_paths, load_checkpoint
from utils.LoadSession import findrootdir

# --- Configuration ---
CHECKPOINT_ID = 100000  # The snapshot ID to load
MAX_NETS_TO_PLOT = 100  # To avoid overly cluttered plots
PLOT_TRUNCATE_AT = 50  # Truncate curves for plotting


def compute_svd_for_model(model):
    """
    Accesses the recurrent weight matrix of the model, computes its SVD,
    and returns the singular values.
    """
    try:
        w_hh_tensor = model._rnn_linear.weight
    except AttributeError:
        print("Error: Could not find 'model._rnn_linear.weight'.")
        print(
            "Please check the name of the RNN module inside your MultiRNN class."
        )
        return None

    w_hh_numpy = w_hh_tensor.detach().cpu().numpy()
    singular_values = np.linalg.svd(w_hh_numpy, compute_uv=False)
    return singular_values


def get_cumulative_variances_for_config(id, model_type, require_grad):
    """
    Loads all snapshots for a given configuration and returns their cumulative
    variance curves and effective dimensionalities.

    Returns:
        tuple: (list_of_cumvar_curves, list_of_eff_dims, n_nets_plotted)
    """
    root_dir = findrootdir()
    checkpoint_path_base = f"{root_dir}/rnn_snapshots/{id}"
    paths_with_ids = extract_paths(
        checkpoint_path_base, type=model_type, require_grad=require_grad
    )

    if not paths_with_ids:
        print(
            f"No models found for config: id={id}, type={model_type}, grad={require_grad}. Skipping."
        )
        return [], [], 0

    all_cumvar_curves = []
    eff_dims = []
    n_nets_plotted = 0

    for log_path, run_id in paths_with_ids:
        if n_nets_plotted >= MAX_NETS_TO_PLOT:
            break

        checkpoint_path = f"{log_path}/snapshots/{CHECKPOINT_ID}"
        if not os.path.exists(checkpoint_path):
            continue

        model = MultiRNN(
            hidden_size=200, input_feature_len=3, activity_decay=0.2
        )
        try:
            load_checkpoint(model, checkpoint_path)
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}. Error: {e}")
            continue

        singular_values = compute_svd_for_model(model)
        if singular_values is None:
            continue

        # 1. Compute the FULL cumulative variance
        variances = singular_values**2
        variance_explained = variances / np.sum(variances)
        full_cumvar_pct = np.cumsum(variance_explained) * 100

        # 2. Use the FULL curve to calculate effective dimension correctly
        idx80 = np.where(full_cumvar_pct >= 80)[0]
        eff_dim = int(idx80[0]) + 1 if idx80.size else len(full_cumvar_pct)
        eff_dims.append(eff_dim)

        # 3. Truncate the curve before adding to the list for plotting
        truncated_curve = full_cumvar_pct[:PLOT_TRUNCATE_AT]
        all_cumvar_curves.append(truncated_curve)

        idx80 = np.where(truncated_curve >= 80)[0]
        eff_dim = int(idx80[0]) + 1 if idx80.size else len(truncated_curve)
        eff_dims.append(eff_dim)

        n_nets_plotted += 1

    return all_cumvar_curves, eff_dims, n_nets_plotted


def main(model_type, require_grad, id="39553470"):
    """
    Process one configuration: compute & plot cumulative variance explained,
    then save a PDF and a .txt report of effective dimensionality stats.
    (This function's behavior is unchanged from the user's perspective).
    """
    print(
        f"--- Processing: model_type={model_type}, require_grad={require_grad} ---"
    )

    # Use the new helper function to get the data
    all_cumvar_curves, eff_dims, n_nets_plotted = (
        get_cumulative_variances_for_config(id, model_type, require_grad)
    )

    if n_nets_plotted == 0:
        return

    # Finalize and save plot
    root_dir = findrootdir()
    stats_dir = f"{root_dir}/stats_paper"

    # Compute and write statistics report
    eff = np.array(eff_dims)
    mean_dim = float(np.mean(eff))
    std_err = float(np.std(eff, ddof=1) / np.sqrt(len(eff)))
    ci95 = 1.96 * std_err

    report_filename = f"svd_{id}_{model_type}_{require_grad}_report.txt"
    report_path = os.path.join(stats_dir, report_filename)
    with open(report_path, "w") as f:
        f.write(
            "Effective dimensionality (components to reach 80% cumulative variance explained)\n"
        )
        f.write(f"Range: {int(np.min(eff))} - {int(np.max(eff))}\n")
        f.write(f"Mean: {mean_dim:.2f}\n")
        f.write(
            f"95% confidence interval of mean: [{mean_dim - ci95:.2f}, {mean_dim + ci95:.2f}]\n"
        )
    print(f"Report saved to: {report_path}\n")


def plot_overlay(model_spec1, model_spec2):
    """
    Generates a single plot overlaying the cumulative variance curves
    from two different model specifications.

    Args:
        model_spec1 (tuple): A tuple of (id, model_type, require_grad).
        model_spec2 (tuple): A second tuple for the model to compare against.
    """
    id1, type1, grad1 = model_spec1
    id2, type2, grad2 = model_spec2
    print(f"--- Creating Overlay Plot: {model_spec1} vs {model_spec2} ---")

    # Get data for both models using the helper function
    curves1, _, _ = get_cumulative_variances_for_config(id1, type1, grad1)
    curves2, _, _ = get_cumulative_variances_for_config(id2, type2, grad2)

    if not curves1 and not curves2:
        print(
            "No data found for either model configuration. Skipping overlay plot."
        )
        return

    # Plotting Setup
    fig, ax = plt.subplots(figsize=(4, 4))
    color1, color2 = "dodgerblue", "orangered"
    label1 = f"{type1}, grad={grad1}"
    label2 = f"{type2}, grad={grad2}"

    # Plot curves for model 1
    for i, curve in enumerate(curves1):
        ax.plot(
            curve,
            color=color1,
            alpha=0.3,
            linewidth=1.5,
            label=label1 if i == 0 else "",
        )

    # Plot curves for model 2
    for i, curve in enumerate(curves2):
        ax.plot(
            curve,
            color=color2,
            alpha=0.3,
            linewidth=1.5,
            label=label2 if i == 0 else "",
        )

    # Finalize and save plot
    root_dir = findrootdir()
    fig_dir = f"{root_dir}/plots_paper"
    os.makedirs(fig_dir, exist_ok=True)

    ax.axhline(80, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Cumulative Variance Explained", fontsize=12)
    ax.set_xlabel("Component Index", fontsize=12)
    ax.set_ylabel("Cumulative Variance (%)", fontsize=12)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    beutify(ax)

    plot_filename = f"FigS13D_overlay_{type1}_{grad1}_vs_{type2}_{grad2}.pdf"
    plot_path = os.path.join(fig_dir, plot_filename)
    rcParams["pdf.fonttype"] = 42
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay plot saved to: {plot_path}\n")


if __name__ == "__main__":
    # --- Part 1: Run the original, individual analysis ---
    print("--- Running Individual Model Analyses ---")
    for model_type in ["ortho", "paral"]:
        for require_grad in ["True", "False"]:
            main(
                model_type=model_type, require_grad=require_grad, id="39553470"
            )
        main(model_type=model_type, require_grad=model_type, id="41838359")
        main(model_type=model_type, require_grad=model_type, id="41905874")

    # --- Part 2: Run the new overlay plots for comparison ---
    print("\n--- Running Overlay Comparison Plots ---")

    # Example 1: Compare 'ortho' models with grad=True vs grad=False
    spec1 = ("41838359", "ortho", "ortho")
    spec2 = ("41838359", "paral", "paral")
    plot_overlay(spec1, spec2)

    print("--- SVD analysis complete. ---")
