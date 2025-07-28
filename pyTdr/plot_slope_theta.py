# For each session, plot the theta(actor, switch) on the x-axis
# and the slope (switch, history) on the y axis.
# Plot the Actor slope of one animal against Observer slope of the other.

import os
import sys
import pandas as pd
import glob
import numpy as np
import json
from scipy.stats import sem, t
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.LoadSession import findrootdir
from utils.plots import beutify
from single_unit.tls_regression import fit_with_weighted_tls
import statsmodels.api as sm


def generate_vectors_with_angle(dims, angle_deg):
    """
    Generates two random unit vectors in a space of `dims` dimensions
    with a specific angle between them.
    """
    angle_rad = np.deg2rad(angle_deg)
    v1 = np.random.randn(dims)
    v1 = v1 / np.linalg.norm(v1)
    v2_initial = np.random.randn(dims)
    v2_ortho = v2_initial - np.dot(v1, v2_initial) * v1
    v2_ortho = v2_ortho / np.linalg.norm(v2_ortho)
    v2 = np.cos(angle_rad) * v1 + np.sin(angle_rad) * v2_ortho
    return v1, v2


def simulate_subsampled_angles(
    total_dims, subsample_dims, original_angle_deg, num_trials=5000
):
    """
    Simulates the distribution of angles between two vectors after
    randomly subsampling their dimensions.
    """
    if subsample_dims > total_dims:
        raise ValueError(
            "Subsample dimensions cannot be greater than total dimensions."
        )
    if subsample_dims == total_dims:
        return 0

    v1, v2 = generate_vectors_with_angle(total_dims, original_angle_deg)
    all_indices = np.arange(total_dims)
    calculated_angles = []

    for _ in range(num_trials):
        subsample_indices = np.random.choice(
            all_indices, size=subsample_dims, replace=False
        )
        v1_sub, v2_sub = v1[subsample_indices], v2[subsample_indices]

        dot_product = np.dot(v1_sub, v2_sub)
        norm_v1 = np.linalg.norm(v1_sub)
        norm_v2 = np.linalg.norm(v2_sub)

        if norm_v1 == 0 or norm_v2 == 0:
            continue

        cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(cosine_angle))
        calculated_angles.append(angle_deg)

    return np.std(calculated_angles)


def load_theta_by_session(data_dir, animal, event):
    """
    Load theta data for a specific animal and event from the given data directory.

    :param data_dir: Directory containing the theta data files.
    :param animal: Animal identifier (e.g., 'O').
    :param event: Event type (e.g., 'fdbk').
    :return: DataFrame containing the loaded theta data.
    """
    file_pattern = os.path.join(
        data_dir, f"{animal}_{event}_act_obs_dimensions_*.json"
    )
    files = glob.glob(file_pattern)
    if not files:
        print(
            f"Warning: No theta files found for pattern {file_pattern}. Plot may be empty."
        )
        return pd.DataFrame(
            columns=["session", "theta_switch_actor", "theta_switch_observer"]
        )

    records = []
    for f in files:
        # if file ends with 'NR.json', skip it
        if f.endswith("NR.json"):
            continue
        with open(f, "r") as infile:
            try:
                data = json.load(infile)
            except json.JSONDecodeError:
                infile.seek(0)
                data = eval(infile.read())
            filename = os.path.basename(f)
            session = filename.split("_")[-1].replace(".json", "")
            actor_theta = data.get("theta_switch_actor")
            observer_theta = data.get("theta_switch_observer")
            unit_idx = data.get("unit_idx_master", None)
            if len(unit_idx) < 10:
                continue
            if actor_theta is not None and observer_theta is not None:
                records.append(
                    {
                        "session": session,
                        "theta_switch_actor": actor_theta,
                        "theta_switch_observer": observer_theta,
                        "n_units": len(unit_idx),
                    }
                )
    if not records:
        print(
            "Warning: No valid theta_switch_actor and theta_switch_observer data found in files."
        )
        return pd.DataFrame(
            columns=["session", "theta_switch_actor", "theta_switch_observer"]
        )

    return pd.DataFrame.from_records(records)


def load_slopes(animal, event):
    """
    Load slope data for a specific animal and event.

    :param animal: Animal identifier (e.g., 'O').
    :param event: Event type (e.g., 'fdbk').
    :return: DataFrame containing the loaded slope data.
    """
    animal_name = "Offenbach" if animal == "O" else "Lalo"
    root_dir = findrootdir()
    data_dir = os.path.join(root_dir, "stats_paper")
    file_path = os.path.join(data_dir, f"slopes_{animal_name}.csv")

    if not os.path.exists(file_path):
        print(f"Warning: Slope file not found: {file_path}. Plot may be empty.")
        return pd.DataFrame(
            columns=["session", "slopes_actor", "slopes_observer"]
        )

    return pd.read_csv(file_path)


def get_all_merged_data(animal, event, slope_only=False):
    """
    Loads and merges theta, slope, and stats data into a single DataFrame.
    """
    root_dir = findrootdir()
    data_dir = os.path.join(root_dir, "stats_paper")

    # Load p_switch data for filtering
    tstart, tend = 0.0, 0.6
    formatted_window = [round(tstart, 1), round(tend, 1)]
    stats_file = os.path.join(
        data_dir, f"{animal}_{event}_{formatted_window}_switch_dir_stats.json"
    )
    try:
        with open(stats_file, "r") as f:
            stat_session = json.load(f)
        stat_df = pd.DataFrame(
            [
                {"session": sess, "pval_switch": data["pval_ranksum"]}
                for sess, data in stat_session.items()
            ]
        )
    except FileNotFoundError:
        print(
            f"Warning: Stats file not found: {stats_file}. Cannot filter by p-value."
        )
        stat_df = pd.DataFrame()

    # Load primary data
    theta_df = load_theta_by_session(data_dir, animal, event)
    slope_df = load_slopes(animal, event)

    if theta_df.empty or slope_df.empty:
        print("Theta or Slope DataFrame is empty. Cannot proceed with merging.")
        return pd.DataFrame()

    theta_df["session"] = theta_df["session"].astype(str)
    slope_df["session"] = slope_df["session"].astype(str)

    # If only slope data is needed, return the slope DataFrame
    if slope_only:
        return slope_df

    # Merge dataframes
    merged_df = pd.merge(theta_df, slope_df, on="session", how="inner")

    if not stat_df.empty:
        stat_df["session"] = stat_df["session"].astype(str)
        merged_df = pd.merge(merged_df, stat_df, on="session", how="inner")

    return merged_df


def process_role_data(merged_df, role):
    """
    From a merged DataFrame, process data for a specific role to get plotting values.
    Partitions data into significant (p<0.05) and non-significant (p>=0.05) groups.
    """
    if merged_df.empty:
        return (([], [], [], []), ([], [], [], []))

    # Create lists for both significant and non-significant data
    sig_means, sig_lowers, sig_uppers, sig_slopes, sig_n_units = (
        [],
        [],
        [],
        [],
        [],
    )
    nsig_means, nsig_lowers, nsig_uppers, nsig_slopes, nsig_n_units = (
        [],
        [],
        [],
        [],
        [],
    )

    for idx, row in merged_df.iterrows():
        theta_col, slope_col = f"theta_switch_{role}", f"slopes_{role}"

        # Check for column existence and NaN values robustly
        if (
            theta_col not in row
            or slope_col not in row
            or pd.isna(row[slope_col])
            or not isinstance(row[theta_col], (list, np.ndarray))
            or len(row[theta_col]) == 0
        ):
            continue

        theta_arr = np.array(row[theta_col])
        theta_arr = theta_arr[~np.isnan(theta_arr)]
        if theta_arr.size == 0:
            continue

        n = len(theta_arr)
        mean = np.mean(theta_arr)
        std_err = sem(theta_arr)
        ci = std_err * t.ppf(0.975, n - 1) if n > 1 else 0

        # Decide which group the data point belongs to.
        is_significant = "pval_switch" in row and row["pval_switch"] < 0.05

        if is_significant:
            sig_means.append(mean)
            sig_lowers.append(mean - ci)
            sig_uppers.append(mean + ci)
            sig_slopes.append(row[slope_col])
            sig_n_units.append(row["n_units"])
        else:
            nsig_means.append(mean)
            nsig_lowers.append(mean - ci)
            nsig_uppers.append(mean + ci)
            nsig_slopes.append(row[slope_col])
            nsig_n_units.append(row["n_units"])

    sig_data = (sig_means, sig_lowers, sig_uppers, sig_slopes, sig_n_units)
    nsig_data = (
        nsig_means,
        nsig_lowers,
        nsig_uppers,
        nsig_slopes,
        nsig_n_units,
    )

    return sig_data, nsig_data


def save_role_data(animal, event, role="actor"):
    """
    This function reproduces the original script's behavior:
    1. Loads and merges all data.
    2. Saves the merged data to a role-specific CSV file.
    3. Creates and saves a plot for that single role.
    """
    root_dir = findrootdir()
    data_dir = os.path.join(root_dir, "stats_paper")

    if animal == "both":
        df_1 = get_all_merged_data("O", event)
        df_2 = get_all_merged_data("L", event)
        merged_df = pd.concat([df_1, df_2], ignore_index=True)
    else:
        merged_df = get_all_merged_data(animal, event)

    if merged_df.empty:
        print(f"No data to process for {animal}, {event}, {role}.")
        return

    # Save merged data to CSV (as in original script)
    csv_path = os.path.join(
        data_dir, f"{animal}_{event}_{role}_theta_slope.csv"
    )
    merged_df.to_csv(csv_path, index=False)
    print(f"Saved merged data to {csv_path}")


def process_diff_data(merged_df):
    """
    Process the merged DataFrame to compute the difference between actor and observer data.
    Returns the differences in theta and slope.
    """
    if merged_df.empty:
        return ([], [])

    # First, create column in df for the difference in theta and slope
    merged_df["theta_diff"] = np.mean(
        np.array(merged_df["theta_switch_actor"].tolist()), axis=1
    ) - np.mean(np.array(merged_df["theta_switch_observer"].tolist()), axis=1)
    merged_df["slope_diff"] = np.array(merged_df["slopes_actor"]) - np.array(
        merged_df["slopes_observer"]
    )
    # Next, select rows with significant p-values
    sig_df = merged_df[merged_df["pval_switch"] < 0.05]
    if sig_df.empty:
        print("No significant data found for difference calculation.")
        return ([], [])
    # Finally, extract the differences
    theta_diff = sig_df["theta_diff"].tolist()
    slope_diff = sig_df["slope_diff"].tolist()
    return (theta_diff, slope_diff)


def plot_combined_roles(animal, event, plot_actor=True, plot_observer=True):
    """
    Generates and saves a new plot combining actor and observer data.
    - Actor data is plotted with solid circles.
    - Observer data is plotted with open circles.
    The plot is saved with high resolution and tight bounding box.
    """
    root_dir = findrootdir()
    data_dir = os.path.join(root_dir, "stats_paper")
    plot_dir = os.path.join(root_dir, "plots_paper")

    if animal == "both":
        df_1 = get_all_merged_data("O", event)
        df_2 = get_all_merged_data("L", event)
        merged_df = pd.concat([df_1, df_2], ignore_index=True)
    else:
        merged_df = get_all_merged_data(animal, event)
    if merged_df.empty:
        print(f"No data to process for combined plot for {animal}, {event}.")
        return

    # Unpack the partitioned data for both roles
    (actor_sig_data, actor_nsig_data) = process_role_data(merged_df, "actor")
    (observer_sig_data, observer_nsig_data) = process_role_data(
        merged_df, "observer"
    )
    if not any(d for d in actor_sig_data) and not any(
        d for d in observer_sig_data
    ):
        return

    # Compute std based on largest n_units for each role
    actor_sig_n_units = np.array(actor_sig_data[-1])
    observer_sig_n_units = np.array(observer_sig_data[-1])
    idx_largest_n_actor, largest_n_actor = np.argmax(actor_sig_n_units), np.max(
        actor_sig_n_units
    )
    idx_largest_n_observer, largest_n_observer = np.argmax(
        observer_sig_n_units
    ), np.max(observer_sig_n_units)
    theta_largest_actor = actor_sig_data[0][idx_largest_n_actor]
    theta_largest_observer = observer_sig_data[0][idx_largest_n_observer]
    # Compute std based on largest n_units and theta
    std_theta_actor = [
        simulate_subsampled_angles(
            total_dims=largest_n_actor,
            subsample_dims=actor_sig_n_units[i],
            original_angle_deg=theta_largest_actor,
            num_trials=5000,
        )
        + 1  # Add 1 to avoid division by zero
        for i in range(len(actor_sig_n_units))
    ]
    # normalize to the smallest std
    std_theta_actor = np.array(std_theta_actor) / np.min(std_theta_actor)
    std_theta_observer = [
        simulate_subsampled_angles(
            total_dims=largest_n_observer,
            subsample_dims=observer_sig_n_units[i],
            original_angle_deg=theta_largest_observer,
            num_trials=5000,
        )
        + 1  # Add 1 to avoid division by zero
        for i in range(len(observer_sig_n_units))
    ]
    # normalize to the smallest std
    std_theta_observer = np.array(std_theta_observer) / np.min(
        std_theta_observer
    )

    # compute the difference between actor and observer data
    theta_diff, slope_diff = process_diff_data(merged_df)

    # Unpack tuples for easier access
    (
        actor_sig_means,
        actor_sig_lowers,
        actor_sig_uppers,
        actor_sig_slopes,
        actor_sig_n_units,
    ) = actor_sig_data
    (
        actor_nsig_means,
        actor_nsig_lowers,
        actor_nsig_uppers,
        actor_nsig_slopes,
        actor_nsig_n_units,
    ) = actor_nsig_data
    (
        observer_sig_means,
        observer_sig_lowers,
        observer_sig_uppers,
        observer_sig_slopes,
        observer_sig_n_units,
    ) = observer_sig_data
    (
        observer_nsig_means,
        observer_nsig_lowers,
        observer_nsig_uppers,
        observer_nsig_slopes,
        observer_nsig_n_units,
    ) = observer_nsig_data

    if not any(
        [
            actor_sig_means,
            actor_nsig_means,
            observer_sig_means,
            observer_nsig_means,
        ]
    ):
        print(
            f"No data points to plot for combined plot for {animal}, {event} after processing."
        )
        return

    # --- PLOTTING SECTION ---
    sns.set_theme(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(2, 2))

    # 1. Plot SIGNIFICANT data on top (in color, with labels)
    if actor_sig_means and plot_actor:
        ax.scatter(
            actor_sig_means,
            actor_sig_slopes,
            color="C0",
            label="Actor",
            edgecolor="black",
            s=50 / std_theta_actor,
        )
    if observer_sig_means and plot_observer:
        ax.scatter(
            observer_sig_means,
            observer_sig_slopes,
            color="C1",
            label="Observer",
            edgecolor="black",
            s=50 / std_theta_observer,
            facecolor="white",
        )
    # plot regression line for combined (sig) data and print r squared
    if (
        actor_sig_means
        and observer_sig_means
        and (plot_actor and plot_observer)
    ):
        combined_means = actor_sig_means + observer_sig_means
        combined_slopes = actor_sig_slopes + observer_sig_slopes
        combined_std = std_theta_actor.tolist() + std_theta_observer.tolist()
        inverse_std = 1 / np.array(combined_std)
        combined_n_units = (
            actor_sig_n_units + observer_sig_n_units
        )  # Combine n_units for both roles
        combined_data = pd.DataFrame(
            {
                "theta": combined_means,
                "slope": combined_slopes,
                "n_units": combined_n_units,
                "inverse_std": inverse_std,
            }
        )

        fit_with_weighted_tls(
            combined_data,
            x_col="theta",
            y_col="slope",
            weights_col="inverse_std",
            ax=ax,
            title="Fit (Combined Actor & Observer)",
            alternative="less",
        )
    elif actor_sig_means and plot_actor:
        combined_data = pd.DataFrame(
            {
                "theta": actor_sig_means,
                "slope": actor_sig_slopes,
                "n_units": actor_sig_n_units,
                "inverse_std": 1 / np.array(std_theta_actor),
            }
        )
        fit_with_weighted_tls(
            combined_data,
            x_col="theta",
            y_col="slope",
            weights_col="inverse_std",
            ax=ax,
            title="Fit (Actor Only)",
            alternative="less",
        )
    elif observer_sig_means and plot_observer:
        combined_data = pd.DataFrame(
            {
                "theta": observer_sig_means,
                "slope": observer_sig_slopes,
                "n_units": observer_sig_n_units,
                "inverse_std": 1 / np.array(std_theta_observer),
            }
        )
        fit_with_weighted_tls(
            combined_data,
            x_col="theta",
            y_col="slope",
            weights_col="inverse_std",
            ax=ax,
            title="Fit (Observer Only)",
            alternative="less",
        )

    # Customize the aesthetics of the plot
    sns.set_theme(font_scale=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_facecolor("white")
    ax.grid(visible=True, which="major", axis="y", color="0.9", linestyle="-")
    beutify(ax)
    ax.set_title(f"Theta vs Slope for {animal}", fontsize=14)
    ax.set_xlabel("Theta (switch)", fontsize=12)
    ax.set_ylabel("Slope (switch, history)", fontsize=12)
    # Save the plot
    prefix_combined = {"O": "FigS14d", "L": "FigS14g", "both": "FigS14a"}
    prefix_act = {"O": "FigS14e", "L": "FigS14h", "both": "FigS14b"}
    prefix_obs = {"O": "FigS14f", "L": "FigS14i", "both": "FigS14c"}
    output_filename = os.path.join(
        plot_dir,
        f"{prefix_combined[animal]}_{animal}_{event}_combined_theta_slope.pdf",
    )
    if plot_actor and not plot_observer:
        output_filename = os.path.join(
            plot_dir,
            f"{prefix_act[animal]}_{animal}_{event}_actor_theta_slope.pdf",
        )
    elif plot_observer and not plot_actor:
        output_filename = os.path.join(
            plot_dir,
            f"{prefix_obs[animal]}_{animal}_{event}_observer_theta_slope.pdf",
        )
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close()

    # plot the difference between actor and observer data in a new plot
    std_theta = std_theta_actor + std_theta_observer
    std_theta = np.array(std_theta) / np.min(std_theta)
    inverse_std = 1 / std_theta
    fig_diff, ax_diff = plt.subplots(figsize=(2, 2))
    ax_diff.scatter(
        theta_diff, slope_diff, color="C2", edgecolor="black", s=50 / std_theta
    )

    # Fit a linear regression model
    diff_data = pd.DataFrame(
        {
            "theta_diff": theta_diff,
            "slope_diff": slope_diff,
            "inverse_std": inverse_std,
        }
    )
    fit_with_weighted_tls(
        diff_data,
        x_col="theta_diff",
        y_col="slope_diff",
        weights_col="inverse_std",
        ax=ax_diff,
        title="Fit (Difference)",
    )

    ax_diff.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax_diff.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax_diff.set_xlabel("Theta (Actor - Observer)", fontsize=12)
    ax_diff.set_ylabel("Slope (Actor - Observer)", fontsize=12)
    ax_diff.set_title(f"Difference in Theta vs Slope for {animal}", fontsize=14)
    ax_diff.spines["right"].set_visible(False)
    ax_diff.spines["top"].set_visible(False)
    ax_diff.spines["left"].set_color("black")
    ax_diff.spines["bottom"].set_color("black")
    ax_diff.set_facecolor("white")
    beutify(ax_diff)
    output_diff_filename = os.path.join(
        data_dir, f"{animal}_{event}_diff_theta_slope.pdf"
    )
    plt.savefig(output_diff_filename, bbox_inches="tight", dpi=300)
    plt.close(fig_diff)

    print(f"Saved combined plot to {output_filename}")
    print(f"Saved difference plot to {output_diff_filename}")


# --- Function to create a combined cross-animal slope comparison plot ---
def plot_cross_animal_slope_comparison(event):
    """
    Generates one plot comparing slope values across animals for the same sessions,
    with both conditions on the same axes.
    - O(Actor) vs. L(Observer) is plotted in green.
    - L(Actor) vs. O(Observer) is plotted in magenta.
    """
    root_dir = findrootdir()
    data_dir = os.path.join(root_dir, "stats_paper")
    plot_dir = os.path.join(root_dir, "plots_paper")
    print("\n[INFO] Generating combined cross-animal slope comparison plot...")

    # --- 1. Load data for both animals ---
    df_O = get_all_merged_data("O", event, slope_only=True)
    df_L = get_all_merged_data("L", event, slope_only=True)

    if df_O.empty or df_L.empty:
        print(
            "Data for one or both animals is missing. Cannot create cross-animal plots."
        )
        return

    # --- 2. Helper function to process data ---
    def prep_slope_df(df, animal_char):
        df_copy = df.copy()
        df_actor = (
            df_copy[["session", "slopes_actor"]]
            .dropna()
            .rename(columns={"slopes_actor": f"slope_actor_{animal_char}"})
        )
        df_observer = (
            df_copy[["session", "slopes_observer"]]
            .dropna()
            .rename(
                columns={"slopes_observer": f"slope_observer_{animal_char}"}
            )
        )
        return df_actor, df_observer

    actor_O_df, observer_O_df = prep_slope_df(df_O, "O")
    actor_L_df, observer_L_df = prep_slope_df(df_L, "L")

    # --- 3. Merge data for both comparisons ---
    merged_O_actor_L_observer = pd.merge(
        actor_O_df, observer_L_df, on="session", how="inner"
    )
    merged_L_actor_O_observer = pd.merge(
        actor_L_df, observer_O_df, on="session", how="inner"
    )

    if merged_O_actor_L_observer.empty and merged_L_actor_O_observer.empty:
        print(
            "No common significant sessions found for cross-animal slope plot."
        )
        return

    # --- 4. Create a single plot for both comparisons ---
    fig, ax = plt.subplots(figsize=(2, 2))

    # Plot O(Actor) vs L(Observer) in GREEN
    if not merged_O_actor_L_observer.empty:
        ax.scatter(
            merged_O_actor_L_observer["slope_actor_O"],
            merged_O_actor_L_observer["slope_observer_L"],
            color="green",
            edgecolor="black",
            s=50,
            label="O (Actor) vs L (Observer)",
        )
        fit_with_weighted_tls(
            merged_O_actor_L_observer,
            x_col="slope_actor_O",
            y_col="slope_observer_L",
            weights_col=None,
            ax=ax,
            alternative="greater",
        )

    # Plot L(Actor) vs O(Observer) in MAGENTA
    if not merged_L_actor_O_observer.empty:
        ax.scatter(
            merged_L_actor_O_observer["slope_actor_L"],
            merged_L_actor_O_observer["slope_observer_O"],
            color="magenta",
            edgecolor="black",
            s=50,
            label="L (Actor) vs O (Observer)",
        )
        fit_with_weighted_tls(
            merged_L_actor_O_observer,
            x_col="slope_actor_L",
            y_col="slope_observer_O",
            weights_col=None,
            ax=ax,
        )

    # --- 5. Beautify and save the combined plot ---
    beutify(ax)
    ax.set_xlabel("Slope (Actor)", fontsize=12)
    ax.set_ylabel("Slope (Observer)", fontsize=12)
    ax.set_title("Cross-Animal Slope Comparison", fontsize=14)
    ax.legend()

    output_filename = os.path.join(
        plot_dir, f"FigS14j_{event}_cross_animal_slope_comparison_combined.pdf"
    )
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved combined cross-animal plot to {output_filename}")


class ReportTee:
    """A context manager to write stdout to a file and the console."""

    def __init__(self, filepath, mode="w"):
        self.file = open(filepath, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


if __name__ == "__main__":
    # --- 1. Setup File Paths and Reporting ---
    root_dir = findrootdir()
    output_dir = os.path.join(root_dir, "stats_paper")
    os.makedirs(output_dir, exist_ok=True)
    report_filepath = os.path.join(output_dir, "analysis_report.txt")

    # fix random seed for reproducibility
    np.random.seed(0)

    # --- 2. Main Execution with Report Generation ---
    with ReportTee(report_filepath) as report:
        print(f"--- Analysis Report Generated on {pd.Timestamp.now()} ---\n")

        animals = ["O", "L", "both"]
        event = "fdbk"
        roles = ["actor", "observer"]

        # --- Generate original plots and data files ---
        print("--- Generating single-role data files ---")
        for animal in animals:
            print(f"\n----- Processing Animal: {animal} -----")
            for role in roles:
                save_role_data(animal, event, role)

        print("\n" + "=" * 80 + "\n")

        # --- Generate new combined plots (one for each animal) ---
        print("--- Generating combined-role and difference plots ---")
        for animal in animals:
            print(f"\n----- Analysis for Animal: {animal} -----")

            print("\n[INFO] Generating plot: Combined Actor & Observer")
            plot_combined_roles(animal, event)

            print("\n[INFO] Generating plot: Actor-Only")
            plot_combined_roles(
                animal,
                event,
                plot_actor=True,
                plot_observer=False,
            )

            print("\n[INFO] Generating plot: Observer-Only")
            plot_combined_roles(
                animal,
                event,
                plot_actor=False,
                plot_observer=True,
            )

        # --- Generate cross-animal plots ---
        plot_cross_animal_slope_comparison(event)

    # Final message to the console after the report is closed
    print(f"\nAnalysis complete. Full report saved to: {report_filepath}")
