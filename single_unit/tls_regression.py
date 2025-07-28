"""
This module contains a function that performs total least squares regression.
"""

import numpy as np
import pandas as pd
from scipy.odr import Model, RealData, ODR
import matplotlib.pyplot as plt
from scipy import stats


def tls_regression(x, y):
    # Define the model function for the regression line (linear in this case)
    def linear_model(B, x):
        return B[0] * x + B[1]

    # Function to calculate orthogonal distances from points to a line
    def orthogonal_distances(x, y, B):
        m, c = B
        distances = np.abs(m * x - y + c) / np.sqrt(m**2 + 1)
        return np.sum(distances**2)

    # Create a Model object and RealData object
    model = Model(linear_model)
    data = RealData(x, y)

    # Set up ODR with the model and data
    odr = ODR(data, model, beta0=[1.0, 2.0])

    # Run the regression
    out = odr.run()

    # Calculate the total and residual orthogonal variances
    x_mean, y_mean = np.mean(x), np.mean(y)
    total_variance = orthogonal_distances(x, y, [0, y_mean])
    residual_variance = orthogonal_distances(x, y, out.beta)
    explained_variance = total_variance - residual_variance
    explained_variance_ratio = explained_variance / total_variance

    # Prepare the plot
    x_plot = np.linspace(min(x), max(x), 100)
    y_plot = linear_model(out.beta, x_plot)

    fig = plt.figure(figsize=(2, 2))
    plt.scatter(x, y, color="black", label="")
    plt.plot(
        x_plot,
        y_plot,
        color="blue",
        label=f"Slope: {out.beta[0]:.2f}\nExplained Variance Ratio: {explained_variance_ratio:.2f}",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Total Least Squares Regression with Explained Variance")
    plt.legend()
    return fig, out.beta, out.sd_beta, explained_variance_ratio


def weighted_total_least_squares(x, y, w):
    """
    Performs Weighted Total Least Squares (Orthogonal Regression) for a line.

    Finds the coefficients (a, b, c) of the line a*x + b*y + c = 0 that
    minimizes the weighted sum of squared perpendicular distances from the
    points (x, y) to the line.

    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.
        w (array-like): The weights associated with each data point.

    Returns:
        tuple: The coefficients (a, b, c) of the line.
    """
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)

    sum_w = np.sum(w)
    if sum_w == 0:
        raise ValueError("Sum of weights cannot be zero.")

    x_mean_w = np.sum(w * x) / sum_w
    y_mean_w = np.sum(w * y) / sum_w

    x_centered = x - x_mean_w
    y_centered = y - y_mean_w

    sqrt_w = np.sqrt(w)
    A = np.vstack([sqrt_w * x_centered, sqrt_w * y_centered]).T

    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    a, b = Vh[-1, :]

    c = -(a * x_mean_w + b * y_mean_w)

    return a, b, c


def _bootstrap_tls_slope(x, y, w, n_bootstraps=1000):
    """
    Estimates the standard error of the TLS slope using bootstrapping.
    """
    n_points = len(x)
    bootstrap_slopes = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        # Resample the data with replacement
        indices = np.random.choice(range(n_points), size=n_points, replace=True)
        x_sample, y_sample, w_sample = x[indices], y[indices], w[indices]

        # In rare cases, a bootstrap sample might be all identical points or a vertical line
        # which would make TLS fail. We'll handle this.
        if np.all(x_sample == x_sample[0]) or np.all(y_sample == y_sample[0]):
            bootstrap_slopes[i] = np.nan  # Mark as invalid
            continue

        try:
            a, b, c = weighted_total_least_squares(x_sample, y_sample, w_sample)
            if abs(b) < 1e-9:  # Avoid division by zero for vertical lines
                bootstrap_slopes[i] = np.nan
            else:
                bootstrap_slopes[i] = -a / b
        except (np.linalg.LinAlgError, ValueError):
            bootstrap_slopes[i] = np.nan  # Mark as invalid if TLS fails

    # Calculate standard error from the valid bootstrap results
    valid_slopes = bootstrap_slopes[~np.isnan(bootstrap_slopes)]
    if len(valid_slopes) < 2:
        return np.nan, np.nan  # Cannot compute std error or p-value

    slope_se = np.std(
        valid_slopes, ddof=1
    )  # ddof=1 for sample standard deviation

    return slope_se


def fit_with_weighted_tls(
    df,
    x_col="theta",
    y_col="slope",
    weights_col=None,
    ax=None,
    title="",
    calculate_p_value=True,  # Control for this potentially slow calculation
    n_bootstraps=1000,  # Number of bootstrap iterations
    alternative="two-sided",  # NEW: 'two-sided', 'greater', or 'less'
):
    """
    Fits a line using Weighted Total Least Squares (TLS) and estimates a p-value
    for the slope using bootstrapping.

    Args:
        df (pd.DataFrame): The input data.
        x_col (str): Column for the independent variable (x).
        y_col (str): Column for the dependent variable (y).
        weights_col (str, optional): Column for weights. If None, uses equal weights.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        title (str, optional): A title for context.
        calculate_p_value (bool): If True, performs bootstrapping to estimate the
                                  slope's p-value. Can be slow for large datasets.
        n_bootstraps (int): Number of bootstrap samples to generate.
        alternative (str): Specifies the alternative hypothesis for the p-value calculation.
        Must be one of {"two-sided", "greater", "less"}.

    Returns:
        dict: A dictionary named 'model' with fitted coefficients and statistics.
    """
    # ---------- 1. Prepare data ---------------------------------------------
    x = df[x_col].values
    y = df[y_col].values
    weights = (
        df[weights_col].values
        if weights_col and weights_col in df.columns
        else np.ones_like(x)
    )

    # ---------- 2. Fit the main TLS model -----------------------------------
    a, b, c = weighted_total_least_squares(x, y, weights)

    slope_se = None
    p_value = None

    if abs(b) < 1e-9:
        slope = np.inf
        intercept = -c / a
        pseudo_r2 = 0
    else:
        slope = -a / b
        intercept = -c / b
        y_pred = slope * x + intercept
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        pseudo_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # ---------- 3. Calculate p-value via bootstrapping ------------------
        if calculate_p_value:
            slope_se = _bootstrap_tls_slope(x, y, weights, n_bootstraps)

            if slope_se is not None and not np.isnan(slope_se) and slope_se > 0:
                # Perform a t-test for the null hypothesis that the slope is 0
                t_statistic = slope / slope_se
                # Degrees of freedom is n-2 (n data points, 2 estimated parameters for a line)
                dof = len(x) - 2
                if alternative == "two-sided":
                    p_value = 2 * stats.t.sf(np.abs(t_statistic), df=dof)
                elif alternative == "greater":
                    p_value = stats.t.sf(t_statistic, df=dof)
                elif alternative == "less":
                    p_value = stats.t.cdf(t_statistic, df=dof)
            else:
                p_value = np.nan  # Could not be calculated

    # Create the model dictionary to return
    model_results = {
        "a": a,
        "b": b,
        "c": c,
        "slope": slope,
        "intercept": intercept,
        "slope_std_err": slope_se,
        "p_value_slope": p_value,
        "pseudo_r2": pseudo_r2,
    }

    # ---------- 4. Plot --------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_range = np.linspace(x_min, x_max, 100)

    # Build the label string
    label = f"TLS Fit (pseudo-R²={pseudo_r2:.2f}"
    if p_value is not None and not np.isnan(p_value):
        label += f", p={p_value:.2g}"
    label += ")"

    if np.isinf(slope):
        ax.axvline(
            x=intercept,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Vertical Fit (x={intercept:.2f})",
        )
    else:
        y_range = slope * x_range + intercept
        ax.plot(
            x_range,
            y_range,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ---------- 5. Console report ---------------------------------------------
    prefix = f"[{title}] " if title else ""
    p_string = (
        f"{p_value:.3g}"
        if p_value is not None and not np.isnan(p_value)
        else "N/A"
    )
    print(
        f"{prefix}TLS Fit | pseudo-R²: {pseudo_r2:.3f} | slope: {slope:.3f} | intercept: {intercept:.3f} "
        f"| p-value (slope): {p_string}"
    )

    return model_results
