"""
This script plots heatmap of eye gaze positions during collect phase of task.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap_from_trials(X, Y):
    """
    Plots a 2D histogram (heatmap) of all (x, y) points aggregated from the
    list-like objects X, Y. Returns the figure and axes handle for optional saving.
    """
    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Concatenate all X and Y arrays
    x = np.concatenate(X)
    y = np.concatenate(Y)

    bins = 20
    xlim = (-0.5, 1.5)
    ylim = (-0.5, 1.5)
    black_cutoff = 1e-5
    n_points = len(x)
    weights = np.ones_like(x) / n_points  # for proportion

    # First, do a quick call to np.histogram2d to find the max bin value.
    # (Alternatively, you could rely on the counts from hist2d directly.)
    hist, _, _ = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim], weights=weights
    )
    max_val = hist.max()  # This is the highest bin value (proportion or raw)

    # Create a copy of the viridis colormap and set the "under" color to black
    # so that any bin < black_cutoff is painted black.
    cmap_custom = plt.get_cmap("viridis").copy()
    cmap_custom.set_under("black")

    # Now plot the 2D histogram using LogNorm.
    im = ax.hist2d(
        x,
        y,
        bins=bins,
        range=[xlim, ylim],
        cmap="viridis",
        weights=weights,
    )

    # Colorbar
    cb = fig.colorbar(im[3], ax=ax)
    cb.set_label(
        "Proportion of Points (log scale)"
        if weights is not None
        else "Counts (log scale)"
    )

    # Optional: Turn off minor ticks on log colorbar & set a custom major tick set
    cb.ax.minorticks_off()

    # Axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    ax.set_title(
        f"Log Heatmap with < {black_cutoff} shown in black\n"
        f"Max bin: {max_val:.2e}"
    )

    return fig, ax
