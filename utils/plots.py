"""
This module contains functions for plotting results.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns


def beutify(ax, direction="left"):
    # Make side and bottom spines more prominent by increasing their width
    sns.set_theme(font_scale=0.8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_facecolor("white")
    ax.grid(visible=True, which="major", axis="y", color="0.9", linestyle="-")
    # make sure grid is behind the plot
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["right"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Separate the x and y axis by setting the spine positions
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["right"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Adjust the ticks to be outside and include both major and minor ticks
    ax.tick_params(
        axis="both", direction="out", length=6, width=0.8, which="major"
    )
    ax.tick_params(
        axis="both", direction="out", length=4, width=0.8, which="minor"
    )

    # Set the ticks to only appear on the bottom and direction sides
    if direction == "left":
        ax.yaxis.tick_left()
        ax.spines["right"].set_visible(False)
    else:
        ax.yaxis.tick_right()
        ax.spines["left"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.spines["top"].set_visible(False)


def align_zero_axes(ax, ax2):
    ylim1 = ax.get_ylim()
    ylim2 = ax2.get_ylim()

    range1 = ylim1[1] - ylim1[0]
    range2 = ylim2[1] - ylim2[0]

    # Calculate the scale factors to align zeros
    scale1 = ylim1[1] / range1
    scale2 = ylim2[1] / range2

    if ylim1[0] >= 0 and ylim2[0] >= 0:
        # Both axes start at zero or above
        ax.set_ylim(0, ylim1[1])
        ax2.set_ylim(0, ylim2[1])
    elif ylim1[0] < 0 and ylim2[0] < 0:
        # Both axes include zero
        ratio = max(scale1, scale2)
        ax.set_ylim(ylim1[0] * ratio / scale1, ylim1[1] * ratio / scale1)
        ax2.set_ylim(ylim2[0] * ratio / scale2, ylim2[1] * ratio / scale2)
    elif ylim1[0] < 0:
        # ax includes zero, ax2 is positive
        ax2_zero = ylim2[0] * scale1 / scale2
        ax2.set_ylim(ax2_zero, ylim2[1])
        ax.set_ylim(ylim1[0], ylim1[1])
    else:
        # ax2 includes zero, ax is positive
        ax_zero = ylim1[0] * scale2 / scale1
        ax.set_ylim(ax_zero, ylim1[1])
        ax2.set_ylim(ylim2[0], ylim2[1])


def plot_results(x, y1, yerr1, y2, yerr2, filename):
    fig, ax = plt.subplots(figsize=(5, 4))

    cmap1 = cm.get_cmap("Reds")  # Get the 'Reds' colormap
    cmap2 = cm.get_cmap("Blues")  # Get the 'Blues' colormap
    colors_actor = cmap1(np.linspace(0.3, 1, len(x)))
    colors_observer = cmap1(np.linspace(0.3, 1, len(x)))
    markers = ["o", "^", "s"]
    for i in range(len(x)):
        # plot actor (all)
        ax.errorbar(
            x[i],
            y1[i],
            yerr=yerr1[i],
            fmt=markers[0],
            color=colors_actor[i],
            markersize=15,
        )
        ax.errorbar(
            x[i] + 0.2,
            y2[i],
            yerr=yerr2[i],
            fmt=markers[1],
            color=colors_observer[i],
            markersize=15,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    plt.savefig(filename, dpi=300)
    plt.close()
