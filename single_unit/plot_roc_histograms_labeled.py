"""
This script generates scatter plot and histograms of ROC selectivity
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from tls_regression import tls_regression
from utils.LoadSession import findrootdir
from matplotlib_venn import venn2

pd.options.mode.chained_assignment = None


def make_plots_combined(
    data, factor, event, window, subject, example_L=None, example_O=None
):
    mpl.rcParams["pdf.fonttype"] = 42
    field_self_selectivity = f"{event}_{factor}_{window}_selectivity_self"
    field_other_selectivity = f"{event}_{factor}_{window}_selectivity_other"
    field_n_self = f"{event}_{factor}_{window}_n_self"
    field_n_other = f"{event}_{factor}_{window}_n_other"
    field_p_self = f"{event}_{factor}_{window}_p_self"
    field_p_other = f"{event}_{factor}_{window}_p_other"
    # find neurons that have enough trials for both self and other
    data = data[(data[field_n_self] > 10) & (data[field_n_other] > 10)]
    # Separate the data based on the significance of p-values
    data["self_sig"] = data[field_p_self] < 0.05
    data["other_sig"] = data[field_p_other] < 0.05
    significant_self = data[data[field_p_self] < 0.05][field_self_selectivity]
    non_significant_self = data[data[field_p_self] >= 0.05][
        field_self_selectivity
    ]
    significant_other = data[data[field_p_other] < 0.05][
        field_other_selectivity
    ]
    non_significant_other = data[data[field_p_other] >= 0.05][
        field_other_selectivity
    ]

    # calculate the r_squared value between self and other selectivity
    x = np.array(data[field_self_selectivity])
    y = np.array(data[field_other_selectivity])
    mask = ~np.isnan(x) & ~np.isnan(y) & (data["self_sig"] | data["other_sig"])
    x = x[mask]
    y = y[mask]
    _, beta, sd_beta, expvar = tls_regression(x, y)
    # save beta and sd_beta to a csv
    beta_sd = pd.DataFrame(
        {
            "beta": beta,
            "sd_beta": sd_beta,
            "expvar": expvar,
            "length": len(x),
        },
    )
    beta_sd.to_csv(
        f"{root_dir}/stats_paper/Fig3D_E_beta_sd_{subject}_{factor}_{event}_{window}.csv",
        index=False,
    )

    # Define the bin edges
    bin_edges = np.arange(-0.5, 0.5 + 0.05, 0.025)

    # Create a grid of subplots
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])

    # Histogram for selectivity_self
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(
        [significant_self, non_significant_self],
        bins=bin_edges,
        alpha=0.7,
        color=["black", "grey"],
        edgecolor="black",
        stacked=True,
    )
    ax0.set_xlabel("Selectivity")
    ax0.set_ylabel("Number of cells")
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_xticks(np.arange(-0.5, 0.5 + 0.1, 0.5))
    # set y tick to be 0 and maximum of y ticks
    ax0.set_yticks([0, max(ax0.get_yticks())])
    ax0.set_facecolor("white")
    ax0.text(
        0.95,
        0.95,
        f"N={len(significant_self)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax0.transAxes,
        color="black",
    )
    ax0.text(
        0.95,
        0.85,
        f"N={len(non_significant_self)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax0.transAxes,
        color="grey",
    )
    # remove x label and ticklabels
    ax0.set_xlabel("")
    ax0.set_xticklabels([])
    # Histogram for selectivity_other
    ax1 = fig.add_subplot(gs[1, 1])
    ax1.hist(
        [significant_other, non_significant_other],
        bins=bin_edges,
        alpha=0.7,
        color=["black", "grey"],
        edgecolor="black",
        orientation="horizontal",
        stacked=True,
    )
    ax1.set_ylabel("Selectivity")
    ax1.set_xlabel("Number of cells")
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks(np.arange(-0.5, 0.5 + 0.1, 0.5))
    # set x tick to be 0 and maximum of x ticks
    ax1.set_xticks([0, max(ax1.get_xticks())])
    ax1.set_facecolor("white")
    ax1.text(
        0.95,
        0.95,
        f"N={len(significant_other)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax1.transAxes,
        color="black",
    )
    ax1.text(
        0.95,
        0.85,
        f"N={len(non_significant_other)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax1.transAxes,
        color="grey",
    )
    # remove y label and ticklabels
    ax1.set_ylabel("")
    ax1.set_yticklabels([])

    # scatter plot for self vs other selectivity
    # Create a new column 'overlap_sig' that is True where both 'self_sig' and 'other_sig' are True
    data["overlap_sig"] = data["self_sig"] & data["other_sig"]
    data["either_sig"] = data["self_sig"] | data["other_sig"]
    # Create a color map where points that are significant in both 'self' and 'other' are black and others are grey
    colors = np.where(data["either_sig"], "black", "grey")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(
        data[field_self_selectivity],
        data[field_other_selectivity],
        color=colors,
        alpha=0.7,
        s=10,
    )
    # add dashed line at x=0 and y=0
    ax2.axvline(0, color="black", linestyle="--")
    ax2.axhline(0, color="black", linestyle="--")
    # add red points for specific neurons where unit_index is in example_idx
    if example_L is not None:
        example_data = data[
            (data["subject"] == "L") & (data["unit_index"].isin(example_L))
        ]
        ax2.scatter(
            example_data[field_self_selectivity],
            example_data[field_other_selectivity],
            color=["red"],
            alpha=0.7,
            s=50,
        )
        # add a text string next to the data point indicating the unit_index
        for i, txt in enumerate(example_data["unit_index"]):
            ax2.annotate(
                f"L{txt}",
                (
                    example_data[field_self_selectivity].iloc[i] + 0.02,
                    example_data[field_other_selectivity].iloc[i],
                ),
                fontsize=12,
                color="red",
            )
    if example_O is not None:
        example_data = data[
            (data["subject"] == "O") & (data["unit_index"].isin(example_O))
        ]
        ax2.scatter(
            example_data[field_self_selectivity],
            example_data[field_other_selectivity],
            color=["red"],
            alpha=0.7,
            s=50,
        )
        # add a text string next to the data point indicating the unit_index
        for i, txt in enumerate(example_data["unit_index"]):
            ax2.annotate(
                f"O{txt}",
                (
                    example_data[field_self_selectivity].iloc[i] + 0.02,
                    example_data[field_other_selectivity].iloc[i],
                ),
                fontsize=12,
                color="red",
            )

    # add regression line to the scatter plot from beta
    x_plot = np.linspace(-0.5, 0.5, 100)
    y_plot = beta[0] * x_plot + beta[1]
    ax2.plot(x_plot, y_plot, color="black")
    # plot a text with total number of cells
    # Add the text annotation to the plot
    ax2.text(
        0.95,
        0.95,
        f"N={len(data)}\nExplained Variance Ratio: {expvar:.2f}\nSlope: {beta[0]:.2f}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax2.transAxes,
        color="black",
    )
    ax2.set_xlabel("Self Selectivity")
    ax2.set_ylabel("Other Selectivity")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xticks(np.arange(-0.5, 0.5 + 0.1, 0.5))
    ax2.set_yticks(np.arange(-0.5, 0.5 + 0.1, 0.5))
    ax2.set_facecolor("white")
    # Adjust style to match the example
    for ax in [ax0, ax1, ax2]:
        # Setting the style of ticks and spines to match the example
        ax.tick_params(
            direction="out",
            length=6,
            width=2,
            colors="black",
            grid_color="black",
            grid_alpha=0.5,
        )
        # Setting spines to be visible and a bit distanced from the axis lines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_position(("outward", 5))
        # Turning off the top and right spines to match the example style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # increase the font size of all axes
    for ax in fig.get_axes():
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(20)
    # Tight layout
    plt.tight_layout()

    return fig


def make_plots_separate(data, factor, event, window):
    # make three histograms in three panels, for self, other, and both
    field_self_selectivity = f"{event}_{factor}_{window}_selectivity_self"
    field_other_selectivity = f"{event}_{factor}_{window}_selectivity_other"
    field_both_selectivity = f"{event}_{factor}_{window}_selectivity_both"
    field_n_self = f"{event}_{factor}_{window}_n_self"
    field_n_other = f"{event}_{factor}_{window}_n_other"
    field_p_self = f"{event}_{factor}_{window}_p_self"
    field_p_other = f"{event}_{factor}_{window}_p_other"
    field_p_both = f"{event}_{factor}_{window}_p_both"
    # find neurons that have enough trials for both self and other
    data = data[(data[field_n_self] > 10) & (data[field_n_other] > 10)]
    # Separate the data based on the significance of p-values
    data["self_sig"] = data[field_p_self] < 0.05
    data["other_sig"] = data[field_p_other] < 0.05
    data["both_sig"] = data[field_p_both] < 0.05
    significant_self = data[data[field_p_self] < 0.05][field_self_selectivity]
    non_significant_self = data[data[field_p_self] >= 0.05][
        field_self_selectivity
    ]
    significant_other = data[data[field_p_other] < 0.05][
        field_other_selectivity
    ]
    non_significant_other = data[data[field_p_other] >= 0.05][
        field_other_selectivity
    ]
    significant_both = data[data[field_p_both] < 0.05][field_both_selectivity]
    non_significant_both = data[data[field_p_both] >= 0.05][
        field_both_selectivity
    ]

    # Define the bin edges
    bin_edges = np.arange(-0.5, 0.5 + 0.05, 0.025)

    # Create three subplots
    fig = plt.figure(figsize=(12, 6))
    # Histogram for selectivity_self
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.hist(
        [significant_self, non_significant_self],
        bins=bin_edges,
        alpha=0.7,
        color=["black", "grey"],
        edgecolor="black",
        stacked=True,
    )
    ax0.set_xlabel("Selectivity - Self")
    ax0.set_ylabel("Number of cells")
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_xticks(np.arange(-0.5, 0.5 + 0.1, 0.5))
    ax0.set_facecolor("white")
    ax0.text(
        0.95,
        0.95,
        f"N={len(significant_self)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax0.transAxes,
        color="black",
    )
    ax0.text(
        0.95,
        0.85,
        f"N={len(non_significant_self)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax0.transAxes,
        color="grey",
    )

    # Histogram for selectivity_other
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.hist(
        significant_other,
        bins=bin_edges,
        alpha=0.7,
        color="black",
        edgecolor="black",
        # orientation="horizontal",
    )
    ax1.hist(
        non_significant_other,
        bins=bin_edges,
        alpha=0.7,
        color="grey",
        edgecolor="black",
        # orientation="horizontal",
    )
    ax1.set_xlabel("Selectivity - Other")
    ax1.set_ylabel("Number of cells")
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_xticks(np.arange(-0.5, 0.5 + 0.1, 0.5))
    ax1.set_facecolor("white")
    ax1.text(
        0.95,
        0.95,
        f"N={len(significant_other)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax1.transAxes,
        color="black",
    )
    ax1.text(
        0.95,
        0.85,
        f"N={len(non_significant_other)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax1.transAxes,
        color="grey",
    )

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.hist(
        significant_both,
        bins=bin_edges,
        alpha=0.7,
        color="black",
        edgecolor="black",
    )
    ax2.hist(
        non_significant_both,
        bins=bin_edges,
        alpha=0.7,
        color="grey",
        edgecolor="black",
    )
    ax2.text(
        0.95,
        0.95,
        f"N={len(significant_both)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax2.transAxes,
        color="black",
    )
    ax2.text(
        0.95,
        0.85,
        f"N={len(non_significant_both)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax2.transAxes,
        color="grey",
    )
    ax2.set_xlabel("Selectivity - Both")

    # Adjust style to match the example
    for ax in [ax0, ax1, ax2]:
        # Setting the style of ticks and spines to match the example
        ax.tick_params(
            direction="out",
            length=6,
            width=2,
            colors="black",
            grid_color="black",
            grid_alpha=0.5,
        )
        # Setting spines to be visible and a bit distanced from the axis lines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_position(("outward", 5))
        # Turning off the top and right spines to match the example style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig


def make_venn_plot(data, event, window):
    # make a venn diagram showing the proportion of significant neurons of each type
    # Define the sets
    mpl.rcParams["pdf.fonttype"] = 42
    data = data.reset_index(drop=True)
    field_p_self = f"{event}_reward_{window}_p_self"
    field_p_other = f"{event}_reward_{window}_p_other"
    idx_self = np.where(data[field_p_self] < 0.05)[0]
    idx_other = np.where(data[field_p_other] < 0.05)[0]
    idx_self_other = np.where(
        (data[field_p_self] < 0.05) & (data[field_p_other] < 0.05)
    )[0]
    # Use the matlab_venn function to create the Venn diagram
    fig = plt.figure()
    # Define your set sizes and their intersections
    # The numbers represent the size of each area in the Venn diagram
    subsets = (
        len(idx_self),
        len(idx_other),
        len(idx_self_other),
    )
    # make a venn diagram showing proportion of significant neurons of each type
    venn = venn2(
        subsets=subsets,
        set_labels=("Self", "Other"),
        set_colors=("r", "b"),
        alpha=0.5,
    )
    # Remove edge lines
    for patch in venn.patches:
        if patch:  # Check if the patch exists
            patch.set_edgecolor("none")  # Remove the edge line
    # add a text to the plot with N= total number
    n_total = sum(subsets)
    plt.text(
        0,
        0.65,
        f"N={n_total}",
        verticalalignment="center",
        horizontalalignment="center",
    )
    # make text large in all ax
    for ax in fig.get_axes():
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(20)

    return fig


if __name__ == "__main__":
    start = -0.6
    end = 0.6
    step = 0.6
    window_list = [
        f"[{round(i+0.001, 1)}, {round(i+step+0.001, 1)}]"
        for i in np.arange(start, end, step)
    ]
    root_dir = findrootdir()
    master_list_both = pd.concat(
        [
            pd.read_csv(f"{root_dir}/master_list_{subject}.csv").assign(
                subject=subject
            )
            for subject in ["L", "O"]
        ]
    )
    # each subject's example neurons follow order of self, other, both, opposite
    example_neurons_L = [75]
    example_neurons_O = [2128]
    # make combined plot using both monkeys
    factor = "reward"
    for phase in [
        "choice",
        "fdbk",
    ]:
        if phase == "choice":
            window = window_list[0]
        else:
            window = window_list[-1]
        figname = f"{root_dir}/plots_paper/Fig3D_E_roc_combined_{phase}_{window}_stacked.pdf"
        fig = make_plots_combined(
            master_list_both,
            factor,
            phase,
            window,
            subject="both",
            example_L=example_neurons_L,
            example_O=example_neurons_O,
        )
        fig.savefig(figname, dpi=300)
        plt.close(fig)

        # make pie chart showing the proportion of significant neurons of each type
        fig = make_venn_plot(master_list_both, phase, window)
        fig.savefig(
            f"{root_dir}/plots_paper/Fig3D_E_venn_{phase}_{window}.pdf",
            dpi=300,
        )
        for subject in ["L", "O"]:
            master_list_subjuct = pd.read_csv(
                f"{root_dir}/master_list_{subject}.csv"
            )
            master_list_subjuct["subject"] = subject
            fig = make_venn_plot(master_list_subjuct, phase, window)
            fig.savefig(
                f"{root_dir}/plots_paper/FigS6_A_D_venn_{phase}_{subject}_{window}.pdf",
                dpi=300,
            )
            figname = f"{root_dir}/plots_paper/FigS6_A_D_roc_combined_{phase}_{window}_{subject}.pdf"
            if subject == "L":
                example_L = example_neurons_L
                example_O = None
            else:
                example_L = None
                example_O = example_neurons_O
            fig = make_plots_combined(
                master_list_subjuct,
                factor,
                phase,
                window,
                subject,
                example_L=example_L,
                example_O=example_O,
            )
            fig.savefig(figname, dpi=300)
