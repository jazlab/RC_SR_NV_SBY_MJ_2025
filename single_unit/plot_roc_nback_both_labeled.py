"""
This script generates histogram of accumulation selectivity ('nback' refers to
number of consecutive unrewarded trials)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matplotlib_venn import venn2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.LoadSession import findrootdir
import os


def save_selectivity_data(
    data,
    sig_self,
    sig_other,
    sig_AA,
    sig_OO,
    sig_OA,
    sig_AO,
    idx_hasother,
    idx_noother,
    all_sig,
    root_dir,
    subject,
):
    # save consistent neuron info
    # build 4 csv files for: sig_self, sig_other, sig_so_integration, all_sig
    # field to include: unit_index, subject, date, group, unit_number, n_trials
    # field to exclude: all other fields
    columns_of_interest = [
        "subject",
        "unit_index",
        "date",
        "group",
        "unit_number",
        "n_trials",
    ]
    columns_to_drop = [
        col for col in data.columns if col not in columns_of_interest
    ]
    # create directory if not exists
    if not os.path.exists(f"{root_dir}/stats_paper"):
        os.makedirs(f"{root_dir}/stats_paper")
    # save data for 'both'
    if subject == "both":
        df_filtered = data.iloc[all_sig]
        df_final = df_filtered.drop(columns=columns_to_drop)
        df_final.to_csv(
            f"{root_dir}/stats_paper/integration_sig_all_{subject}.csv",
            index=False,
        )
        df_filtered = data.iloc[sig_self]
        df_final = df_filtered.drop(columns=columns_to_drop)
        df_final.to_csv(
            f"{root_dir}/stats_paper/integration_sig_self_{subject}.csv",
            index=False,
        )
        df_filtered = data.iloc[sig_other]
        df_final = df_filtered.drop(columns=columns_to_drop)
        df_final.to_csv(
            f"{root_dir}/stats_paper/integration_sig_other_{subject}.csv",
            index=False,
        )

    fig = make_venn_plot(sig_AA, idx_hasother)
    # save figure
    figname_by_subject = {
        "L": "FigS6F",
        "O": "FigS6E",
        "both": "Fig3G",
    }
    fig.savefig(
        f"{root_dir}/plots_paper/{figname_by_subject[subject]}_integration_sig_venn_{subject}.pdf",
        dpi=300,
    )


def plotSelectivityHistogram(data, event, window, subject="both"):
    # make histogram of integration:
    # Rew-NR and 1NR-2NR selective in the same direction
    # 1. all integration
    factor = "reward"
    rew_self_sel = f"{event}_{factor}_{window}_selectivity_self"
    rew_other_sel = f"{event}_{factor}_{window}_selectivity_other"
    rew_self_p = f"{event}_{factor}_{window}_p_self"
    rew_other_p = f"{event}_{factor}_{window}_p_other"
    # Identify selective neurons for all conditions
    sig_self = np.where((data[rew_self_p] < 0.05))[0]
    sig_other = np.where((data[rew_other_p] < 0.05))[0]

    # AA, AO, OA, OO conditions
    factor = "nback_AOOA"
    nr1_nr2_OA_sel = f"{event}_{factor}_{window}_selectivity_self"
    nr1_nr2_AO_sel = f"{event}_{factor}_{window}_selectivity_other"
    nr1_nr2_OA_p = f"{event}_{factor}_{window}_p_self"
    nr1_nr2_AO_p = f"{event}_{factor}_{window}_p_other"
    factor = "nback_all"
    nr1_nr2_AA_sel = f"{event}_{factor}_{window}_selectivity_self"
    nr1_nr2_AA_p = f"{event}_{factor}_{window}_p_self"
    nr1_nr2_OO_sel = f"{event}_{factor}_{window}_selectivity_other"
    nr1_nr2_OO_p = f"{event}_{factor}_{window}_p_other"

    sig_AA = np.where(
        ((data[rew_self_p] < 0.05) | (data[rew_other_p] < 0.05))
        & (data[rew_self_sel] * data[rew_other_sel] > 0)
        & (data[nr1_nr2_AA_p] < 0.05)
        & (data[rew_self_sel] * data[nr1_nr2_AA_sel] > 0)
    )[0]
    sig_OA = np.where(
        ((data[rew_self_p] < 0.05) | (data[rew_other_p] < 0.05))
        & (data[rew_self_sel] * data[rew_other_sel] > 0)
        & (data[nr1_nr2_OA_p] < 0.05)
        & (data[rew_other_sel] * data[nr1_nr2_OA_sel] > 0)
    )[0]
    sig_AO = np.where(
        ((data[rew_self_p] < 0.05) | (data[rew_other_p] < 0.05))
        & (data[rew_self_sel] * data[rew_other_sel] > 0)
        & (data[nr1_nr2_AO_p] < 0.05)
        & (data[rew_self_sel] * data[nr1_nr2_AO_sel] > 0)
    )[0]
    sig_OO = np.where(
        ((data[rew_self_p] < 0.05) | (data[rew_other_p] < 0.05))
        & (data[rew_self_sel] * data[rew_other_sel] > 0)
        & (data[nr1_nr2_OO_p] < 0.05)
        & (data[rew_other_sel] * data[nr1_nr2_OO_sel] > 0)
    )[0]
    all_sig = np.union1d(sig_AA, sig_OO)
    all_sig = np.union1d(all_sig, sig_OA)
    all_sig = np.union1d(all_sig, sig_AO)
    idx_hasother = np.union1d(sig_OO, sig_OA)
    idx_hasother = np.union1d(idx_hasother, sig_AO)
    idx_noother = np.setdiff1d(all_sig, idx_hasother)

    # all reward selective neurons
    sig_rew_self = np.where(data[rew_self_p] < 0.05)[0]
    sig_rew_other = np.where(data[rew_other_p] < 0.05)[0]
    sig_rew_all = np.union1d(sig_rew_self, sig_rew_other)
    same_rew_sel = np.where(data[rew_self_sel] * data[rew_other_sel] > 0)[0]
    sig_rew_all = np.intersect1d(sig_rew_all, same_rew_sel)
    # all reward selective neuron not in all_sig
    sig_rew_no_integration = np.setdiff1d(sig_rew_all, all_sig)
    # save dfs using save_selectivity_data
    save_selectivity_data(
        data,
        sig_self,
        sig_other,
        sig_AA,
        sig_OO,
        sig_OA,
        sig_AO,
        idx_hasother,
        idx_noother,
        all_sig,
        root_dir,
        subject,
    )

    # Create plot
    fig = plt.figure(figsize=(3, 3))
    bin_edges = np.arange(-0.5, 0.5 + 0.1, 0.1)
    # Histogram for selectivity of all_sig
    sel_AO = data.iloc[sig_AO][nr1_nr2_AO_sel].values
    sel_OA = data.iloc[sig_OA][nr1_nr2_OA_sel].values
    sel_OO = data.iloc[sig_OO][nr1_nr2_OO_sel].values
    sel_hasother = np.concatenate([sel_AO, sel_OA, sel_OO])
    sel_noother = data.iloc[idx_noother][nr1_nr2_AA_sel].values
    sel_allsig = np.concatenate([sel_hasother, sel_noother])
    sel_nointegration = data.iloc[sig_rew_no_integration][nr1_nr2_AA_sel].values
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.hist(
        [sel_allsig, sel_nointegration],
        bins=bin_edges,
        alpha=0.5,
        color=["black", "gray"],
        edgecolor="black",
        stacked=True,
    )
    ax0.set_xlabel("Selectivity (1NR-2NR)")
    ax0.set_ylabel("Number of cells")
    ax0.set_xlim(-0.5, 0.5)
    ax0.set_xticks(np.arange(-0.5, 0.5 + 0.1, 0.5))
    ax0.set_facecolor("white")
    # add test with number of each group
    ax0.text(
        0.95,
        0.95,
        f"N={len(all_sig)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax0.transAxes,
        color="black",
    )
    ax0.text(
        0.95,
        0.85,
        f"N={len(sig_rew_no_integration)}",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax0.transAxes,
        color="grey",
    )

    # Adjust style to match the example
    for ax in [ax0]:
        # Setting the style of ticks and spines to match the example
        ax.tick_params(
            direction="out",
            length=6,
            width=0.8,
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


def make_venn_plot(idx_AA, idx_hasother):
    # make a venn diagram showing the proportion of significant neurons of each type
    # Define the sets
    mpl.rcParams["pdf.fonttype"] = 42
    idx_overlap = np.intersect1d(idx_AA, idx_hasother)
    # Use the matlab_venn function to create the Venn diagram
    fig = plt.figure()
    # Define your set sizes and their intersections
    # The numbers represent the size of each area in the Venn diagram
    subsets = (
        len(idx_AA),
        len(idx_hasother),
        len(idx_overlap),
    )
    # make a venn diagram showing proportion of significant neurons of each type
    venn = venn2(
        subsets=subsets,
        set_labels=("AA", "AO/OA/OO"),
        set_colors=("r", "b"),
        alpha=0.5,
    )
    # Remove edge lines
    for patch in venn.patches:
        if patch:  # Check if the patch exists
            patch.set_edgecolor("none")  # Remove the edge line
    # add a text to the plot with N= total number
    n_total = sum(subsets) - len(idx_overlap) * 2
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
    root_dir = findrootdir()
    master_list_both = pd.concat(
        [
            pd.read_csv(f"{root_dir}/master_list_{subject}.csv").assign(
                subject=subject
            )
            for subject in ["L", "O"]
        ]
    )
    window = "[0.0, 0.6]"
    # make combined plot using both monkeys
    phase = "fdbk"
    figname = f"{root_dir}/plots_paper/Fig3G_roc_nr1_nr2.pdf"
    fig = plotSelectivityHistogram(
        master_list_both, phase, window, subject="both"
    )
    fig.savefig(figname, dpi=300)
    plt.close(fig)
    # make plot for each monkey
    for subject in ["L", "O"]:
        master_list = master_list_both[master_list_both["subject"] == subject]
        figname = f"{root_dir}/plots_paper/FigS6E_F_roc_nr1_nr2_{subject}.pdf"
        fig = plotSelectivityHistogram(
            master_list, phase, window, subject=subject
        )
        fig.savefig(figname, dpi=300)
        plt.close(fig)
