"""
This is ported from Matlab code from the original TDR codebase.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy


def tdrVectorTimeAverage(coef_time, vecpars, plotflag=None):
    """
    tdrVectorTimeAverage Define fixed regression vectors by time averaging coefficients

    Inputs:
    coef_time: regression coefficients
        .name: regressor names [nvc 1]
        .response: coefficients [nun npt nvc]
        .time: time axis [1 npt]
    vecpars: times to average. Each field of vecpars contains the time
        window(s) to average for a given coefficient. The fieldname
        corresponds to the coefficient name.
        e.g. vecpars.coefname(i).time_win = [tmin tmax] averages the
             coefficient 'coefname' over the time window [tmin tmax].
        Can specify many time windows (i=1,2,...).
        If the time window is empty, uses the time of maximum norm.
    plotflag: summary plot yes (1) or no (0)

    Outputs:
    coef_fix: fixed (non time-varying) regression vectors
        .name: regression vector name {nvc 1}
        .response: regression vector coefficients [ndm 1 nvc]
        .dimension: dimension labels {ndm 1}
    coef_nrm(i): time-varying norm of the regression coefficients for vector i
        .bnorm: vector norm [npt 1]
        .jpt: time samples that were averaged
        .irg: index of coefficient
        .name: name of coefficient
    """

    if plotflag is None:
        plotflag = 0

    # The vectors
    vecname = list(vecpars.keys())

    # Initialize
    coef_fix = copy.deepcopy(coef_time)
    coef_fix["name"] = []

    # The number of vectors
    nve = 0
    nvc = len(vecname)
    for irg in range(nvc):
        nve += len(vecpars[vecname[irg]])

    # Dimensions
    nun, npt, ncf = coef_time["response"].shape

    # The norm of the vectors
    bnorm = np.zeros((npt, ncf))
    for icf in range(ncf):
        for ipt in range(npt):
            bnorm[ipt, icf] = np.linalg.norm(coef_time["response"][:, ipt, icf])

    # Initialize
    response = np.zeros((nun, 1, nve))
    coef_nrm = [{} for _ in range(nve)]

    # Loop over vectors
    ive = 0
    nvc = len(vecname)
    for irg in range(nvc):
        # The matching vector in the input
        imatch = coef_time["name"].index(vecname[irg])

        # Loop over temporal windows
        nwn = len(vecpars[vecname[irg]])
        for iwn in range(nwn):
            if not vecpars[vecname[irg]]["time_win"]:
                # Find maximum of the norm
                jpt = np.nanargmax(bnorm[:, imatch])
                response[:, 0, ive] = coef_time["response"][:, jpt, imatch]
            else:
                # Time points to average
                jpt = np.where(
                    (coef_time["time"] >= vecpars[vecname[irg]]["time_win"][0])
                    & (
                        coef_time["time"]
                        <= vecpars[vecname[irg]]["time_win"][-1]
                    )
                )[0]

                # Find closest if no match
                if len(list(jpt)) == 0:
                    jpt = np.argmin(
                        np.abs(
                            coef_time["time"]
                            - np.mean(vecpars[vecname[irg]]["time_win"])
                        )
                    )
                # Average vectors over time
                # if jpt is a single value, then mean is just the coef
                if isinstance(jpt, np.ndarray) and jpt.size == 1:
                    response[:, 0, ive] = np.squeeze(
                        coef_time["response"][:, jpt, imatch]
                    )
                else:
                    response[:, 0, ive] = np.mean(
                        coef_time["response"][:, jpt, imatch], axis=1
                    )

            # Vector name
            if nwn == 1:
                coef_fix["name"].append(vecname[irg])
            elif nwn > 1:
                coef_fix["name"].append(f"{vecname[irg]}_{iwn}")

            # Info for plots
            coef_nrm[ive]["bnorm"] = bnorm[:, imatch]
            coef_nrm[ive]["jpt"] = jpt
            coef_nrm[ive]["irg"] = irg
            coef_nrm[ive]["name"] = coef_fix["name"][ive]

            # Update
            ive += 1

    # Keep what you need
    coef_fix["response"] = response
    coef_fix["time"] = []

    # PLOT a summary with the norm of the vectors and the times when they were
    # averaged
    if plotflag:
        # Line colors
        lc = cm.get_cmap("jet")(np.linspace(0, 1, nvc))

        plt.figure()

        # Loop over vectors
        for ive in range(nve):
            # Times that were averaged
            jpt = coef_nrm[ive]["jpt"]

            # The raw vectors
            (hp1,) = plt.plot(coef_time["time"], coef_nrm[ive]["bnorm"], "-")
            hp1.set_color(lc[coef_nrm[ive]["irg"]])

            # The values that were averaged
            (hp2,) = plt.plot(
                coef_time["time"][jpt], coef_nrm[ive]["bnorm"][jpt], "o"
            )
            hp2.set_color(lc[coef_nrm[ive]["irg"]])

            # Axis labels
            plt.xlabel("time (s)")
            plt.ylabel("coefficient norm")

            # The vector name
            ht = plt.text(
                coef_time["time"][jpt[0]],
                coef_nrm[ive]["bnorm"][jpt[0]],
                coef_fix["name"][ive],
            )
            ht.set_horizontalalignment("left")
            ht.set_verticalalignment("top")
            ht.set_color(lc[coef_nrm[ive]["irg"]])

    return coef_fix, coef_nrm
