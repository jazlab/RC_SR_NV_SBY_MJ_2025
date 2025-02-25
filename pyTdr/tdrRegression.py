"""
This is ported from Matlab code from the original TDR codebase.
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils.roc_selectivity import calculate_selectivity


def perform_regression(data, regressors, normalization="max_abs"):
    regpars = {
        "regressor": regressors,
        "regressor_normalization": normalization,
    }
    return tdrRegression(data, regpars)


def tdrRegression(data, regpars, plotflag=False):
    """
    Linear regression on population responses.

    Inputs:
        data: population response (sequential or simultaneous)
        regpars.regressor: list of regressors
            Each entry is a string, corresponding to the name of one of the fields
            in data.task_variable.
        regpars.regressor_normalization:
            (1) normalize regressors by maximum ('max_abs')
            (2) use raw regressors ('none')
        regpars.regularization: regularization parameter
            (1) None: no regularization
            (2) 'l1': L1 regularization (lasso)
            (3) 'l2': L2 regularization (ridge)
        plotflag: summary plot, yes (True) or no (False)

    Outputs:
        coef.name: regressor names (same as regpars.regressor)
        coef.response: regression coefficients over units and times [nun npt nrg]
            nun: n of units
            npt: n of time samples
            nrg: n of regressors
        coef.time: time axis
        coef.dimension: dimension labels

    coef = tdrRegression(data, regpars, plotflag=False)
    """

    # Default inputs
    if plotflag is None:
        plotflag = False
    resp_field = "response"
    task_field = "task_variable"
    # Initialize
    coef = {}
    coef["name"] = regpars["regressor"]

    # Check if data is sequentially or simultaneously recorded
    if "unit" in data and data["unit"]:
        # --- Sequential recordings ---
        # Dimensions
        npt = len(data["time"])
        nun = len(data["unit"])
        nrg = len(regpars["regressor"])

        # Initialize
        bb = np.zeros((nun, npt, nrg))  # regression coefficients

        # Loop over units
        for iun in range(nun):
            # Construct the regressor values from the task variables
            ntr = data["unit"][iun][resp_field].shape[0]
            regmod = np.ones((ntr, nrg))

            # Loop over task variables
            for irg, reg in enumerate(regpars["regressor"]):
                isep = (
                    [0]
                    + [m.start() for m in re.finditer(r"\*", reg)]
                    + [len(reg) + 1]
                )
                for jsep in range(len(isep) - 1):
                    # The task variable
                    varname = reg[isep[jsep] : isep[jsep + 1]]
                    # remove ' and * from varname
                    varname = varname.replace("*", "")

                    # Check that not constant term
                    if varname != "b0":
                        # Update regressor
                        regmod[:, irg] = (
                            regmod[:, irg]
                            * data["unit"][iun][task_field][varname]
                        )

            # Normalize the regressors
            if regpars["regressor_normalization"] == "none":
                regnrm = regmod
            elif regpars["regressor_normalization"] == "max_abs":
                regnrm = np.zeros_like(regmod)
                for irg in range(nrg):
                    regnrm[:, irg] = regmod[:, irg] / np.max(
                        np.abs(regmod[:, irg])
                    )

            # Loop over time points
            bbt = np.zeros((npt, nrg))
            for ipt in range(npt):
                # Responses to predict
                yy = data["unit"][iun][resp_field][:, ipt]

                # Linear regression
                # check if yy contains nan
                if np.any(np.isnan(yy)):
                    print(yy)
                    import pdb

                    pdb.set_trace()
                bbt[ipt, :] = np.linalg.lstsq(regnrm, yy, rcond=None)[0]

            # Keep coefficients
            bb[iun, :, :] = bbt

        # The state space dimensions
        dimension = [data["unit"][i]["dimension"] for i in range(nun)]

    else:
        # --- Simultaneous recordings ---
        # Dimensions
        nun, npt, ntr = data[resp_field].shape
        nrg = len(regpars["regressor"])

        # Initialize
        bb = np.zeros((nun, npt, nrg))  # regression coefficients

        # Construct the regressor values from the task variables
        regmod = np.ones((ntr, nrg))

        # Get the regressors
        for irg, reg in enumerate(regpars["regressor"]):
            isep = (
                [0]
                + [m.start() for m in re.finditer(r"\*", reg)]
                + [len(reg) + 1]
            )
            for jsep in range(len(isep) - 1):
                # The task variable
                varname = reg[isep[jsep] : isep[jsep + 1]]
                # remove ' and * from varname
                varname = varname.replace("*", "")

                # Check that not constant term
                if varname != "b0":
                    # Update regressor
                    regmod[:, irg] = regmod[:, irg] * data[task_field][varname]

        # Normalize the regressors
        if regpars["regressor_normalization"] == "none":
            regnrm = regmod
        elif regpars["regressor_normalization"] == "max_abs":
            regnrm = np.zeros_like(regmod)
            for irg in range(nrg):
                regnrm[:, irg] = regmod[:, irg] / np.max(np.abs(regmod[:, irg]))

        # Loop over units
        for iun in range(nun):
            # Loop over time points
            bbt = np.zeros((npt, nrg))
            for ipt in range(npt):
                # Responses to predict
                yy = data[resp_field][iun, ipt, :]

                # Linear regression
                bbt[ipt, :] = np.linalg.lstsq(regnrm, yy, rcond=None)[0]

            # Keep coefficients
            bb[iun, :, :] = bbt

        # The state space dimensions
        dimension = data["dimension"]

    # The norm of the raw regression vectors
    bnorm = np.zeros((npt, nrg))
    for irg in range(nrg):
        for ipt in range(npt):
            bnorm[ipt, irg] = np.linalg.norm(bb[:, ipt, irg])

    # Plot the norms
    if plotflag:
        # Plot
        fig, ax = plt.subplots()
        ax.plot(data["time"], bnorm)

        # Labels
        for irg in range(nrg):
            ax.text(
                data["time"][-1],
                bnorm[-1, irg],
                f'  {regpars["regressor"][irg]}',
                horizontalalignment="left",
                verticalalignment="center",
                color=ax.lines[irg].get_color(),
            )
        ax.set(
            xlim=[data["time"][0], data["time"][-1]],
            xlabel="time (s)",
            ylabel="regressor norm",
        )

    if "select_timepoint" in regpars:
        idx_timepoint = np.argmin(
            np.abs(data["time"] - regpars["select_timepoint"])
        )
        bb_timepoint = bb[:, idx_timepoint, :]
        bb = np.repeat(bb_timepoint[:, np.newaxis, :], npt, axis=1)

    # Keep what you need
    coef["response"] = bb
    coef["time"] = data["time"]
    coef["dimension"] = dimension
    coef["unit_idx_master"] = data["unit_idx_master"]

    return coef


def plotSelectivity(projection, target, fig_name):
    selectivity, p_val = calculate_selectivity(projection, target)
    projection = projection.reshape(-1, 1)
    model = LinearRegression()
    model.fit(projection, target)
    predictions = model.predict(projection)
    r2 = r2_score(target, predictions)
    # plot data and regression line
    plt.title(f"R-squared: {r2}; AUC: {selectivity}; p-value: {p_val}")
    plt.scatter(projection, target)
    plt.xlabel("projection value")
    plt.ylabel("switch")
    plt.savefig(fig_name, dpi=300)
    plt.close()
