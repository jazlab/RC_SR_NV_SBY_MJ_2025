"""
This is ported from Matlab code from the original TDR codebase.
"""

import numpy as np


def tdrMeanAndStd(data, avgpars=None):
    """
    tdrMeanAndStd compute mean and std of responses

    Inputs:
    data: population response (sequential or simultaneous, see 'help tdr')
    avgpars.trial: logical indices of trials/conditions to use [ntr 1] (def: all)
    avgpars.time: logical indices of times to use [npt 1] (def: all)

    Outputs:
    ravg: mean of responses across trials and times [nun 1]
    rstd: STD of responses across trials and times [nun 1]
    """

    # Inputs
    if avgpars is None:
        avgpars = {}

    # Time indices
    timeSet = ("time" not in avgpars) or (len(avgpars["time"]) == 0)

    # Trial indices
    trialSet = ("trial" not in avgpars) or (len(avgpars["trial"]) == 0)
    # Number of time samples
    npt = len(data["time"])

    # Check if data is sequentially or simultaneously recorded
    if "unit" in data and data["unit"] is not None:
        # --- Sequential recordings ---
        # Number of units
        nun = len(data["unit"])

        # Initialize
        ravg = np.zeros(nun)
        rstd = np.zeros(nun)

        # Time indices
        if timeSet:
            avgpars["time"] = np.ones(npt, dtype=bool)

        # Loop over units
        for iun in range(nun):
            # Number of trials
            ntr = data["unit"][iun]["response"].shape[0]

            # Trial indices
            if trialSet:
                avgpars["trial"] = np.ones(ntr, dtype=bool)

            # Responses to use
            response = data["unit"][iun]["response"][avgpars["trial"], :][
                :, avgpars["time"]
            ]

            # Mean across trials and times
            ravg[iun] = np.nanmean(response)

            # Standard deviation across trials and times
            rstd[iun] = np.nanstd(response)

    else:
        # --- Simultaneous recordings ---
        # Dimensions
        nun, npt, ntr = data["response"].shape

        # Time indices
        if timeSet:
            avgpars["time"] = np.ones(npt, dtype=bool)

        # Trial indices
        if trialSet:
            avgpars["trial"] = np.ones(ntr, dtype=bool)

        # Responses to use
        response = data["response"][:, avgpars["time"], :][
            :, :, avgpars["trial"]
        ]

        # Collapse time and conditions/trials
        rcollapse = response.reshape((nun, -1))

        # Mean across trials/conditions and times
        ravg = np.nanmean(rcollapse, axis=1)

        # Mean across trials/conditions and times
        rstd = np.nanstd(rcollapse, axis=1)

    return ravg, rstd
