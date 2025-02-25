"""
This is ported from Matlab code from the original TDR codebase.
"""

import numpy as np
from math import floor
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def temporal_smooth(data):
    # Smoothing parameters
    smthpars = {}
    smthpars["filter"] = "gauss"  # {'gauss';'box'}
    smthpars["width"] = 0.04
    # Temporal smoothing
    return tdrTemporalSmoothing(data, smthpars)


def tdrTemporalSmoothing(data, tmppars):
    # Temporal smoothing of trial-by-trial or condition-averaged responses
    #
    # Inputs:
    #  data: population response (simultaneous, see 'help tdr')
    #  tmppars.filter: filter type ('gauss' or 'box')
    #  tmppars.width: filter widht (s)
    #
    # Outputs:
    #  data_smooth: smoothed data, same format as data.
    #
    # data_smooth = tdrTemporalSmoothing(data,tmppars)

    # Sample duration
    if len(data["time"]) < 2:
        return data
    dt = data["time"][1] - data["time"][0]

    # Make filter
    if tmppars["filter"] == "gauss":
        # Filter parameters
        fw = tmppars["width"]

        # Time axis
        nt = round(fw * 3.75 / dt)
        tf = np.arange(-nt, nt + 1) * dt

        # Make filter
        rf = gaussian(len(tf), std=fw / dt)
        rf = rf / (np.sum(rf) * dt)

    elif tmppars["filter"] == "box":
        # Filter parameters
        fw = tmppars["width"]

        # Box width
        nt = round(fw / 2 / dt)

        # Make filter
        rf = np.ones(2 * nt + 1)
        rf = rf / (np.sum(rf) * dt)

    # Initialize
    data_smooth = data.copy()
    # Temporal smoothing
    if len(rf) > 1:
        if "unit" in data:
            # Dimensions
            nun = len(data["unit"])
            # loop over units
            for iun in range(nun):
                # Raw response
                npt, ntr = data["unit"][iun]["response"].shape
                # loop over trials
                for itr in range(ntr):
                    rraw = data["unit"][iun]["response"][:, itr]
                    # Pad extremes
                    rpad = np.concatenate(
                        [
                            np.ones(len(rf)) * rraw[0],
                            rraw,
                            np.ones(len(rf)) * rraw[-1],
                        ]
                    )
                    # Filter
                    rlng = convolve(rpad, rf, mode="full") * dt
                    rlng = rlng[: len(rpad)]
                    # Shift
                    rfil = rlng[
                        len(rf)
                        + floor(len(rf) / 2) : len(rf)
                        + floor(len(rf) / 2)
                        + len(rraw)
                    ]
                    # Keep
                    data_smooth["unit"][iun]["response"][:, itr] = rfil
        else:
            # Dimensions
            nun, npt, ntr = data["response"].shape

            # Loop over units
            for iun in range(nun):
                # Loop over trials
                for itr in range(ntr):
                    # Raw response
                    rraw = data["response"][iun, :, itr]

                    # Pad extremes
                    rpad = np.concatenate(
                        [
                            np.ones(len(rf)) * rraw[0],
                            rraw,
                            np.ones(len(rf)) * rraw[-1],
                        ]
                    )

                    # Filter
                    rlng = convolve(rpad, rf, mode="full") * dt
                    rlng = rlng[: len(rpad)]

                    # Shift
                    rfil = rlng[
                        len(rf)
                        + floor(len(rf) / 2) : len(rf)
                        + floor(len(rf) / 2)
                        + len(rraw)
                    ]

                    # Keep
                    data_smooth["response"][iun, :, itr] = rfil

    return data_smooth
