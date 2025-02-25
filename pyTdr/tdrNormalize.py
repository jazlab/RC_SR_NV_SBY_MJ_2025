"""
This is ported from Matlab code from the original TDR codebase.
"""

import numpy as np
from pyTdr.tdrTemporalSmoothing import tdrTemporalSmoothing
from pyTdr.tdrMeanAndStd import tdrMeanAndStd


def smooth_and_normalize(data, filter_type="gauss", width=0.04):
    smthpars = {"filter": filter_type, "width": width}
    data_smth = tdrTemporalSmoothing(data, smthpars)
    avgpars = {"trial": [], "time": []}
    mean, std = tdrMeanAndStd(data_smth, avgpars)
    nrmlpars = {"ravg": mean, "rstd": std}
    return tdrNormalize(data_smth, nrmlpars)


def tdrNormalize(data, nrmlpars=None):
    """
    tdrNormalize Normalize population responses

    Computes R* = (R - m)/(s + c)

    R* are the normalized responses
    R are the raw responses
    m is the mean across trials/conditions and time
    s is the standard deviation across trials/conditions and time
    c is a constant, to prevent R* from exploding for s->0

    Inputs:
        data: population response (sequential or simultaneous)
        nrmlpars.ravg: the mean across trials/conditions and time [nun 1]
        nrmlpars.rstd: the std across trials/conditions and time [nun 1]
        nrmlpars.cnst: the constant [nun 1] or [1 1] (def = 0)

    Outputs:
        data_nrml: normalized responses

    data_nrml = tdrNormalize(data, nrmlpars=None)
    """

    if nrmlpars is None:
        nrmlpars = {}
    # Check if data is sequentially or simultaneously recorded
    if "unit" in data and len(data["unit"]) > 0:
        # Sequential recordings
        # Number of units
        nun = len(data["unit"])

        # Initialize
        data_nrml = data

        # Parameters
        if "ravg" not in nrmlpars or len(nrmlpars["ravg"]) == 0:
            nrmlpars["ravg"] = np.zeros((nun, 1))
        if "rstd" not in nrmlpars or len(nrmlpars["rstd"]) == 0:
            nrmlpars["rstd"] = np.ones((nun, 1))
        if "cnst" not in nrmlpars or len(nrmlpars["cnst"]) == 0:
            nrmlpars["cnst"] = np.zeros((nun, 1))
        elif len(nrmlpars["cnst"]) == 1:
            nrmlpars["cnst"] = np.ones((nun, 1)) * nrmlpars["cnst"]
        # Loop over units
        for iun in range(nun):
            # Z-score
            data_nrml["unit"][iun]["response"] = (
                data["unit"][iun]["response"] - nrmlpars["ravg"][iun]
            ) / (nrmlpars["rstd"][iun] + nrmlpars["cnst"][iun])
            # take care of crossval
            if "response_test" in data["unit"][iun]:
                data_nrml["unit"][iun]["response_test"] = (
                    data["unit"][iun]["response_test"] - nrmlpars["ravg"][iun]
                ) / (nrmlpars["rstd"][iun] + nrmlpars["cnst"][iun])

    else:
        # Simultaneous recordings
        # Initialize
        data_nrml = data

        # Dimensions
        nun, npt, ncd = data["response"].shape

        # Parameters
        if "ravg" not in nrmlpars or len(nrmlpars["ravg"]) == 0:
            nrmlpars["ravg"] = np.zeros((nun, 1))
        if "rstd" not in nrmlpars or len(nrmlpars["rstd"]) == 0:
            nrmlpars["rstd"] = np.ones((nun, 1))
        if "cnst" not in nrmlpars or len(nrmlpars["cnst"]) == 0:
            nrmlpars["cnst"] = np.zeros((nun, 1))
        elif len(nrmlpars["cnst"]) == 1:
            nrmlpars["cnst"] = np.ones((nun, 1)) * nrmlpars["cnst"]

        # Mean and STD
        # Replicate nrmlpars['ravg'] along second and third dimensions of data['response']
        Ravg = np.repeat(
            nrmlpars["ravg"][:, np.newaxis, np.newaxis], npt, axis=1
        )
        Ravg = np.repeat(Ravg, ncd, axis=2)
        Ravg = np.squeeze(Ravg)

        Rstd = np.repeat(
            nrmlpars["rstd"][:, np.newaxis, np.newaxis], npt, axis=1
        )
        Rstd = np.repeat(Rstd, ncd, axis=2)
        Rstd = np.squeeze(Rstd)

        Cnst = np.repeat(
            nrmlpars["cnst"][:, np.newaxis, np.newaxis], npt, axis=1
        )
        Cnst = np.repeat(Cnst, ncd, axis=2)
        Cnst = np.squeeze(Cnst)

        # check if any Rstd is zero and set Cnst to 1 for those
        Cnst[Rstd == 0] = 1

        # make sure Ravg, Cnst has same dimension as data['response']
        if Ravg.shape != data["response"].shape:
            # fix for cases where there is only one condition
            if data["response"].shape[2] == 1:
                Ravg = np.repeat(Ravg[:, :, np.newaxis], ncd, axis=2)
                Rstd = np.repeat(Rstd[:, :, np.newaxis], ncd, axis=2)
                Cnst = np.repeat(Cnst[:, :, np.newaxis], ncd, axis=2)
            # fix for cases where there is only one timepoint
            # which reults in Ravg being [nun, ncd] and response [nun, 1, ncd]
            elif len(Ravg.shape) == 2 and data["response"].shape[1] == 1:
                # add one axis to Ravg at dimension 1
                Ravg = np.expand_dims(Ravg, axis=1)
                Rstd = np.expand_dims(Rstd, axis=1)
                Cnst = np.expand_dims(Cnst, axis=1)
            else:
                raise ValueError(
                    "Ravg and data['response'] have different dimensions"
                )

        # Z-score
        data_nrml["response"] = (data["response"] - Ravg) / (Rstd + Cnst)

    return data_nrml
