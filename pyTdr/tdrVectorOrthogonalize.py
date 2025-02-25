"""
This is ported from Matlab code from the original TDR codebase.
"""

import numpy as np


def tdrVectorOrthogonalize(coef_fix, ortpars):
    """
    tdrVectorOrthogonalize orthogonalize regression vectors

    Inputs:
    coef_fix: regression vectors
        .name: vector names {nvc 1}
        .response: vector coefficients [ndm 1 nvc]
        .dimension: dimension labels {ndm 1}
    ortpars.name: vectors to orthogonalize {nrg 1}

    Outputs:
    coef_ort: orthogonalized regression vectors
        .name: vector names {nrg 1}
        .response: vector coefficients [ndm 1 nrg]
        .dimension: dimension labels {ndm 1}
    lowUN_lowTA: projection matrix from task-related subspace (subspace
        basis) into original state space (original basis)
    """

    # Dimensions
    nrg = len(ortpars["name"])
    nun = coef_fix["response"].shape[0]

    # Initialize
    coef_ort = coef_fix.copy()
    coef_ort["name"] = ortpars["name"]
    coef_ort["response"] = np.zeros((nun, 1, nrg))

    raw = np.zeros((nun, nrg))
    for irg in range(nrg):
        # Find vector
        jmatch = coef_fix["name"].index(ortpars["name"][irg])

        # Keep vector
        raw[:, irg] = coef_fix["response"][:, 0, jmatch]

    # Orthogonalize
    qq, rr = np.linalg.qr(raw)
    ort = qq[:, :nrg]

    # Make sure the vectors are pointing in the same direction
    for irg in range(nrg):
        if np.dot(ort[:, irg], coef_fix["response"][:, 0, irg]) < 0:
            ort[:, irg] = -ort[:, irg]

    # Keep what you need
    coef_ort["response"][:, 0, :] = ort
    lowUN_lowTA = ort

    return coef_ort, lowUN_lowTA
