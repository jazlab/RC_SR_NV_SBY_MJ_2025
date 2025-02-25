import numpy as np


def compute_angle(v1, v2):
    """Compute the angle between two vectors.

    Parameters
    ----------
    v1 : array_like
        First vector.
    v2 : array_like
        Second vector.

    Returns
    -------
    angle : float
        Angle between the two vectors.

    """
    return np.degrees(
        np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    )
