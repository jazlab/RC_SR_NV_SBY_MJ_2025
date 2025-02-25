"""
This script is used to compare the beta values of two orthogonalized regressions
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats


def compare_slopes_odr(file_1, file_2):

    # Read beta and sd_beta from csv file
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)
    slope1 = df1["beta"].values[0]
    slope2 = df2["beta"].values[0]
    se1 = df1["sd_beta"].values[0]
    se2 = df2["sd_beta"].values[0]
    n1 = df1["length"].values[0]
    n2 = df2["length"].values[0]

    # Calculate the standard error of the difference between slopes
    se_diff = np.sqrt(se1**2 + se2**2)

    # Calculate the t-statistic
    t_stat = (slope1 - slope2) / se_diff

    # Calculate the degrees of freedom
    df = n1 + n2 - 4

    # Calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

    return t_stat, p_value


from utils.LoadSession import findrootdir

root_dir = findrootdir()
file_1 = (
    root_dir + "/stats_paper/Fig3D_E_beta_sd_both_reward_choice_[-0.6, 0.0].csv"
)
file_2 = (
    root_dir + "/stats_paper/Fig3D_E_beta_sd_both_reward_fdbk_[0.0, 0.6].csv"
)
print(compare_slopes_odr(file_1, file_2))
