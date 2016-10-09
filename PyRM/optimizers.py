import numpy as np
import pandas as pd

from scipy.stats import norm

from PyRM.fare_transformation import fare_transformation
from .helpers import fill_nan

def EMSRb(fares, demands, sigmas=None):
    """Standard EMSRb algorithm assuming Gaussian distribution of
    demands for the classes.

    Vector `fares` has to be provided in decreasing order.

    If no standard deviations `sigmas` are provided (deterministic demand), the
    cumulative demand is returned as protection levels.

    Notation adopted from book "The Theory and Practice of Revenue Management"
    by Talluri et al, see page 48.
    """

    #check_fares_decreasing(fares)

    # initialize protection levels y
    y = np.zeros(len(fares) - 1)

    if sigmas is None or np.all(sigmas == 0):
        # 'deterministic EMSRb' if no sigmas provided
        y = demands.cumsum()[:-1]

    else:
        # conventional EMSRb
        for j in range(1, len(fares)):
            S_j = demands[:j].sum()
            # eq. 2.13
            p_j_bar = np.sum(demands[:j]*fares[:j]) / demands[:j].sum()
            p_j_plus_1 = fares[j]
            z_alpha = norm.ppf(1 - p_j_plus_1 / p_j_bar)
            # sigma of joint distribution
            sigma = np.sqrt(np.sum(sigmas[:j]**2))
            # mean of joint distribution.
            mu = S_j
            y[j-1] = mu + z_alpha*sigma
        # ensure there are no negative and NaN protection levels
        y[y < 0] = 0
        y[np.isnan(y)] = 0

    # protection level for most expensive class should be always 0
    return np.hstack((0, np.round(y)))


def EMSRb_MR(fares, demands, sigmas=None, cap=None):
    """
    EMSRb_MR algorithm following the paper "Optimization of Mixed Fare
    Structures: Theory and Applications" by Fiig et al (2010).
    Currently only supports fully undifferentiated fare structures.

   """
    if sigmas is None:
        sigmas = np.zeros(fares.shape)

    adjusted_fares, adjusted_demand = \
        fare_transformation(fares, demands, cap=cap)

    notnull = np.where(pd.notnull(adjusted_fares))[0]
    if len(adjusted_fares[notnull]):
        protection_levels_ = EMSRb(adjusted_fares[notnull],
                                   adjusted_demand[notnull], sigmas[notnull])
        protection_levels = fill_nan(fares.shape, notnull, protection_levels_)
    else:
        # if there is no efficient strategy, return zeros as  protection levels
        protection_levels = np.zeros(fares.shape)

    return protection_levels


