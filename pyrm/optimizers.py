import numpy as np

from scipy.stats import norm

from pyrm.fare_transformation import calc_fare_transformation
from pyrm.helpers import fill_nan


def calc_EMSRb(fares, demands, sigmas=None):
    """Standard EMSRb algorithm assuming Gaussian distribution of
    demands for the classes.

    params:
    `fares`: vector of fares, has to be provided in decreasing order.
    `demands`: vector of demands
    `sigmas`: standard deviation of demands

    If no standard deviations `sigmas` are provided (deterministic demand),
    simply the cumulative demand is returned as protection level.

    Notation of variables adopted from book
    "The Theory and Practice of Revenue Management"
    by Talluri et al, see page 48.
    """

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

        # ensure that protection levels are neither negative (e.g. when demand
        # is low and sigma is high) nor NaN (e.g. when demand is 0)
        y[y < 0] = 0
        y[np.isnan(y)] = 0

    # protection level for most expensive class should be always 0
    return np.hstack((0, np.round(y)))


def calc_EMSRb_MR(fares, demands, sigmas=None, cap=None):
    """
    EMSRb_MR algorithm following the research paper "Optimization of Mixed Fare
    Structures: Theory and Applications" by Fiig et al (2010).
    Currently only supports fully undifferentiated fare structures.

   """
    if sigmas is None:
        sigmas = np.zeros(fares.shape)

    adjusted_fares, adjusted_demand = \
        calc_fare_transformation(fares, demands, cap=cap)

    # inefficient strategies correspond NaN adjusted fares
    efficient_indices = np.where(~np.isnan(adjusted_fares))[0]

    # calculate protection levels with EMSRb using efficient strategies
    if adjusted_fares[efficient_indices].size:
        protection_levels_temp = calc_EMSRb(adjusted_fares[efficient_indices],
                                        adjusted_demand[efficient_indices],
                                        sigmas[efficient_indices])
        protection_levels = fill_nan(fares.shape, efficient_indices,
                                     protection_levels_temp)
    else:
        # if there is no efficient strategy, return zeros as  protection levels
        protection_levels = np.zeros(fares.shape)

    return protection_levels


