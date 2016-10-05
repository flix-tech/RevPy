import numpy as np
from scipy.stats import norm

from .fare_transformation import fare_transformation


def EMSRb(fares, demands, sigmas):
    """Standard EMSRb algorithm assuming Gaussian distribution of
    demands for the classes.

    Notation adopted from book "The Theory and Practice of Revenue Management"
    by Talluri et al, see page 48.
    """

    # initialize protection levels y
    y = np.zeros(len(fares) - 1)

    for j in range(1, len(fares)):
        S_j = demands[:j].sum()
        p_j_bar = np.sum(demands[:j]*fares[:j]) / demands[:j].sum()  # eq. 2.13
        p_j_plus_1 = fares[j]
        z_alpha = norm.ppf(1 - p_j_plus_1 / p_j_bar)
        sigma = np.sqrt(np.sum(sigmas[:j]**2))  # sigma of joint distribution
        mu = S_j  # mean of joint distribution.
        y[j-1] = mu + z_alpha*sigma

    y[y < 0] = 0  # ensure there are no negative protection levels
    y[np.isnan(y)] = 0  # set NaN protectionlevels to zero

    # protection level for most expensive class should be always 0
    return np.hstack((0, np.round(y)))


def EMSRb_MR(fares, demands, sigmas):
    """
    EMSRb_MR algorithm following the paper "Optimization of Mixed Fare
    Structures: Theory and Applications" by Fiig et al (2010).
    Currently only supports fully undifferentiated fare structures.

   """
    adjusted_fares, adjusted_demand, __, __, original_fares = \
        fare_transformation(fares, demands, return_all=True)

    if len(adjusted_fares):
        protection_levels = EMSRb(adjusted_fares, adjusted_demand, sigmas)
    else:
        # if there is no efficient strategy, return zeros as  protection levels
        protection_levels = np.zeros(fares.shape)

    return protection_levels, original_fares

