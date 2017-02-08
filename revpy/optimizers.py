import numpy as np
from scipy.stats import norm


def calc_EMSRb(fares, demands, sigmas=None):
    """Standard EMSRb algorithm assuming Gaussian distribution of
    demands for the classes.

    Parameters
    ----------
    fares: np array
           fares provided in decreasing order
    demands: np array
           demands for the fares in `fares`
    sigmas: np array
           standard deviations of demands

    Returns
    -------
    np array containing protection levels


    If no standard deviations `sigmas` are provided (deterministic
    demand), simply the cumulative demand is returned as protection
    level.

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
        # TODO: vectorize this loop
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

        # ensure that protection levels are neither negative (e.g. when
        # demand is low and sigma is high) nor NaN (e.g. when demand is 0)
        y[y < 0] = 0
        y[np.isnan(y)] = 0

        # ensure that protection levels are monotonically increasing.
        # can be violated when adjusted fares after fare transformation
        # are not monotonically decreasing
        # TODO: double-check above reasoning
        y = np.maximum.accumulate(y)

    # protection level for most expensive class should be always 0
    return np.hstack((0, np.round(y)))
