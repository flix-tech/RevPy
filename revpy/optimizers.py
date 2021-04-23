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

    if sigmas is None or np.all(sigmas == 0):
        # 'deterministic EMSRb' if no sigmas provided
        y = demands.cumsum()

    else:
        # conventional EMSRb
        S_j = demands.cumsum()
        # eq. 2.13
        p_j_bar = (demands * fares).cumsum() / S_j
        p_j_plus_1 = np.hstack([fares[1:], 0])
        # last value of z_alpha will be inf; drop later
        z_alpha = norm.ppf(1 - p_j_plus_1 / p_j_bar)
        # sigma of joint distribution
        sigma = np.sqrt((sigmas**2).cumsum())
        # mean of joint distribution
        mu = S_j
        y = mu + z_alpha * sigma

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
    # drop last value corresponding to inf z_alpha
    return np.hstack((0, np.round(y[:-1])))
