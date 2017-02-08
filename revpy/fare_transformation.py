"""
Fare transformation following the paper "Optimization of Mixed Fare
Structures: Theory and Applications" by Fiig et al (2010).
Currently only supports fully undifferentiated fare structures.

Notation of variable names partly follow this paper.
"""


import numpy as np

from revpy.helpers import check_fares_decreasing, fill_nan


def calc_fare_transformation(fares, demands, cap=None,
                             fare_structure='undifferentiated',
                             return_all=False):

    """Transform fares and demands to adjusted fares and adjusted demands.

    Parameters
    ----------
    fares: np array
           fares provided in decreasing order
    demands: np array
           demands for the fares in `fares`
    cap: int, capacity
    fare_structure: str
           only 'undifferentiated' is supported at the moment
    return_all: bool
           when True, return `Q` and `TR`

    Returns
    -------
    adjusted_fares: np array
    adjusted_fares: np array
    Q_eff: np array
           cumulative demands of efficient strategies
    TR_eff: np array
           total revenues of efficient strategies
    """

    if fare_structure != 'undifferentiated':
        raise ValueError('dare structure "{}" not supported'
                         ''.format(fare_structure))

    check_fares_decreasing(fares)

    # cumulative demand
    Q = demands.cumsum()

    # shrink Q when it exceeds capacity
    if cap is not None:
        Q[Q > cap] = cap

    # total revenue
    TR = fares*Q

    # calculate fare adjustment, remove inefficient strategies
    adjusted_fares_temp, adjusted_demand_temp, Q_eff_temp, \
        TR_eff_temp, eff_indices = efficient_strategies(Q, TR, fares[0])

    # ensure that adjusted fares and demands have the same shape as `fares` by
    # filling indices corresponding to inefficient strategies with NaNs.
    size = fares.shape
    adjusted_fares = fill_nan(size, eff_indices, adjusted_fares_temp)
    adjusted_demand = fill_nan(size, eff_indices, adjusted_demand_temp)

    if not return_all:

        return adjusted_fares, adjusted_demand
    else:
        Q_eff = fill_nan(size, eff_indices, Q_eff_temp)
        TR_eff = fill_nan(size, eff_indices, TR_eff_temp)

        return adjusted_fares, adjusted_demand, Q_eff, TR_eff


def efficient_strategies(Q, TR, highest_fare, indices=None):
    """Recursively removing all inefficient strategies.

    For fare transformation, all inefficient strategies have to be
    removed. Inefficient strategies have a negative marginal revenue
    (aka adjusted fare). See p. 7 of the fare transformation paper.

    Parameters
    ----------

    Q: np array
        cumulative demands
    TR: np array
        total revenues
    highest_fare: float, most expensive fare
    indices: np array
        efficient indices, must be initially set to None, used in recursion

    Returns
    -------
    adjusted_fares, adjusted_demand, Q, TR: np arrays
        fares, demand, cumulative demand and total revenue for efficient
        strategies
    indices: np.array
        indices of the original fare classes which correspond to  efficient
        strategies
    """

    adjusted_demand = Q - np.hstack((0, Q[:-1]))
    adjusted_fares = (TR - np.hstack((0, TR[:-1]))) / adjusted_demand

    # class 1 (most expensive class) adjusted fare should always be the
    # original fare (and should correspond to an efficient strategy),
    # even when class 1 demand is zero (the above operation yields
    # adjusted_fares[0] = NaN then).
    if adjusted_demand[0] == 0 or np.isnan(adjusted_demand[0]):
        adjusted_fares[0] = highest_fare

    # if subsequent classes have zero demand, mark the strategies as
    # inefficient
    adjusted_fares[np.isnan(adjusted_fares)] = -1

    # initialize indices
    if indices is None:
        indices = np.arange(0, len(Q))

    # base case
    if all(adjusted_fares >= 0):

        return adjusted_fares, adjusted_demand, Q, TR, indices
    # recursively remove inefficient strategies
    else:
        inefficient = adjusted_fares < 0
        Q = Q[~inefficient]
        TR = TR[~inefficient]
        indices = indices[~inefficient]

        return efficient_strategies(Q, TR, highest_fare, indices)


def fare_trafo_decorator(optimizer):
    """Decorator that wraps the fare trafo around an optimizer."""

    def wrapper(fares, demands, sigmas=None, cap=None):
        if sigmas is None:
            sigmas = np.zeros(fares.shape)

        adjusted_fares, adjusted_demand = \
            calc_fare_transformation(fares, demands, cap=cap)

        # inefficient strategies correspond NaN adjusted fares
        efficient_indices = np.where(~np.isnan(adjusted_fares))[0]
        # calculate protection levels with `optimizer` using efficient
        # strategies only
        if adjusted_fares[efficient_indices].size:
            protection_levels_temp = optimizer(
                adjusted_fares[efficient_indices],
                adjusted_demand[efficient_indices],
                sigmas[efficient_indices])
            protection_levels = fill_nan(fares.shape, efficient_indices,
                                         protection_levels_temp)
        else:
            # if there is no efficient strategy, return zeros as  protection
            #  levels
            protection_levels = np.zeros(fares.shape)

        return protection_levels

    return wrapper
