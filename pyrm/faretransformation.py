"""
Fare transformation following the paper "Optimization of Mixed Fare
Structures: Theory and Applications" by Fiig et al (2010).
Currently only supports fully undifferentiated fare structures.

Notation of variable names partly follow this paper.
"""


import numpy as np

from .helpers import check_fares_decreasing, fill_nan


def fare_transformation(fares, demands, cap=None,
                        fare_structure='undifferentiated', return_all=False):

    """Transform fares and demands to adjusted fares and adjusted demands.

    params:
    `fares`: array of fares
    `demands`: array of demands
     `cap`: maximum capacity
     `fare_structure`: only 'undifferentiated' is supported at the moment
     `return_all`: when True, return `Q` and `TR`

    returns:
    `adjusted_fares`: array of adjusted fares
    `adjusted_fares`: array of adjusted demands
    `Q_eff_`: array of cumulative demands of efficient strategies
    `TR_eff`: array of total revenues  of efficient strategies
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
    adjusted_fares_, adjusted_demand_, Q_eff_, TR_eff_, eff_indices = \
        efficient_strategies(Q, TR, fares[0])

    # ensure that adjusted fares and demands have the same shape as `fares` by
    # filling indices corresponding to inefficient strategies with NaNs.
    size = fares.shape
    adjusted_fares = fill_nan(size, eff_indices, adjusted_fares_)
    adjusted_demand = fill_nan(size, eff_indices, adjusted_demand_)

    if not return_all:

        return adjusted_fares, adjusted_demand
    else:
        Q_eff = fill_nan(size, eff_indices, Q_eff_)
        TR_eff = fill_nan(size, eff_indices, TR_eff_)

        return adjusted_fares, adjusted_demand, Q_eff, TR_eff


def efficient_strategies(Q, TR, highest_fare, indices=None):
    """Recursively removing all inefficient strategies.

    For fare transformation, all inefficient strategies have to be removed.
    Inefficient strategies have a negative marginal revenue (aka adjusted fare).
    See p. 7 of the fare transformation paper.

    parameters:
    `Q`: array of cumulative demands
    `TR`: array of total revenues
    `highest_fare`: most expensive fare
    `indices`: efficient indices, must be initially None, used in recursion

    returns:
    `adjusted_fares`, `adjusted_demand`, `Q`, `TR` of efficient strategies
    `indices`: indices of the original fare classes which correspond to
               efficient strategies
    """

    adjusted_demand = Q - np.hstack((0, Q[:-1]))
    adjusted_fares = (TR - np.hstack((0, TR[:-1]))) / adjusted_demand

    # class 1 (most expensive class) adjusted fare should always be the original
    # fare (and should correspond to an efficient strategy), even when class 1
    # demand is zero (the above operation yields adjusted_fares[0] = NaN then).
    if adjusted_demand[0] == 0:
        adjusted_fares[0] = highest_fare
    # if subsequent classes have zero demand, mark the strategies as inefficient
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