"""
Fare transformation following the paper "Optimization of Mixed Fare
Structures: Theory and Applications" by Fiig et al (2010).
Currently only supports fully undifferentiated fare structures.
"""


import numpy as np

from revenuemanagement.helpers import check_fares_decreasing


def fare_transformation(fares, demands, cap=None,
                        fare_structure='undifferentiated', return_all=False):


    if fare_structure != 'undifferentiated':
        raise ValueError('structure "{}" not supported'.format(fare_structure))

    check_fares_decreasing(fares)

    # cumulative demand
    Q = demands.cumsum()
    # shrink Q when it exceeds capacity
    if cap is not None:
        Q[Q > cap] = cap
    # total revenue
    TR = fares*Q

    adjusted_fares_, adjusted_demand_, Q_eff_, TR_eff_, eff_indices = \
        efficient_strategies(Q, TR)

    adjusted_fares = np.ones(fares.shape)*np.nan
    adjusted_fares[eff_indices] = adjusted_fares_

    adjusted_demand = np.ones(fares.shape)*np.nan
    adjusted_demand[eff_indices] = adjusted_demand_

    Q_eff = np.ones(fares.shape)*np.nan
    Q_eff[eff_indices] = Q_eff_

    TR_eff = np.ones(fares.shape)*np.nan
    TR_eff[eff_indices] = TR_eff_

    if return_all:
        return adjusted_fares, adjusted_demand, Q_eff, TR_eff
    else:
        return adjusted_fares, adjusted_demand


# def efficient_strategies(Q, TR, fares):
#     """Recursively removing all inefficient strategies.
#
#     For fare transformation, all inefficient strategies have to be removed.
#     Inefficient strategies have a negative marginal revenue (aka adjusted fare).
#     See p. 7 of the fare transformation paper.
#
#     """

#
#     adjusted_demand = Q - np.hstack((0, Q[:-1]))
#     adjusted_fares = (TR - np.hstack((0, TR[:-1]))) / adjusted_demand
#
#     # if class 1 demand is zero, make the corresponding adjusted fare
#     # (which is NaN due to zero by zero division) negative, which marks the
#     # corresponding strategy as inefficient
#     adjusted_fares[np.isnan(adjusted_fares)] = -np.inf
#
#
#
#     # base case
#     if all(adjusted_fares >= 0):
#         return adjusted_fares, adjusted_demand, Q, TR, fares
#     # recursively remove inefficient strategies
#     else:
#         inefficient = adjusted_fares < 0
#         Q = Q[~inefficient]
#         TR = TR[~inefficient]
#         fares = fares[~inefficient]
#         return efficient_strategies(Q, TR, fares)

def efficient_strategies(Q, TR, indices=None):
    """Recursively removing all inefficient strategies.

    For fare transformation, all inefficient strategies have to be removed.
    Inefficient strategies have a negative marginal revenue (aka adjusted fare).
    See p. 7 of the fare transformation paper.

    """

    adjusted_demand = Q - np.hstack((0, Q[:-1]))
    adjusted_fares = (TR - np.hstack((0, TR[:-1]))) / adjusted_demand

    # if class 1 demand is zero, make the corresponding adjusted fare
    # (which is NaN due to zero by zero division) negative, which marks the
    # corresponding strategy as inefficient
    adjusted_fares[np.isnan(adjusted_fares)] = -np.inf

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
        return efficient_strategies(Q, TR, indices)