"""
Fare transformation following the paper "Optimization of Mixed Fare
Structures: Theory and Applications" by Fiig et al (2010).
Currently only supports fully undifferentiated fare structures.
"""


import numpy as np

from .helpers import check_fares_decreasing


def fare_transformation(fares, demands, fare_structure='undifferentiated',
                        return_all=False):


    if fare_structure != 'undifferentiated':
        raise ValueError('structure "{}" not supported'.format(fare_structure))

    check_fares_decreasing(fares)

    Q = demands.cumsum()  # cumulative demand
    TR = fares*Q  # total revenue

    adjusted_fares, adjusted_demand, Q_eff, TR_eff, original_fares = \
        efficient_strategies(Q, TR, fares)

    if return_all:
        return adjusted_fares, adjusted_demand, Q_eff, TR_eff, original_fares
    else:
        return adjusted_fares, adjusted_demand


def efficient_strategies(Q, TR, fares):
    """Recursively removing all inefficient strategies.

    For fare transformation, all inefficient strategies have to be removed.
    Inefficient strategies have a negative marginal revenue (aka adjusted fare).
    See p. 7 of the fare transformation paper.

    """

    #
    adjusted_demand = Q - np.hstack((0, Q[:-1]))
    adjusted_fares = (TR - np.hstack((0, TR[:-1]))) / adjusted_demand

    # if class 1 demand is zero, make the corresponding adjusted fare
    # (which is NaN due to zero by zero division) negative, which marks the
    # corresponding strategy as inefficient
    adjusted_fares[np.isnan(adjusted_fares)] = -np.inf

    # base case
    if all(adjusted_fares >= 0):
        return adjusted_fares, adjusted_demand, Q, TR, fares
    # recursively remove inefficient strategies
    else:
        inefficient = adjusted_fares < 0
        Q = Q[~inefficient]
        TR = TR[~inefficient]
        fares = fares[~inefficient]
        return efficient_strategies(Q, TR, fares)
