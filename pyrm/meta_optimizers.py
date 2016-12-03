from functools import wraps

import numpy as np

from pyrm.optimizers import calc_EMSRb
from pyrm.fare_transformation import calc_fare_transformation
from pyrm.helpers import fill_nan


def calc_EMSRb_MR(fares, demands, sigmas=None, cap=None):
    """
    EMSRb_MR algorithm following the research paper "Optimization of Mixed Fare
    Structures: Theory and Applications" by Fiig et al (2010).
    Currently only supports fully undifferentiated fare structures.

    parameters:
    ----------

    `fares`: array of fares, has to be provided in decreasing order.
    `demands`: array of demands
    `sigmas`: array of standard deviations of demands
    `cap`: capacity of the resource (e.g. number of seats)

    returns:
    -------

    array of protection levels
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


# TODO: wrap_fare_trafo fails. fix that
def wrap_fare_trafo(optimizer):

    def wrapper(fares, demands, sigmas=None, cap=None):
        if sigmas is None:
            sigmas = np.zeros(fares.shape)

        adjusted_fares, adjusted_demand = \
            calc_fare_transformation(fares, demands, cap=cap)

        # inefficient strategies correspond NaN adjusted fares
        efficient_indices = np.where(~np.isnan(adjusted_fares))[0]

        # calculate protection levels with EMSRb using efficient strategies
        if adjusted_fares[efficient_indices].size:
            protection_levels_temp = optimizer(adjusted_fares[efficient_indices],
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

@wrap_fare_trafo
def calc_EMSRb_MR2(fares, demands, sigmas=None, cap=None):
    return calc_EMSRb(fares, demands, sigmas=None)
