"""
High-level revenue management functions for calculating protection
levels and booking limits.
"""
import numpy as np
import pandas as pd

from .helpers import check_fares_decreasing, \
    cumulative_booking_limits, incremental_booking_limits
from .optimizers import EMSRb, EMSRb_MR


def protection_levels(fares, demands, sigmas=None, cap=None, method='ESMRMb'):
    """Calculate protection levels.\

     params:
     `fares`: array of fares (decreasing order)
     `demands`: array of predicted demands for the fares in `fares`
     `sigmas`: standard deviation of the demand predictions
     `cap`: number of seats available
     `method`: optimization method ('ESMRMb', 'ESMRMb_MR')
    """
    check_fares_decreasing(fares)

    if method == 'ESMRMb':
        return EMSRb(fares, demands, sigmas)

    elif method == 'ESMRMb_MR':
        protection_levels_ = EMSRb_MR(fares, demands, sigmas,
                                                      cap)
        return protection_levels_

    else:
        raise ValueError('method "{}" not supported'.format(method))


def booking_limits(fares, demands, sigmas=None, cap=None, method='ESMRMb'):
    """Calculate bookings limits.

     params:
     `fares`: array of fares (decreasing order)
     `demands`: array of predicted demands for the fares in `fares`
     `sigmas`: standard deviation of the demand predictions
     `cap`: number of seats available
     `method`: optimization method ('ESMRMb', 'ESMRMb_MR', 'ESMRMb_MR_step')
    """
    if cap is None:
        raise ValueError('No capacity specified')

    if method == 'ESMRMb_MR_step':
        booking_limits_ = iterative_booking_limits(fares, demands, sigmas,
                                                   cap, 'ESMRMb_MR')
    else:
        protection_levels_ = protection_levels(fares, demands, sigmas,
                                               cap, method)
        cumulative_booking_limits_ = \
            cumulative_booking_limits(protection_levels_, cap)
        booking_limits_ = incremental_booking_limits(cumulative_booking_limits_)

    return booking_limits_


def iterative_booking_limits(fares, demands, sigmas=None, cap=None,
                             method='ESMRMb_MR'):


    # iterate through all possible capacities (remaining seats) and calculate
    # lowest open fare class (fc)
    lowest_open_fc_list = []
    for remaining_cap in range(1, int(cap)):
        protection_levels_ = \
            protection_levels(fares, demands, sigmas, remaining_cap, method)
        lowest_open_fc = max(np.where(pd.notnull(protection_levels_))[0])
        lowest_open_fc_list.append(lowest_open_fc)


    fcs = np.array(range(0, len(fares)))
    booking_limits_ = np.zeros(len(fcs))
    for fc in fcs:
        booking_limits_[fc] = lowest_open_fc_list.count(fc)

    return booking_limits_