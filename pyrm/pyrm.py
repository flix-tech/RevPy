"""
High-level revenue management functions for calculating protection
levels and booking limits.

Currently includes EMSRb, EMSRb + fare transformation (EMSRb-MR), and a custom
heuristic (EMSRb_MR_step).
"""
import numpy as np

from .helpers import check_fares_decreasing, \
    cumulative_booking_limits, incremental_booking_limits
from .optimizers import EMSRb, EMSRb_MR


def protection_levels(fares, demands, sigmas=None, cap=None, method='EMSRb'):
    """Calculate protection levels.\

     params:
     `fares`: array of fares (decreasing order)
     `demands`: array of predicted demands for the fares in `fares`
     `sigmas`: standard deviation of the demand predictions
     `cap`: number of seats available
     `method`: optimization method ('EMSRb' or 'EMSRb_MR')
    """
    check_fares_decreasing(fares)

    if method == 'EMSRb':
        return EMSRb(fares, demands, sigmas)

    elif method == 'EMSRb_MR':
        protection_levels_ = EMSRb_MR(fares, demands, sigmas, cap)
        return protection_levels_

    else:
        raise ValueError('method "{}" not supported'.format(method))


def booking_limits(fares, demands, sigmas=None, cap=None, method='EMSRb'):
    """Calculate bookings limits.

     params:
     `fares`: array of fares (decreasing order)
     `demands`: array of predicted demands for the fares in `fares`
     `sigmas`: standard deviation of the demand predictions
     `cap`: number of seats available
     `method`: optimization method ('EMSRb', 'EMSRb_MR' or 'EMSRb_MR_step')
    """
    if cap is None:
        raise ValueError('No capacity specified')

    if method == 'EMSRb_MR_step':
        booking_limits_ = iterative_booking_limits(fares, demands, sigmas,
                                                   cap, 'EMSRb_MR')
    else:
        protection_levels_ = protection_levels(fares, demands, sigmas,
                                               cap, method)
        cumulative_booking_limits_ = \
            cumulative_booking_limits(protection_levels_, cap)
        booking_limits_ = incremental_booking_limits(cumulative_booking_limits_)

    return booking_limits_


def iterative_booking_limits(fares, demands, sigmas=None, cap=None,
                             method='EMSRb_MR'):
    """Custom heuristic for calculating booking limits.

    Assume you have a certain demand forecast `demands`. When bookings for a
    resource are made, the capacity reduces. Assuming that the demand forecast
    does not change when the bookings are made (could be e.g. the case for early
    group bookings), we can re-calculate the booking limits for the new
    capacity. Doing this iteratively for all possible remaining capacities and
    recording the cheapest open fare class for each situation allows for the
    calculation of booking limits.
    The use case are e.g. early bookings, that should not influence the demand
    forecast.
    """

    # iterate through all possible capacities (remaining seats) and calculate
    # cheapest open fare class (fc)
    cheapest_open_fc_list = []
    for remaining_cap in range(1, int(cap) + 1):
        protection_levels_ = \
            protection_levels(fares, demands, sigmas, remaining_cap, method)
        cheapest_open_fc = max(np.where(~np.isnan(protection_levels_))[0])
        cheapest_open_fc_list.append(cheapest_open_fc)

    fcs = np.array(range(0, len(fares)))
    booking_limits_ = np.zeros(len(fcs))
    # count the number of times a particular fare class was the cheapest open
    for fc in fcs:
        booking_limits_[fc] = cheapest_open_fc_list.count(fc)

    return booking_limits_