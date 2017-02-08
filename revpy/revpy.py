"""
High-level revenue management functions for calculating booking limits.

Currently includes EMSRb, EMSRb + fare transformation (EMSRb-MR), and a
custom heuristic (EMSRb_MR_step).

"""

import numpy as np

from revpy.helpers import check_fares_decreasing, \
    cumulative_booking_limits, incremental_booking_limits
from revpy.optimizers import calc_EMSRb
from revpy.meta_optimizers import calc_EMSRb_MR


def booking_limits(fares, demands, cap, sigmas=None, method='EMSRb'):
    """Calculate bookings limits.

    Parameters
    ----------
    fares: np array
           fares provided in decreasing order
    demands: np array
           demands for the fares in `fares`
    cap: int, capacity
    sigmas: np array
           standard deviations of demands
    method: str
           optimization method ('EMSRb', 'EMSRb_MR' or 'EMSRb_MR_step')

    Returns
    -------
    np array of booking limits for each fare class

    """
    if method == 'EMSRb_MR_step':
        book_lim = iterative_booking_limits(fares, demands, cap, sigmas,
                                            'EMSRb_MR')
    else:
        prot_levels = protection_levels(fares, demands, sigmas, cap, method)
        cum_book_lim = cumulative_booking_limits(prot_levels, cap)
        book_lim = incremental_booking_limits(cum_book_lim)

    return book_lim


def protection_levels(fares, demands, sigmas=None, cap=None, method='EMSRb'):
    """Calculate protection levels.

    Parameters
    ----------
    fares: np array
           fares provided in decreasing order
    demands: np array
           demands for the fares in `fares`
    cap: int, capacity
    sigmas: np array
           standard deviations of demands
    method: str
           optimization method ('EMSRb'or 'EMSRb_MR')

    Returns
    -------
    np array of protection levels for each fare class

    """
    check_fares_decreasing(fares)

    if method == 'EMSRb':
        return calc_EMSRb(fares, demands, sigmas)

    elif method == 'EMSRb_MR':
        prot_levels = calc_EMSRb_MR(fares, demands, sigmas, cap)
        return prot_levels

    else:
        raise ValueError('method "{}" not supported'.format(method))


def iterative_booking_limits(fares, demands, cap, sigmas=None,
                             method='EMSRb_MR'):
    """Custom heuristic for iteratively calculating booking limits.

    Parameters
    ----------
    fares: np array
           fares provided in decreasing order
    demands: np array
           demands for the fares in `fares`
    cap: int, capacity
    sigmas: np array
           standard deviations of demands
    method: str
           optimization method ('EMSRb'or 'EMSRb_MR')

    Returns
    -------
    np array of booking limits for each fare class

    Assume you have a certain demand forecast `demands`. When bookings
    for a resource are made, the capacity reduces. Assuming that the
    demand forecast does not change when the bookings are made (could be
    e.g. the case for early group bookings), we can re-calculate the
    booking limits for the new capacity. Doing this iteratively for all
    possible remaining capacities and recording the cheapest open fare
    class for each situation allows for the calculation of booking
    limits. The use case are e.g. early bookings, that should not
    influence the demand forecast.

    """

    # iterate through all possible capacities (remaining seats) and
    # calculate cheapest open fare class (fc)
    cheapest_open_fc_list = []
    for remaining_cap in range(1, int(cap) + 1):
        temp_book_lims = \
            booking_limits(fares, demands, remaining_cap, sigmas, method)
        cheapest_open_fc = max(np.where(temp_book_lims > 0)[0])
        cheapest_open_fc_list.append(cheapest_open_fc)
    fcs = np.array(range(0, len(fares)))
    book_lims = np.zeros(len(fcs))

    # count the number of times a particular fare class was the cheapest
    # open
    for fc in fcs:
        book_lims[fc] = cheapest_open_fc_list.count(fc)

    return book_lims
