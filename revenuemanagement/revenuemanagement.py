"""
High-level revenue management functions
"""

from .helpers import check_fares_decreasing, fill_up, \
    cumulative_booking_limits, incremental_booking_limits
from .optimizers import EMSRb, EMSRb_MR


def protection_levels(fares, demands, sigmas, method='ESMRMb'):

    check_fares_decreasing(fares)

    if method == 'ESMRMb':
        return EMSRb(fares, demands, sigmas)

    elif method == 'ESMRMb_MR':
        protection_levels_, original_fares = EMSRb_MR(fares, demands, sigmas)
        protection_levels_ = fill_up(protection_levels_, fares, original_fares)

        return protection_levels_

    else:
        raise ValueError('Unknown method: "{}"'.format(method))


def booking_limits(fares, demands, sigmas, capacity, method='ESMRMb'):

    check_fares_decreasing(fares)

    protection_levels_ = protection_levels(fares, demands, sigmas, method)
    cumulative_booking_limits_ = cumulative_booking_limits(protection_levels_,
                                                           capacity)
    return incremental_booking_limits(cumulative_booking_limits_)