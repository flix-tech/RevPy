import numpy as np


def is_decreasing(array):

    return all(x >= y for x, y in zip(array, array[1:]))


def is_increasing(array):

    return is_decreasing(array[::-1])


def check_fares_decreasing(fares):
    if not is_decreasing(fares):
        raise ValueError('fares must be provided in decreasing order')


def fill_nan(array_size, indices, values):
    """
    Return array of size `array_size`, that contains values `values` at
    positions `indices` and nan everywhere else.
    """
    out = np.full(array_size, np.nan)
    out[indices] = values

    return out


def cumulative_booking_limits(protection_levels, capacity):
    """Convert protection level into cumulative booking limits."""

    if np.all(protection_levels == 0):
        # if all protection levels are zero, protect everything for
        # lowest class
        book_lim = np.zeros(protection_levels.shape)
        book_lim[0] = capacity
    else:
        book_lim = capacity - protection_levels
        book_lim[book_lim < 0] = 0

    return book_lim


def incremental_booking_limits(cum_book_lim):
    """Convert cumulative booking limits to incremental booking limits.

    If element in `cum_book_lim` is null, set the resulting incremental
    booking limit to zero.
    """

    incremental_limits = np.zeros(cum_book_lim.shape)
    notnull = ~np.isnan(cum_book_lim)
    book_lim_notnull = cum_book_lim[notnull]
    incremental_limits_notnull = np.diff(- np.hstack((book_lim_notnull, 0)))
    incremental_limits[notnull] = incremental_limits_notnull

    return incremental_limits
