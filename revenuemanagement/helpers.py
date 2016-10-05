import pandas as pd
import numpy as np


def is_decreasing(array):
    return all(x >= y for x, y in zip(array, array[1:]))


def check_fares_decreasing(fares):
    if not is_decreasing(fares):
        raise ValueError('fares must be provided in decreasing order')


def fill_up(values, fares, original_fares):
    """
    Fill up the `values` vector (which corresponds to efficient strategies
    indicated by `original_fares`) with NaNs such that it matches the size
    of the original fare structure, denoted in `fares`.
    """

    ind = np.where(pd.Series(fares).isin(original_fares).values)[0]
    filled = np.ones(fares.shape)*np.nan
    filled[ind] = values

    return filled


def cumulative_booking_limits(protection_levels, capacity):
    book_lim = capacity - protection_levels
    book_lim[book_lim < 0] = 0

    return book_lim


def incremental_booking_limits(book_lim):
    """Convert cumulative booking limits to incremental booking limits"""

    incremental_limits_ = np.zeros(book_lim.shape)
    notnull = ~np.isnan(book_lim)
    book_lim_notnull = book_lim[notnull]
    incremental_limits_notnull = np.diff(- np.hstack((book_lim_notnull, 0)))
    incremental_limits_[notnull] = incremental_limits_notnull
    return incremental_limits_
