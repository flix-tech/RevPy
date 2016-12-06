import unittest

import numpy as np

from pyrm import helpers


class HelpersTest(unittest.TestCase):

    def test_fill_nan(self):
        test_array = np.array([0, 1, 2, 3])
        indices = np.array([0, 2])
        values = np.array([10, 100])
        out = helpers.fill_nan(test_array.shape, indices, values)
        expected_out = np.array([10, np.nan, 100, np.nan])
        np.testing.assert_equal(out, expected_out)

    def test_incremental_booking_limits(self):
        cum_book_lim = np.array([40, 10, 10, 0])
        incremental_lim = helpers.incremental_booking_limits(cum_book_lim)
        np.testing.assert_equal(incremental_lim, np.array([30, 0, 10, 0]))

    def test_cumulative_booking_limits(self):
        pass

    def test_is_decreasing(self):
        array1 = np.array([-10, 0, 1, 2, 100])
        array2 = np.array([1000, 123, -1, -2, -100])
        array3 = np.array([0, 0, 0, 0])

        self.assertFalse(helpers.is_decreasing(array1))
        self.assertTrue(helpers.is_decreasing(array2))
        self.assertTrue(helpers.is_decreasing(array3))
