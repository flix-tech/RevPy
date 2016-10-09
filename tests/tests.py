import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath('..'))


from PyRM import optimizers, fare_transformation
from PyRM import helpers
from PyRM import pyrm as rm

class OptimizersTest(unittest.TestCase):

    def setUp(self):
        self.fares = np.array([1200, 1000, 800, 600, 400, 200])
        self.demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])
        self.sigmas = np.array([11.2, 6.6, 7.7, 8.9, 10.4, 12])

    def test_emsrb_stochastic_demand(self):
        # example from page 13 of paper
        # "Optimization of Mixed Fare Structures: Theory and Applications"
        # by Fiig et al.

        p = optimizers.EMSRb(self.fares, self.demands, self.sigmas)

        self.assertEqual([0., 20., 35., 54., 80., 117.], p.tolist())

    def test_emsrb_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        p = optimizers.EMSRb(self.fares, demands, self.sigmas)
        self.assertEqual(np.zeros(demands.shape).tolist(), p.tolist())

    def test_emsrb_partly_zero_demand(self):
        demands = np.array([31.2, 0, 14.8, 19.9, 0, 36.3])
        sigmas = np.zeros(demands.shape)
        p = optimizers.EMSRb(self.fares, demands, sigmas)
        self.assertEqual([0.0, 31.0, 31.0, 46.0, 66.0, 66.0], p.tolist())

    def test_emsrbmr_stochastic_demand(self):
        # example from page 13 of paper
        # "Optimization of Mixed Fare Structures: Theory and Applications"
        # by Fiig et al.

        p = optimizers.EMSRb_MR(self.fares, self.demands, self.sigmas)

        np.testing.assert_equal(np.array([0.0, 35.0, 52.0, 84.0,
                                          np.nan, np.nan]), p)

    def test_emsrbmr_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        p = optimizers.EMSRb_MR(self.fares, demands, self.sigmas)
        self.assertEqual(np.zeros(demands.shape).tolist(), p.tolist())


class FareTransformationTest(unittest.TestCase):

    def setUp(self):
        self.fares = np.array([1200, 1000, 800, 600, 400, 200])
        self.demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])

    def test_faretrafo_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        adjusted_fares, adjusted_demand =  \
            fare_transformation.fare_transformation(self.fares, demands)
        nans = np.ones(demands.shape)*np.nan

        np.testing.assert_equal(nans, adjusted_fares)
        np.testing.assert_equal(nans, adjusted_demand)

    def test_example1(self):
        adjusted_fares, adjusted_demand =  \
            fare_transformation.fare_transformation(self.fares, self.demands)

        np.testing.assert_almost_equal(adjusted_fares, [1200, 427, 231, 28,
                                                        np.nan, np.nan], 0)

    def test_example2(self):
        demands = np.array([0, 15, 0, 30, 2, 60])
        adjusted_fares, adjusted_demand =  \
            fare_transformation.fare_transformation(self.fares, demands)

        np.testing.assert_almost_equal(adjusted_fares, [np.nan, 1000, np.nan,
                                                        400, np.nan, np.nan, ])

    def test_efficient_strategies(self):
        fares = np.array([ 69.5,  59.5,  48.5,  37.5,  29. ])
        demands = np.array([3, 1, 0, 0, 10])
        Q = demands.cumsum()
        TR = Q*fares
        __, __, __, __,  eff_indices = \
            fare_transformation.efficient_strategies(Q, TR)
        self.assertEqual(eff_indices.tolist(), [0, 1, 4])


class RevenueManagementTest(unittest.TestCase):

    def setUp(self):
        self.fares = np.array([1200, 1000, 800, 600, 400, 200])
        self.demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])
        self.sigmas = np.array([11.2, 6.6, 7.7, 8.9, 10.4, 12])
        self.cap = 100

    def test_protection_levels_esmrmb(self):
        p = rm.protection_levels(self.fares, self.demands, self.sigmas,
                                 method='EMSRb')
        self.assertEqual([0., 20., 35., 54., 80., 117.], p.tolist())

    def test_booking_limits_esmrmb(self):
        bl = rm.booking_limits(self.fares, self.demands, self.sigmas,
                                 cap=self.cap, method='EMSRb')
        self.assertEqual([20., 15., 19., 26., 20., 0.], bl.tolist())

    def test_protection_levels_esmrmb_mr(self):
        p = rm.protection_levels(self.fares, self.demands, self.sigmas,
                                 method='EMSRb_MR')
        np.testing.assert_equal(np.array([0.0, 35.0, 52.0, 84.0,
                                          np.nan, np.nan]), p)

    def test_booking_limits_esmrmb_mr(self):
        bl = rm.booking_limits(self.fares, self.demands, self.sigmas,
                                 method='EMSRb_MR', cap=self.cap)
        np.testing.assert_equal(np.array([35., 17., 32., 16., 0., 0.]), bl)

    def test_protection_levels_esmrmb_mr_with_cap(self):
        p = rm.protection_levels(self.fares, self.demands, self.sigmas,
                                 method='EMSRb_MR', cap=40)
        np.testing.assert_equal(np.array([0.0, 39.0, np.nan, np.nan,
                                          np.nan, np.nan]), p)

    def test_esmrmb_mr_stepwise(self):
        bl = rm.booking_limits(self.fares, self.demands, self.sigmas,
                               method='EMSRb_MR_step', cap=40)
        np.testing.assert_equal(bl, np.array([ 37., 2., 0., 0., 0., 0.]))

class HelpersTest(unittest.TestCase):

    def test_fill_nan(self):
        test_array = np.array([0, 1, 2, 3])
        indices = np.array([0, 2])
        values = np.array([10, 100])
        out = helpers.fill_nan(test_array.shape, indices, values)
        expected_out = np.array([10, np.nan, 100, np.nan ])
        np.testing.assert_equal(out, expected_out)

