import unittest

import numpy as np

from pyrm import pyrm


class PyRMTest(unittest.TestCase):

    def setUp(self):
        self.fares = np.array([1200, 1000, 800, 600, 400, 200])
        self.demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])
        self.sigmas = np.array([11.2, 6.6, 7.7, 8.9, 10.4, 12])
        self.cap = 100

    def test_protection_levels_esmrmb(self):
        p = pyrm.protection_levels(self.fares, self.demands, self.sigmas,
                                   method='EMSRb')
        self.assertEqual([0., 20., 35., 54., 80., 117.], p.tolist())

    def test_booking_limits_esmrmb(self):
        bl = pyrm.booking_limits(self.fares, self.demands, cap=self.cap,
                                 sigmas=self.sigmas, method='EMSRb')
        self.assertEqual([20., 15., 19., 26., 20., 0.], bl.tolist())

    def test_booking_limits_esmrmb_nan_demand(self):
        demands = np.full(self.demands.shape, np.nan)
        bl = pyrm.booking_limits(self.fares, demands, cap=self.cap,
                                 sigmas=self.sigmas, method='EMSRb')
        self.assertEqual([self.cap, 0., 0., 0., 0., 0.], bl.tolist())

    def test_booking_limits_esmrmb_zero_demand(self):
        demands = np.zeros(self.demands.shape)
        bl = pyrm.booking_limits(self.fares, demands, cap=self.cap,
                                 sigmas=self.sigmas, method='EMSRb')
        self.assertEqual([self.cap, 0., 0., 0., 0., 0.], bl.tolist())

    def test_protection_levels_esmrmb_mr(self):
        p = pyrm.protection_levels(self.fares, self.demands, self.sigmas,
                                   method='EMSRb_MR')
        np.testing.assert_equal(np.array([0.0, 35.0, 52.0, 84.0,
                                          np.nan, np.nan]), p)

    def test_booking_limits_esmrmb_mr(self):
        bl = pyrm.booking_limits(self.fares, self.demands, cap=self.cap,
                                 sigmas=self.sigmas, method='EMSRb_MR')
        np.testing.assert_equal(np.array([35., 17., 32., 16., 0., 0.]), bl)

    def test_protection_levels_esmrmb_mr_with_cap(self):
        p = pyrm.protection_levels(self.fares, self.demands, self.sigmas,
                                   method='EMSRb_MR', cap=40)
        np.testing.assert_equal(np.array([0.0, 39.0, np.nan, np.nan,
                                          np.nan, np.nan]), p)

    def test_esmrmb_mr_stepwise(self):
        bl = pyrm.booking_limits(self.fares, self.demands, cap=40,
                                 sigmas=self.sigmas, method='EMSRb_MR_step')
        np.testing.assert_equal(bl, np.array([39., 1., 0., 0., 0., 0.]))

    def test_esmrmb_mr_stepwise_zero_demand(self):
        demands = np.zeros(self.demands.shape)
        bl = pyrm.booking_limits(self.fares, demands, self.cap,
                                 sigmas=self.sigmas, method='EMSRb_MR_step')
        np.testing.assert_equal(bl, np.array([self.cap, 0., 0., 0., 0., 0.]))

    def test_esmrmb_mr_stepwise_nan_demand(self):
        demands = np.full(self.demands.shape, np.nan)
        bl = pyrm.booking_limits(self.fares, demands, self.cap,
                                 sigmas=self.sigmas, method='EMSRb_MR_step')
        np.testing.assert_equal(bl, np.array([self.cap, 0., 0., 0., 0., 0.]))

    def test_esmrmb_mr_stepwise_cap_sum(self):
        bl = pyrm.booking_limits(self.fares, self.demands, cap=self.cap,
                                 sigmas=self.sigmas, method='EMSRb_MR_step')
        np.testing.assert_equal(bl.sum(), self.cap)
