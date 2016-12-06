import unittest

import numpy as np

from pyrm import optimizers, meta_optimizers
from pyrm.helpers import is_increasing


class OptimizersTest(unittest.TestCase):

    def setUp(self):
        # example data from page 13 of research paper
        # "Optimization of Mixed Fare Structures: Theory and Applications"
        # by Fiig et al. (2010)
        self.fares = np.array([1200, 1000, 800, 600, 400, 200])
        self.demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])
        self.sigmas = np.array([11.2, 6.6, 7.7, 8.9, 10.4, 12])

    def test_emsrb_stochastic_demand(self):
        # test example data from above mentioned paper
        p = optimizers.calc_EMSRb(self.fares, self.demands, self.sigmas)
        self.assertEqual([0., 20., 35., 54., 80., 117.], p.tolist())

    def test_emsrb_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        p = optimizers.calc_EMSRb(self.fares, demands, self.sigmas)
        self.assertEqual(np.zeros(demands.shape).tolist(), p.tolist())

    def test_emsrb_partly_zero_demand(self):
        demands = np.array([31.2, 0, 14.8, 19.9, 0, 36.3])
        sigmas = np.zeros(demands.shape)
        p = optimizers.calc_EMSRb(self.fares, demands, sigmas)
        self.assertEqual([0.0, 31.0, 31.0, 46.0, 66.0, 66.0], p.tolist())

    def test_emsrbmr_stochastic_demand(self):
        # test example data from above mentioned paper
        p = meta_optimizers.calc_EMSRb_MR(self.fares, self.demands, self.sigmas)
        np.testing.assert_equal(np.array([0.0, 35.0, 52.0, 84.0,
                                          np.nan, np.nan]), p)

    def test_emsrbmr_stochastic_demand_with_cap(self):
        cap = 50
        # test example data from above mentioned paper
        p = meta_optimizers.calc_EMSRb_MR(self.fares, self.demands,
                                          self.sigmas, cap)
        np.testing.assert_equal(np.array([0.0, 35.0, np.nan, np.nan,
                                          np.nan, np.nan]), p)

    def test_emsrbmr_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        p = meta_optimizers.calc_EMSRb_MR(self.fares, demands, self.sigmas)
        np.testing.assert_equal(np.array([0, np.nan, np.nan, np.nan,
                                          np.nan, np.nan]), p)

    def test_emsrb_nan_demand(self):
        demands = np.full(self.demands.shape, np.nan)
        p = optimizers.calc_EMSRb(self.fares, demands, self.sigmas)
        self.assertEqual(np.zeros(demands.shape).tolist(), p.tolist())

    def test_emsrbmr_protecion_levels_increasing(self):
        demands = np.array([1,  0,  0,  1,  0, 17,  2,  6])
        sigmas = np.array([4, 4, 4, 4, 4, 4, 4, 4])
        fares = np.array([12.97, 9.07, 8.65, 7.92, 7.66, 6.25, 6.08, 5.91])
        cap = 55
        y = meta_optimizers.calc_EMSRb_MR(fares, demands, sigmas)
        self.assertTrue(is_increasing(y[~np.isnan(y)]))
