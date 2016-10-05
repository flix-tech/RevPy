import unittest
import numpy as np

# As suggested in http://docs.python-guide.org/en/latest/writing/structure/
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


from revenuemanagement import optimizers, helpers, revenuemanagement,\
    fare_transformation


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

        p, original_fares \
            = optimizers.EMSRb_MR(self.fares, self.demands, self.sigmas)

        self.assertEqual([0.0, 35.0, 52.0, 84.0], p.tolist())
        self.assertEqual([1200, 1000, 800, 600], original_fares.tolist())

    def test_emsrbmr_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        p, __ = optimizers.EMSRb_MR(self.fares, demands, self.sigmas)
        self.assertEqual(np.zeros(demands.shape).tolist(), p.tolist())


class FareTransformationTest(unittest.TestCase):

    def setUp(self):
        self.fares = np.array([1200, 1000, 800, 600, 400, 200])
        self.demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])

    def test_faretrafo_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        adjusted_fares, adjusted_demand =  \
            fare_transformation.fare_transformation(self.fares, demands)

        self.assertEqual([], adjusted_fares.tolist())
        self.assertEqual([], adjusted_demand.tolist())

    def test_example1(self):
        adjusted_fares, adjusted_demand =  \
            fare_transformation.fare_transformation(self.fares, self.demands)

        np.testing.assert_almost_equal(adjusted_fares, [1200, 427, 231, 28], 0)

    def test_example2(self):
        demands = np.array([0, 15, 0, 30, 2, 60])
        adjusted_fares, adjusted_demand =  \
            fare_transformation.fare_transformation(self.fares, demands)

        np.testing.assert_almost_equal(adjusted_fares, [1000, 400])


class HelpersTest(unittest.TestCase):

    pass

