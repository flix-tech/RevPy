import unittest

import numpy as np

from revpy import fare_transformation


class FareTransformationTest(unittest.TestCase):

    def setUp(self):
        # example data from page 13 of research paper
        # "Optimization of Mixed Fare Structures: Theory and Applications"
        # by Fiig et al. (2010)
        self.fares = np.array([1200, 1000, 800, 600, 400, 200])
        self.demands = np.array([31.2, 10.9, 14.8, 19.9, 26.9, 36.3])

    def test_faretrafo_zero_demand(self):
        demands = np.zeros(self.fares.shape)
        adjusted_fares, adjusted_demand =  \
            fare_transformation.calc_fare_transformation(self.fares, demands)

        np.testing.assert_equal([1200, np.nan, np.nan, np.nan, np.nan, np.nan],
                                adjusted_fares)
        np.testing.assert_equal([0, np.nan, np.nan, np.nan, np.nan, np.nan],
                                adjusted_demand)

    def test_example1(self):
        # test example from above mentioned paper
        adjusted_fares, adjusted_demand =  \
            fare_transformation.calc_fare_transformation(self.fares,
                                                         self.demands)

        np.testing.assert_almost_equal(adjusted_fares, [1200, 427, 231, 28,
                                                        np.nan, np.nan], 0)

    def test_example2(self):
        # example containing some zero demands
        demands = np.array([0, 15, 0, 30, 2, 60])
        adjusted_fares, adjusted_demand =  \
            fare_transformation.calc_fare_transformation(self.fares, demands)

        np.testing.assert_almost_equal(adjusted_fares, [1200, 1000, np.nan,
                                                        400, np.nan, np.nan, ])

    def test_efficient_strategies(self):
        fares = np.array([69.5,  59.5,  48.5,  37.5,  29.])
        demands = np.array([3, 1, 0, 0, 10])
        Q = demands.cumsum()
        TR = Q*fares
        __, __, __, __,  eff_indices = \
            fare_transformation.efficient_strategies(Q, TR, fares[0])
        self.assertEqual(eff_indices.tolist(), [0, 1, 4])
