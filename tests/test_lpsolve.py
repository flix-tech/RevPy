import unittest

import numpy as np
import pandas as pd

from revpy import lp_solve


class test_lp_sover(unittest.TestCase):
    def setUp(self):
        # data from:
        # figure 18 and tables 14 and 15 from the article "
        # Airline network revenue management: Considerations for
        # implementation" (2014):
        # http://link.springer.com/article/10.1057/rpm.2013.33.
        class_names = ['H', 'N']
        trip_names = ['SFO_BOS', 'SFO_CLT', 'CLT_BOS', 'ABE_BOS', 'ABE_CLT']
        leg_names = ['SFO_CLT', 'CLT_BOS', 'ABE_CLT']

        # fares and demands for each itinerary
        fares = np.array([[800, 500, 580, 350, 120],   # class H
                          [450, 380, 400, 250, 100]])  # class N

        demands =  np.array([[6, 4, 5, 4, 3],      # class H
                             [15, 14, 8, 11, 5]])  # class N

        # capacities for the 3 legs
        self.cap = [10, 10, 8] # SFO_CLT, CLT_BOS, ABE_CLT

        # trip matrix defining the network (size: n_trips * n_legs)
        A = np.array([[1, 1, 0], # SFO_BOS
                      [1, 0, 0], # SFO_CLT
                      [0, 1, 0], # ...
                      [0, 1, 1],
                      [0, 0, 1]])
        self.fares = pd.DataFrame(fares, index=class_names, columns=trip_names)
        self.demands = pd.DataFrame(demands, index=class_names,
                                    columns=trip_names)
        self.incidence_matrix = pd.DataFrame(A, index=trip_names, columns=leg_names)

    def test_reproduce_paper_example_allocs(self):
        allocation, bid_prices = lp_solve.solve_network_lp_df(self.fares,
                                                              self.demands,
                                                              self.cap,
                                                              self.incidence_matrix)
        expected_alloc = np.array([[ 5,  0],
                                   [ 4,  1],
                                   [ 5,  0],
                                   [ 0,  0],
                                   [ 3,  5]])

        np.testing.assert_allclose(allocation.values, expected_alloc)

    def test_reproduce_paper_example_bid_prices(self):
        allocation, bid_prices = lp_solve.solve_network_lp_df(self.fares,
                                                        self.demands, self.cap,
                                                        self.incidence_matrix)

        expected_bid_prices = np.array([[ 380], [ 420], [  0]])
        np.testing.assert_allclose(bid_prices.values, expected_bid_prices)

    def test_some_nulls(self):
        fares = self.fares
        demands = self.demands
        A = self.incidence_matrix
        demands['test'] = [np.nan, None]
        fares['test'] = [None, np.nan]
        A.loc['test'] = [None, np.nan, 1]

        allocation, bid_prices = lp_solve.solve_network_lp_df(fares, demands,
                                                              self.cap, A)

        expected_alloc = np.array([[ 5,  0],
                                   [ 4,  1],
                                   [ 5,  0],
                                   [ 0,  0],
                                   [ 3,  5],
                                   [ 0,  0]])
        expected_bid_prices = np.array([[ 380], [ 420], [  0]])

        np.testing.assert_equal(allocation.values, expected_alloc)
        np.testing.assert_equal(bid_prices.values, expected_bid_prices)


    def test_all_nulls(self):
        fares = self.fares
        fares[fares.notnull()] = np.nan
        demands = self.demands
        demands[demands.notnull()] = np.nan
        A = self.incidence_matrix
        demands['test'] = [np.nan, None]
        fares['test'] = [None, np.nan]
        A.loc['test'] = [None, np.nan, 1]

        allocation, bid_prices = lp_solve.solve_network_lp_df(fares, demands,
                                                              self.cap, A)

        expected_alloc = np.array([[ 0,  0],
                                   [ 0,  0],
                                   [ 0,  0],
                                   [ 0,  0],
                                   [ 0,  0],
                                   [ 0,  0]])

        np.testing.assert_equal(allocation.values, expected_alloc)