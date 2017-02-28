import unittest

from revpy import mfrm
from revpy.exceptions import InvalidInputParameters


class MFRMTestHost(unittest.TestCase):

    def test_empty_estimate_host_level(self):
        estimations = mfrm.estimate_host_level({}, {}, {}, 0.9)
        self.assertEqual(estimations, (0, 0, 0))

    def test_estimate_paper_ex1(self):
        """This test is based on the exaple #1 in the paper"""

        estimations = mfrm.estimate_host_level({'fare1': 3, 'fare2': 0},
                                               {'fare1': -2.8564,
                                                'fare2': -2.5684},
                                               {'fare1': 1, 'fare2': 0.}, 0.5)
    
        self.assertIsInstance(estimations, tuple)
        self.assertEqual(len(estimations), 3)

        # round numbers
        estimations = round_tuple(estimations)

        self.assertTupleEqual(estimations, (5, 2.86, 0.86))

    def test_demand_mass_balance_h(self):
        estimations = mfrm.demand_mass_balance_c(3, 3, 1, 0.86)
        self.assertTupleEqual(round_tuple(estimations), (2.14, 0., 0.86))

    def test_invalid_parameters(self):
        """Should not rise any exception
        """

        mfrm.estimate_host_level({'fare2': 3},
                                 {'fare1': -2.8564, 'fare2': -2.5684},
                                 {'fare2': 1.}, 0.5)


class MFRMTestClass(unittest.TestCase):

    def test_empty_estimate_class_level(self):
        estimations = mfrm.estimate_class_level({}, {}, {}, 0.9)
        self.assertEqual(estimations, {})

    def test_demand_mass_balance_c_ex1(self):
        estimations = mfrm.demand_mass_balance_c(3, 3, 1, 0.86)

        self.assertIsInstance(estimations, tuple)
        self.assertEqual(len(estimations), 3)

        self.assertTupleEqual(round_tuple(estimations), (2.14, 0., 0.86))

    def test_demand_mass_balance_c_ex2_a11(self):
        estimations = mfrm.demand_mass_balance_c(3, 0., 0.1, 0.61)
        self.assertTupleEqual(round_tuple(estimations), (0, 0, 0))

    def test_demand_mass_balance_c_ex2_a21(self):
        estimations = mfrm.demand_mass_balance_c(3, 2, 1, 0.61)
        self.assertTupleEqual(round_tuple(estimations), (1.59, 0, 0.41))

    def test_estimate_class_level_struct(self):
        estimations = mfrm.estimate_class_level({'fare1': 3, 'fare2': 0},
                                                {'fare1': -2.8564,
                                                 'fare2': -2.5684},
                                                {'fare1': 1, 'fare2': 0.}, 0.5)

        self.assertIsInstance(estimations, dict)
        self.assertEqual(len(estimations), 2)
        self.assertEqual(sorted(estimations.keys()), ['fare1', 'fare2'])
        self.assertEqual(sorted(estimations['fare1'].keys()),
                         ['demand', 'recapture', 'spill'])

    def test_estimate_class_level_ex1(self):
        estimations = mfrm.estimate_class_level({'fare1': 3, 'fare2': 0},
                                                {'fare1': -2.8564,
                                                 'fare2': -2.5684},
                                                {'fare1': 1, 'fare2': 0.}, 0.5)

        self.assertAlmostEqual(estimations['fare1']['spill'], 0, places=2)
        self.assertAlmostEqual(estimations['fare1']['recapture'], 0.86, 2)
        self.assertAlmostEqual(estimations['fare1']['demand'], 2.14, 2)

        self.assertAlmostEqual(estimations['fare2']['spill'], 2.86, places=2)
        self.assertAlmostEqual(estimations['fare2']['recapture'], 0, 2)
        self.assertAlmostEqual(estimations['fare2']['demand'], 2.86, 2)

    def test_invalid_parameters(self):
        """Should not rise any exception
        """

        mfrm.estimate_class_level({'fare2': 3},
                                  {'fare1': -2.8564, 'fare2': -2.5684},
                                  {'fare2': 1.}, 0.5)

    def test_non_zero_demand_zero_availability(self):
        with self.assertRaises(InvalidInputParameters):
            mfrm.estimate_class_level({'fare1': 3, 'fare2': 1},
                                      {'fare1': -2.8564, 'fare2': -2.5684},
                                      {'fare2': 1.}, 0.5)


def round_tuple(tlp, level=2):
    return tuple([round(e, level) for e in tlp])
