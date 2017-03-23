import unittest

from revpy import mfrm
from revpy.exceptions import InvalidInputParameters


class MFRMTestHost(unittest.TestCase):

    def test_empty_estimate_host_level(self):
        estimations = mfrm.estimate_host_level({}, {}, {}, 0.9)
        self.assertEqual(estimations, (0, 0, 0))

    def test_estimate_paper_ex1(self):
        """This test is based on the exaple #1 in the paper"""
        utilities = {'fare1': -2.8564, 'fare2': -2.5684}
        probs, nofly_prob = mfrm.selection_probs(utilities, 0.5)

        estimations = mfrm.estimate_host_level({'fare1': 3, 'fare2': 0},
                                               {'fare1': 1, 'fare2': 0.},
                                               probs, nofly_prob)
    
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

        mfrm.estimate_host_level({'fare2': 3}, {'fare2': 1.},
                                 {'fare1': 0.1, 'fare2': 0.4}, 0.5)


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
        utilities = {'fare1': -2.8564, 'fare2': -2.5684}
        probs, nofly_prob = mfrm.selection_probs(utilities, 0.5)

        estimations = mfrm.estimate_class_level({'fare1': 3, 'fare2': 0},
                                                {'fare1': 1, 'fare2': 0.},
                                                probs, nofly_prob)

        self.assertIsInstance(estimations, dict)
        self.assertEqual(len(estimations), 2)
        self.assertEqual(sorted(estimations.keys()), ['fare1', 'fare2'])
        self.assertEqual(sorted(estimations['fare1'].keys()),
                         ['demand', 'recapture', 'spill'])

    def test_estimate_class_level_ex1(self):
        utilities = {'fare1': -2.8564, 'fare2': -2.5684}
        probs, nofly_prob = mfrm.selection_probs(utilities, 0.5)

        estimations = mfrm.estimate_class_level({'fare1': 3, 'fare2': 0},
                                                {'fare1': 1, 'fare2': 0.},
                                                probs, nofly_prob)

        self.assertAlmostEqual(estimations['fare1']['spill'], 0, places=2)
        self.assertAlmostEqual(estimations['fare1']['recapture'], 0.86, 2)
        self.assertAlmostEqual(estimations['fare1']['demand'], 2.14, 2)

        self.assertAlmostEqual(estimations['fare2']['spill'], 2.86, places=2)
        self.assertAlmostEqual(estimations['fare2']['recapture'], 0, 2)
        self.assertAlmostEqual(estimations['fare2']['demand'], 2.86, 2)

    def test_invalid_parameters(self):
        """Should not rise any exception
        """

        mfrm.estimate_class_level({'fare2': 3}, {'fare2': 1.},
                                  {'fare1': 0.1, 'fare2': 0.4}, 0.5)

    def test_non_zero_demand_zero_availability(self):
        with self.assertRaises(InvalidInputParameters):
            mfrm.estimate_class_level({'fare1': 3, 'fare2': 1},
                                      {'fare2': 1.},
                                      {'fare1': 0.1, 'fare2': 0.4}, 0.5)

    def test_estimate_class_level_ex3(self):
        """Example 3 from MFRM paper"""

        probs = {
            'a11': 0.0256,
            'a12': 0.0513,
            'a13': 0.0769,
            'a21': 0.041,
            'a22': 0.0615,
            'a23': 0.0821,
            'a31': 0.0154,
            'a32': 0.0205,
            'a33': 0.0256
        }

        observed = {
            'a11': 2,
            'a12': 5,
            'a13': 0,
            'a21': 4,
            'a22': 0,
            'a23': 0,
            'a31': 0,
            'a32': 3,
            'a33': 6
        }

        availability = {
            'a11': 1,
            'a12': 1,
            'a13': 0.25,
            'a21': 1,
            'a22': 0.5,
            'a23': 0,
            'a31': 1,
            'a32': 1,
            'a33': 0.5
        }

        nofly_prob = 0.6
        host_estimations = mfrm.estimate_host_level(observed, availability,
                                                    probs, nofly_prob)
        estimations = round_tuple(host_estimations)
        self.assertTupleEqual(estimations, (30.16, 13.83, 3.67))

        class_estimations = mfrm.estimate_class_level(observed, availability,
                                                      probs, nofly_prob)
        
        # ensure that class demand, spill and recapture in total
        # equals host level estimations
        total_class = sum([v['demand'] for v in class_estimations.values()])
        self.assertAlmostEqual(total_class, host_estimations[0], 2)

        total_class = sum([v['spill'] for v in class_estimations.values()])
        self.assertAlmostEqual(total_class, host_estimations[1], 2)

        total_class = sum([v['recapture'] for v in class_estimations.values()])
        self.assertAlmostEqual(total_class, host_estimations[2], 2)

        expected = {
            'a11': {'demand': 1.63, 'spill': 0, 'recapture': 0.37},
            'a12': {'demand': 4.08, 'spill': 0, 'recapture': 0.92},
            # 'a13': {'demand': 4.35, 'spill': 4.35, 'recapture': 0.},
            'a13': {'demand': 3.02, 'spill': 3.02, 'recapture': 0.},
            'a21': {'demand': 3.27, 'spill': 0, 'recapture': 0.73},
            # 'a22': {'demand': 2.32, 'spill': 2.32, 'recapture': 0.},
            'a22': {'demand': 1.61, 'spill': 1.61, 'recapture': 0.},
            # 'a23': {'demand': 6.19, 'spill': 6.19, 'recapture': 0.},
            'a23': {'demand': 4.3, 'spill': 4.3, 'recapture': 0.},
            'a31': {'demand': 0, 'spill': 0, 'recapture': 0.},
            'a32': {'demand': 2.45, 'spill': 0, 'recapture': 0.55},
            # NOTE: this example case is different from the paper. Somehow
            # it doesn't satisfy (14): s = d*k, k33 = 0.5 ...
            # It also affects a13, a22, a23 that have 0 bookings and
            # calibrated according to the unaccounted spill
            # 'a33': {'demand': 5.87, 'spill': 0.97, 'recapture': 1.1},
            'a33': {'demand': 9.798, 'spill': 4.899, 'recapture': 1.1}
        }

        for element, values in expected.items():
            for key, value in values.items():
                self.assertAlmostEqual(
                    class_estimations[element][key], value, 2)

    def test_estimate_class_level_regression_1(self):
        utilities = {
            '29255588_3950': 1.4659572,
            '27330948_3950': 2.16431,
            '29255588_2490': 1.1630461,
            '29255578_2990': 1.3300509,
            '29255508_3950': 0.43902999,
            '29255578_3590': 0.83872116,
            '29255578_3950': 0.70265454,
            '29255528_3590': 0.52609205,
            '29255518_3590': 0.52609205,
            '30920848_3950': -0.19642138,
            '27331928_3950': 0.096954226,
            '27337358_3590': 0.52609205,
            '27334478_3590': -0.12226862,
            '29255548_3950': 1.4128097,
            '29255558_3590': 0.2330209,
            '29255588_2990': 0.99219722,
            '29255538_3590': 0.76555932,
            '27341178_3590': 0.61577702,
            '29255548_3590': 1.3927615,
            '29255558_2990': 0.72435057
        }

        observed = {
            '29255588_3950': 3,
            '27330948_3950': 1,
            '29255578_2990': 7,
            '27331928_2450': 1,
            '27331928_3200': 4,
            # missing in 'utilities'
            '27331928_2490': 1,
            '29255588_3018': 1,
            '29255588_2490': 6,
            '29255518_3950': 1,
            '29255578_3018': 1,
            '27331928_3950': 2,
            '29255578_3950': 3,
            '29255548_3950': 2,
            '29255588_2518': 1,
            '29255538_3590': 1,
            '29255578_3590': 1,
            '29255588_2241': 1
        }

        availability = {
            '29255588_3950': 0.650909090909091,
            '27330948_3950': 0.634146341463415,
            '29255578_2990': 0.1875,
            '27331928_2450': 0.00436681222707424,
            '30920848_3950': 0.5,
            '29255538_3950': 0.0411764705882353,
            '29255528_3590': 0.796875,
            '29255518_3950': 0.032967032967033,
            '29255588_3590': 0.0254545454545455,
            '29255518_3590': 0.648351648351648,
            '29255558_3950': 0.00436681222707424,
            '27334478_3590': 0.753246753246753,
            '27334478_3950': 0.0779220779220779,
            '29255528_3950': 0.0703125,
            '29255548_3950': 0.521951219512195,
            '29255558_3590': 0.148471615720524,
            '29255578_3018': 0.00436681222707424,
            '27331928_2490': 0.00436681222707424,
            '29255548_3590': 0.268292682926829,
            '29255588_2241': 0.00436681222707424,
            '29255508_3950': 0.714285714285714,
            '27331928_3200': 0.00436681222707424,
            '29255588_2518': 0.00436681222707424,
            '29255588_3018': 0.00436681222707424,
            '29255588_2490': 0.0872727272727273,
            '27331928_3950': 0.370517928286853,
            '27337358_3590': 0.774193548387097,
            '29255578_3950': 0.213235294117647,
            '29255578_3590': 0.132352941176471,
            '29255588_2990': 0.236363636363636,
            '29255538_3590': 0.911764705882353,
            '27341178_3590': 0.971428571428571,
            '29255558_2990': 0.152838427947598
        }

        probs, nofly_prob = mfrm.selection_probs(utilities, 0.7)

        class_level = mfrm.estimate_class_level(observed, availability,
                                                probs, nofly_prob)

        # no utility
        self.assertTrue('27331928_2490' not in class_level)

        # low utility and no observed demand
        self.assertEqual(class_level['27334478_3590'],
                         {'demand': 0, 'spill': 0, 'recapture': 0})

        # one observation and low utility
        self.assertLess(class_level['29255538_3590']['demand'],
                        observed['29255538_3590'])
        self.assertLess(class_level['29255538_3590']['spill'], 0.1)

        # high utility, many bookings, low availability
        self.assertGreater(class_level['29255578_2990']['demand'],
                           observed['29255578_2990'])
        self.assertGreater(class_level['29255578_2990']['spill'],
                           observed['29255578_2990'])

        # moderate utility, many bookings, moderate availability
        self.assertGreater(class_level['29255578_3950']['demand'],
                           observed['29255578_3950'])
        

def round_tuple(tlp, level=2):
    return tuple([round(e, level) for e in tlp])
