import unittest

from pyrm import mfrm


class MFRMTest(unittest.TestCase):

    def test_estimate_host_level(self):
        estimations = mfrm.estimate_host_level({}, {}, {}, 0.9)
        self.assertEqual(estimations, (0, 0, 0))

    def test_estimate_paper_example_1(self):
        """This test is based on the exaple #1 in the paper"""

        estimations = mfrm.estimate_host_level({'fare1': 3, 'fare2': 0},
                                               {'fare1': -2.8564,
                                                'fare2': -2.5684},
                                               {'fare1': 1, 'fare2': 0.}, 0.5)
    
        self.assertIsInstance(estimations, tuple)
        self.assertEqual(len(estimations), 3)

        # round numbers
        estimations = tuple([round(e, 2) for e in estimations])

        self.assertTupleEqual(estimations, (5, 2.86, 0.86))
