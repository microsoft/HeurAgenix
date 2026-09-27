import unittest

from scripts.verify_laya_order_control import compare_json


class OrderControlJsonVerificationTests(unittest.TestCase):
    def test_integer_histogram_keys_round_trip_as_json_strings(self):
        compare_json({"indices": {0: 10, 1: 0}, "scores": (1., 2.)},
                     {"indices": {"0": 10, "1": 0}, "scores": [1., 2.]})

    def test_real_mapping_or_score_changes_are_not_hidden(self):
        with self.assertRaises(AssertionError):
            compare_json({"indices": {0: 10}}, {"indices": {"1": 10}})
        with self.assertRaises(AssertionError):
            compare_json({"scores": [1., 2.]}, {"scores": [2., 1.]})


if __name__ == "__main__":
    unittest.main()
