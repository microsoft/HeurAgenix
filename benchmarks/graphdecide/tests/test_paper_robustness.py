from unittest import TestCase

from paper.build_robustness import common_valid, loss, number
from scripts import graph_probability_audit as audit


class RobustnessTableTests(TestCase):
    def test_empty_common_valid_rows_are_undefined(self):
        result = common_valid([{"instance_id": "missing", "task": "adjacency"}],
                              {p: {} for p in audit.PROFILES}, {})
        self.assertEqual(result, {"denominator": 0, "status": "undefined_no_common_valid_rows",
                                  "profiles": None})

    def test_infinite_loss_is_not_rendered_as_missing(self):
        self.assertEqual(loss({"mean_log_loss_is_infinite": True, "mean_log_loss": None}),
                         r"$+\infty$")
        self.assertEqual(loss({"mean_log_loss_is_infinite": False, "mean_log_loss": None}), "--")
        self.assertEqual(number(.1254), "0.125")
