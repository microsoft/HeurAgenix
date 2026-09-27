import unittest

from scripts.summarize_trained_jevlike_probabilities import trace


class ProbabilityTraceTests(unittest.TestCase):
    def runtime(self):
        return {"runtime": {"candidate": "kev", "dtype": "float32",
                            "temperature": 2.406050072164233}}

    def test_native_temperature_and_explicit_confidence_semantics(self):
        attempt = {"instance_id": "authored", "step": 0,
                   "request": {"request_id": "authored-0"}, "status": "success",
                   "selected_option_id": "10", "probabilities": {"0": .2, "10": .8},
                   "probability_kind": "native", "strict_status": "success",
                   "action_observation": {"eligible": True},
                   "backend": {"result": {"logits": {"0": 0., "10": 1.38629436}}}}
        row = trace(attempt, self.runtime())
        self.assertEqual(row["native_temperature"], 2.406050072164233)
        self.assertEqual(row["confidence"], .8)
        self.assertEqual(row["probabilities"], attempt["probabilities"])
        self.assertEqual(row["logits"], attempt["backend"]["result"]["logits"])
        self.assertIn("uncalibrated", row["confidence_semantics"])

    def test_failure_does_not_invent_confidence(self):
        attempt = {"instance_id": "authored", "step": 0,
                   "request": {"request_id": "authored-0"}, "status": "failure",
                   "selected_option_id": None}
        row = trace(attempt, self.runtime())
        self.assertIsNone(row["confidence"])
        self.assertIsNone(row["probabilities"])
