from copy import deepcopy
import math
from unittest import TestCase

from scripts.summarize_release_matrix import same_aggregates, verify_token_scores, suite


class MatrixScoreVerificationTests(TestCase):
    def test_only_aggregate_float_roundoff_is_tolerated(self):
        self.assertTrue(same_aggregates({"rate": 0.3060628607503607, "n": 136},
                                        {"rate": 0.30606286075036077, "n": 136}))
        for other in ({"rate": 0.306, "n": 136},
                      {"rate": 0.3060628607503607, "n": 135},
                      {"rate": 0.3060628607503607, "n": 136.0}):
            self.assertFalse(same_aggregates({"rate": 0.3060628607503607, "n": 136}, other))
        self.assertFalse(same_aggregates(float("nan"), float("nan")))

    def row(self):
        p = 1/(1+math.exp(-1))
        return {"selected_option_id": "9", "request": {"options": [{"id": "9"}, {"id": "2"}]},
                "usage": {"input_tokens": 100, "output_tokens": 1},
                "backend_accounting": {
                    "option_ids": ["9", "2"], "candidate_logprobs": [-1, -2],
                    "probabilities": {"9": p, "2": 1-p}, "candidate_count": 2,
                    "candidate_prompts": 1, "prefill_tokens": 100, "shared_prefix_tokens": 100,
                    "generated_tokens": 1, "generated_tokens_used_for_decision": False}}

    def test_checks_mapping_probability_argmax_and_accounting(self):
        valid = self.row()
        verify_token_scores(valid)
        for key, value in (("option_ids", ["2", "9"]),
                           ("candidate_logprobs", [float("nan"), -2]),
                           ("probabilities", {"9": 0.5, "2": 0.5}),
                           ("prefill_tokens", 200),
                           ("generated_tokens_used_for_decision", True)):
            bad = deepcopy(valid)
            bad["backend_accounting"][key] = value
            with self.subTest(key=key), self.assertRaises(suite.Refusal):
                verify_token_scores(bad)
        bad = deepcopy(valid)
        bad["selected_option_id"] = "2"
        with self.assertRaises(suite.Refusal):
            verify_token_scores(bad)
