from unittest import TestCase

from scripts.fixed_prefix_readout_control import (
    ARMS, bridge, pair_result, parse_arm, payload_for, suite, verify_result,
)


class FixedPrefixTests(TestCase):
    def test_only_mask_temperature_and_transport_identity_change(self):
        a = payload_for([1, 2], [10, 11], "a", "scores")
        b = payload_for([1, 2], [10, 11], "b", "masked_greedy")
        self.assertEqual({k: v for k, v in a.items() if k not in {"request_id", "temperature"}},
                         {k: v for k, v in b.items()
                          if k not in {"request_id", "temperature", "allowed_token_ids"}})
        self.assertEqual(b["allowed_token_ids"], [10, 11])
        self.assertEqual(b["temperature"], 0)
        self.assertEqual(a["max_tokens"], 1)

    def test_ties_are_not_silently_converted_to_agreement(self):
        case = {"id": "fixture", "task": "degree_exact",
                "request": {"options": [{"id": "7"}, {"id": "3"}]}}
        result = {"option_ids": ["7", "3"], "selected_option_id": "7",
                  "probabilities": {"7": .5, "3": .5}, "candidate_logprobs": [-1., -1.],
                  "generated_action": "3"}
        rows = [{"arm": arm, "status": "success", "result": dict(result)} for arm in ARMS]
        measured = pair_result(case, rows)
        self.assertFalse(measured["choice_agreement"])
        self.assertTrue(measured["greedy_in_original_argmax_set"])
        self.assertTrue(measured["repeat_choice_agreement"])

    def test_missing_and_failed_arms_are_not_success(self):
        case = {"id": "fixture", "task": "degree_exact"}
        with self.assertRaises(suite.Refusal):
            pair_result(case, [])
        rows = [{"arm": arm, "status": "failure"} for arm in ARMS]
        self.assertEqual(pair_result(case, rows),
                         {"case_id": "fixture", "task": "degree_exact", "complete": False})

    def test_generated_token_maps_to_nonordered_action_not_argmax_substitution(self):
        request = bridge.DecisionRequest("x", {}, "Choose", (
            bridge.DecisionOption("7"), bridge.DecisionOption("3")))
        payload = payload_for([1, 2], [10, 11], "x", "masked_greedy")
        body = {"id": "cmpl-x", "model": bridge.MODEL_ID, "choices": [{
            "index": 0, "prompt_token_ids": [1, 2], "token_ids": [11], "finish_reason": "length",
            "logprobs": {"top_logprobs": [{"token_id:10": -3., "token_id:11": -1.}]},
        }], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
        result = parse_arm(body, request, payload, "masked_greedy")
        self.assertEqual(result["generated_action"], "3")
        case = {"request": {"options": [{"id": "7"}, {"id": "3"}]},
                "prefix": [1, 2], "slots": [10, 11]}
        row = {"arm": "masked_greedy", "result": result}
        verify_result(case, row)
        result["generated_action"] = "7"
        with self.assertRaises(suite.Refusal):
            verify_result(case, row)
        body["choices"][0]["token_ids"] = [10]
        with self.assertRaises(suite.Refusal):
            parse_arm(body, request, payload, "masked_greedy")
