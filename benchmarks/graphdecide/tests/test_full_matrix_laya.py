import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts import full_matrix_laya as full


class FullMatrixLayaTests(unittest.TestCase):
    def test_all_960_and_1920_budget(self):
        instances = full.panel()
        self.assertEqual(len(instances), 960)
        self.assertEqual(sum(i["query_budget"] for i in instances), 1920)
        rows = full.evaluation.replay(instances, [],
                                     SimpleNamespace(model=full.pilot.MODEL), native_action=False)
        self.assertEqual(len(rows), 960)
        self.assertTrue(all(not r["feasible"] for r in rows))

    def test_own_history_and_json_replay(self):
        instance = next(i for i in full.panel() if i["task"] == "tsp_construct")
        request = full.public_request(instance, [])
        chosen = request["options"][-1]["id"]
        following = full.public_request(instance, [chosen])
        self.assertNotEqual(following, full.public_request(
            instance, [request["options"][0]["id"]]))
        attempts = []
        history = []
        while (request := full.public_request(instance, history)) is not None:
            action = request["options"][-1]["id"]
            attempts.append({"instance_id": instance["id"], "step": len(history),
                             "request": request, "status": "success",
                             "selected_option_id": action,
                             "resolved_model": full.pilot.MODEL, "fatal": False})
            history.append(action)
        persisted = json.loads(json.dumps(attempts))
        scores = full.evaluation.replay([instance], persisted,
                                       SimpleNamespace(model=full.pilot.MODEL), native_action=False)
        self.assertTrue(scores[0]["feasible"])
        self.assertEqual(scores[0]["decisions"], history)
        persisted[1]["request"] = full.public_request(instance, [])
        with self.assertRaises(Exception):
            full.evaluation.replay([instance], persisted,
                                  SimpleNamespace(model=full.pilot.MODEL), native_action=False)

    def test_tuple_list_histogram_normalization(self):
        value = {"options": ({"id": "0"},), "histogram": dict(Counter([2, 2, 12]))}
        normalized = full.canonical(value)
        self.assertEqual(normalized, json.loads(json.dumps(value)))
        self.assertEqual(normalized["histogram"], {"2": 2, "12": 1})
        self.assertIsInstance(normalized["options"], list)

    def test_unsupported_retains_entire_denominator(self):
        instances = full.panel()
        root = MagicMock()
        root.__truediv__.return_value.read_text.return_value = json.dumps({
            "rows": [{"instance_id": i["id"], "compatible": False} for i in instances]})
        attempts = [{"instance_id": i["id"], "step": 0,
                     "request": full.public_request(i, []), "status": "failure",
                     "selected_option_id": None, "resolved_model": None, "fatal": False,
                     "diagnostic_code": "unsupported", "forward_count": 0} for i in instances]
        scores, summary = full.derive(root, attempts, None, 0.)
        self.assertEqual(len(scores), 960)
        self.assertEqual(summary["scheduled_episodes"], 960)
        self.assertEqual(summary["official_forwards"], 0)
        self.assertEqual(summary["execution_categories"], {"unsupported_initial": 960})
        self.assertEqual(summary, json.loads(json.dumps(summary)))

    def test_budgets_and_cpu_contract(self):
        self.assertEqual(full.LIMITS["max_forwards"], 1920)
        self.assertEqual(full.LIMITS["job_seconds"], 86400)
        self.assertEqual(full.LIMITS["forward_seconds"], 30)
        self.assertEqual(full.pilot.environment()["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(full.LIMITS["context_tokens"], 1024)
        self.assertEqual(full.LIMITS["head_tokens"], 256)
        self.assertEqual(full.LIMITS["option_tokens"], 48)

    def test_lossless_overflow_gates(self):
        import sys
        sys.path.insert(0, str(full.pilot.ROOT / "upstream"))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(full.pilot.ROOT / "model/tokenizer",
                                           local_files_only=True, trust_remote_code=False)
        request = full.pilot.smoke_requests()[0]
        self.assertTrue(full.pilot.coverage(tok, request)["compatible"])
        request["state"] = " graph" * 1100
        self.assertIn("context_overlength", full.pilot.coverage(tok, request)["reasons"])
        request["state"] = "small"
        request["question"] = " question" * 300
        self.assertIn("head_truncation", full.pilot.coverage(tok, request)["reasons"])
        request["question"] = "short"
        request["options"][0]["description"] = " option" * 60
        self.assertIn("option_truncation", full.pilot.coverage(tok, request)["reasons"])


if __name__ == "__main__":
    unittest.main()
