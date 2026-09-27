from copy import deepcopy
import unittest

from scripts import laya_order_control as control
from scripts import laya_cpu_pilot as pilot


class LayaOrderControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances, _ = pilot.panel()
        cls.plan = control.select_plan(cls.instances)

    def test_selection_is_fixed_balanced_lexicographic_and_order_only(self):
        self.assertEqual(len(self.plan), 20)
        self.assertEqual(len({r["instance_id"] for r in self.plan}), 10)
        self.assertEqual(sum(r["reference_answer"] == "yes" for r in self.plan), 10)
        for a, b in zip(self.plan[::2], self.plan[1::2]):
            pool = [i["id"] for i in self.instances if i["task"] == a["task"]
                    and len(i["state"]["nodes"]) == 8
                    and str(i["private"]["answer"]) == a["reference_answer"]]
            self.assertEqual(a["instance_id"], min(pool))
            original = deepcopy(a["request"])
            original["options"].reverse()
            self.assertEqual(original, b["request"])
            self.assertNotIn("reference_answer", original)
            self.assertEqual(a["request_sha256"], control.hash_value(a["request"]))

    def test_unattempted_retains_all_denominators(self):
        attempts = [{"index": p["index"], "request": p["request"],
                     "status": "unattempted", "choice": None} for p in self.plan]
        result = control.analyze(self.plan, attempts)
        self.assertEqual(result["scheduled_calls"], 20)
        self.assertEqual(result["scheduled_paired_cases"], 10)
        self.assertEqual(result["completed_pairs"], 0)
        self.assertEqual(result["by_order"]["original"]["scheduled"], 10)
        self.assertEqual(result["by_order"]["reversed"]["scheduled"], 10)

    def test_changed_graph_or_mapping_fails_replay(self):
        attempts = [{"index": p["index"], "request": deepcopy(p["request"]),
                     "status": "unattempted", "choice": None} for p in self.plan]
        attempts[0]["request"]["state"]["edges"] = []
        with self.assertRaises(AssertionError):
            control.analyze(self.plan, attempts)

    def test_cpu_budgets_not_main_run_budgets(self):
        self.assertEqual(control.LIMITS["max_forwards"], 20)
        self.assertEqual(control.LIMITS["total_seconds"], 600)
        self.assertEqual(control.LIMITS["forward_seconds"], 30)
        self.assertEqual(control.LIMITS["load_seconds"], 300)
        self.assertEqual(pilot.environment()["CUDA_VISIBLE_DEVICES"], "")

    def test_position_and_semantic_constants_are_distinguished(self):
        for semantic_constant in (False, True):
            plan = deepcopy(self.plan)
            attempts = []
            for p in plan:
                p["token_audit"] = {"native_sequence_sha256": "test-sequence"}
                ids = [o["id"] for o in p["request"]["options"]]
                choice = "yes" if semantic_constant else ids[0]
                logits = [1. if i == choice else -1. for i in ids]
                result = {
                    "answer": {"choice": choice, "probabilities": {
                        i: .8 if i == choice else .2 for i in ids}},
                    "raw_logits": logits, "forward_count": 1, "cuda_initialized": False,
                    "coverage": {"compatible": True, "native_sequence_sha256": "test-sequence"}}
                attempts.append({"index": p["index"], "request": p["request"],
                                 "status": "success", "choice": choice, "backend": result})
            result = control.analyze(plan, attempts)
            self.assertEqual(result["semantic_preserved_pairs"], 10 if semantic_constant else 0)
            self.assertEqual(result["position_preserved_pairs"], 0 if semantic_constant else 10)


if __name__ == "__main__":
    unittest.main()
