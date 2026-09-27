import unittest
from dataclasses import asdict
import json
from types import SimpleNamespace

from scripts.summarize_laya_cpu_pilot import metrics


class LayaSummaryTests(unittest.TestCase):
    def test_empty_conditional_subset(self):
        result = metrics([])
        self.assertEqual(result["scheduled"], 0)
        self.assertEqual(result["optimization_scheduled"], 0)

    def test_infeasible_is_not_counted_correct_or_optimal(self):
        rows = [
            {"kind": "exact", "feasible": True, "correct": True},
            {"kind": "exact", "feasible": False, "correct": None},
            {"kind": "optimization", "feasible": False, "optimal": None},
            {"kind": "optimization", "feasible": True, "optimal": False},
        ]
        result = metrics(rows)
        self.assertEqual(result["scheduled"], 4)
        self.assertEqual(result["feasible"], 2)
        self.assertEqual(result["exact_correct"], 1)
        self.assertEqual(result["exact_scheduled"], 2)
        self.assertEqual(result["optimization_feasible"], 1)
        self.assertEqual(result["optimization_scheduled"], 2)
        self.assertEqual(result["optimization_optimal"], 0)

    def test_durable_json_replay_fixes_only_representation(self):
        from scripts import laya_cpu_pilot as pilot
        from scripts import evaluate_graph_release as evaluation
        instances, _ = pilot.panel()
        instance = instances[0]
        request = evaluation.public_request(instance, [])
        row = {"instance_id": instance["id"], "step": 0, "request": asdict(request),
               "status": "success", "selected_option_id": request.options[0].id,
               "resolved_model": pilot.MODEL, "fatal": False}
        with self.assertRaises(evaluation.suite.shared.Refusal):
            evaluation.replay([instance], [row], SimpleNamespace(model=pilot.MODEL),
                              native_action=False)
        durable = json.loads(json.dumps(row))
        scores = evaluation.replay([instance], [durable], SimpleNamespace(model=pilot.MODEL),
                                   native_action=False)
        self.assertEqual(len(scores), 1)
        self.assertTrue(scores[0]["feasible"])
        self.assertEqual(durable["request"]["options"], [asdict(o) for o in request.options])


if __name__ == "__main__":
    unittest.main()
