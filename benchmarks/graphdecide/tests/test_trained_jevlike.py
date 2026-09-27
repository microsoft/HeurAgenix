import math
import json
from contextlib import nullcontext
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from scripts import trained_jevlike_adapter as adapter
from src.clients.base import DecisionOption, DecisionRequest, InvalidResponseError


class TrainedAdapterTests(unittest.TestCase):
    def request(self, ids=("yes", "10", "0", "A")):
        return DecisionRequest("authored", {"nodes": ["A", "10"]}, "Select a candidate.",
                               tuple(DecisionOption(i, f"Semantic description {i}") for i in ids))

    def result(self):
        return {"selected_option_id": "10",
                "probabilities": {"yes": .1, "10": .5, "0": .2, "A": .2},
                "logits": {"yes": -2., "10": 1., "0": 0., "A": 0.},
                "input_tokens": 50}

    def test_mixed_ids_and_permutations_preserve_semantics(self):
        for ids in [("yes", "10", "0", "A"), ("A", "0", "yes", "10")]:
            request = self.request(ids)
            payload = adapter.public_payload(request)
            self.assertEqual(len(payload["options"]), 4)
            self.assertIn("Semantic description 10", payload["options"][ids.index("10")])
            response, diagnostics = adapter.validate_result(request, self.result(), "trained")
            self.assertEqual(response.selected_option_id, "10")
            self.assertEqual(response.probabilities, self.result()["probabilities"])
            self.assertTrue(diagnostics["action_observation"]["eligible"])

    def test_twelve_numeric_degree_options_roundtrip(self):
        from dataclasses import asdict
        request = self.request(tuple(str(i) for i in range(12)))
        restored = adapter.request_from_dict(asdict(request))
        self.assertEqual(restored, request)
        payload = adapter.public_payload(restored)
        self.assertIn('"id":"11"', payload["options"][11])

    def test_rejects_missing_duplicate_extra_or_nonfinite_scores(self):
        for value in [math.nan, math.inf, -1, 1.1, True, "0.5"]:
            result = self.result()
            result["probabilities"]["10"] = value
            with self.assertRaises(InvalidResponseError):
                adapter.validate_result(self.request(), result, "trained")
        result = self.result()
        del result["probabilities"]["A"]
        with self.assertRaises(InvalidResponseError):
            adapter.validate_result(self.request(), result, "trained")

    def test_no_silent_renormalization_or_choice_repair(self):
        for field, value in [("probabilities", {"yes": .1, "10": .5, "0": .1, "A": .1}),
                             ("selected_option_id", "yes"),
                             ("logits", {"yes": 2., "10": 1., "0": 0., "A": 0.})]:
            result = self.result()
            result[field] = value
            with self.assertRaises(InvalidResponseError) as context:
                adapter.validate_result(self.request(), result, "trained")
            self.assertTrue(context.exception.action_observation["eligible"])

    def test_invalid_action_separate_from_strict_diagnostic(self):
        result = self.result()
        result["selected_option_id"] = "unknown"
        with self.assertRaises(InvalidResponseError) as context:
            adapter.validate_result(self.request(), result, "trained")
        self.assertFalse(context.exception.action_observation["eligible"])
        self.assertEqual(context.exception.diagnostic_code, "invalid_choice")

    def test_invalid_logits_and_usage(self):
        result = self.result()
        result["input_tokens"] = True
        with self.assertRaises(InvalidResponseError):
            adapter.validate_result(self.request(), result, "trained")
        for value in [math.nan, True, "1"]:
            result = self.result()
            result["logits"]["10"] = value
            with self.assertRaises(InvalidResponseError):
                adapter.validate_result(self.request(), result, "trained")


class TrainedSupervisorTests(unittest.TestCase):
    def setUp(self):
        from src.benchmark import extended_tasks as tasks
        self.root = Path(__file__).resolve().parents[1] / "output" / ("trained-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        graph = {"nodes": list(range(5)), "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
                 "directed": False}
        self.instances = [
            tasks._record("maxcut_construct", 0, graph),
            tasks._record("degree_exact", 0, {**graph, "vertex": 0}),
            tasks._record("cycle_detection", 0, graph),
        ]

    def execute(self, fatal=False, load_fail=False):
        from scripts import run_trained_jevlike_pilot as pilot
        owner = self
        owner.seen = []

        class FakeWorker:
            def __init__(self, *args):
                pass

            def receive(self, seconds):
                return {"status": "fatal" if load_fail else "ready"}

            def predict(self, request):
                owner.seen.append(request)
                if fatal:
                    raise TimeoutError("Authored test deadline")
                ids = [o.id for o in request.options]
                return {"status": "success", "result": {
                    "selected_option_id": ids[-1],
                    "probabilities": {i: 1 / len(ids) for i in ids},
                    "input_tokens": None}}

            def close(self):
                owner.closed = True

        with patch.object(pilot, "Worker", FakeWorker):
            pilot.evaluate(self.root, "decider", self.instances, {"code_sha256": {}})
        return json.loads((self.root / "run-decider/status.json").read_text())

    def test_own_history_complete_and_public_worker_requests(self):
        status = self.execute()
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["unattempted_episodes"], 0)
        self.assertEqual(len(self.seen), sum(i["query_budget"] for i in self.instances))
        self.assertTrue(self.closed)
        self.assertTrue(all(type(r) is DecisionRequest for r in self.seen))

    def test_timeout_has_one_failed_attempt_and_all_scheduled_rows(self):
        status = self.execute(fatal=True)
        self.assertEqual(status["status"], "aborted")
        self.assertEqual(status["calls"], 1)
        self.assertEqual(status["unattempted_episodes"], 2)
        self.assertEqual(status["failure_count"], 1)
        rows = (self.root / "run-decider/scores.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(self.closed)

    def test_load_failure_preserves_all_unattempted(self):
        status = self.execute(load_fail=True)
        self.assertEqual(status["calls"], 0)
        self.assertEqual(status["unattempted_episodes"], 3)
        self.assertEqual(status["status"], "aborted")

    def test_parent_launch_uses_unchanged_standard_locks(self):
        from scripts import run_trained_jevlike_pilot as pilot
        from scripts import extended_graph_suite as suite
        registration = {"packages": {}, "code_sha256": {}, "models": {},
                        "selection": str(self.root), "selection_sha256": "authored"}
        path = self.root / "benchmark-registration-parent-locks.json"
        pilot.save(path, registration)
        pilot.save(self.root / "benchmark-registration-parent-locks.sha256", pilot.digest(path))
        with patch.object(pilot.importlib.metadata, "distributions", return_value=[]), \
                patch.object(pilot, "verify_handoff") as handoff, \
                patch.object(pilot, "verify_artifacts"), \
                patch.object(pilot, "panel", return_value=(self.instances, {
                    "selection_sha256": "authored"})), \
                patch.object(suite, "lane_locks", return_value=nullcontext()) as locks, \
                patch.object(pilot, "run_locked") as run:
            pilot.run_gpu(self.root, self.root / "handoff.json")
        locks.assert_called_once_with(("qwen08", "qwen2"))
        handoff.assert_called_once_with(self.root / "handoff.json", path)
        run.assert_called_once_with(self.root, self.instances, registration)
