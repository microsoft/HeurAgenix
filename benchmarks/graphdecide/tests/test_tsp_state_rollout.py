import asyncio
from contextlib import nullcontext
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from scripts import tsp_state_rollout as rollout
from scripts.analyze_tsp_state_rollout import analyze
from src.benchmark import extended_tasks as tasks
from src.clients.base import InvalidResponseError, ProviderHTTPError
from test_extended_graph_suite import FakeClient


class TspStateRolloutTests(TestCase):
    def instances(self):
        return [i for i in tasks.build_curriculum(node_counts=(5,), samples_per_size=1)
                if i["task"] == "tsp_construct"]

    def execute(self, root, factory, **kwargs):
        async def preflight(_):
            return None
        with patch.object(rollout.suite, "lane_locks", return_value=nullcontext()):
            return asyncio.run(rollout.run(root, self.instances(), ("jev",), {},
                                          client_factory=factory, preflight=preflight, **kwargs))

    def test_native_action_policy_preserves_strict_failure_without_choice_repair(self):
        class Rounded(FakeClient):
            async def predict(self, request):
                ids = [o.id for o in request.options]
                body = {"model": rollout.suite.model_configs(("jev",))["jev"].model,
                        "answers": {"decision": {
                            "type": "choice", "choice": ids[-1], "confidence": 0.5,
                            "probabilities": {key: 0.99/len(ids) for key in ids}}},
                        "usage": {}}
                raise InvalidResponseError("probabilities must sum to 1", raw_output=body,
                                           diagnostic_code="invalid_probability_sum")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "observed"
            result = self.execute(root, lambda _, **kw: Rounded(kw["model"]),
                                  native_action_observation=True)
            self.assertEqual(result["jev"]["status"], "completed")
            self.assertEqual(result["jev"]["strict_validation_failures"], 6)
            rows = rollout.suite.read_rows(root / "jev-attempts.jsonl")
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row["status"], "success")
                self.assertEqual(row["strict_status"], "failure")
                self.assertEqual(row["strict_diagnostic_code"], "invalid_probability_sum")
                self.assertEqual(row["selected_option_id"], row["request"]["options"][-1]["id"])
                self.assertNotIn("correct", row["action_observation"])
            analysis = analyze(root)["models"]["jev"]
            self.assertEqual(analysis["replay_and_independent_tour_mismatches"], 0)
            self.assertTrue(all(cell["feasible"] == 0 for cell in
                                analysis["strict_projection_of_observed_traces"].values()))
            strict = self.execute(Path(directory) / "strict", lambda _, **kw: Rounded(kw["model"]))
            self.assertEqual(strict["jev"]["calls"], 2)
            self.assertTrue(all(c["feasible"] == 0 for c in strict["jev"]["by_arm"].values()))

    def test_actual_history_and_transformed_question_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            result = self.execute(root, lambda _, **kw: FakeClient(kw["model"]))
            self.assertEqual(result["jev"]["status"], "completed")
            self.assertEqual(result["jev"]["calls"], 6)
            rows = rollout.suite.read_rows(root / "jev-attempts.jsonl")
            previous = {arm: [] for arm in rollout.ARMS}
            for row in rows:
                request = row["request"]
                self.assertEqual(request["state"]["partial_tour"], [0, *previous[row["arm"]]])
                self.assertEqual("NOW at vertex" in request["question"], row["arm"] == "state_explicit")
                self.assertNotIn("private", request["state"])
                previous[row["arm"]].append(int(row["selected_option_id"]))
            for row in rollout.suite.read_rows(root / "jev-scores.jsonl"):
                self.assertTrue(row["feasible"])
                self.assertEqual(row["decisions"], list(map(str, previous[row["arm"]])))
            analysis = analyze(root)["models"]["jev"]
            self.assertEqual(analysis["relative_gap_reduction_common_feasible"]["paired_instances"], 1)

    def test_fatal_failure_preserves_full_scheduled_denominator(self):
        class Fatal(FakeClient):
            async def predict(self, request):
                raise ProviderHTTPError(401)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            result = self.execute(root, lambda _, **kw: Fatal(kw["model"]))
            self.assertEqual(result["jev"]["status"], "aborted")
            self.assertEqual(result["jev"]["calls"], 1)
            self.assertEqual(sum(c["scheduled"] for c in result["jev"]["by_arm"].values()), 2)
            self.assertEqual(sum(c["attempted"] for c in result["jev"]["by_arm"].values()), 1)
            self.assertEqual(sum(c["feasible"] for c in result["jev"]["by_arm"].values()), 0)

    def test_action_protocol_requires_inspectable_native_body(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(Path(directory) / "run",
                                  lambda _, **kw: FakeClient(kw["model"]),
                                  native_action_observation=True)
            self.assertEqual(result["jev"]["status"], "aborted")
            self.assertEqual(result["jev"]["diagnostic_code"], "invalid_response")
            self.assertEqual(result["jev"]["calls"], 1)

    def test_cancellation_is_sealed_and_propagated(self):
        class Cancel(FakeClient):
            async def predict(self, request):
                raise asyncio.CancelledError()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with self.assertRaises(asyncio.CancelledError):
                self.execute(root, lambda _, **kw: Cancel(kw["model"]))
            status = rollout.suite.read_json(root / "jev-status.json")
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual(status["calls"], 1)
