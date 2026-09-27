import asyncio
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from scripts import build_graph_release as builder
from scripts import evaluate_graph_release as evaluation
from src.clients.base import ProviderHTTPError
from test_extended_graph_suite import FakeClient
import test_graph_release as release_tests


class ReleaseEvaluationTests(TestCase):
    def bundle(self, base):
        source = release_tests.GraphReleaseTests().source(base)
        bundle = base / "bundle"
        builder.build(bundle, {"development": (source, None)}, base / "bundle.tgz")
        return bundle

    def execute(self, base, profile, factory):
        with patch.object(evaluation.suite, "lane_locks", return_value=nullcontext()):
            return asyncio.run(evaluation.run(
                self.bundle(base), "development", base / "run", profile,
                client_factory=factory, preflight=AsyncMock()))

    def test_ten_task_rollouts_replay_and_keep_native_body_private(self):
        class Native(FakeClient):
            async def predict(self, request):
                response = await super().predict(request)
                ids = [o.id for o in request.options]
                body = {"model": response.resolved_model, "secret": "NOT_FOR_ARTIFACTS",
                        "answers": {"decision": {"type": "choice", "choice": response.selected_option_id,
                            "probabilities": {i: 1/len(ids) for i in ids}, "confidence": 1/len(ids)}},
                        "usage": {}}
                return replace(response, raw_output=body)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self.execute(base, "jev_action", lambda _, **kw: Native(kw["model"]))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["calls"], 28)
            summary = evaluation.suite.read_json(base / "run/summary.json")
            self.assertEqual(len(summary["by_task"]), 10)
            text = (base / "run/attempts.jsonl").read_text()
            self.assertNotIn("NOT_FOR_ARTIFACTS", text)
            rows = evaluation.suite.read_rows(base / "run/attempts.jsonl")
            self.assertTrue(all(all(o["description"] for o in r["request"]["options"]) for r in rows))
            self.assertTrue(all("return only" not in r["request"]["question"].lower() for r in rows))

    def test_fatal_failure_keeps_all_planned_tasks_and_no_retry(self):
        class Fatal(FakeClient):
            async def predict(self, request):
                raise ProviderHTTPError(401)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self.execute(base, "qwen4_grammar", lambda _, **kw: Fatal(kw["model"]))
            self.assertEqual(result["status"], "aborted")
            self.assertEqual(result["calls"], 1)
            scores = evaluation.suite.read_rows(base / "run/scores.jsonl")
            self.assertEqual(len(scores), 10)
            self.assertEqual(sum(r["query_count"] == 0 for r in scores), 9)

    def test_cancellation_is_recorded_and_propagated(self):
        class Cancel(FakeClient):
            async def predict(self, request):
                raise asyncio.CancelledError()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaises(asyncio.CancelledError):
                self.execute(base, "qwen4_grammar", lambda _, **kw: Cancel(kw["model"]))
            result = evaluation.suite.read_json(base / "run/status.json")
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(result["calls"], 1)
