import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from scripts import full_matrix_qwen27 as q
from src.clients.base import DecisionResponse, InvalidResponseError, ProviderHTTPError, TokenUsage


class FakeClient:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]

    async def initialize(self):
        pass

    async def aclose(self):
        pass

    async def predict(self, request):
        return DecisionResponse(request_id=request.request_id,
                                selected_option_id=request.options[0].id,
                                resolved_model=self.model, usage=TokenUsage(),
                                raw_output={"test": True})


class FullQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = q.release.labeled_instances(q.BUNDLE, q.SPLIT,
                                                   q.release.manifest_at(q.BUNDLE))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jev-full-qwen-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "registration.json").write_text("{}")
        self.registration = {"model": asdict(q.configuration())}

    def run_with(self, factory):
        with patch.object(q, "load", return_value=(self.registration, self.instances)), \
                patch.object(q, "verify_sources"):
            return asyncio.run(q.run(self.root, client_factory=factory, preflight=AsyncMock()))

    def test_fixed_system_one_budget(self):
        config = q.configuration()
        self.assertFalse(config.think)
        self.assertTrue(config.constrain_choices)
        self.assertEqual(config.max_tokens, 64)
        self.assertEqual(config.timeout_seconds, 180)

    def test_complete_task_counts_not_pilot_or_filtered(self):
        q.check_instances(self.instances)
        for bad in (self.instances[:64], self.instances[:-1],
                    [self.instances[0]] + self.instances[1:-1] + [self.instances[0]]):
            with self.assertRaises(q.suite.shared.Refusal):
                q.check_instances(bad)

    def test_full_own_history_replay_and_real_evidence_gate(self):
        state = self.run_with(FakeClient)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["calls"], 1920)
        rows = q.suite.read_rows(self.root / "run/scores.jsonl")
        self.assertEqual(len(rows), 960)
        self.assertTrue(all(row["feasible"] for row in rows))
        with patch.object(q, "load", return_value=(self.registration, self.instances)):
            with self.assertRaisesRegex(q.suite.shared.Refusal, "Injected"):
                q.verify(self.root)
            self.assertTrue(q.verify(self.root, allow_injected=True)["verified"])

    def test_fatal_failure_retains_all_denominators(self):
        class Fatal(FakeClient):
            async def predict(self, request):
                raise ProviderHTTPError(401)
        state = self.run_with(Fatal)
        self.assertEqual(state["status"], "aborted")
        self.assertEqual(state["calls"], 1)
        rows = q.suite.read_rows(self.root / "run/scores.jsonl")
        self.assertEqual(len(rows), 960)
        self.assertEqual(sum(row["query_count"] == 0 for row in rows), 959)

    def test_parse_failure_no_retry_or_repair(self):
        class Invalid(FakeClient):
            async def predict(self, request):
                raise InvalidResponseError("truncated", diagnostic_code="finish_length")
        state = self.run_with(Invalid)
        self.assertEqual(state["status"], "completed_with_failures")
        self.assertEqual(state["calls"], 960)
        self.assertEqual(state["failure_counts"], {"finish_length": 960})
        rows = q.suite.read_rows(self.root / "run/scores.jsonl")
        self.assertEqual(sum(row["feasible"] for row in rows), 0)

    def test_initialization_failure_has_960_unattempted(self):
        class Init(FakeClient):
            async def initialize(self):
                raise RuntimeError("test init failure")
        state = self.run_with(Init)
        self.assertEqual(state["status"], "aborted")
        self.assertEqual(state["calls"], 0)
        self.assertEqual(len(q.suite.read_rows(self.root / "run/scores.jsonl")), 960)

    def test_cancellation_persists_inflight_attempt(self):
        class Cancel(FakeClient):
            async def predict(self, request):
                raise asyncio.CancelledError()
        state = self.run_with(Cancel)
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["calls"], 1)
        self.assertEqual(state["failure_counts"], {"interrupted": 1})

    def test_run_deadline_remains_explicit(self):
        class Slow(FakeClient):
            async def predict(self, request):
                await asyncio.sleep(1)
        with patch.object(q, "MAX_SECONDS", .02):
            state = self.run_with(Slow)
        self.assertEqual(state["status"], "aborted")
        self.assertEqual(state["diagnostic_code"], "run_deadline")

    def test_resume_refused(self):
        self.run_with(FakeClient)
        with self.assertRaises(q.suite.shared.Refusal):
            self.run_with(FakeClient)

    def test_tampered_results_refused(self):
        self.run_with(FakeClient)
        path = self.root / "run/summary.json"
        path.write_text(json.dumps({"fake_success": True}))
        with patch.object(q, "load", return_value=(self.registration, self.instances)):
            with self.assertRaisesRegex(q.suite.shared.Refusal, "seal"):
                q.verify(self.root, allow_injected=True)
