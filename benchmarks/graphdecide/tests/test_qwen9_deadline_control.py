import asyncio
from contextlib import nullcontext
from dataclasses import asdict
import io
import json
import shutil
import unittest
from unittest.mock import AsyncMock
import uuid

from scripts import qwen9_deadline_control as control
from src.clients.base import ClientTimeoutError, DecisionResponse, InvalidResponseError, TokenUsage


class FakeClient:
    seen = []

    def __init__(self, **kwargs):
        self.model = kwargs["model"]

    async def initialize(self):
        pass

    async def aclose(self):
        pass

    async def predict(self, request):
        self.seen.append(request)
        return DecisionResponse(
            request_id=request.request_id, selected_option_id=request.options[0].id,
            resolved_model=self.model, usage=TokenUsage(input_tokens=2, output_tokens=3),
            raw_output={"arbitrary_provider_text": "PRIVATE_BODY_SENTINEL", "choices": [{
                "finish_reason": "stop", "message": {"role": "assistant",
                    "content": "FINAL_TEXT_SENTINEL", "reasoning": "REASONING_SENTINEL"}}]})


class DeadlineControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.all_instances = control.shared.load_panel(control.panel.DEFAULT_ROOT)
        cls.instances = control.select_instances(cls.all_instances)

    def setUp(self):
        self.root = control.REPO / "output" / ("deadline-control-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        (self.root / "registration.json").write_text("{}")
        (self.root / "checkpoint.json").write_text("{}")
        self.registration = {"predecessor_status": str(self.root / "old/status.json"),
                             "service_identity": {}}
        FakeClient.seen = []

    def tearDown(self):
        shutil.rmtree(self.root)

    def execute(self, factory=FakeClient, **changes):
        def lease(names):
            self.assertEqual(names, ("qwen9",))
            return nullcontext()
        options = dict(
            loader=lambda root: (self.registration, self.instances),
            wait=AsyncMock(return_value="completed_with_failures"), lock_factory=lease,
            preflight=AsyncMock(), service_guard=lambda _: None,
            checkpoint_guard=lambda _: None, client_factory=factory)
        options.update(changes)
        return asyncio.run(control.controller(self.root, **options))

    def scores(self):
        return control.suite.read_rows(self.root / "scores.jsonl")

    def test_all_construction_cases_only_fixed_order_16_112(self):
        self.assertEqual(len(self.instances), 16)
        self.assertEqual(sum(i["query_budget"] for i in self.instances), 112)
        self.assertTrue(all(i["kind"] == "optimization" for i in self.instances))
        self.assertEqual(self.instances, [i for i in self.all_instances if i["kind"] == "optimization"])

    def test_only_timeout_changed(self):
        old = asdict(control.panel.configurations()["qwen9_thinking"])
        new = asdict(control.configuration())
        self.assertEqual({k for k in new if old[k] != new[k]}, {"timeout_seconds"})
        self.assertEqual((old["timeout_seconds"], new["timeout_seconds"]), (180.0, 600.0))
        self.assertEqual(new["max_tokens"], 8192)
        self.assertEqual(new["seed"], 20261008)

    def test_own_history_all_112_calls_replay_and_redacted_outputs(self):
        state = self.execute()
        self.assertEqual(state["status"], "completed")
        rows = self.scores()
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(r["query_count"] for r in rows), 112)
        self.assertTrue(all(r["feasible"] for r in rows))
        attempts = control.suite.read_rows(self.root / "attempts.jsonl")
        self.assertEqual(len(attempts), 112)
        text = "\n".join(p.read_text() for p in self.root.iterdir() if p.is_file())
        self.assertNotIn("SENTINEL", text)
        self.assertNotIn('"raw_output"', text)
        self.assertTrue(all(a["response_diagnostics"]["reasoning_present"] for a in attempts))

    def test_length_failure_no_retry_all_sixteen_denominators(self):
        class Length(FakeClient):
            async def predict(self, request):
                error = InvalidResponseError("PRIVATE_EXCEPTION_SENTINEL", diagnostic_code="finish_length")
                error.response_diagnostics = {"finish_reason": "length"}
                raise error
        state = self.execute(Length)
        self.assertEqual(state["status"], "completed_with_failures")
        rows = self.scores()
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r["query_count"] == 1 and not r["feasible"] for r in rows))
        report = control.suite.read_json(self.root / "diagnostics.json")
        self.assertEqual(report["failures"], {"finish_length": 16})

    def test_timeouts_are_explicit_and_not_retried(self):
        class Timeout(FakeClient):
            async def predict(self, request):
                raise ClientTimeoutError("injected timeout")
        self.execute(Timeout)
        rows = self.scores()
        self.assertTrue(all(r["query_count"] == 1 for r in rows))
        report = control.suite.read_json(self.root / "diagnostics.json")
        self.assertEqual(report["failures"], {"timeout": 16})

    def test_prestart_wait_reads_status_only(self):
        states, paths = iter(("running", "completed_with_failures")), []
        status_path = self.root / "old/status.json"
        def reader(path):
            paths.append(path)
            self.assertEqual(path, status_path)
            return {"status": next(states)}
        result = asyncio.run(control.wait_terminal(status_path, deadline=10, clock=lambda: 0,
                                                   sleep=AsyncMock(), reader=reader))
        self.assertEqual(result, "completed_with_failures")
        self.assertEqual(paths, [status_path, status_path])

    def test_prestart_wait_deadline_no_status_or_score_read(self):
        reader = unittest.mock.Mock()
        with self.assertRaises(control.WaitDeadline):
            asyncio.run(control.wait_terminal(self.root / "old/status.json", deadline=1,
                                               clock=lambda: 1, reader=reader))
        reader.assert_not_called()

    def test_wait_failure_never_acquires_lock_and_keeps_all_denominators(self):
        lease = unittest.mock.Mock()
        state = self.execute(wait=AsyncMock(side_effect=control.WaitDeadline("expired")),
                             lock_factory=lease)
        lease.assert_not_called()
        self.assertEqual(state["status"], "wait_deadline_exceeded")
        self.assertEqual(len(self.scores()), 16)
        self.assertTrue(all(r["query_count"] == 0 for r in self.scores()))

    def test_lock_refusal_is_once_only_no_client_or_preflight(self):
        lease = unittest.mock.Mock(side_effect=control.suite.shared.Refusal("busy"))
        preflight, factory = AsyncMock(), unittest.mock.Mock()
        state = self.execute(lock_factory=lease, preflight=preflight, client_factory=factory)
        lease.assert_called_once_with(("qwen9",))
        preflight.assert_not_called()
        factory.assert_not_called()
        self.assertEqual(state["status"], "blocked_or_aborted")
        self.assertEqual(len(self.scores()), 16)

    def test_busy_or_wrong_endpoint_preflight_prevents_inference(self):
        factory = unittest.mock.Mock()
        self.execute(preflight=AsyncMock(side_effect=control.suite.shared.Refusal("busy")),
                     client_factory=factory)
        factory.assert_not_called()
        self.assertTrue(all(r["query_count"] == 0 for r in self.scores()))

    def test_run_deadline_does_not_start_truncated_budget_request(self):
        client = FakeClient(model="Qwen3.5-9B")
        with self.assertRaises(control.RunDeadline):
            asyncio.run(control.calls(self.instances, control.configuration(), client, io.StringIO(), [],
                                      guard=lambda: None, deadline=599, clock=lambda: 0))
        self.assertEqual(FakeClient.seen, [])

    def test_controller_no_resume_or_overwrite(self):
        self.execute()
        with self.assertRaises(control.suite.shared.Refusal):
            self.execute()

    def test_source_and_checkpoint_verified_before_and_after(self):
        checkpoint = unittest.mock.Mock()
        source = unittest.mock.Mock(return_value=(self.registration, self.instances))
        state = self.execute(checkpoint_guard=checkpoint, loader=source)
        self.assertEqual(checkpoint.call_count, 2)
        self.assertGreater(source.call_count, 112)
        self.assertTrue(state["source_and_service_verified_after"])

    def test_wait_and_run_deadlines_are_separate(self):
        timer = unittest.mock.Mock()
        self.execute(arm_deadline=timer)
        self.assertEqual(timer.call_args_list, [
            unittest.mock.call(86400, "wait"), unittest.mock.call(72000, "run"),
            unittest.mock.call(0, "done")])

    def test_in_request_hard_deadline_retains_failure_and_all_denominators(self):
        class HardDeadline(FakeClient):
            async def predict(self, request):
                raise control.RunDeadline("injected hard deadline")
        state = self.execute(HardDeadline)
        self.assertEqual(state["status"], "run_deadline_exceeded")
        rows = self.scores()
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(r["query_count"] for r in rows), 1)
        diagnostic = control.suite.read_json(self.root / "diagnostics.json")
        self.assertEqual(diagnostic["failures"], {"run_deadline_exceeded": 1})

    def test_client_cancellation_keeps_all_denominators(self):
        class Cancel(FakeClient):
            async def predict(self, request):
                raise asyncio.CancelledError()
        state = self.execute(Cancel)
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(len(self.scores()), 16)
        self.assertEqual(sum(r["query_count"] for r in self.scores()), 1)

    def test_old_nonterminal_state_does_not_admit_or_read_accuracy(self):
        paths = []
        ticks = iter((0, 0, 31))
        def read(path):
            paths.append(path)
            return {"status": "running"}
        with self.assertRaises(control.WaitDeadline):
            asyncio.run(control.wait_terminal(self.root / "old/status.json", deadline=30,
                clock=lambda: next(ticks), sleep=AsyncMock(), reader=read))
        self.assertEqual(paths, [self.root / "old/status.json"])


if __name__ == "__main__":
    unittest.main()
