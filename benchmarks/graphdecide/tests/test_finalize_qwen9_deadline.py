import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

from scripts import finalize_qwen9_deadline as final


class FinalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, all_instances = final.shared.load_panel(final.control.panel.DEFAULT_ROOT)
        cls.instances = final.control.select_instances(all_instances)
        cls.config = final.control.configuration()

    def setUp(self):
        self.root = final.REPO / "output" / ("deadline-finalizer-test-" + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.root)

    def fixture(self, failure=None):
        attempts = []
        for instance in self.instances:
            history = []
            while (request := final.canonical.public_request(instance, history)) is not None:
                option = request.options[0].id
                attempts.append(final.suite.json_value({
                    "instance_id": instance["id"], "step": len(history), "request": asdict(request),
                    "response_diagnostics": {"response_is_object": True,
                                             "finish_reason": "length" if failure == "finish_length" else "stop"},
                    "latency_seconds": 1.5, "status": "failure" if failure else "success",
                    "selected_option_id": None if failure else option,
                    "resolved_model": None if failure else self.config.model,
                    "usage": {"input_tokens": 5, "output_tokens": None if failure else 10,
                              "reasoning_tokens": None},
                    "diagnostic_code": failure, "http_status": None, "fatal": False,
                    "exception_type": "ClientTimeoutError" if failure else None}))
                if failure:
                    break
                history.append(option)
        return self.payload(attempts, "completed_with_failures" if failure else "completed")

    def payload(self, attempts, status):
        rows = final.canonical.replay(self.instances, attempts, self.config, native_action=False)
        diagnostics = {
            "planned_episodes": 16, "calls": len(attempts),
            "failures": dict(final.Counter(a["diagnostic_code"] for a in attempts if a["status"] == "failure")),
            "unattempted_episodes": sum(r["query_count"] == 0 for r in rows),
            "infeasible_episodes": sum(not r["feasible"] for r in rows),
            "length_cap_note": "8192-token cap unchanged; finish_length remains a failure, never retried."}
        return [self.instances, attempts, rows,
                {"status": status, "calls": len(attempts), "planned_episodes": 16,
                 "source_and_service_verified_after": True},
                final.suite.summarize(rows), diagnostics, self.config]

    def test_valid_full_own_history_and_independent_objectives(self):
        data = self.fixture()
        rows = final.replay_control(*data)
        self.assertEqual(len(data[1]), 112)
        audits = final.independent_objectives(self.instances, rows)
        self.assertEqual(len(audits), 16)
        self.assertTrue(all(r["feasible"] for r in rows))

    def test_timeout_denominators_and_no_empty_gap_mean(self):
        rows = final.replay_control(*self.fixture("timeout"))
        self.assertEqual(len(rows), 16)
        for metric in final.metrics(rows):
            self.assertEqual(metric["conditional_gap_denominator"], 0)
            self.assertIsNone(metric["conditional_mean_absolute_gap"])
            self.assertEqual(metric["optimal_all_planned"], 0)
        pairs = final.paired(final.replay_control(*self.fixture()), rows)
        self.assertTrue(all(r["both_feasible_denominator"] == 0
                            and r["paired_mean_gap_delta_600_minus_180"] is None for r in pairs))

    def test_length_and_timeout_failure_diagnostics_distinct(self):
        length, timeout = self.fixture("finish_length"), self.fixture("timeout")
        self.assertEqual(final.usage(length[1])["failure_diagnostics"], {"finish_length": 16})
        self.assertEqual(final.usage(timeout[1])["failure_diagnostics"], {"timeout": 16})
        self.assertIsNone(final.usage(timeout[1])["successful_call_token_lengths"]["output_tokens"]["sum"])

    def test_wrong_saved_score_or_summary_or_diagnostics_fails(self):
        for index in (2, 4, 5):
            data = self.fixture()
            if index == 2:
                data[index][0]["query_count"] += 1
            else:
                data[index]["unexpected"] = True
            with self.assertRaises(ValueError):
                final.replay_control(*data)

    def test_duplicate_score_id_rejected(self):
        data = self.fixture()
        data[2][1] = deepcopy(data[2][0])
        with self.assertRaises(ValueError):
            final.replay_control(*data)
        with self.assertRaises(ValueError):
            final.subset(data[2], self.instances)

    def test_history_request_mutation_rejected(self):
        data = self.fixture()
        data[1][1]["request"]["question"] += " modified"
        with self.assertRaises((ValueError, final.suite.shared.Refusal)):
            final.replay_control(*data)

    def test_partial_completed_ledger_fails(self):
        data = self.fixture()
        partial = self.payload(data[1][:1], "completed")
        with self.assertRaises(ValueError):
            final.replay_control(*partial)

    def test_explicit_abort_with_full_planned_replay_allowed(self):
        data = self.payload([], "blocked_or_aborted")
        rows = final.replay_control(*data)
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r["query_count"] == 0 for r in rows))
        data[2] = data[2][:-1]
        with self.assertRaises(ValueError):
            final.replay_control(*data)

    def test_full_body_and_nonallowlisted_diagnostics_rejected(self):
        data = self.fixture()
        data[1][0]["raw_output"] = {"reasoning": "hidden"}
        with self.assertRaises(ValueError):
            final.replay_control(*data)
        data = self.fixture()
        data[1][0]["response_diagnostics"]["reasoning_text"] = "hidden"
        with self.assertRaises(ValueError):
            final.replay_control(*data)

    def test_malformed_latency_usage_and_digest_rejected(self):
        for mutate in (
            lambda row: row.update(latency_seconds=float("nan")),
            lambda row: row["usage"].update(output_tokens=-1),
            lambda row: row["response_diagnostics"].update(final_content_sha256="invalid")):
            data = self.fixture()
            mutate(data[1][0])
            with self.assertRaises(ValueError):
                final.replay_control(*data)

    def test_input_hash_drift_and_partial_seal_fail(self):
        path = self.root / "input.json"
        path.write_text("{}")
        evidence = final.Evidence()
        evidence.track(path)
        path.write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            evidence.recheck()
        (self.root / "export-seal.json").write_text(json.dumps(
            {"status": "completed", "artifact_sha256": {"missing.json": "0" * 64}}))
        with self.assertRaises(ValueError):
            final.sealed_files(self.root, final.Evidence())

    def test_wait_reads_status_only_until_actual_terminal(self):
        values = iter(("running", "run_deadline_exceeded"))
        paths = []
        path = self.root / "status.json"
        def reader(given):
            paths.append(given)
            return {"status": next(values)}
        result = asyncio.run(final.wait_terminal(path, deadline=1, clock=lambda: 0,
                            sleeper=AsyncMock(), reader=reader))
        self.assertEqual(result["status"], "run_deadline_exceeded")
        self.assertEqual(paths, [path, path])

    def test_wait_deadline_no_results_read(self):
        reader = Mock()
        with self.assertRaises(TimeoutError):
            asyncio.run(final.wait_terminal(self.root / "status.json", deadline=1,
                                            clock=lambda: 1, reader=reader))
        reader.assert_not_called()

    def test_unknown_controller_state_fails(self):
        with self.assertRaises(ValueError):
            asyncio.run(final.wait_terminal(self.root / "status.json", deadline=1, clock=lambda: 0,
                        reader=lambda _: {"status": "unknown"}))

    def test_finalize_start_is_exclusive_and_contains_pid_start(self):
        with patch.object(final, "frozen", return_value={}), \
                patch.object(final, "wait_terminal", AsyncMock(side_effect=TimeoutError("expired"))):
            (self.root / "registration.json").write_text("{}")
            result = asyncio.run(final.run(self.root))
            self.assertEqual(result["status"], "deadline_exceeded")
            marker = final.suite.read_json(self.root / "finalizer-start.json")
            self.assertEqual(marker["pid"], final.os.getpid())
            self.assertIn("start_ticks", marker)
            with self.assertRaises(FileExistsError):
                asyncio.run(final.run(self.root))

    def test_analysis_async_timeout_terminates_only_own_child(self):
        child = Mock(pid=final.os.getpid(), returncode=None)
        async def stall():
            await asyncio.sleep(10)
        child.communicate = stall
        child.wait = AsyncMock(return_value=-15)
        with patch.object(final.asyncio, "create_subprocess_exec", AsyncMock(return_value=child)):
            with self.assertRaises(TimeoutError):
                asyncio.run(final.bounded_child(self.root, seconds=0.001))
        child.terminate.assert_called_once()
        child.kill.assert_not_called()

    def test_table_empty_gaps_show_dash_not_zero(self):
        rows = final.replay_control(*self.fixture("timeout"))
        tex = final.table({"qwen9_600": rows})
        self.assertIn("-- (0)", tex)
        self.assertNotIn("0 (0)", tex)

    def test_config_only_timeout_matches_frozen_original(self):
        old = final.control.panel.configurations()["qwen9_thinking"]
        current = final.control.configuration()
        self.assertEqual({k for k, v in asdict(old).items() if v != asdict(current)[k]},
                         {"timeout_seconds"})
        self.assertEqual((old.timeout_seconds, current.timeout_seconds), (180.0, 600.0))
        registration = {"original_configuration": asdict(old), "configuration": asdict(current)}
        final.verify_configurations(registration)
        registration["configuration"]["max_tokens"] = 16384
        with self.assertRaises(ValueError):
            final.verify_configurations(registration)

    def test_pending_candidate_never_reads_ledger(self):
        reader = Mock(return_value={"status": "running"})
        with patch.object(final, "frozen", return_value={}), \
                patch.object(final.suite, "read_json", reader):
            with self.assertRaises(ValueError):
                final.analyze(self.root)
        reader.assert_called_once_with(final.RUN / "status.json")

    def test_malformed_status_json_fails_without_analysis(self):
        path = self.root / "status.json"
        path.write_text("{invalid")
        with self.assertRaises(json.JSONDecodeError):
            asyncio.run(final.wait_terminal(path, deadline=1, clock=lambda: 0))

    def test_incomplete_abort_output_cannot_be_published(self):
        observed = {"status": "blocked_or_aborted"}
        with patch.object(final, "frozen", return_value={
                    "fixed_input_sha256": {}, "source_sha256": {}}), \
                patch.object(final.suite, "read_json", return_value=observed), \
                patch.object(final, "RUN", self.root):
            with self.assertRaises(ValueError):
                final.analyze(self.root)
        self.assertFalse((self.root / "export").exists())


if __name__ == "__main__":
    unittest.main()
