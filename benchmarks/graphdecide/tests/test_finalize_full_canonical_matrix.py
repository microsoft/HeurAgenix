import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock, patch
import uuid

from scripts import finalize_full_canonical_matrix as finalizer


class Clock:
    def __init__(self):
        self.value = 1000.
        self.sleeps = []

    def time(self):
        return self.value

    def sleep(self, value):
        self.sleeps.append(value)
        self.value += value


class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.container = finalizer.REPO / "output" / ("full-matrix-wait-tests-" + uuid.uuid4().hex)
        self.container.mkdir()
        self.addCleanup(shutil.rmtree, self.container)
        self.root = self.container / "watcher"
        self.code = self.container / "source.py"
        self.code.write_text("# frozen fixture\n")
        self.clock = Clock()
        self.hashes = {str(self.code.relative_to(finalizer.REPO)): finalizer.suite.digest(self.code)}

    def prepare(self, duration=120):
        with patch.object(finalizer, "source_freeze", return_value=self.hashes), \
                patch.object(finalizer.time, "time", self.clock.time):
            return finalizer.prepare(self.root, duration)

    def state(self, status="running", ready=False, blocked=False):
        return {"fixture": {"status": status, "ready": ready, "blocked": blocked}}

    def run_waiter(self, states, export=None):
        return finalizer.run(self.root, clock=self.clock.time, monotonic=self.clock.time,
                             sleep=self.clock.sleep, read_statuses=states,
                             export=export or Mock())

    def status(self):
        return finalizer.suite.read_json(self.root / "status.json")

    def fake_export(self, astra, output):
        self.assertEqual(astra, finalizer.ASTRA)
        output.mkdir()
        names = ("decision_matrix-episodes.csv", "decision_matrix-task-size.csv",
                 "strong_anchors-episodes.csv", "strong_anchors-task-size.csv",
                 "full-matrix-tables.tex", "verified-summary.json")
        for name in names:
            (output / name).write_text("{}\n")
        finalizer.write(output / "export-seal.json", {
            "status": "completed", "artifact_sha256": {
                name: finalizer.suite.digest(output / name) for name in names}}, exclusive=True)
        return {"status": "verified_full_canonical_matrix", "new_model_calls": 0}

    def test_preparation_is_exclusive_and_bounded(self):
        for seconds in (0, -1, finalizer.MAX_WAIT + 1, 1.5):
            with self.assertRaises(finalizer.suite.shared.Refusal):
                self.prepare(seconds)
        registration = self.prepare()
        self.assertEqual(registration["interval_seconds"], 60)
        self.assertEqual(registration["deadline_unix"], 1120)
        with self.assertRaises(finalizer.suite.shared.Refusal):
            self.prepare()

    def test_pending_does_not_open_correctness_or_call_export(self):
        self.prepare()
        export = Mock()
        with patch.object(finalizer.exporter.suite, "read_rows", side_effect=AssertionError):
            code = self.run_waiter(lambda: self.state(), export)
        self.assertEqual(code, 2)
        self.assertEqual(self.clock.sleeps, [60, 60])
        self.assertEqual(self.status()["status"], "deadline_expired")
        export.assert_not_called()
        self.assertFalse((self.root / "completion.json").exists())

    def test_aborted_source_blocks_even_when_other_sources_ready(self):
        self.prepare()
        export = Mock()
        self.assertEqual(self.run_waiter(
            lambda: {**self.state("aborted", blocked=True),
                     "other": {"status": "completed", "ready": True, "blocked": False}}, export), 2)
        self.assertEqual(self.status()["status"], "blocked_terminal_source")
        self.assertEqual(self.clock.sleeps, [])
        export.assert_not_called()

    def test_completed_without_seal_never_exports(self):
        self.prepare(61)
        export = Mock()
        self.assertEqual(self.run_waiter(lambda: self.state("completed"), export), 2)
        self.assertEqual(self.clock.sleeps, [60, 1])
        export.assert_not_called()

    def test_source_mutation_before_run_refuses(self):
        self.prepare()
        self.code.write_text("# altered\n")
        export = Mock()
        self.assertEqual(self.run_waiter(lambda: self.state("completed", ready=True), export), 2)
        self.assertEqual(self.status()["status"], "failed_verification")
        export.assert_not_called()

    def test_source_mutation_while_waiting_refuses(self):
        self.prepare()
        def states():
            self.code.write_text("# altered while waiting\n")
            return self.state()
        self.assertEqual(self.run_waiter(states), 2)
        self.assertEqual(self.status()["status"], "failed_verification")

    def test_mutation_during_export_cannot_produce_completion(self):
        self.prepare()
        def export(astra, output):
            result = self.fake_export(astra, output)
            self.code.write_text("# altered during verification\n")
            return result
        self.assertEqual(self.run_waiter(lambda: self.state("completed", ready=True), export), 2)
        self.assertFalse((self.root / "completion.json").exists())
        self.assertEqual(self.status()["status"], "failed_verification")

    def test_success_seals_durable_output_and_duplicate_runner_refused(self):
        self.prepare()
        self.assertEqual(self.run_waiter(lambda: self.state("completed", ready=True), self.fake_export), 0)
        self.assertEqual(self.status()["status"], "completed")
        proof = finalizer.suite.read_json(self.root / "completion.json")
        self.assertEqual(proof["export_seal_sha256"],
                         finalizer.suite.digest(self.root / "export/export-seal.json"))
        with self.assertRaises(FileExistsError):
            self.run_waiter(lambda: self.state("completed", ready=True), self.fake_export)

    def test_completed_with_source_failures_has_explicit_nonzero_outcome(self):
        self.prepare()
        self.assertEqual(self.run_waiter(
            lambda: self.state("completed_with_failures", ready=True), self.fake_export), 1)
        self.assertEqual(self.status()["status"], "completed_with_source_failures")
        self.assertEqual(self.status()["source_failures"], ["fixture"])

    def test_export_verification_failure_does_not_claim_completion(self):
        self.prepare()
        export = Mock(side_effect=RuntimeError("bad objective"))
        self.assertEqual(self.run_waiter(lambda: self.state("completed", ready=True), export), 2)
        self.assertEqual(self.status()["status"], "failed_verification")
        self.assertFalse((self.root / "completion.json").exists())

    def test_verification_deadline_refuses_completion(self):
        self.prepare()
        export = Mock(side_effect=finalizer.DeadlineExpired("offline deadline"))
        self.assertEqual(self.run_waiter(lambda: self.state("completed", ready=True), export), 2)
        self.assertEqual(self.status()["status"], "deadline_expired")
        self.assertFalse((self.root / "completion.json").exists())

    def test_export_returning_after_deadline_is_not_published_as_complete(self):
        self.prepare()
        def export(astra, output):
            result = self.fake_export(astra, output)
            self.clock.value += 121
            return result
        self.assertEqual(self.run_waiter(lambda: self.state("completed", ready=True), export), 2)
        self.assertEqual(self.status()["status"], "deadline_expired")
        self.assertFalse((self.root / "completion.json").exists())

    def test_expired_preparation_refuses_even_ready_sources(self):
        self.prepare()
        self.clock.value = 2000
        export = Mock()
        self.assertEqual(self.run_waiter(lambda: self.state("completed", ready=True), export), 2)
        export.assert_not_called()
        self.assertEqual(self.status()["status"], "deadline_expired")

    def test_watcher_registration_tamper_refused(self):
        self.prepare()
        (self.root / "registration.json").write_text("{}\n")
        self.assertEqual(self.run_waiter(lambda: self.state()), 2)
        self.assertEqual(self.status()["status"], "failed_verification")

    def test_status_reader_uses_only_lifecycle_and_seal_existence(self):
        lane = finalizer.exporter.Lane("fixture", "Qwen", "qwen27", self.container / "lane")
        lane.run.mkdir(parents=True)
        finalizer.write(lane.run / "status.json", {"status": "running"}, exclusive=True)
        with patch.object(finalizer.exporter, "lanes", return_value=[lane]), \
                patch.object(finalizer.exporter.suite, "read_rows", side_effect=AssertionError):
            status = finalizer.statuses()["fixture"]
            self.assertEqual(status["status"], "running")
            self.assertFalse(status["ready"])
            finalizer.write(lane.terminal, {"status": "completed"}, exclusive=True)
            self.assertFalse(finalizer.statuses()["fixture"]["ready"])
            finalizer.write(lane.seal, {"unread": True}, exclusive=True)
            self.assertTrue(finalizer.statuses()["fixture"]["ready"])

    def test_final_export_tamper_is_refused(self):
        output = self.container / "export"
        self.fake_export(finalizer.ASTRA, output)
        (output / "full-matrix-tables.tex").write_text("tampered")
        with self.assertRaises(finalizer.suite.shared.Refusal):
            finalizer.durable_export(output)


if __name__ == "__main__":
    unittest.main()
