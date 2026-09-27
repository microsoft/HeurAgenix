import json
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from scripts import full_matrix_trained as full
from src.benchmark import extended_tasks as tasks


class FullTrainedTests(unittest.TestCase):
    def setUp(self):
        self.root = full.REPO / "output" / ("full-trained-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        graph = {"nodes": list(range(5)), "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
                 "directed": False}
        self.instances = [
            tasks._record("maxcut_construct", 0, graph),
            tasks._record("degree_exact", 0, {**graph, "vertex": 0}),
            tasks._record("cycle_detection", 0, graph),
        ]
        self.registration = {
            "assets": str(self.root), "code_sha256": {},
            "loading_timeout_seconds": 600, "inference_timeout_seconds": 180,
            "model_settings": {"decider": {"dtype": "bfloat16", "temperature": 1.164}}}
        self.seen = []

    def worker(self, *, timeout=False, unsupported=False, load_failure=False, interrupted=False):
        owner = self

        class FakeWorker:
            def __init__(self, assets, candidate, output):
                self.process = SimpleNamespace(poll=lambda: None if not owner.closed else 0)
                owner.closed = False

            def receive(self, seconds):
                owner.assertEqual(seconds, 600)
                if load_failure:
                    raise TimeoutError("loading deadline")
                return {"status": "ready", "runtime": {
                    "candidate": "decider", "device": "cuda", "context_limit": 8192,
                    "dtype": "bfloat16", "temperature": 1.164}}

            def predict(self, request, seconds):
                owner.assertEqual(seconds, 180)
                if len(owner.instances) < 960 or not owner.seen:
                    dispatched = full.suite.read_rows(owner.root / "run-decider/dispatch.jsonl")
                    owner.assertEqual(len(dispatched), len(owner.seen) + 1)
                    owner.assertEqual(dispatched[-1]["request"], full.suite.json_value(full.asdict(request)))
                owner.seen.append(request)
                if timeout:
                    raise TimeoutError("inference deadline")
                if interrupted:
                    raise KeyboardInterrupt
                if unsupported:
                    return {"status": "failure", "fatal": False,
                            "error_type": "UnsupportedRequestError", "error": "overlength"}
                ids = [o.id for o in request.options]
                return {"status": "success", "result": {
                    "selected_option_id": ids[-1],
                    "probabilities": {i: 1 / len(ids) for i in ids},
                    "input_tokens": 123}}

            def close(self):
                owner.closed = True

        return FakeWorker

    def execute(self, **kwargs):
        interrupted = full.evaluate(self.root, "decider", self.instances, self.registration,
                                    worker_factory=self.worker(**kwargs))
        return full.read(self.root / "run-decider/status.json"), interrupted

    def test_public_own_history_and_durable_pre_dispatch(self):
        outcome, interrupted = self.execute()
        self.assertEqual(outcome["status"], "completed")
        self.assertFalse(interrupted)
        self.assertTrue(self.closed)
        self.assertEqual(len(self.seen), sum(i["query_budget"] for i in self.instances))
        attempts = full.suite.read_rows(self.root / "run-decider/attempts.jsonl")
        self.assertGreater(attempts[1]["step"], 0)
        self.assertNotIn("private", attempts[1]["request"])
        self.assertEqual(full.verify_terminal(self.root, "decider", self.instances)["episodes"], 3)

    def test_no_retry_resume_or_overwrite(self):
        self.execute()
        with self.assertRaises(FileExistsError):
            full.evaluate(self.root, "decider", self.instances, self.registration,
                          worker_factory=self.worker())
        with self.assertRaisesRegex(RuntimeError, "no retries"):
            full.launch(self.root, self.root / "unused.json")
        with self.assertRaisesRegex(RuntimeError, "cannot overwrite"):
            full.register(self.root, self.root, self.root)

    def test_timeout_preserves_every_denominator(self):
        outcome, _ = self.execute(timeout=True)
        self.assertEqual(outcome["status"], "aborted")
        self.assertEqual(outcome["calls"], 1)
        self.assertEqual(outcome["unattempted_episodes"], 2)
        self.assertEqual(len(full.suite.read_rows(self.root / "run-decider/scores.jsonl")), 3)
        self.assertTrue(self.closed)

    def test_loading_timeout_has_all_unattempted(self):
        outcome, _ = self.execute(load_failure=True)
        self.assertEqual(outcome["calls"], 0)
        self.assertEqual(outcome["unattempted_episodes"], 3)
        self.assertEqual(outcome["status"], "aborted")

    def test_nonfatal_unsupported_kept_and_next_episode_attempted(self):
        outcome, _ = self.execute(unsupported=True)
        self.assertEqual(outcome["calls"], 3)
        self.assertEqual(outcome["failure_count"], 3)
        self.assertEqual(outcome["unattempted_episodes"], 0)
        self.assertEqual(outcome["status"], "completed_with_failures")

    def test_interrupt_is_logged_and_blocks_remaining_dispatch(self):
        outcome, interrupted = self.execute(interrupted=True)
        self.assertTrue(interrupted)
        self.assertEqual(outcome["calls"], 1)
        self.assertTrue(self.closed)
        full.evaluate(self.root, "kev", self.instances, self.registration,
                      blocked="interrupted", worker_factory=self.worker())
        outcome = full.read(self.root / "run-kev/status.json")
        self.assertEqual(outcome["unattempted_episodes"], 3)
        self.assertEqual(outcome["calls"], 0)

    def test_request_and_dispatch_tampering_rejected(self):
        self.execute()
        path = self.root / "run-decider/dispatch.jsonl"
        rows = full.suite.read_rows(path)
        rows[0]["request_sha256"] = "tampered"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            full.verify_terminal(self.root, "decider", self.instances)

    def test_unresolved_dispatch_rejected(self):
        self.execute()
        with (self.root / "run-decider/dispatch.jsonl").open("a") as stream:
            full.suite.append_row(stream, {"call_index": 999})
        with self.assertRaisesRegex(RuntimeError, "unresolved call"):
            full.verify_terminal(self.root, "decider", self.instances)

    def test_native_configuration_gate(self):
        for changed in ({"temperature": 1.0}, {"dtype": "float16"}, {"context_limit": 4096}):
            ready = {"status": "ready", "runtime": {
                "candidate": "decider", "device": "cuda", "context_limit": 8192,
                "dtype": "bfloat16", "temperature": 1.164, **changed}}
            with self.assertRaisesRegex(RuntimeError, "configuration gate"):
                full.runtime_gate(ready, "decider", self.registration)

    def test_deadline_rejects_partial_line_without_blocking(self):
        worker = full.Worker.__new__(full.Worker)
        worker.buffer = b""
        reader, writer = os.pipe()
        self.addCleanup(os.close, writer)
        stdout = os.fdopen(reader, "rb")
        self.addCleanup(stdout.close)
        worker.process = SimpleNamespace(stdout=stdout)
        os.write(writer, b'{"status":')
        with self.assertRaises(TimeoutError):
            worker.receive(.02)

    def test_deadline_covers_blocked_request_write(self):
        worker = full.Worker.__new__(full.Worker)
        reader, writer = os.pipe()
        self.addCleanup(os.close, reader)
        stdin = os.fdopen(writer, "wb")
        self.addCleanup(stdin.close)
        os.set_blocking(writer, False)
        try:
            while True:
                os.write(writer, b"x" * 4096)
        except BlockingIOError:
            pass
        worker.process = SimpleNamespace(stdin=stdin)
        request = full.evaluation.public_request(self.instances[0], [])
        with self.assertRaises(TimeoutError):
            worker.predict(request, .02)

    def test_complete_canonical_panel_count_and_order(self):
        instances = full.canonical_panel(full.DEFAULT_BUNDLE)
        public = full.release.load_public(
            full.DEFAULT_BUNDLE, full.SPLIT, full.release.manifest_at(full.DEFAULT_BUNDLE))
        self.assertEqual(len(instances), 960)
        self.assertEqual(sum(i["query_budget"] for i in instances), 1920)
        self.assertEqual([i["id"] for i in instances], [i["id"] for i in public])
        self.assertEqual(sum(i["kind"] == "exact" for i in instances), 800)
        self.assertEqual(sum(i["kind"] != "exact" for i in instances), 160)

    def test_full_canonical_fake_rollout_preserves_all_1920_requests(self):
        self.instances = full.canonical_panel(full.DEFAULT_BUNDLE)
        outcome, _ = self.execute()
        self.assertEqual(outcome["scheduled_episodes"], 960)
        self.assertEqual(outcome["calls"], 1920)
        self.assertEqual(outcome["unattempted_episodes"], 0)
        self.assertEqual(outcome["status"], "completed")
        scores = full.suite.read_rows(self.root / "run-decider/scores.jsonl")
        self.assertEqual(len(scores), 960)
        self.assertEqual(sum(s["query_count"] for s in scores), 1920)

    def test_registration_source_tampering_fails_before_assets(self):
        full.write(self.root / "registration.json", {"code_sha256": {"bad": "hash"}})
        full.write(self.root / "registration.sha256", full.digest(self.root / "registration.json"))
        with patch.object(full.pilot, "verify_files", side_effect=RuntimeError("source drift")):
            with self.assertRaisesRegex(RuntimeError, "source drift"):
                full.verify_registration(self.root)

    def test_parent_uses_standard_locks_and_uuid_only_after_handoff(self):
        observation = {"physical_gpus": {0: {"uuid": "GPU-authored"}}}
        events = []

        @contextmanager
        def locks(resources):
            self.assertEqual(resources, ("qwen08", "qwen2"))
            events.append("locked")
            yield
            events.append("unlocked")

        def evaluate(root, candidate, instances, registration, *, blocked):
            self.assertEqual(events[-1], "locked")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-authored")
            self.assertEqual(os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"], "approved")
            output = root / f"run-{candidate}"
            output.mkdir()
            full.write(output / "status.json", {"worker_released": True})
            return False

        full.write(self.root / "registration.json", {})
        full.write(self.root / "handoff.json", {"approved": True})
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(full, "verify_registration",
                             return_value=({"code_sha256": {}, "models": {}}, self.instances)), \
                patch.object(full.pilot, "verify_handoff") as handoff, \
                patch.object(full.suite, "lane_locks", side_effect=locks), \
                patch.object(full.parent, "inspect_empty_devices", return_value=observation), \
                patch.object(full, "evaluate", side_effect=evaluate) as execute, \
                patch.object(full.pilot, "verify_artifacts"), \
                patch.object(full, "verify_terminal", return_value={"status": "verified"}):
            full.launch(self.root, self.root / "handoff.json")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        handoff.assert_called_once_with(self.root / "handoff.json", self.root / "registration.json")
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(events, ["locked", "unlocked"])

    def test_occupied_device_refuses_before_launch_or_workers(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(full, "verify_registration", return_value=({}, self.instances)), \
                patch.object(full.pilot, "verify_handoff"), \
                patch.object(full.suite, "lane_locks") as locks, \
                patch.object(full.parent, "inspect_empty_devices", side_effect=RuntimeError("occupied")), \
                patch.object(full, "evaluate") as execute:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                full.launch(self.root, self.root / "handoff.json")
        locks.assert_called_once_with(("qwen08", "qwen2"))
        execute.assert_not_called()
        self.assertFalse((self.root / "launch.json").exists())


if __name__ == "__main__":
    unittest.main()
