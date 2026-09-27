from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from scripts import queue_trained_pilot as queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.root = queue.REPO / "output" / ("queue-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.start = {"locks": list(queue.LANES), "target_gpu_uuids": ["GPU-zero", "GPU-one"],
                      "protected": [{"pid": 121, "start_ticks": "old-protected"}], "services": {}}
        for offset, (lane, (gpu, port, model)) in enumerate(queue.LANES.items()):
            self.start["services"][lane] = {
                "pid": 111 + offset, "start_ticks": f"old-{lane}", "process_tree": [],
                "argv": [queue.PYTHON, "/fastdata/xianya/vllm-venv/bin/vllm", "serve",
                         f"/fastdata/xianya/models/{model}", "--port", str(port),
                         "--host", "127.0.0.1", "--served-model-name", model],
                "cwd": str(queue.REPO), "cuda_environment": {"CUDA_VISIBLE_DEVICES": str(gpu)}}
        queue.save(self.root / "supervisor-start.json", self.start)
        self.manifest = {
            "prior_root": str(self.root), "trained_root": str(self.root),
            "prior_supervisor": {"pid": 91, "start_ticks": "supervisor"},
            "prior_server": {"pid": 92, "start_ticks": "server"},
            "start_sha256": queue.digest(self.root / "supervisor-start.json"),
            "restore_startup_timeout_seconds": 1}

    def handoff(self):
        return {"owned_processes_stopped": True, "target_gpus_free": True,
                "protected_services_alive": True, "ready_after_supervisor_exit": True,
                "old_services_restored": False}

    def test_invalid_handoff_refused_before_gpu_inspection(self):
        for key in self.handoff():
            handoff = self.handoff()
            handoff[key] = not handoff[key]
            with self.subTest(key=key), patch.object(queue.trained, "inspect_empty_devices") as gpu, \
                    self.assertRaises(RuntimeError):
                queue.validate_release(self.manifest, handoff, [])
            gpu.assert_not_called()

    def test_live_prior_supervisor_is_not_handoff(self):
        with patch.object(queue, "alive", return_value=True), \
                patch.object(queue.trained, "inspect_empty_devices") as gpu, \
                self.assertRaisesRegex(RuntimeError, "supervisor is still alive"):
            queue.validate_release(self.manifest, self.handoff(), [])
        gpu.assert_not_called()

    def test_handoff_while_prior_alive_waits_only_until_deadline(self):
        queue.save(self.root / "handoff.json", self.handoff())
        manifest = {**self.manifest, "max_wait_seconds": 1,
                    "initial_server_tree": [], "poll_seconds": 30}
        with patch.object(queue, "tree", return_value=[]), \
                patch.object(queue, "alive", return_value=True), \
                patch.object(queue.time, "monotonic", side_effect=[0, 2, 2]), \
                patch.object(queue, "validate_release") as release, \
                self.assertRaisesRegex(RuntimeError, "deadline exceeded"):
            queue.wait_for_release(self.root, manifest)
        release.assert_not_called()

    def test_live_prior_tree_is_not_handoff(self):
        with patch.object(queue, "alive", side_effect=[False, False, True]), \
                self.assertRaisesRegex(RuntimeError, "server tree"):
            queue.validate_release(self.manifest, self.handoff(), [{"pid": 93}])

    def test_pid_reuse_is_not_original_liveness(self):
        with patch.object(queue, "identity", return_value={
            "pid": 91, "start_ticks": "new-start", "state": "S"}):
            self.assertFalse(queue.alive(self.manifest["prior_supervisor"]))

    def test_unknown_gpu_process_refuses_restoration(self):
        with patch.object(queue.trained, "nvidia_query",
                          return_value=[["GPU-zero", "999", "unknown"]]), \
                self.assertRaisesRegex(RuntimeError, "Unknown target-GPU process"):
            queue.restore_inventory(self.start, {})

    def test_wrong_restored_gpu_rejected(self):
        with patch.object(queue, "restore_inventory", return_value=[["GPU-one", "201", "worker"]]), \
                patch.object(queue, "tree", return_value=[{"pid": 201}]), \
                self.assertRaisesRegex(RuntimeError, "wrong GPU"):
            queue.verify_restored_mapping("qwen08", self.start, {"qwen08": {"pid": 201}})

    def test_wrong_port_identity_and_replacement_owner_rejected(self):
        with patch.object(queue, "listeners", return_value=[201]), \
                patch.object(queue, "http_get", return_value='{"data":[{"id":"wrong-model"}]}'), \
                self.assertRaisesRegex(RuntimeError, "wrong model"):
            queue.check_endpoint(8000, "Qwen3.5-0.8B", {201})
        with patch.object(queue, "listeners", return_value=[999]), \
                patch.object(queue, "http_get") as http, \
                self.assertRaisesRegex(RuntimeError, "replacement/unknown"):
            queue.check_endpoint(8000, "Qwen3.5-0.8B", {201})
        http.assert_not_called()

    def test_verified_endpoint_needs_idle_metrics(self):
        models = '{"data":[{"id":"Qwen3.5-0.8B"}]}'
        for metric in ["vllm:num_requests_running 0\nvllm:num_requests_waiting 1",
                       "vllm:num_requests_running nan\nvllm:num_requests_waiting 0",
                       "vllm:num_requests_running 0"]:
            with patch.object(queue, "listeners", return_value=[201]), \
                    patch.object(queue, "http_get", side_effect=[models, metric]), \
                    self.assertRaises(RuntimeError):
                queue.check_endpoint(8000, "Qwen3.5-0.8B", {201})

    def test_missing_registrations_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Missing trained registrations"):
            queue.verify_trained_registration(self.root)

    def test_no_overwrite_or_resume(self):
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            queue.prepare(self.root, self.root, self.root)
        marker = self.root / "queue-start.json"
        marker.write_text("do-not-overwrite")
        with patch.object(queue, "load_registration", return_value=self.manifest), \
                patch.object(queue, "wait_for_release") as wait, \
                self.assertRaises(FileExistsError):
            queue.run(self.root)
        wait.assert_not_called()
        self.assertEqual(marker.read_text(), "do-not-overwrite")

    def test_no_release_means_no_trained_or_restore_actions(self):
        with patch.object(queue, "load_registration", return_value=self.manifest), \
                patch.object(queue, "identity", return_value={"pid": 1}), \
                patch.object(queue, "wait_for_release", side_effect=RuntimeError("no handoff")), \
                patch.object(queue, "invoke_trained") as trained, \
                patch.object(queue, "restore_originals") as restore, \
                self.assertRaisesRegex(RuntimeError, "no handoff"):
            queue.run(self.root)
        trained.assert_not_called()
        restore.assert_not_called()

    def test_failed_trained_run_still_restores_preserving_partial_rows(self):
        attempts = self.root / "partial-attempts.jsonl"
        attempts.write_text('{"status":"failure","fatal":true}\n')
        with patch.object(queue, "invoke_trained", side_effect=RuntimeError("frozen failure")), \
                patch.object(queue, "restore_originals") as restore:
            result = queue.follow_through(self.root, self.manifest)
        restore.assert_called_once_with(self.root, self.manifest)
        self.assertIn("failed", result["status"])
        self.assertIsNotNone(result["trained_error"])
        self.assertEqual(attempts.read_text(), '{"status":"failure","fatal":true}\n')

    def test_failed_trained_exit_also_restores(self):
        with patch.object(queue, "invoke_trained", return_value=1), \
                patch.object(queue, "restore_originals") as restore:
            result = queue.follow_through(self.root, self.manifest)
        restore.assert_called_once()
        self.assertEqual(result["trained_returncode"], 1)

    def test_restore_refusal_is_explicit_and_not_retried(self):
        with patch.object(queue, "invoke_trained", return_value=1), \
                patch.object(queue, "restore_originals", side_effect=RuntimeError("unknown occupant")) as restore:
            result = queue.follow_through(self.root, self.manifest)
        restore.assert_called_once()
        self.assertEqual(result["status"], "restoration_failed")
        self.assertTrue((self.root / "restoration-failure.json").is_file())

    def test_restoration_rechecks_empty_devices_inside_standard_leases(self):
        held = []

        @contextmanager
        def locks(resources):
            self.assertEqual(resources, ("qwen08", "qwen2"))
            held.append(True)
            try:
                yield
            finally:
                held.pop()

        def occupied():
            self.assertTrue(held)
            raise RuntimeError("unknown occupant after lock")

        with patch.object(queue.trained.suite, "lane_locks", locks), \
                patch.object(queue.trained, "inspect_empty_devices", side_effect=occupied), \
                patch.object(queue.subprocess, "Popen") as spawn, \
                self.assertRaisesRegex(RuntimeError, "unknown occupant after lock"):
            queue.restore_originals(self.root, self.manifest)
        spawn.assert_not_called()
        self.assertFalse(held)

    def test_unverified_service_launch_is_never_repeated(self):
        @contextmanager
        def locks(resources):
            yield

        queue.save(self.root / "restore-qwen08-launch.json", {"status": "launch_intent"})
        with patch.object(queue.trained.suite, "lane_locks", locks), \
                patch.object(queue.trained, "inspect_empty_devices", return_value={
                    "physical_gpus": {0: {"uuid": "GPU-zero"}, 1: {"uuid": "GPU-one"}}}), \
                patch.object(queue, "alive", side_effect=lambda r: r["pid"] == 121), \
                patch.object(queue, "port_free"), \
                patch.object(queue.subprocess, "Popen") as spawn, \
                self.assertRaisesRegex(RuntimeError, "no retry"):
            queue.restore_originals(self.root, self.manifest)
        spawn.assert_not_called()

    def test_restore_uses_exact_saved_commands_detaches_and_verifies(self):
        held, calls = [], []

        @contextmanager
        def locks(resources):
            held.append(True)
            try:
                yield
            finally:
                held.pop()

        def spawn(argv, **kwargs):
            self.assertTrue(held)
            lane = list(queue.LANES)[len(calls)]
            saved = self.start["services"][lane]
            self.assertEqual(argv, saved["argv"])
            self.assertEqual(kwargs["cwd"], saved["cwd"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"],
                             saved["cuda_environment"]["CUDA_VISIBLE_DEVICES"])
            self.assertTrue(kwargs["start_new_session"])
            self.assertNotIn("preexec_fn", kwargs)
            calls.append(lane)
            return SimpleNamespace(pid=200 + len(calls), poll=lambda: None)

        with patch.object(queue.trained.suite, "lane_locks", locks), \
                patch.object(queue.trained, "inspect_empty_devices", return_value={
                    "physical_gpus": {0: {"uuid": "GPU-zero"}, 1: {"uuid": "GPU-one"}}}), \
                patch.object(queue, "alive", side_effect=lambda r: r["pid"] not in {111, 112}), \
                patch.object(queue, "identity", side_effect=lambda pid: {
                    "pid": pid, "start_ticks": str(pid), "state": "S"}), \
                patch.object(queue, "tree", side_effect=lambda r: [r]), \
                patch.object(queue, "port_free"), \
                patch.object(queue, "restore_inventory", return_value=[]), \
                patch.object(queue, "verify_restored_mapping"), \
                patch.object(queue, "check_endpoint", return_value=True), \
                patch.object(queue.subprocess, "Popen", side_effect=spawn):
            queue.restore_originals(self.root, self.manifest)
        self.assertEqual(calls, ["qwen08", "qwen2"])
        self.assertEqual(queue.read(self.root / "restoration.json")["status"], "verified")
        self.assertTrue(queue.read(self.root / "restore-qwen08-process.json")["verified_ready"])
