import unittest
from unittest.mock import Mock, patch

from scripts import stronger_local_coordinator_v2 as c
from tests.test_stronger_local_execution_v1 import ProjectFixture


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class PostExecTests(unittest.TestCase):
    def setUp(self):
        self.anchor = {"pid": 123, "start_ticks": "456"}
        self.expected = {
            "argv": ["python", "vllm", "serve", "original-model"],
            "cwd": str(c.REPO),
            "cuda_environment": {"CUDA_VISIBLE_DEVICES": "0"},
            "restore_environment": {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "7"}}
        self.match = {**self.anchor, **self.expected, "state": "S"}
        self.clock = FakeClock()

    def wait(self, reader, **kwargs):
        return c.wait_post_exec(
            self.anchor, self.expected, reader=reader, clock=self.clock,
            sleep=self.clock.sleep, is_alive=kwargs.pop("is_alive", lambda _: True),
            timeout=kwargs.pop("timeout", 1), **kwargs)

    def test_empty_preexec_then_matching_is_confirmed(self):
        early = {**self.match, "argv": [], "cuda_environment": {}, "restore_environment": {}}
        result, evidence = self.wait(Mock(side_effect=[early, self.match]))
        self.assertEqual(result, self.match)
        self.assertEqual(evidence["observations"], 2)
        self.assertTrue(evidence["definitive_identity"])

    def test_parent_argv_then_matching_exec_is_confirmed(self):
        early = {**self.match, "argv": ["python", "coordinator.py"]}
        _, evidence = self.wait(Mock(side_effect=[early, self.match]))
        self.assertEqual(evidence["observations"], 2)

    def test_transient_proc_reads_are_bounded_and_recorded(self):
        reader = Mock(side_effect=[FileNotFoundError(), PermissionError(), self.match])
        _, evidence = self.wait(reader)
        self.assertEqual(evidence["transient_read_errors"], 2)
        with self.assertRaisesRegex(c.suite.shared.Refusal, "deadline"):
            self.wait(Mock(side_effect=ProcessLookupError()), timeout=.2)

    def test_same_pid_different_startticks_is_immediate_refusal(self):
        reader = Mock(return_value={**self.match, "start_ticks": "other"})
        with self.assertRaisesRegex(c.suite.shared.Refusal, "identity changed"):
            self.wait(reader)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(self.clock.now, 0)

    def test_different_pid_is_immediate_refusal(self):
        with self.assertRaisesRegex(c.suite.shared.Refusal, "identity changed"):
            self.wait(Mock(return_value={**self.match, "pid": 124}))

    def test_exited_process_is_not_retried(self):
        reader = Mock()
        with self.assertRaisesRegex(c.suite.shared.Refusal, "exited"):
            self.wait(reader, is_alive=lambda _: False)
        reader.assert_not_called()

    def test_zombie_is_not_confirmed(self):
        with self.assertRaisesRegex(c.suite.shared.Refusal, "exited"):
            self.wait(Mock(return_value={**self.match, "state": "Z"}))

    def test_wrong_argv_environment_or_cwd_reaches_explicit_deadline(self):
        for field in ("argv", "cwd", "cuda_environment", "restore_environment"):
            self.clock = FakeClock()
            wrong = {**self.match, field: [] if field == "argv" else "wrong"}
            with self.assertRaisesRegex(c.suite.shared.Refusal, "deadline"):
                self.wait(Mock(return_value=wrong), timeout=.2)
            self.assertAlmostEqual(self.clock.now, .2)


class LeaseV2Tests(ProjectFixture):
    def test_all_four_restore_after_empty_initial_capture(self):
        originals, current, launched = {}, {}, []
        for lane, (gpu, port, _) in c.LANES.items():
            originals[lane] = {
                "pid": 100 + gpu, "start_ticks": "1", "argv": ["original", str(port)],
                "cwd": str(c.REPO), "cuda_environment": {"CUDA_VISIBLE_DEVICES": str(gpu)},
                "restore_environment": {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "7"}}
            current[lane] = {**originals[lane], "pid": 200 + gpu, "start_ticks": "2",
                             "state": "S", "process_tree": [{"pid": 200 + gpu}]}
        def popen(command, **kwargs):
            gpu = len(launched)
            self.assertEqual(command, originals[list(c.LANES)[gpu]]["argv"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(gpu))
            launched.append(gpu)
            return Mock(pid=200 + gpu)
        def initial(pid):
            return {"anchor": {"pid": pid, "start_ticks": "2"},
                    "metadata": {"argv": []}, "definitive_identity": False}
        original_wait = c.wait_post_exec
        def handshake(anchor, old):
            matching = current[list(c.LANES)[anchor["pid"] - 200]]
            early = {**matching, "argv": [], "restore_environment": {}}
            clock = FakeClock()
            return original_wait(
                anchor, old, reader=Mock(side_effect=[early, matching]),
                is_alive=lambda _: True, clock=clock, sleep=clock.sleep)
        def inventory():
            return {"gpus": [[str(i), f"GPU-{i}"] for i in range(4)],
                    "processes": [[str(200 + i), f"GPU-{i}", "10"] for i in launched]}
        lease = c.Lease(self.root, originals, [f"GPU-{i}" for i in range(4)])
        with patch.object(c.runtime, "alive", side_effect=lambda identity: identity["pid"] >= 200), \
                patch.object(c, "initial_identity", side_effect=initial), \
                patch.object(c, "wait_post_exec", side_effect=handshake), \
                patch.object(c.runtime, "gpu_inventory", side_effect=inventory), \
                patch.object(c.resources, "port_free"), \
                patch.object(c.v1, "idle_originals"), \
                patch.object(c.v1, "originals", return_value=current), \
                patch.object(c.subprocess, "Popen", side_effect=popen):
            lease.restore()
        self.assertEqual(launched, [0, 1, 2, 3])
        self.assertTrue(c.suite.read_json(self.root / "restoration.json")["original_services_restored"])
        for lane in c.LANES:
            self.assertEqual(c.suite.read_json(self.root / f"restore-{lane}-initial-process.json")[
                "metadata"]["argv"], [])
            self.assertEqual(c.suite.read_json(self.root / f"restore-{lane}-process.json")[
                "argv"], originals[lane]["argv"])
            self.assertEqual(c.suite.read_json(self.root / f"restore-{lane}-post-exec.json")[
                "observations"], 2)

    def test_initial_capture_preserved_definitive_capture_postexec_only(self):
        lane = "qwen08"
        initial = {"anchor": {"pid": 123, "start_ticks": "456"},
                   "metadata": {"argv": []}, "definitive_identity": False}
        c.suite.write_json(self.root / f"restore-{lane}-initial-process.json", initial)
        final = {"pid": 123, "start_ticks": "456", "argv": ["expected"]}
        evidence = {"definitive_identity": True}
        lease = c.Lease(self.root, {}, [])
        with patch.object(c, "wait_post_exec", return_value=(final, evidence)):
            self.assertEqual(lease.confirmed_restoration(lane, {}, initial), final)
        self.assertEqual(c.suite.read_json(self.root / f"restore-{lane}-initial-process.json"), initial)
        self.assertEqual(c.suite.read_json(self.root / f"restore-{lane}-process.json"), final)
        self.assertTrue(c.suite.read_json(self.root / f"restore-{lane}-post-exec.json")[
            "definitive_identity"])

    def test_wrong_identity_never_writes_definitive_capture(self):
        initial = {"anchor": {"pid": 123, "start_ticks": "456"}}
        lease = c.Lease(self.root, {}, [])
        with patch.object(c, "wait_post_exec", side_effect=c.suite.shared.Refusal("wrong identity")):
            with self.assertRaisesRegex(c.suite.shared.Refusal, "wrong identity"):
                lease.confirmed_restoration("qwen08", {}, initial)
        self.assertFalse((self.root / "restore-qwen08-process.json").exists())

    def test_recovery_after_confirmation_before_definitive_write(self):
        lane = "qwen08"
        initial = {"anchor": {"pid": 123, "start_ticks": "456"}, "metadata": {"argv": []}}
        path = self.root / f"restore-{lane}-initial-process.json"
        c.suite.write_json(path, initial)
        evidence = {"definitive_identity": True, "confirmed_anchor": initial["anchor"],
                    "initial_capture_sha256": c.suite.digest(path)}
        c.suite.write_json(self.root / f"restore-{lane}-post-exec.json", evidence)
        final = {"pid": 123, "start_ticks": "456", "argv": ["expected"]}
        lease = c.Lease(self.root, {}, [])
        with patch.object(c, "wait_post_exec", return_value=(final, {"definitive_identity": True})):
            lease.confirmed_restoration(lane, {}, initial)
        self.assertEqual(c.suite.read_json(self.root / f"restore-{lane}-process.json"), final)
        self.assertEqual(c.suite.read_json(self.root / f"restore-{lane}-post-exec.json"), evidence)

    def test_acquisition_uses_v2_lease_and_restores_after_interrupt(self):
        lock = Mock()
        lock.__enter__ = Mock()
        lock.__exit__ = Mock()
        originals = {lane: {"pid": i + 1, "process_tree": []}
                     for i, lane in enumerate(c.LANES)}
        lease = Mock()
        with patch.object(c.suite, "lane_locks", return_value=lock), \
                patch.object(c.v1, "lock_ownership", return_value={}), \
                patch.object(c.v1, "snapshot_originals", return_value=(originals, ["a", "b", "c", "d"])), \
                patch.object(c.v1, "idle_originals"), \
                patch.object(c.resources, "port_free"), \
                patch.object(c.resources, "service"), \
                patch.object(c.resources, "same_identity", return_value=True), \
                patch.object(c.admission, "four_gpu_mapping"), \
                patch.object(c.runtime, "gpu_inventory"), \
                patch.object(c.runtime, "stop_tree", return_value=True), \
                patch.object(c, "Lease", return_value=lease) as factory:
            with self.assertRaises(KeyboardInterrupt):
                with c.acquire(self.root / "resources"):
                    raise KeyboardInterrupt("fake SIGTERM")
        factory.assert_called_once()
        lease.cleanup.assert_called_once()
        lease.restore.assert_called_once()
        lock.__exit__.assert_called_once()

    def test_frozen_v1_sources_are_not_rebound(self):
        from scripts import stronger_local_coordinator_v1
        self.assertIs(stronger_local_coordinator_v1.Lease, c.v1.Lease)
        self.assertIsNot(c.Lease, c.v1.Lease)
        self.assertIsNot(c.acquire, c.v1.acquire)


if __name__ == "__main__":
    unittest.main()
