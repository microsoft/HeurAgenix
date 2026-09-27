from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from scripts import full_matrix_trained_parent as entry

full = entry.full


class ParentLeaseTests(unittest.TestCase):
    def setUp(self):
        self.root = full.REPO / "output" / ("trained-parent-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        full.write(self.root / "registration.json", {"authored": True})
        with patch.object(full, "verify_registration"):
            entry.register(self.root)
        self.parent = {"pid": 123, "start_ticks": 987654,
                       "argv": [full.pilot.PYTHON, str(full.REPO / entry.COORDINATOR), "launch"]}
        self.locks = {lane: {"path": path, "owner_pid": 123,
                            "device_inode": f"00:35:{index}"}
                      for index, (lane, path) in enumerate(entry.lock_paths().items())}
        self.handoff = {
            "approved": True, "prior_gpu_owner_released": True,
            "parent_owns_standard_locks": True,
            "registration_sha256": full.digest(self.root / "registration.json"),
            "parent_lease_registration_sha256": full.digest(self.root / entry.AMENDMENT),
            "parent_pid": 123, "parent_start_ticks": 987654,
            "cuda_visible_devices": "GPU-authored-0", "resource_models": list(entry.LANES),
            "locks": self.locks, "coordination_registration": str(self.root / "coordinator/registration.json"),
            "coordination_registration_sha256": "authored-coordinator-hash"}

    def lease_context(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-authored-0"}))
        stack.enter_context(patch.object(entry.os, "getppid", return_value=123))
        stack.enter_context(patch.object(entry, "process_identity", return_value=self.parent))
        stack.enter_context(patch.object(entry, "lock_evidence", return_value=self.locks))
        stack.enter_context(patch.object(entry, "verify_coordinator"))
        return stack

    def test_register_is_resource_only_immutable_and_no_inference(self):
        registration = entry.verify_entry(self.root)
        self.assertTrue(registration["parent_owned_lease_required"])
        self.assertTrue(registration["model_protocol_panel_unchanged"])
        self.assertFalse(registration["child_acquires_locks"])
        self.assertFalse(registration["child_changes_services"])
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            entry.register(self.root)
        self.assertEqual(full.read(self.root / "status.json")["benchmark_calls"], 0)

    def test_valid_parent_identity_and_explicit_frozen_handoff(self):
        with self.lease_context():
            self.assertEqual(entry.verify_lease(self.root, self.handoff), self.parent)

    def test_pid_startticks_and_direct_parent_gates(self):
        for key, value in (("parent_pid", 456), ("parent_start_ticks", 1),
                           ("parent_pid", True), ("parent_start_ticks", "987654")):
            with self.subTest(key=key, value=value), self.lease_context():
                with self.assertRaises(RuntimeError):
                    entry.verify_lease(self.root, {**self.handoff, key: value})

    def test_explicit_approval_hashes_and_single_uuid_required(self):
        for key, value in (
                ("approved", False), ("parent_owns_standard_locks", False),
                ("prior_gpu_owner_released", False),
                ("registration_sha256", "drift"), ("parent_lease_registration_sha256", "drift"),
                ("cuda_visible_devices", "0"), ("resource_models", ["qwen08"])):
            with self.subTest(key=key), self.lease_context():
                with self.assertRaises(RuntimeError):
                    entry.verify_lease(self.root, {**self.handoff, key: value})
        with self.lease_context(), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                entry.verify_lease(self.root, {**self.handoff, "cuda_visible_devices": "GPU-a,GPU-b"})

    def test_parent_must_be_expected_coordinator(self):
        with self.lease_context(), patch.object(entry, "process_identity", return_value={
                **self.parent, "argv": [full.pilot.PYTHON, "scripts/unrelated.py"]}):
            with self.assertRaisesRegex(RuntimeError, "registered matrix coordinator"):
                entry.verify_lease(self.root, self.handoff)

    def test_actual_lock_fdinfo_verification_without_opening_lock_paths(self):
        fds = [Path("/proc/123/fd/10"), Path("/proc/123/fd/11")]
        targets = {fd: self.locks[lane]["path"] for fd, lane in zip(fds, entry.LANES)}
        info = [f"pos:\t0\nlock:\t1: FLOCK ADVISORY WRITE 123 {self.locks[lane]['device_inode']} 0 EOF\n"
                for lane in entry.LANES]
        with patch.object(Path, "iterdir", return_value=iter(fds)), \
                patch.object(entry.os, "readlink", side_effect=lambda fd: targets[fd]), \
                patch.object(Path, "read_text", side_effect=info) as reads:
            self.assertEqual(entry.lock_evidence(123, self.locks), self.locks)
        self.assertEqual(reads.call_count, 2)

    def test_missing_stale_or_wrong_lock_owner_is_refused(self):
        fd = Path("/proc/123/fd/10")
        for line in ("", "lock: 1: FLOCK ADVISORY WRITE 456 00:35:0 0 EOF",
                     "lock: 1: POSIX ADVISORY WRITE 123 00:35:0 0 EOF"):
            with patch.object(Path, "iterdir", return_value=iter([fd])), \
                    patch.object(entry.os, "readlink", return_value=self.locks["qwen08"]["path"]), \
                    patch.object(Path, "read_text", return_value=line):
                with self.assertRaisesRegex(RuntimeError, "standard FLOCK"):
                    entry.lock_evidence(123, self.locks)

    def test_real_self_identity_reads_start_ticks(self):
        identity = entry.process_identity(os.getpid())
        self.assertEqual(identity["pid"], os.getpid())
        self.assertGreater(identity["start_ticks"], 0)
        self.assertTrue(identity["argv"])

    def test_parent_launch_does_not_reacquire_locks_or_change_cuda_uuid(self):
        full.write(self.root / "handoff.json", self.handoff)
        observation = {"physical_gpus": {0: {"uuid": "GPU-authored-0"}}}

        def evaluate(root, candidate, instances, registration, *, blocked, worker_factory):
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-authored-0")
            self.assertEqual(os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"], "approved")
            self.assertIsNone(blocked)
            output = root / f"run-{candidate}"
            output.mkdir()
            full.write(output / "status.json", {"worker_released": True, "calls": 0})
            return False

        with self.lease_context(), \
                patch.object(entry, "bind_parent") as bind, \
                patch.object(full, "verify_registration", return_value=({"code_sha256": {}, "models": {}}, [])), \
                patch.object(full.parent, "inspect_empty_devices", return_value=observation), \
                patch.object(full, "evaluate", side_effect=evaluate) as execute, \
                patch.object(full.pilot, "verify_artifacts"), \
                patch.object(full, "verify_terminal", return_value={"status": "verified"}), \
                patch.object(full.suite, "lane_locks", side_effect=AssertionError("must not reacquire")):
            result = entry.launch(self.root, self.root / "handoff.json")
        bind.assert_called_once_with(123)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(result["status"], "terminal_verified")
        self.assertTrue(result["parent_owns_service_restoration"])
        self.assertEqual(full.read(self.root / "launch.json")["parent"], self.parent)
        with self.assertRaisesRegex(RuntimeError, "no retry/resume"):
            entry.launch(self.root, self.root / "handoff.json")

    def test_gpu_identity_mismatch_refuses_before_launch(self):
        full.write(self.root / "handoff.json", self.handoff)
        with self.lease_context(), patch.object(entry, "bind_parent"), \
                patch.object(full, "verify_registration", return_value=({}, [])), \
                patch.object(full.parent, "inspect_empty_devices", return_value={
                    "physical_gpus": {0: {"uuid": "GPU-other"}}}), \
                patch.object(full, "evaluate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "physical GPU0"):
                entry.launch(self.root, self.root / "handoff.json")
        evaluate.assert_not_called()
        self.assertFalse((self.root / "launch.json").exists())

    def test_coordinator_registration_source_and_snapshot_gate(self):
        root = self.root / "coordinator"
        root.mkdir()
        source = {entry.COORDINATOR: full.digest(full.REPO / entry.COORDINATOR)}
        snap = root / "source-snapshot" / entry.COORDINATOR
        snap.parent.mkdir(parents=True)
        snap.write_bytes((full.REPO / entry.COORDINATOR).read_bytes())
        full.write(root / "registration.json", {
            "source_sha256": source, "resources": list(entry.LANES), "restore_originals": True,
            "thinking": False})
        digest = full.digest(root / "registration.json")
        full.write(root / "registration.sha256", digest)
        entry.verify_coordinator(root / "registration.json", digest)
        snap.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "frozen source changed"):
            entry.verify_coordinator(root / "registration.json", digest)


if __name__ == "__main__":
    unittest.main()
