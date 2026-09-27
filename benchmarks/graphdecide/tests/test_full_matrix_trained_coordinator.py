from contextlib import contextmanager
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch
import uuid

from scripts import full_matrix_trained_coordinator as coordinator

full = coordinator.full


class TrainedCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.root = full.REPO / "output" / ("trained-coordinator-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        full.write(self.root / "registration.json", {"authored": True})
        full.write(self.root / coordinator.leased.AMENDMENT, {"authored": True})
        self.base = {"scheduled_episodes_per_model": 960, "maximum_calls_per_model": 1920}
        sources = {coordinator.ENTRY: full.digest(full.REPO / coordinator.ENTRY)}
        with patch.object(full, "verify_registration", return_value=(self.base, [])), \
                patch.object(coordinator.leased, "verify_entry"), \
                patch.object(coordinator, "source_hashes", return_value=sources), \
                patch.object(coordinator.resources, "acquire", side_effect=AssertionError("offline")):
            coordinator.register(self.root)
        self.out = self.root / coordinator.DIRECTORY
        self.approval = {
            "approved": True,
            "coordinator_registration_sha256": full.digest(self.out / "registration.json"),
            "base_registration_sha256": full.digest(self.root / "registration.json")}
        full.write(self.root / "approval.json", self.approval)
        self.events = []

    def test_frozen_registration_keeps_original_counts_and_has_no_thinking(self):
        registration = coordinator.verify(self.root, base=False)
        self.assertEqual(registration["scheduled_episodes_per_model"], 960)
        self.assertEqual(registration["maximum_calls_per_model"], 1920)
        self.assertFalse(registration["thinking"])
        self.assertTrue(registration["restore_originals"])
        self.assertFalse(registration["child_acquires_locks"])
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            coordinator.register(self.root)

    def test_changed_source_snapshot_rejected(self):
        snapshot = self.out / "source-snapshot" / coordinator.ENTRY
        snapshot.chmod(0o644)
        snapshot.write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "source drift"):
            coordinator.verify(self.root, base=False)

    def launch(self, *, code=0, timeout=False):
        owner = self
        proof = {name: {"verified": True} for name in full.CANDIDATES}

        class Child:
            def wait(self, timeout):
                owner.assertEqual(timeout, coordinator.RUN_SECONDS)
                owner.events.append("wait")
                if owner.timeout:
                    raise subprocess.TimeoutExpired("authored", timeout)
                full.write(owner.root / "terminal-verification.json", proof)
                return code

        class Lease:
            targets = ["GPU-test0", "GPU-test1"]

            def start(self, command, environment, label):
                owner.events.append("start")
                owner.assertIn("worker-phase", command)
                owner.assertIn(str(full.REPO / coordinator.ENTRY), command)
                owner.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-test0")
                owner.assertEqual(label, "trained-supervisor")
                return Child()

            def stop(self, child):
                owner.events.append("stop_owned")

        @contextmanager
        def acquire(root, *, wait_seconds):
            owner.assertEqual(root, owner.out / "resources")
            owner.assertEqual(wait_seconds, coordinator.WAIT_SECONDS)
            owner.events.append("acquire")
            try:
                yield Lease()
            finally:
                owner.events.append("restore")

        self.timeout = timeout
        with patch.object(full, "verify_registration", return_value=(self.base, [])), \
                patch.object(coordinator.leased, "verify_entry"), \
                patch.object(coordinator.resources, "acquire", side_effect=acquire), \
                patch.object(coordinator.resources, "lock_owner_evidence", return_value={}), \
                patch.object(full, "verify_terminal", return_value={"verified": True}):
            return coordinator.launch(self.root, self.root / "approval.json")

    def test_success_uses_fresh_context_and_restores_after_one_supervisor(self):
        result = self.launch()
        self.assertEqual(self.events, ["acquire", "start", "wait", "restore"])
        self.assertTrue(result["original_services_restored"])
        self.assertEqual(result["status"], "terminal_verified_services_restored")
        handoff = full.read(self.out / "worker-handoff.json")
        self.assertEqual(handoff["parent_pid"], os.getpid())
        self.assertIs(type(handoff["parent_start_ticks"]), int)

    def test_child_failure_still_restores(self):
        with self.assertRaisesRegex(RuntimeError, "supervisor failed"):
            self.launch(code=1)
        self.assertEqual(self.events[-1], "restore")
        self.assertEqual(full.read(self.out / "status.json")["status"], "failed")

    def test_finite_supervisor_deadline_stops_only_owned_child_then_restores(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.launch(timeout=True)
        self.assertEqual(self.events, ["acquire", "start", "wait", "stop_owned", "restore"])

    def test_explicit_parent_approval_precedes_resource_acquisition(self):
        bad = self.root / "bad-approval.json"
        full.write(bad, {**self.approval, "approved": False})
        with patch.object(full, "verify_registration", return_value=(self.base, [])), \
                patch.object(coordinator.leased, "verify_entry"), \
                patch.object(coordinator.resources, "acquire") as acquire:
            with self.assertRaisesRegex(RuntimeError, "Explicit parent approval"):
                coordinator.launch(self.root, bad)
        acquire.assert_not_called()
        self.assertFalse((self.out / "start.json").exists())

    def test_new_coordinator_identity_required_without_qwen_handoff(self):
        handoff = {
            "parent_pid": 123, "parent_start_ticks": 456,
            "coordinator_registration_sha256": full.digest(self.out / "registration.json"),
            "base_registration_sha256": full.digest(self.root / "registration.json"),
            "cuda_visible_devices": "GPU-test0", "locks": {}}
        owner = {"pid": 123, "start_ticks": 456,
                 "argv": [full.pilot.PYTHON, str(full.REPO / coordinator.ENTRY), "run"]}
        with patch.object(coordinator.os, "getppid", return_value=123), \
                patch.object(coordinator.leased, "process_identity", return_value=owner), \
                patch.object(coordinator.leased, "lock_evidence") as locks, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test0"}):
            self.assertEqual(coordinator.check_handoff(self.root, handoff), owner)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                coordinator.check_handoff(self.root, {**handoff, "parent_start_ticks": 999})
        locks.assert_called_once_with(123, {})

    def test_no_coordinator_resume_even_before_model_outputs(self):
        full.write(self.out / "start.json", {})
        with patch.object(full, "verify_registration", return_value=(self.base, [])), \
                patch.object(coordinator.leased, "verify_entry"), \
                patch.object(coordinator.resources, "acquire") as acquire:
            with self.assertRaisesRegex(RuntimeError, "No coordinator"):
                coordinator.launch(self.root, self.root / "approval.json")
        acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
