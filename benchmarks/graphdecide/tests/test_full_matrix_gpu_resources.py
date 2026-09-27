from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import full_matrix_gpu_resources as r


class FullResourceTests(unittest.TestCase):
    def test_identity_includes_start_ticks_arguments_and_cuda(self):
        identity = {"pid": 12, "start_ticks": "42", "argv": ["server"],
                    "cuda_environment": {"CUDA_VISIBLE_DEVICES": "0"}, "cwd": "/repo"}
        self.assertTrue(r.same_identity(identity, dict(identity)))
        for key, replacement in (("start_ticks", "43"), ("pid", 13),
                                 ("argv", ["other"]), ("cuda_environment", {}), ("cwd", "/")):
            self.assertFalse(r.same_identity(identity, {**identity, key: replacement}))

    def test_empty_checks_only_registered_gpu_uuids(self):
        lease = r.Lease(Path("/unused"), {}, {}, ["A", "B"])
        with patch.object(r.runtime, "gpu_inventory", return_value={
                "processes": [["10", "C", "100"]]}):
            self.assertTrue(lease.empty())
        with patch.object(r.runtime, "gpu_inventory", return_value={
                "processes": [["10", "A", "100"]]}):
            self.assertFalse(lease.empty())

    def test_existing_resource_root_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(r.suite.shared.Refusal):
                with r.acquire(Path(tmp)):
                    self.fail("Existing resource root accepted")

    def test_busy_lease_does_not_inspect_or_stop_services(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = Mock()
            manager.__enter__ = Mock(side_effect=r.suite.shared.Refusal("busy"))
            with patch.object(r.suite, "lane_locks", return_value=manager), \
                    patch.object(r, "service") as service, \
                    patch.object(r.runtime, "stop_tree") as stop:
                with self.assertRaisesRegex(RuntimeError, "wait deadline"):
                    with r.acquire(Path(tmp) / "resources", wait_seconds=0):
                        self.fail("Busy lease accepted")
                service.assert_not_called()
                stop.assert_not_called()

    def test_unregistered_child_cannot_be_stopped(self):
        lease = r.Lease(Path("/unused"), {}, {}, ["A", "B"])
        with patch.object(r.runtime, "stop_tree") as stop:
            with self.assertRaises(StopIteration):
                lease.stop(Mock())
            stop.assert_not_called()
