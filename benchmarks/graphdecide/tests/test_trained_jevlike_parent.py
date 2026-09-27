from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from scripts import launch_trained_jevlike_parent as parent


class ParentEntryTests(unittest.TestCase):
    def setUp(self):
        self.root = parent.REPO / "output" / ("trained-parent-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def test_empty_selected_gpus_ignore_untouched_other_services(self):
        with patch.object(parent, "nvidia_query", side_effect=[
            [["0", "GPU-zero", "6"], ["1", "GPU-one", "6"], ["2", "GPU-two", "42000"]],
            [["GPU-two", "789", "existing-4B-service"]],
        ]):
            observation = parent.inspect_empty_devices()
        self.assertEqual(observation["compute_occupants"], [])
        self.assertEqual(set(observation["physical_gpus"]), {0, 1})

    def test_refuses_live_occupant_excess_memory_and_missing_gpu(self):
        inventories = [
            ([["0", "GPU-zero", "6"], ["1", "GPU-one", "6"]],
             [["GPU-zero", "123", "existing-service"]]),
            ([["0", "GPU-zero", "2000"], ["1", "GPU-one", "6"]], []),
            ([["0", "GPU-zero", "6"]], []),
        ]
        for inventory, processes in inventories:
            with self.subTest(inventory=inventory, processes=processes), \
                    patch.object(parent, "nvidia_query", side_effect=[inventory, processes]), \
                    self.assertRaises(RuntimeError):
                parent.inspect_empty_devices()

    def test_standard_locks_held_across_admission_run_and_release(self):
        held, calls = [], []

        @contextmanager
        def locks(resources):
            self.assertEqual(resources, ("qwen08", "qwen2"))
            held.append(True)
            try:
                yield
            finally:
                held.pop()

        def inspect():
            self.assertTrue(held)
            calls.append("inspect")
            return {"compute_occupants": [], "physical_gpus": {0: {"uuid": "GPU-zero"}}}

        def run(root, instances, registration):
            self.assertTrue(held)
            self.assertEqual(parent.os.environ["CUDA_VISIBLE_DEVICES"], "GPU-zero")
            calls.append("run")
            (root / "status.json").write_text(json.dumps({
                "outcomes": {"decider": {"status": "completed"}, "kev": {"status": "completed"}}}))

        with patch.object(parent.suite, "lane_locks", locks), \
                patch.object(parent, "inspect_empty_devices", side_effect=inspect), \
                patch.object(parent, "admit", return_value=([], {})), \
                patch.object(parent.pilot, "run_locked", side_effect=run):
            parent.launch(self.root, self.root / "handoff.json")
        self.assertEqual(calls, ["inspect", "run", "inspect"])
        self.assertFalse(held)
        status = json.loads((self.root / "parent-entry-status.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertTrue(status["gpu_workers_released"])

    def test_occupied_gpu_never_starts_worker(self):
        @contextmanager
        def locks(resources):
            yield

        with patch.object(parent.suite, "lane_locks", locks), \
                patch.object(parent, "inspect_empty_devices", side_effect=RuntimeError("occupied")), \
                patch.object(parent.pilot, "run_locked") as run, \
                self.assertRaises(RuntimeError):
            parent.launch(self.root, self.root / "handoff.json")
        run.assert_not_called()
        status = json.loads((self.root / "parent-entry-status.json").read_text())
        self.assertEqual(status["status"], "refused_before_inference")
