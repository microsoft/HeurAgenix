import asyncio
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from scripts import full_matrix_qwen27_coordinator as c


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jev-full-coordinator-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "registration.json").write_text("{}")

    def test_registration_has_only_bounded_no_thinking_profile(self):
        with patch.object(c.qwen, "load"):
            registration = c.register(self.root)
            self.assertFalse(registration["thinking"])
            self.assertEqual(registration["run_only"], "qwen27_grammar")
            self.assertEqual(registration["resources"], ["qwen08", "qwen2"])
            self.assertTrue(registration["restore_originals"])
            self.assertEqual(c.verify(self.root), registration)
            with self.assertRaises(c.suite.shared.Refusal):
                c.register(self.root)

    def test_source_or_input_drift_precedes_resource_admission(self):
        with patch.object(c.qwen, "load"):
            c.register(self.root)
            (self.root / "registration.json").write_text('{"changed":true}')
            with patch.object(c.resources, "acquire") as acquire:
                with self.assertRaises(c.suite.shared.Refusal):
                    c.launch(self.root)
                acquire.assert_not_called()

    def test_async_signal_handler_restored_for_cleanup(self):
        previous = signal.getsignal(signal.SIGTERM)
        with patch.object(c.qwen, "run", new=AsyncMock(return_value={"status": "completed"})):
            result = asyncio.run(c.run_cancellable(self.root))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
