import unittest
from unittest.mock import patch

from scripts import monitor_paper_completion as monitor


class MonitorTests(unittest.TestCase):
    def snapshot(self, source="running", final=None, live=True):
        registered = {name: {role: {"start_ticks": "1"} for role in ("worker", "finalizer")}
                      for name in monitor.JOBS}

        def read(path, optional=False):
            return final if optional else {"status": source, "calls": 5}

        with patch.object(monitor, "read", side_effect=read), \
                patch.object(monitor, "alive", return_value=live), \
                patch.object(monitor.Path, "exists", return_value=False):
            return monitor.snapshot(registered)

    def test_live_waiting_is_not_an_error(self):
        result = self.snapshot()
        self.assertEqual(result["alerts"], [])
        self.assertFalse(result["all_analyses_finished"])

    def test_dead_worker_and_finalizer_raise_alerts(self):
        self.assertEqual(len(self.snapshot(live=False)["alerts"]), 4)

    def test_failed_finalizer_is_not_success(self):
        result = self.snapshot(source="completed_with_failures", final={"status": "failed"})
        self.assertEqual(len(result["alerts"]), 2)
        self.assertFalse(result["all_analyses_finished"])

    def test_pid_reuse_is_not_live(self):
        with patch.object(monitor, "identity", return_value={"start_ticks": "2", "state": "S"}):
            self.assertFalse(monitor.alive(123, {"start_ticks": "1"}))


if __name__ == "__main__":
    unittest.main()
