from copy import deepcopy
from unittest import TestCase

from paper.finalize_gpt import execution_hashes, ready, render, suite, verify_execution_hashes
from scripts.build_paper_evidence import EXACT, OPTIMIZATION


class PaperFinalizerTests(TestCase):
    def test_waiting_analysis_cannot_silently_change_sources(self):
        hashes = execution_hashes()
        verify_execution_hashes(hashes)
        hashes["paper/finalize_gpt.py"] = "changed"
        with self.assertRaises(suite.Refusal):
            verify_execution_hashes(hashes)

    def test_pending_and_unsealed_results_never_publish(self):
        for status in ("registered", "waiting", "running"):
            self.assertFalse(ready({"status": status}, True))
        self.assertFalse(ready({"status": "completed"}, False))
        self.assertTrue(ready({"status": "completed_with_failures"}, True))

    def test_interruption_and_unknown_status_require_attention(self):
        for status in ("aborted", "interrupted", "unknown"):
            with self.assertRaises(suite.Refusal):
                ready({"status": status}, True)

    def test_failed_construction_uses_scheduled_denominator_and_undefined_gap(self):
        summary = {"by_task": {
            **{task: {"macro_size_accuracy": .75} for task in EXACT},
            **{task: {"instances": 40, "feasible": 0, "optimal_count": 0,
                      "mean_absolute_gap_feasible": None} for task in OPTIMIZATION},
        }}
        original = deepcopy(summary)
        paired = {task: {"percentile_95": [-.1, .2]} for task in EXACT}
        text = render(summary, summary, paired, [{"latency_seconds": .5, "status": "failure"}])
        self.assertIn("0/40", text)
        self.assertIn("--", text)
        self.assertIn("1 failed attempts", text)
        self.assertIn("[-10.00, 20.00]", text)
        self.assertEqual(summary, original)
