from collections import Counter
import asyncio
from contextlib import nullcontext
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from scripts import extended_action_controls as controls
from scripts.extended_action_controls import cases
from scripts.extended_response_diagnostic import diagnostics
from test_extended_graph_suite import FakeClient


class ActionControlTests(TestCase):
    def test_opt_in_constraints_are_frozen_for_the_same_local_checkpoint(self):
        kwargs_seen = []

        def factory(_, **kwargs):
            kwargs_seen.append(kwargs)
            return FakeClient(kwargs["model"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "controls"
            with patch.object(controls.suite, "lane_locks", return_value=nullcontext()), \
                    patch.object(controls.suite.shared, "vllm_preflight", new=AsyncMock()), \
                    patch.object(controls, "create_client", side_effect=factory):
                asyncio.run(controls.run(root, ("qwen4",), constrain_local_choices=True))
            manifest = controls.suite.read_json(root / "manifest.json")
            self.assertTrue(manifest["constrain_local_choices"])
            self.assertTrue(manifest["models"]["qwen4"]["constrain_choices"])
            self.assertEqual(manifest["models"]["qwen4"]["model"], "Qwen3.5-4B")
            self.assertTrue(kwargs_seen[0]["constrain_choices"])

    def test_forced_actions_and_balanced_labels(self):
        rows = cases()
        self.assertEqual(len(rows), 12)
        self.assertEqual(Counter(label["expected"] for _, label in rows), {"A": 6, "B": 6})
        for request, label in rows:
            expected = ("B", "A")[label["case"]] if label["task"] == "maxcut_construct" \
                else ("A", "B")[label["case"]]
            self.assertEqual(label["expected"], expected)
            self.assertNotIn("expected", request.state)
            if label["arm"] == "reversed_options":
                self.assertEqual([o.id for o in request.options], ["B", "A"])

    def test_native_diagnostics_are_safe_for_malformed_shapes(self):
        request, _ = cases()[0]
        for body in (None, {}, {"answers": None}, {"answers": {"decision": []}}):
            row = diagnostics(request, body)
            self.assertIsNone(row["probability_sum"])
            self.assertFalse(row["choice_is_supplied"])
