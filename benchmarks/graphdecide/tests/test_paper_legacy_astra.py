from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from paper import build_legacy_astra as paper


class LegacyAstraTableTests(TestCase):
    def test_correctness_and_optimality_columns_remain_distinct(self):
        tasks = (*paper.EXACT, "maxcut_construct", "tsp_construct",
                 "lt_influence_construct", "community_bipartition_construct")
        summary = {"by_task": {
            task: {"instances": 40, "feasible": 40, "correct": 30, "optimal_count": 20,
                   "mean_absolute_gap_feasible": .125} for task in tasks}}
        rows = paper.table_rows(summary)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row[1] == "30/40" and row[3] == "--" for row in rows[:6]))
        self.assertTrue(all(row[1] == "20/40" and row[3] == "0.125000" for row in rows[6:]))

    def test_existing_evidence_is_not_overwritten_or_reverified(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(paper.subprocess, "run", side_effect=AssertionError("subprocess")):
                with self.assertRaises(paper.suite.Refusal):
                    paper.export(Path("unused"), Path("unused"), Path(temporary))
