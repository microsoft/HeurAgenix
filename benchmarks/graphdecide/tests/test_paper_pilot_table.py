from unittest import TestCase

from paper import build_pilot_table as paper


class PilotTableTests(TestCase):
    def metrics(self):
        return [{"task": task, "profile": profile, "node_count": "all",
                 "episodes": 8 if task in paper.EXACT else 4,
                 "feasible": 8 if task in paper.EXACT else 4,
                 "failed_episodes": 0, "failure_calls": 0, "hits": 2,
                 "mean_absolute_gap_feasible": .125}
                for task in (*paper.EXACT, *paper.CONSTRUCTIONS) for profile in paper.PROFILES]

    def test_exact_and_construction_denominators_and_gaps(self):
        output = paper.tables(self.metrics())
        self.assertEqual(output["exact.tex"].count("2/8"), 18)
        self.assertEqual(output["construction.tex"].count("2/4"), 12)
        self.assertEqual(output["construction.tex"].count("0.125"), 12)

    def test_missing_duplicate_or_infeasible_cells_are_refused(self):
        for mode in ("missing", "duplicate", "infeasible"):
            rows = self.metrics()
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = rows[0]
            else:
                rows[-1]["feasible"] = 3
            with self.assertRaises(paper.suite.Refusal):
                paper.tables(rows)
