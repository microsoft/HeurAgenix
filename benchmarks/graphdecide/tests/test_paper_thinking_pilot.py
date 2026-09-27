import unittest

from paper import build_thinking_pilot as build
from scripts.paired_graph_ablation import Refusal


class ThinkingTableTests(unittest.TestCase):
    def reports(self):
        bounded, thinking = {"metrics": []}, {"metrics": []}
        for task in (*build.EXACT, *build.CONSTRUCTIONS):
            n = 8 if task in build.EXACT else 4
            for profile in ("jev_action", "qwen27_grammar", "qwen27_thinking"):
                row = {"task": task, "profile": profile, "node_count": "all",
                       "episodes": n, "feasible": n, "hits": n,
                       "failure_calls": 0, "mean_absolute_gap_feasible": 0}
                if profile == "qwen27_thinking" and task not in build.EXACT:
                    row.update(feasible=0, hits=0, failure_calls=4,
                               mean_absolute_gap_feasible=None)
                (thinking if profile.endswith("thinking") else bounded)["metrics"].append(row)
        return thinking, bounded

    def test_unavailable_quality_is_not_zero(self):
        tables = build.tables(*self.reports())
        self.assertIn("0/4 & 4 & --", tables["construction.tex"])
        self.assertIn("8/8 & 8/8 & 8/8 & 0", tables["exact.tex"])

    def test_failed_exact_stays_in_denominator(self):
        thinking, bounded = self.reports()
        thinking["metrics"][0].update(feasible=7, hits=7, failure_calls=1)
        self.assertIn("8/8 & 8/8 & 7/8 & 1", build.tables(thinking, bounded)["exact.tex"])

    def test_rejects_invented_gap(self):
        thinking, bounded = self.reports()
        thinking["metrics"][-1]["mean_absolute_gap_feasible"] = 0
        with self.assertRaises(Refusal):
            build.tables(thinking, bounded)

    def test_rejects_missing_cells(self):
        thinking, bounded = self.reports()
        thinking["metrics"].pop()
        with self.assertRaises(Refusal):
            build.tables(thinking, bounded)


if __name__ == "__main__":
    unittest.main()
