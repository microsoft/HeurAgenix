import unittest

from paper import build_laya_table as build
from scripts.paired_graph_ablation import Refusal


class LayaTableTests(unittest.TestCase):
    def panel(self):
        rows, references = [], []
        for task in (*build.EXACT, *build.CONSTRUCTIONS):
            for index in range(8 if task in build.EXACT else 4):
                row = {"instance_id": f"{task}-{index}", "task": task,
                       "initial_context_compatible": True, "feasible": True,
                       "correct": False, "optimal": False, "absolute_gap": 3}
                references.append({**row, "correct": True, "optimal": True, "absolute_gap": 0})
                rows.append(row)
        return rows, references

    def test_unsupported_keeps_coverage_denominator(self):
        rows, references = self.panel()
        for row in rows:
            if row["task"] == "lt_influence_construct" and row["instance_id"].endswith(("-2", "-3")):
                row.update(initial_context_compatible=False, feasible=False)
        tables = build.tables(rows, references)
        self.assertIn("2/4 & 0/2 & 2/2 & 3.000 & 0.000", tables["construction.tex"])
        self.assertIn("0/8 & 8/8", tables["exact.tex"])

    def test_empty_supported_subset_is_not_zero_gap(self):
        rows, references = self.panel()
        for row in rows:
            if row["task"] == "lt_influence_construct":
                row.update(initial_context_compatible=False, feasible=False)
        self.assertIn("0/4 & 0/0 & 0/0 & -- & --",
                      build.tables(rows, references)["construction.tex"])

    def test_later_failure_is_not_silently_excluded(self):
        rows, references = self.panel()
        rows[-1]["feasible"] = False
        with self.assertRaises(Refusal):
            build.tables(rows, references)

    def test_duplicate_reference_is_rejected(self):
        rows, references = self.panel()
        references[-1] = references[0]
        with self.assertRaises(Refusal):
            build.tables(rows, references)


if __name__ == "__main__":
    unittest.main()
