from copy import deepcopy
import unittest

from paper import build_full_matrix_table as build
from scripts.paired_graph_ablation import Refusal


class FullMatrixPaperTests(unittest.TestCase):
    def fixture(self):
        metrics = []
        for profile, _ in build.PROFILES:
            for task in build.COUNTS:
                exact = task in build.EXACT
                cells = []
                for size in range(8, 13):
                    count = 8 * size if task == "degree_exact" else 16 if exact else 8
                    cells.append({
                        "profile": profile, "task": task, "node_count": size,
                        "group": "strong_anchors" if profile.startswith("gpt") else "decision_matrix",
                        "scheduled": count, "feasible": count, "hits": count // 2 if exact else 0,
                        "initial_unsupported": 0, "execution_failures": 0, "unattempted": 0,
                        "hit_definition": "correct" if exact else "optimal",
                        "all_scheduled_hit_rate": .5 if exact else 0,
                        "macro_size_accuracy": .5 if exact else None,
                        "conditional_objective_denominator": 0 if exact else count,
                        "mean_objective_feasible": None if exact else 10.0,
                        "mean_absolute_gap_feasible": None if exact else 1.25,
                    })
                aggregate = {**cells[0], "node_count": "all"}
                for key in ("scheduled", "feasible", "hits", "conditional_objective_denominator"):
                    aggregate[key] = sum(c[key] for c in cells)
                metrics.extend([aggregate, *cells])
        return {
            "status": "verified_full_canonical_matrix", "scheduled_per_profile": 960,
            "maximum_decisions_per_profile": 1920, "initial_requests_identical": True,
            "protocol": "explicit_actions", "split": "structural_challenge",
            "profiles": {p: {} for p, _ in build.PROFILES}, "task_counts": build.COUNTS,
            "metrics": metrics,
        }

    def test_all_ten_profiles_and_separate_reasoning_group(self):
        source = self.fixture()
        before = deepcopy(source)
        texts = build.render(source)
        self.assertEqual(set(texts), {"exact.tex", "optimal.tex", "gaps.tex", "coverage.tex"})
        for text in texts.values():
            self.assertIn("\\midrule\nGPT-5.4 &", text)
            for _, name in build.PROFILES:
                self.assertIn(name + " &", text)
        self.assertEqual(source, before)

    def test_missing_or_duplicate_metric_rejected(self):
        for mode in ("missing", "duplicate"):
            source = self.fixture()
            if mode == "missing":
                source["metrics"].pop()
            else:
                source["metrics"].append(source["metrics"][0])
            with self.assertRaises(Refusal):
                build.render(source)

    def test_pending_summary_rejected(self):
        source = self.fixture()
        source["status"] = "waiting"
        with self.assertRaises(Refusal):
            build.render(source)

    def test_unsupported_inputs_retained(self):
        source = self.fixture()
        for row in source["metrics"]:
            if row["profile"] == "laya" and row["task"] == "lt_influence_construct" \
                    and row["node_count"] in ("all", 12):
                row["feasible"] -= 1
                row["initial_unsupported"] = 1
                row["conditional_objective_denominator"] -= 1
        text = build.render(source)["coverage.tex"]
        self.assertIn("Laya & 800/800 & 159/160 & 1 & 0 & 0", text)

    def test_zero_feasible_does_not_invent_zero_gap(self):
        source = self.fixture()
        for row in source["metrics"]:
            if row["profile"] == "laya" and row["task"] == "maxcut_construct":
                row.update(feasible=0, conditional_objective_denominator=0,
                           execution_failures=row["scheduled"],
                           mean_objective_feasible=None, mean_absolute_gap_feasible=None)
        self.assertIn("Laya & -- &", build.render(source)["gaps.tex"])
        next(r for r in source["metrics"] if r["profile"] == "laya"
             and r["task"] == "maxcut_construct")["mean_absolute_gap_feasible"] = 0
        with self.assertRaises(Refusal):
            build.render(source)

    def test_macro_not_pooled_accuracy(self):
        source = self.fixture()
        for row in source["metrics"]:
            if row["profile"] == "jev_action" and row["task"] == "degree_exact":
                if row["node_count"] == 8:
                    row["hits"] = 64
                    row["all_scheduled_hit_rate"] = 1
                elif row["node_count"] == "all":
                    row["hits"] = 232
                    row["all_scheduled_hit_rate"] = .58
                    row["macro_size_accuracy"] = .6
        self.assertIn("Jev & 50.00 & 60.00 &", build.render(source)["exact.tex"])
        for row in source["metrics"]:
            if row["profile"] == "jev_action" and row["task"] == "degree_exact" \
                    and row["node_count"] == "all":
                row["macro_size_accuracy"] = .58
        with self.assertRaises(Refusal):
            build.render(source)

    def test_wrong_budget_or_group_rejected(self):
        source = self.fixture()
        source["maximum_decisions_per_profile"] = 160
        with self.assertRaises(Refusal):
            build.render(source)
        source = self.fixture()
        next(r for r in source["metrics"] if r["profile"].startswith("gpt"))["group"] = "decision_matrix"
        with self.assertRaises(Refusal):
            build.render(source)

    def test_conditional_denominator_rejected(self):
        source = self.fixture()
        next(r for r in source["metrics"] if r["task"] == "maxcut_construct")[
            "conditional_objective_denominator"] = 0
        with self.assertRaises(Refusal):
            build.render(source)

    def test_zero_scheduled_is_explicit_refusal(self):
        source = self.fixture()
        source["metrics"][0]["scheduled"] = 0
        with self.assertRaises(Refusal):
            build.render(source)
