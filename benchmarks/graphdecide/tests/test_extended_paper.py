from copy import deepcopy
import csv
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
from xml.etree import ElementTree

from scripts import extended_graph_suite as suite
from scripts.summarize_extended_paper import (
    detail_tables, paired_intervals, size_plot, structural_clusters, verified_inputs,
)
from src.benchmark.extended_tasks import build_curriculum


class PaperTableTests(TestCase):
    def test_cycle_twins_and_shared_graphs_form_transitive_clusters(self):
        metadata = {"cases": {
            "cycle-no": {"cluster_id": "forest", "pair_id": "twin"},
            "cycle-yes": {"cluster_id": "cyclic", "pair_id": "twin"},
            "query": {"cluster_id": "cyclic", "pair_id": "queries"},
            "query-twin": {"cluster_id": "another", "pair_id": "queries"},
            "independent": {"cluster_id": "separate", "pair_id": None},
        }}
        clusters = structural_clusters(metadata)
        self.assertEqual(len({clusters[key] for key in metadata["cases"]
                              if key != "independent"}), 1)
        self.assertNotEqual(clusters["independent"], clusters["cycle-no"])

    def test_ratios_do_not_impute_failed_or_zero_optimum_objectives(self):
        rows = [
            {"instance_id": "zero", "task": "maxcut_construct", "kind": "optimization",
             "node_count": 5, "feasible": True, "objective": 0, "optimum": 0,
             "absolute_gap": 0, "direction": "maximize"},
            {"instance_id": "failed", "task": "tsp_construct", "kind": "optimization",
             "node_count": 5, "feasible": False, "objective": None, "optimum": 12,
             "absolute_gap": None, "direction": "minimize"},
            {"instance_id": "tour", "task": "tsp_construct", "kind": "optimization",
             "node_count": 5, "feasible": True, "objective": 18, "optimum": 12,
             "absolute_gap": 6, "direction": "minimize"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detail_tables(root, {"model": rows})
            with (root / "optimization-quality.csv").open() as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual([r["relative_gap"] for r in saved], ["", "", "0.5"])
        self.assertEqual([r["objective_ratio"] for r in saved], ["", "", "1.5"])
        self.assertEqual(saved[1]["objective"], "")

    def test_cluster_not_action_denominator_and_paired_sign(self):
        instances = build_curriculum(node_counts=(5, 6), samples_per_size=2)
        left = [{"instance_id": i["id"], "correct": True, "optimal": True} for i in instances]
        right = [{"instance_id": i["id"], "correct": False, "optimal": False} for i in instances]
        report = paired_intervals(instances, left, right, repetitions=100)
        self.assertEqual(len(report), 7)
        for cell in report.values():
            self.assertEqual(cell["generation_clusters"], 4)
            self.assertEqual(cell["difference_left_minus_right"], 1)
            self.assertEqual(cell["percentile_95"], [1, 1])
            self.assertTrue(cell["degenerate_empirical_interval"])
        self.assertEqual(report, paired_intervals(instances, left, right, repetitions=100))
        zero = paired_intervals(instances, left, deepcopy(left), repetitions=100)
        self.assertTrue(all(c["difference_left_minus_right"] == 0 for c in zero.values()))

    def test_size_plot_is_valid_xml_and_escapes_labels(self):
        summaries = {"model<&": {"by_task": {"degree_exact": {}}, "by_task_and_size": {
            "degree_exact": {"5": {"accuracy": 0.5}, "6": {"accuracy": 1.0}}}}}
        root = ElementTree.fromstring(size_plot(summaries))
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("model&lt;&amp;", size_plot(summaries))

    def test_analysis_verifies_with_frozen_not_live_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "plan"
            suite.plan(root, design="curriculum", node_counts=(5,), samples_per_size=1,
                       models=("jev",))
            with patch.object(suite, "load_plan", side_effect=AssertionError("live source")), \
                    patch.object(suite, "report", side_effect=AssertionError("live source")):
                _, manifest, instances, report = verified_inputs(root, ("jev",))
            self.assertEqual(manifest["design"], "curriculum")
            self.assertEqual(len(instances), 13)
            self.assertEqual(report["models"]["jev"]["status"], "not_started")
