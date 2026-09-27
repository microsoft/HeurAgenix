import unittest

from scripts.audit_task_shortcuts import audit
from src.benchmark import extended_tasks as tasks


class ShortcutTests(unittest.TestCase):
    def test_matched_cycle_counts_and_degrees_are_uninformative(self):
        rows = [i for i in tasks.build_curriculum(node_counts=(8,), samples_per_size=2)
                if i["task"] == "cycle_detection"]
        metrics = audit(rows)["by_task"]["cycle_detection"]
        self.assertEqual(metrics["empirical_lookup_ceiling"], 0.5)
        self.assertEqual(metrics["mixed_label_instance_fraction"], 1)

    def test_degree_shortcut_in_role_controls_is_explicit(self):
        graph = tasks._graph(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
        rows = [tasks._record("articulation_point", v, {**graph, "vertex": v}) for v in (0, 1)]
        result = audit(rows)["by_task"]["articulation_point"]
        self.assertEqual(result["empirical_lookup_ceiling"], 1)
        self.assertEqual(result["by_size"]["5"]["majority_rate"], 0.5)

    def test_degree_is_not_mistakenly_flagged_for_its_definition(self):
        row = tasks._record("degree_exact", 0, {**tasks._graph(5, [(0, 1)]), "vertex": 0})
        self.assertNotIn("empirical_lookup_ceiling", audit([row])["by_task"]["degree_exact"])
