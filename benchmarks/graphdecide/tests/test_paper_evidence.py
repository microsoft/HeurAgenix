from itertools import permutations, product
import math
from unittest import TestCase

from scripts.build_paper_evidence import latex_table, trajectory_regrets
from scripts.construction_prefix_controls import continuation_values, optimal_history, suite
from src.benchmark import extended_tasks as tasks


class PaperEvidenceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = [i for i in tasks.build_curriculum(node_counts=(5,), samples_per_size=1)
                         if i["kind"] == "optimization"]

    def test_all_small_trajectories_telescope_and_retain_first_loss(self):
        for instance in self.instances:
            task = instance["task"]
            if task == "tsp_construct":
                histories = permutations(range(1, 5), 3)
            elif task == "lt_influence_construct":
                histories = permutations(range(5), 2)
            else:
                histories = product(("A", "B"), repeat=4)
            for history in histories:
                decisions = list(map(str, history))
                rows = trajectory_regrets(instance, decisions)
                score = tasks.score(instance, decisions)
                self.assertAlmostEqual(math.fsum(r["incremental_regret"] for r in rows),
                                       score["absolute_gap"])
                self.assertEqual(all(r["locally_optimal"] for r in rows), score["optimal"])
                self.assertTrue(all(r["incremental_regret"] >= -1e-12 for r in rows))

    def test_optimal_reference_never_loses_optimality(self):
        for instance in self.instances:
            rows = trajectory_regrets(instance, optimal_history(instance))
            self.assertTrue(all(r["locally_optimal"] for r in rows))

    def test_incomplete_history_is_not_a_zero_regret_success(self):
        for instance in self.instances:
            with self.assertRaises(suite.Refusal):
                trajectory_regrets(instance, [])

    def test_corrupted_continuation_is_rejected(self):
        instance = self.instances[0]

        def corrupted(inst, history):
            return {key: value + 1 for key, value in continuation_values(inst, history).items()}

        with self.assertRaises(suite.Refusal):
            trajectory_regrets(instance, optimal_history(instance), corrupted)

    def test_table_shape_and_precision(self):
        table = latex_table(("Task", "Rate"), [["Degree", "61.43"]])
        self.assertIn("Degree & 61.43", table)
        with self.assertRaises(suite.Refusal):
            latex_table(("Task", "Rate"), [["Degree"]])
