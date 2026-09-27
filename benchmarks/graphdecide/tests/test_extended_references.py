from itertools import combinations, permutations, product
from unittest import TestCase

from scripts import extended_reference_analysis as analysis
from src.benchmark import extended_tasks as tasks


class ReferenceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = tasks.build_curriculum(node_counts=(5, 6), samples_per_size=1)

    def test_uniform_expectations_match_exhaustive_legal_histories(self):
        for instance in self.instances:
            task, state = instance["task"], instance["state"]
            if instance["kind"] != "optimization":
                continue
            n = len(state["nodes"])
            if task == "tsp_construct":
                solutions = ([0, *p, 0] for p in permutations(range(1, n)))
            elif task == "lt_influence_construct":
                solutions = combinations(range(n), 2)
            else:
                solutions = (["A", *p] for p in product(("A", "B"), repeat=n - 1))
            values = [float(tasks._objective(task, state, s)) for s in solutions]
            self.assertAlmostEqual(analysis.uniform_objective(instance), sum(values) / len(values))

    def test_heuristics_preserve_feasibility_and_improve_greedy(self):
        for instance in self.instances:
            if instance["kind"] != "optimization":
                continue
            task, state, private = instance["task"], instance["state"], instance["private"]
            original = private["baseline"]["solution"].copy()
            _, solution = analysis.improved_solution(instance)
            tasks._validate_solution(task, state, solution)
            objective = tasks._objective(task, state, solution)
            if task == "tsp_construct":
                self.assertLessEqual(objective, private["baseline"]["objective"])
            else:
                self.assertGreaterEqual(float(objective) + 1e-12, private["baseline"]["objective"])
            self.assertGreaterEqual(abs(objective - private["objective"]), 0)
            self.assertEqual(private["baseline"]["solution"], original)

    def test_analysis_shape(self):
        result = analysis.analyze(self.instances)
        self.assertEqual(len(result["instances"]), 8)
        self.assertEqual(set(result["summary"]), set(tasks._OPTIMIZATION))

    def test_modularity_multistart_escapes_all_one_community(self):
        state = tasks._graph(6, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)])
        instance = tasks._record("community_bipartition_construct", 0, state)
        self.assertEqual(instance["private"]["baseline"]["objective"], 0)
        _, solution = analysis.improved_solution(instance)
        self.assertEqual(tasks._objective(instance["task"], state, solution), 0.5)
