"""Continuous-size strata and independent oracle regression tests."""

from collections import Counter
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

import networkx as nx

from src.benchmark import extended_tasks as tasks


class CurriculumTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = tasks.build_curriculum()

    def test_exact_size_coverage_and_budgets(self):
        self.assertEqual(len(self.instances), 264)
        self.assertEqual(sum(i["query_budget"] for i in self.instances), 576)
        self.assertEqual(len({i["id"] for i in self.instances}), 264)
        for n in range(5, 13):
            rows = [i for i in self.instances if len(i["state"]["nodes"]) == n]
            for task in tasks._EXACT:
                answers = Counter(i["private"]["answer"] for i in rows if i["task"] == task)
                self.assertEqual(answers, Counter({d: 2 for d in range(n)})
                                 if task == "degree_exact" else Counter(yes=2, no=2))
            for task in tasks._OPTIMIZATION:
                self.assertEqual(sum(i["task"] == task for i in rows), 2)

    def test_boolean_controls_do_not_leak_from_counts(self):
        for instance in self.instances:
            if instance["task"] not in ("cycle_detection", "pair_connectivity"):
                continue
            if instance["private"]["answer"] != "yes":
                continue
            negative = next(i for i in self.instances
                            if i["id"].rsplit("-", 1)[0] == instance["id"].rsplit("-", 1)[0]
                            and i["private"]["answer"] == "no")
            graphs = []
            for row in (instance, negative):
                g = nx.Graph()
                g.add_nodes_from(row["state"]["nodes"])
                g.add_edges_from(row["state"]["edges"])
                graphs.append(g)
            self.assertEqual(graphs[0].number_of_edges(), graphs[1].number_of_edges())
            self.assertEqual(sorted(d for _, d in graphs[0].degree),
                             sorted(d for _, d in graphs[1].degree))
            if instance["task"] == "pair_connectivity":
                self.assertEqual(instance["state"]["edges"], negative["state"]["edges"])
                for row, graph in zip((instance, negative), graphs):
                    self.assertEqual([graph.degree[v] for v in row["state"]["pair"]], [1, 1])

    def test_reproducible_and_seed_changes_graphs(self):
        self.assertEqual(tasks.build_curriculum(), self.instances)
        self.assertNotEqual(tasks.build_curriculum(seed=20260929), self.instances)
        one = tasks.build_curriculum(node_counts=(6,), samples_per_size=1)
        self.assertEqual(len(one), 14)
        self.assertEqual(sum(i["query_budget"] for i in one), 26)
        for kwargs in ({"seed": True}, {"node_counts": ()}, {"node_counts": (4,)},
                       {"node_counts": (13,)}, {"node_counts": (6, 5)},
                       {"node_counts": (5, 5)}, {"samples_per_size": 0},
                       {"samples_per_size": True}):
            with self.assertRaises(ValueError):
                tasks.build_curriculum(**kwargs)

    def test_independent_oracles_full_curriculum(self):
        with patch.object(tasks, "_held_karp", side_effect=AssertionError("production DP")), \
                patch.object(tasks, "_optimize", side_effect=AssertionError("production oracle")):
            totals = tasks.validate_instances(self.instances)
        self.assertEqual(totals["queries_per_model"], 576)

    def test_lt_objective_is_not_constant(self):
        for instance in self.instances:
            if instance["task"] != "lt_influence_construct":
                continue
            evaluate, candidates = tasks._independent_objectives(instance["task"], instance["state"])
            values = [evaluate(candidate) for candidate in candidates]
            self.assertLess(min(values), max(values))
            self.assertGreater(max(values), 2)

    def test_large_tsp_dp_and_tamper_detection(self):
        small = next(i for i in self.instances if i["task"] == "tsp_construct"
                     and len(i["state"]["nodes"]) == 8)
        self.assertEqual(tasks._independent_tsp_optimum(small["state"]), small["private"]["objective"])
        large = deepcopy(next(i for i in self.instances if i["task"] == "tsp_construct"
                              and len(i["state"]["nodes"]) == 12))
        self.assertEqual(tasks._independent_tsp_optimum(large["state"]), large["private"]["objective"])
        large["private"]["objective"] += 1
        with self.assertRaisesRegex(ValueError, "objective"):
            tasks.validate_instances([large])
