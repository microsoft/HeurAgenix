from dataclasses import asdict
from itertools import permutations, product
from unittest import TestCase

import networkx as nx

from scripts.construction_prefix_controls import (
    ARMS, build_controls, build_tsp_state_controls, continuation_values, optimal_history, suite,
)
from src.benchmark import extended_tasks as tasks


class ConstructionPrefixTests(TestCase):
    def test_state_clarification_changes_only_question_and_not_action_labels(self):
        instances = tasks.build_curriculum(node_counts=(6, 9, 11), samples_per_size=2)
        controls, coverage = build_tsp_state_controls(instances)
        self.assertEqual(len(controls), 18)
        self.assertEqual(len(coverage), 6)
        grouped = {}
        for request, label in controls:
            grouped.setdefault(label["case"], {})[label["arm"]] = (request, label)
        for arms in grouped.values():
            baseline, baseline_label = arms["baseline"]
            explicit, explicit_label = arms["state_explicit"]
            self.assertEqual(baseline.state, explicit.state)
            self.assertEqual(baseline.options, explicit.options)
            self.assertEqual(baseline_label["optimal_option_ids"], explicit_label["optimal_option_ids"])
            self.assertTrue(explicit.question.startswith(baseline.question))
            self.assertIn(f"NOW at vertex {baseline.state['current_vertex']}", explicit.question)

    def test_tsp_continuations_against_independent_permutations(self):
        instance = next(i for i in tasks.build_curriculum(
            node_counts=(5,), samples_per_size=1) if i["task"] == "tsp_construct")
        distance = {(min(u, v), max(u, v)): w for u, v, w in instance["state"]["edges"]}
        reference = optimal_history(instance)
        for step in range(instance["query_budget"]):
            history = reference[:step]
            prefix = [0, *map(int, history)]
            values = continuation_values(instance, history)
            for option, actual in values.items():
                start = prefix + [int(option)]
                remaining = set(instance["state"]["nodes"]) - set(start)
                costs = []
                for tail in permutations(remaining):
                    tour = [*start, *tail, 0]
                    costs.append(sum(distance[min(u, v), max(u, v)]
                                     for u, v in zip(tour, tour[1:])))
                self.assertEqual(actual, min(costs))

    def test_partition_continuations_against_networkx(self):
        for task in ("maxcut_construct", "community_bipartition_construct"):
            state = tasks._graph(5, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])
            instance = tasks._record(task, 0, state)
            graph = nx.Graph()
            graph.add_nodes_from(state["nodes"])
            graph.add_edges_from(state["edges"])
            for history in ([], ["A"], ["B", "A"]):
                for option, actual in continuation_values(instance, history).items():
                    objectives = []
                    for tail in product(("A", "B"), repeat=4-len(history)-1):
                        assignment = ["A", *history, option, *tail]
                        groups = [{v for v in graph if assignment[v] == side} for side in ("A", "B")]
                        value = (nx.cut_size(graph, *groups) if task == "maxcut_construct" else
                                 nx.community.modularity(graph, [g for g in groups if g]))
                        objectives.append(value)
                    self.assertAlmostEqual(float(actual), max(objectives))

    def test_control_matrix_is_bounded_reproducible_and_private(self):
        instances = tasks.build_curriculum(node_counts=(5, 8, 12), samples_per_size=2)
        controls, coverage = build_controls(instances)
        self.assertEqual(len(coverage), 24)
        self.assertLessEqual(len(controls) * 2, 384)
        self.assertEqual((controls, coverage), build_controls(instances))
        groups = {}
        for request, label in controls:
            self.assertTrue(label["optimal_option_ids"])
            self.assertLess(len(label["optimal_option_ids"]), len(request.options))
            self.assertTrue(set(label["optimal_option_ids"]) <= {o.id for o in request.options})
            self.assertNotIn("private", request.state)
            self.assertNotIn("continuation_values", request.state)
            groups.setdefault((label["case"], label["step"]), {})[label["arm"]] = request
        for arms in groups.values():
            self.assertEqual(set(arms), set(ARMS))
            baseline, repeat = asdict(arms["baseline"]), asdict(arms["repeat_baseline"])
            baseline.pop("request_id")
            repeat.pop("request_id")
            self.assertEqual(baseline, repeat)
            self.assertEqual(arms["baseline"].options, arms["reversed_options"].options[::-1])

    def test_complete_or_exact_episode_is_rejected(self):
        instances = tasks.build_curriculum(node_counts=(5,), samples_per_size=1)
        exact = next(i for i in instances if i["kind"] == "exact")
        with self.assertRaises(suite.Refusal):
            continuation_values(exact, [])
        construction = next(i for i in instances if i["kind"] == "optimization")
        with self.assertRaises(suite.Refusal):
            continuation_values(construction, optimal_history(construction))
