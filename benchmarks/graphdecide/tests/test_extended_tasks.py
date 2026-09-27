"""Offline tests for the standalone extended-task development pilot."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations, permutations, product
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import networkx as nx

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.benchmark import extended_tasks as tasks
from src.clients.base import DecisionRequest


def graph(n, edges, directed=False):
    return {"nodes": list(range(n)), "edges": [list(e) for e in sorted(edges)],
            "directed": directed}


def lt_graph(n, edges):
    state = graph(n, edges, directed=True)
    state.update(threshold={"numerator": 1, "denominator": 2}, seed_count=2,
                 weights=[{"source": u, "target": v, "numerator": 1,
                           "denominator": sum(b == v for _, b in edges)}
                          for u, v in state["edges"]])
    return state


def tsp_graph(points):
    edges = [(u, v, abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1]))
             for u, v in combinations(range(len(points)), 2)]
    return {**graph(len(points), edges), "points": points, "metric": "Manhattan"}


def decisions_for(instance, solution):
    if instance["task"] == "tsp_construct":
        return [str(v) for v in solution[1:-2]]
    if instance["task"] == "lt_influence_construct":
        return [str(v) for v in solution]
    return solution[1:]


def matrix_modularity(state, sides):
    """Independent adjacency-matrix definition, with exact rational arithmetic."""
    m = len(state["edges"])
    adjacency = {tuple(edge) for edge in state["edges"]}
    degrees = [sum(v in edge for edge in state["edges"]) for v in state["nodes"]]
    return sum((Fraction(int(tuple(sorted((u, v))) in adjacency)) -
                Fraction(degrees[u] * degrees[v], 2 * m)
                for u in state["nodes"] for v in state["nodes"] if sides[u] == sides[v]),
               Fraction(0)) / (2 * m)


class ExtendedTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = tasks.build_instances()
        cls.by_task = {name: [i for i in cls.instances if i["task"] == name]
                       for name in {i["task"] for i in cls.instances}}

    def test_matrix_shape_and_budgets(self):
        expected = {name: 16 if name in tasks._EXACT else 8 for name in self.by_task}
        self.assertEqual(Counter(i["task"] for i in self.instances), expected)
        self.assertEqual(len(self.instances), 80)
        self.assertEqual(len({i["id"] for i in self.instances}), 80)
        self.assertEqual(sum(i["query_budget"] for i in self.instances), 256)
        self.assertEqual(7 * sum(i["query_budget"] for i in self.instances), 1792)
        budgets = dict(degree_exact=1, cycle_detection=1, pair_connectivity=1,
                       maxcut_construct=9, tsp_construct=6, lt_influence_construct=2,
                       community_bipartition_construct=9)
        for instance in self.instances:
            self.assertEqual(set(instance), {"id", "task", "kind", "state", "query_budget", "private"})
            self.assertEqual(instance["query_budget"], budgets[instance["task"]])
            self.assertEqual(instance["kind"], "exact" if instance["task"] in tasks._EXACT
                             else "optimization")
            state = instance["state"]
            n = len(state["nodes"])
            self.assertEqual(state["nodes"], list(range(n)))
            self.assertEqual(len(state["edges"]), len({tuple(e[:2]) for e in state["edges"]}))
            self.assertTrue(all(e[0] != e[1] for e in state["edges"]))
            if instance["kind"] == "exact":
                self.assertIn(n, (8, 12))
        for task in ("cycle_detection", "pair_connectivity"):
            self.assertEqual(Counter(i["private"]["answer"] for i in self.by_task[task]),
                             {"yes": 8, "no": 8})
        degrees = {i["private"]["answer"] for i in self.by_task["degree_exact"]}
        self.assertGreaterEqual(len(degrees), 10)
        self.assertTrue({0, 7, 11} <= degrees)
        connected = []
        for instance in self.by_task["pair_connectivity"]:
            g = nx.Graph()
            g.add_nodes_from(instance["state"]["nodes"])
            g.add_edges_from(instance["state"]["edges"])
            connected.append(nx.is_connected(g))
        self.assertTrue(any(connected))
        self.assertFalse(all(connected))

    def test_reproducible_json_and_no_global_rng_or_network(self):
        before = random.getstate()
        with patch("socket.create_connection", side_effect=AssertionError("offline")), \
                patch("socket.socket.connect", side_effect=AssertionError("offline")):
            rebuilt = tasks.build_instances(20260926)
            different = tasks.build_instances(20260927)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(rebuilt, self.instances)
        self.assertNotEqual(different, self.instances)
        self.assertEqual(json.loads(json.dumps(rebuilt, allow_nan=False)), rebuilt)
        for seed in (True, "20260926", None, 1.5):
            with self.assertRaises(ValueError):
                tasks.build_instances(seed)

    def test_independent_validation_and_totals(self):
        # Validation must not reuse the production optimum search/DP.
        with patch.object(tasks, "_optimize", side_effect=AssertionError("not independent")), \
                patch.object(tasks, "_held_karp", side_effect=AssertionError("not independent")), \
                patch("socket.socket.connect", side_effect=AssertionError("offline")):
            totals = tasks.validate_instances(self.instances)
        self.assertEqual(totals["instances"], 80)
        self.assertEqual(totals["queries_per_model"], 256)
        self.assertEqual(totals["by_task"], Counter(i["task"] for i in self.instances))
        json.dumps(totals, allow_nan=False)
        self.assertEqual(tasks.validate_instances([])["queries_per_model"], 0)

    def test_request_templates_full_graph_options_and_no_private_dependency(self):
        questions = {}
        for original in self.instances:
            instance = deepcopy(original)
            task = instance["task"]
            if instance["kind"] == "exact":
                complete = [str(instance["private"]["answer"])]
            else:
                complete = decisions_for(instance, instance["private"]["reference"])
            # Deliberately non-JSON private data would break naive record copying.
            instance["private"] = {"oracle": object(), "secret": "PRIVATE_SENTINEL"}
            instance["state"]["private"] = "INJECTED_STATE_SENTINEL"
            for step in range(len(complete)):
                request = tasks.next_request(instance, complete[:step])
                self.assertIsInstance(request, DecisionRequest)
                serialized = json.dumps(asdict(request), allow_nan=False)
                self.assertNotIn("PRIVATE_SENTINEL", serialized)
                self.assertNotIn("INJECTED_STATE_SENTINEL", serialized)
                self.assertNotIn("baseline", serialized)
                self.assertNotIn("reference", serialized)
                self.assertNotIn("oracle", serialized)
                self.assertNotIn("private", request.state)
                self.assertEqual(request.state["edges"], original["state"]["edges"])
                self.assertEqual(request.state["nodes"], original["state"]["nodes"])
                self.assertEqual(request.state["remaining_queries"], len(complete) - step)
                self.assertGreaterEqual(len(request.options), 2)
                self.assertLessEqual(len(request.options), 255)
                ids = [o.id for o in request.options]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertIn(complete[step], ids)
                self.assertEqual(request.question, questions.setdefault(task, request.question))
                expected_keys = tasks._STATE_KEYS[task] | {"remaining_queries"}
                if task in ("maxcut_construct", "community_bipartition_construct"):
                    expected_keys |= {"partial_partition", "current_vertex"}
                    self.assertEqual(request.state["partial_partition"], ["A"] + complete[:step])
                    self.assertEqual(request.state["current_vertex"], step + 1)
                elif task == "tsp_construct":
                    expected_keys |= {"partial_tour", "current_vertex", "unvisited"}
                    self.assertEqual(request.state["partial_tour"], [0] + list(map(int, complete[:step])))
                    self.assertEqual(list(map(int, ids)), request.state["unvisited"])
                elif task == "lt_influence_construct":
                    expected_keys |= {"selected_seeds", "remaining_seeds"}
                    self.assertEqual(request.state["selected_seeds"], list(map(int, complete[:step])))
                    self.assertEqual(request.state["remaining_seeds"], 2 - step)
                elif task == "degree_exact":
                    self.assertEqual(ids, list(map(str, range(len(original["state"]["nodes"])))))
                self.assertEqual(set(request.state), expected_keys)
                request.state["edges"].clear()
                self.assertEqual(instance["state"]["edges"], original["state"]["edges"])
            self.assertIsNone(tasks.next_request(instance, complete))
        self.assertEqual(len(questions), 7)
        self.assertIn("AT MOST TWO", questions["community_bipartition_construct"])
        self.assertIn("indegree-zero", questions["lt_influence_construct"])

    def test_exact_and_optimization_metrics_are_distinct(self):
        for instance in self.instances:
            if instance["kind"] == "exact":
                answer = str(instance["private"]["answer"])
                result = tasks.score(instance, [answer])
                self.assertEqual(result, {"feasible": True, "correct": True, "status": "complete"})
                wrong = next(o.id for o in tasks.next_request(instance, []).options if o.id != answer)
                self.assertEqual(tasks.score(instance, [wrong]),
                                 {"feasible": True, "correct": False, "status": "complete"})
                self.assertNotIn("objective", result)
            else:
                for label in ("reference", "baseline"):
                    solution = (instance["private"]["reference"] if label == "reference"
                                else instance["private"]["baseline"]["solution"])
                    result = tasks.score(instance, decisions_for(instance, solution))
                    self.assertNotIn("correct", result)
                    self.assertTrue(result["feasible"])
                    self.assertEqual(result["solution"], solution)
                    self.assertGreaterEqual(result["normalized_quality"], 0)
                    self.assertLessEqual(result["normalized_quality"], 1)
                    self.assertEqual(result["absolute_gap"], abs(result["optimum"] - result["objective"]))
                    self.assertEqual(result["normalized_quality"], 1 / (1 + result["absolute_gap"]))
                    if result["direction"] == "maximize":
                        self.assertLessEqual(result["objective"], result["optimum"] + 1e-12)
                    else:
                        self.assertEqual(instance["task"], "tsp_construct")
                        self.assertGreaterEqual(result["objective"], result["optimum"])
                    if label == "reference":
                        self.assertTrue(result["optimal"])
                        self.assertEqual(result["normalized_quality"], 1)
                    else:
                        self.assertEqual(result["objective"], instance["private"]["baseline"]["objective"])
                    json.dumps(result, allow_nan=False)

    def test_invalid_and_incomplete_histories_never_repaired(self):
        for instance in self.instances:
            complete = ([str(instance["private"]["answer"])] if instance["kind"] == "exact" else
                        decisions_for(instance, instance["private"]["reference"]))
            for length in range(len(complete)):
                result = tasks.score(instance, complete[:length])
                self.assertFalse(result["feasible"])
                self.assertEqual(result["status"], "incomplete")
                if instance["kind"] == "optimization":
                    for key in ("objective", "normalized_quality", "absolute_gap", "solution"):
                        self.assertIsNone(result[key])
                    self.assertFalse(result["optimal"])
                else:
                    self.assertFalse(result["correct"])
                self.assertIsNotNone(tasks.next_request(instance, complete[:length]))
            invalid = [None, (), "A", [True], [1], ["-1"], ["999"], ["01"], [" A"],
                       ["yes."], ["YES"], ["a"], ["unknown"], complete + [complete[-1]],
                       complete[:-1] + ["unknown"]]
            if instance["task"] in ("lt_influence_construct", "tsp_construct"):
                invalid.append(["1", "1"])
            if instance["task"] == "tsp_construct":
                invalid.append(["0"])
            for history in invalid:
                with self.subTest(task=instance["task"], history=history):
                    with self.assertRaises(ValueError):
                        tasks.next_request(instance, history)
                    result = tasks.score(instance, history)
                    self.assertFalse(result["feasible"])
                    self.assertEqual(result["status"], "invalid")
                    if instance["kind"] == "optimization":
                        self.assertIsNone(result["objective"])
                        self.assertIsNone(result["normalized_quality"])
                        self.assertIsNone(result["solution"])
                    json.dumps(result, allow_nan=False)

    def test_tsp_forced_last_vertex_and_reverse_optimum(self):
        for instance in self.by_task["tsp_construct"]:
            tour = instance["private"]["reference"]
            reverse = list(reversed(tour))
            self.assertNotEqual(tour, reverse)
            for candidate in (tour, reverse):
                history = decisions_for(instance, candidate)
                self.assertEqual(len(history), 6)
                final_request = tasks.next_request(instance, history[:-1])
                self.assertEqual(len(final_request.options), 2)
                self.assertIsNone(tasks.next_request(instance, history))
                result = tasks.score(instance, history)
                self.assertTrue(result["optimal"])
                self.assertEqual(result["solution"], candidate)
            state = instance["state"]
            distances = tasks._distances(state)
            self.assertEqual(len(state["edges"]), 28)
            self.assertEqual(len({tuple(p) for p in state["points"]}), 8)
            for u, v, w in product(range(8), repeat=3):
                self.assertLessEqual(distances[u][w], distances[u][v] + distances[v][w])

    def test_multiple_optimum_partitions_and_seed_orders_accepted(self):
        cut = tasks._record("maxcut_construct", 0, graph(10, [(0, 1)]))
        first, other = ["B"] + ["A"] * 8, ["B"] * 9
        self.assertNotEqual(first, other)
        self.assertTrue(tasks.score(cut, first)["optimal"])
        self.assertTrue(tasks.score(cut, other)["optimal"])
        modularity = tasks._record("community_bipartition_construct", 0,
                                  graph(10, [(0, 1), (2, 3)]))
        optimum = decisions_for(modularity, modularity["private"]["reference"])
        alternate = optimum.copy()
        alternate[-1] = "B" if optimum[-1] == "A" else "A"  # isolated node
        self.assertTrue(tasks.score(modularity, optimum)["optimal"])
        self.assertTrue(tasks.score(modularity, alternate)["optimal"])
        for instance in self.by_task["lt_influence_construct"]:
            seeds = instance["private"]["reference"]
            self.assertTrue(tasks.score(instance, list(map(str, seeds[::-1])))["optimal"])
        isolated_lt = tasks._record("lt_influence_construct", 0, lt_graph(8, []))
        self.assertTrue(tasks.score(isolated_lt, ["6", "7"])["optimal"])
        self.assertEqual(tasks.score(isolated_lt, ["6", "7"])["objective"], 2)

    def test_standard_modularity_all_one_and_negative_quality(self):
        instance = tasks._record("community_bipartition_construct", 0,
                                 graph(10, list(combinations(range(10), 2))))
        self.assertEqual(instance["private"]["objective"], 0)
        self.assertTrue(tasks.score(instance, ["A"] * 9)["optimal"])
        result = tasks.score(instance, ["B"] * 9)
        self.assertLess(result["objective"], 0)
        self.assertEqual(result["objective"], float(matrix_modularity(instance["state"], ["A"] + ["B"] * 9)))
        self.assertGreater(result["normalized_quality"], 0)
        self.assertLess(result["normalized_quality"], 1)
        self.assertIn("additive", result["quality_definition"])
        self.assertEqual(tasks.validate_instances([instance])["instances"], 1)

    def test_exhaust_all_four_node_undirected_motifs(self):
        possible = list(combinations(range(4), 2))
        for mask in range(1 << len(possible)):
            edges = [edge for bit, edge in enumerate(possible) if mask & (1 << bit)]
            state = graph(4, edges)
            g = nx.Graph()
            g.add_nodes_from(range(4))
            g.add_edges_from(edges)
            self.assertEqual(tasks._exact_answer("cycle_detection", state),
                             "yes" if nx.cycle_basis(g) else "no")
            for v in range(4):
                self.assertEqual(tasks._exact_answer("degree_exact", {**state, "vertex": v}), g.degree[v])
            for u, v in possible:
                self.assertEqual(tasks._exact_answer("pair_connectivity", {**state, "pair": [u, v]}),
                                 "yes" if nx.has_path(g, u, v) else "no")
            cuts, modularities = [], []
            for tail in product(("A", "B"), repeat=3):
                sides = ["A", *tail]
                groups = [{v for v in range(4) if sides[v] == c} for c in ("A", "B")]
                cut = nx.cut_size(g, *groups)
                cuts.append(cut)
                self.assertEqual(tasks._objective("maxcut_construct", state, sides), cut)
                if edges:
                    value = matrix_modularity(state, sides)
                    modularities.append(value)
                    self.assertEqual(tasks._objective("community_bipartition_construct", state, sides), value)
            self.assertEqual(tasks._optimize("maxcut_construct", state)[0], max(cuts))
            if edges:
                self.assertEqual(tasks._optimize("community_bipartition_construct", state)[0], max(modularities))
        # In particular, one undirected edge is acyclic; a triangle is not.
        self.assertEqual(tasks._exact_answer("cycle_detection", graph(4, [(0, 1)])), "no")
        self.assertEqual(tasks._exact_answer("cycle_detection", graph(4, [(0, 1), (0, 2), (1, 2)])), "yes")

    def test_exhaust_all_three_node_directed_lt_motifs(self):
        possible = [(u, v) for u in range(3) for v in range(3) if u != v]
        for mask in range(1 << len(possible)):
            edges = [edge for bit, edge in enumerate(possible) if mask & (1 << bit)]
            state = lt_graph(3, edges)
            values = []
            for size in range(4):
                for seeds in combinations(range(3), size):
                    rational = tasks._independent_lt(state, seeds)
                    self.assertEqual(tasks._lt_active(state, seeds), rational)
                    self.assertTrue(set(seeds) <= rational)
                    if size == 2:
                        values.append(len(rational))
            self.assertEqual(tasks._optimize("lt_influence_construct", state)[0], max(values))

    def test_lt_threshold_equality_cascades_and_zero_indegree(self):
        state = lt_graph(6, [(0, 2), (1, 2), (2, 3), (3, 4)])
        self.assertEqual(tasks._lt_active(state, []), set())
        self.assertEqual(tasks._lt_active(state, [0]), {0, 2, 3, 4})
        self.assertEqual(tasks._lt_active(state, [0, 5]), {0, 2, 3, 4, 5})
        self.assertNotIn(1, tasks._lt_active(state, [0]))
        odd = lt_graph(4, [(0, 3), (1, 3), (2, 3)])
        self.assertEqual(tasks._lt_active(odd, [0]), {0})
        self.assertEqual(tasks._lt_active(odd, [0, 1]), {0, 1, 3})
        cycle = lt_graph(3, [(0, 1), (1, 2), (2, 0)])
        self.assertEqual(tasks._lt_active(cycle, []), set())
        self.assertEqual(tasks._lt_active(cycle, [0]), {0, 1, 2})

    def test_held_karp_matches_bruteforce_small_metrics(self):
        points = [[0, 0], [1, 0], [1, 1], [0, 1], [4, 2], [3, 5]]
        for n in range(3, 7):
            state = tsp_graph(points[:n])
            distances = tasks._distances(state)
            candidates = [[0, *tail, 0] for tail in permutations(range(1, n))]
            expected = min((sum(distances[u][v] for u, v in zip(tour, tour[1:])), tour)
                           for tour in candidates)
            self.assertEqual(tasks._held_karp(state), expected)
        square = tasks._record("tsp_construct", 0, tsp_graph(points[:4]))
        self.assertEqual(square["private"]["objective"], 4)
        self.assertEqual(tasks.score(square, ["1", "2"])["solution"], [0, 1, 2, 3, 0])
        self.assertFalse(tasks.score(square, ["1"])["feasible"])
        self.assertFalse(tasks.score(square, ["1", "1"])["feasible"])

    def test_corrupt_records_are_rejected(self):
        exact = self.by_task["degree_exact"][0]
        for key, value in (("query_budget", 2), ("query_budget", True), ("kind", "optimization"),
                           ("task", "unknown"), ("id", "")):
            corrupt = deepcopy(exact)
            corrupt[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                tasks.validate_instances([corrupt])
        for key, value in (("answer", -1), ("answer", True), ("answer", float("nan"))):
            corrupt = deepcopy(exact)
            corrupt["private"][key] = value
            with self.assertRaises(ValueError):
                tasks.validate_instances([corrupt])
        for edges in ([[0, 0]], [[0, 1], [0, 1]], [[1, 0]], [[0, 99]], [[0, True]], [[0, 1, 2]]):
            corrupt = deepcopy(exact)
            corrupt["state"]["edges"] = edges
            with self.assertRaises(ValueError):
                tasks.validate_instances([corrupt])
        corrupt = deepcopy(exact)
        corrupt["state"]["answer"] = "leak"
        with self.assertRaises(ValueError):
            tasks.validate_instances([corrupt])
        with self.assertRaises(ValueError):
            tasks.validate_instances([exact, exact])
        with self.assertRaises(ValueError):
            tasks.validate_instances(tuple(self.instances))
        for task in tasks._OPTIMIZATION:
            original = self.by_task[task][0]
            for field in ("objective", "reference", "baseline-objective", "baseline-solution", "baseline-method"):
                corrupt = deepcopy(original)
                if field == "objective":
                    corrupt["private"][field] += 1
                elif field == "reference":
                    corrupt["private"][field] = []
                elif field == "baseline-objective":
                    corrupt["private"]["baseline"]["objective"] += 1
                elif field == "baseline-solution":
                    corrupt["private"]["baseline"]["solution"] = []
                else:
                    corrupt["private"]["baseline"]["method"] = "fake"
                with self.subTest(task=task, field=field), self.assertRaises(ValueError):
                    tasks.validate_instances([corrupt])
        lt = deepcopy(self.by_task["lt_influence_construct"][0])
        lt["state"]["weights"][0]["denominator"] += 1
        with self.assertRaises(ValueError):
            tasks.validate_instances([lt])
        tsp = deepcopy(self.by_task["tsp_construct"][0])
        tsp["state"]["edges"][0][2] += 1
        with self.assertRaises(ValueError):
            tasks.validate_instances([tsp])

    def test_independent_validation_detects_bad_production_evaluators(self):
        for task in tasks._OPTIMIZATION:
            with patch.object(tasks, "_objective", return_value=-999), self.assertRaises(ValueError):
                tasks.validate_instances([self.by_task[task][0]])
        with patch.object(tasks, "_lt_active", return_value=set(range(8))), self.assertRaises(ValueError):
            tasks.validate_instances([tasks._record("lt_influence_construct", 0, lt_graph(8, []))])


if __name__ == "__main__":
    unittest.main()