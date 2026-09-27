from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import json
import unittest

import networkx as nx

from src.benchmark import extended_tasks as tasks


class FoundationalTests(unittest.TestCase):
    def test_exhaust_all_five_node_motifs(self):
        edges = list(combinations(range(5), 2))
        for mask in range(1 << len(edges)):
            state = tasks._graph(5, [e for i, e in enumerate(edges) if mask & (1 << i)])
            graph = nx.Graph()
            graph.add_nodes_from(range(5))
            graph.add_edges_from(state["edges"])
            for vertex in range(5):
                expected = "yes" if vertex in set(nx.articulation_points(graph)) else "no"
                self.assertEqual(tasks._exact_answer("articulation_point", {**state, "vertex": vertex}),
                                 expected)
            for u, v in edges:
                pair = [u, v]
                self.assertEqual(tasks._exact_answer("adjacency", {**state, "pair": pair}),
                                 "yes" if graph.has_edge(u, v) else "no")
                distance = nx.shortest_path_length(graph, u, v) if nx.has_path(graph, u, v) else None
                for threshold in (1, 2, 3, 4):
                    self.assertEqual(tasks._exact_answer("distance_threshold", {
                        **state, "pair": pair, "threshold": threshold}),
                        "yes" if distance is not None and distance <= threshold else "no")

    def test_record_request_validation_scoring_and_private_isolation(self):
        graph = tasks._graph(6, [(0, 1), (1, 2), (2, 3), (4, 5)])
        for task, extra, expected in (
            ("adjacency", {"pair": [0, 2]}, "no"),
            ("distance_threshold", {"pair": [0, 3], "threshold": 3}, "yes"),
            ("distance_threshold", {"pair": [0, 4], "threshold": 5}, "no"),
            ("articulation_point", {"vertex": 1}, "yes"),
            ("articulation_point", {"vertex": 4}, "no"),
        ):
            record = tasks._record(task, 0, {**graph, **extra})
            self.assertEqual(record["private"]["answer"], expected)
            self.assertEqual(tasks.validate_instances([record])["queries_per_model"], 1)
            self.assertTrue(tasks.score(record, [expected])["correct"])
            self.assertFalse(tasks.score(record, [])["correct"])
            self.assertIsNone(tasks.next_request(record, [expected]))
            public = deepcopy(record)
            public["private"] = {"sentinel": object()}
            request = tasks.next_request(public, [])
            self.assertNotIn("private", json.dumps(asdict(request)))
            self.assertEqual([o.id for o in request.options], ["yes", "no"])
        bad = tasks._record("distance_threshold", 1, {**graph, "pair": [0, 2], "threshold": 2})
        for value in (0, 6, True, 1.5):
            bad["state"]["threshold"] = value
            with self.assertRaisesRegex(ValueError, "threshold"):
                tasks.validate_instances([bad])
