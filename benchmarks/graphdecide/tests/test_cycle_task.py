"""Offline cycle task invariants; does not acquire live model lane locks."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
from scripts import cycle_task as cycle
from src.clients.base import DecisionRequest, DecisionOption
from src.clients._llm_choice import prompt_for


def source(graph):
    return {"graph_id": "fixture", "dataset": "facebook", "node_count": len(graph),
            "topology": {"nodes": list(graph), "edges": [list(e) for e in graph.edges()]}}


class CycleTaskTests(unittest.TestCase):
    def test_dfs_matches_cycle_basis(self):
        for n in range(1, 25):
            for p in (0, 0.05, 0.2, 0.8):
                g = nx.gnp_random_graph(n, p, seed=123)
                self.assertEqual(cycle.has_cycle(source(g)["topology"]), bool(nx.cycle_basis(g)))

    def test_single_edge_is_not_cycle_and_disconnected_cycle_is(self):
        self.assertFalse(cycle.has_cycle({"nodes": [0, 1], "edges": [[0, 1]]}))
        self.assertTrue(cycle.has_cycle({"nodes": list(range(5)), "edges": [[0, 1], [2, 3], [3, 4], [4, 2]]}))

    def test_rejects_duplicate_and_selfloop(self):
        for edges in ([[0, 1], [1, 0]], [[0, 0]]):
            with self.assertRaises(ValueError):
                cycle.has_cycle({"nodes": [0, 1], "edges": edges})

    def test_pair_matches_counts_and_edge_swap(self):
        g = nx.complete_graph(16)
        pair = cycle.build_pair(source(g))
        self.assertEqual(pair, cycle.build_pair(source(g)))
        self.assertEqual([r["label"]["answer"] for r in pair], ["no", "yes"])
        edge_sets = []
        for row in pair:
            state = row["request"]["state"]
            self.assertEqual(len(state["nodes"]), 16)
            self.assertEqual(len(state["edges"]), 14)
            self.assertEqual(cycle.has_cycle(state), row["label"]["answer"] == "yes")
            edge_sets.append({frozenset(e) for e in state["edges"]})
            r = row["request"]
            request = DecisionRequest(**{**r, "options": tuple(DecisionOption(**o) for o in r["options"])})
            visible = json.loads(prompt_for(request))
            self.assertEqual(set(visible), {"state", "question", "options"})
            self.assertEqual(set(visible["state"]), {"nodes", "edges"})
        self.assertEqual(len(edge_sets[0] ^ edge_sets[1]), 2)
        self.assertEqual([r["label"]["components"] for r in pair], [2, 3])

    def test_acyclic_source_not_silently_replaced(self):
        self.assertIsNone(cycle.build_pair(source(nx.path_graph(32))))

    def test_offline_plan_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            prior = Path(tmp) / "graphs.jsonl"
            prior.write_text(json.dumps(source(nx.complete_graph(32))) + "\n")
            root = Path(tmp) / "plan"
            with patch("httpx.AsyncClient", side_effect=AssertionError("offline")):
                manifest = cycle.plan(root, prior)
            self.assertEqual(manifest["questions_per_model"], 2)
            self.assertEqual(manifest["seven_model_call_budget"], 14)
            self.assertEqual(manifest["status"], "offline-prepared-not-evaluated")
            for name, digest in manifest["artifact_sha256"].items():
                self.assertEqual(cycle.digest(root / name), digest)
            with self.assertRaises(FileExistsError):
                cycle.plan(root, prior)


if __name__ == "__main__":
    unittest.main()