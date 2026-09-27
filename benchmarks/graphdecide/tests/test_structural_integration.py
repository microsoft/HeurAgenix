import asyncio
from copy import deepcopy
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import extended_graph_suite as suite
from scripts.summarize_extended_paper import export, paired_intervals
from src.benchmark import extended_tasks as tasks
from test_extended_graph_suite import FakeClient


def fixture():
    graph = tasks._graph(8, [(0, 1), (1, 2), (3, 4), (4, 5), (3, 5)])
    rows = []
    for task in tasks._ALL_EXACT:
        state = deepcopy(graph)
        if task in ("degree_exact", "articulation_point"):
            state["vertex"] = 1
        elif task in ("adjacency", "pair_connectivity", "distance_threshold"):
            state["pair"] = [0, 2]
        if task == "distance_threshold":
            state["threshold"] = 2
        rows.append(tasks._record(task, 0, state))
    for task in tasks._OPTIMIZATION:
        state = deepcopy(graph)
        if task == "tsp_construct":
            points = [[v, v % 3] for v in range(8)]
            edges = [(u, v, sum(abs(a - b) for a, b in zip(points[u], points[v])))
                     for u, v in combinations(range(8), 2)]
            state = {**tasks._graph(8, edges), "points": points, "metric": "Manhattan"}
        elif task == "lt_influence_construct":
            state = tasks._graph(8, [(v, v + 1) for v in range(7)], directed=True)
            state.update(seed_count=2, threshold={"numerator": 1, "denominator": 2},
                         weights=[{"source": u, "target": v, "numerator": 1, "denominator": 1}
                                  for u, v in state["edges"]])
        rows.append(tasks._record(task, 0, state))
    metadata = {"cases": {i["id"]: {"cluster_id": i["task"], "family": "offline-fixture"}
                          for i in rows}, "coverage": []}
    return rows, metadata


class StructuralIntegrationTests(unittest.TestCase):
    def test_prior_pool_and_metadata_are_frozen_and_portable(self):
        instances, metadata = fixture()
        calls = []

        def build(**kwargs):
            calls.append(kwargs)
            return deepcopy(instances), deepcopy(metadata)

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            pool = workspace / "pool.jsonl"
            pool.write_text(json.dumps({"id": "prior", "task": "degree_exact",
                                       "state": tasks._graph(8, [])}) + "\n")
            root = workspace / "run"
            with patch.dict("sys.modules", {"src.benchmark.structural_tasks": SimpleNamespace(
                    build_structural_bank=build)}):
                result = suite.plan(root, design="structural_holdout", prior_pool=pool,
                                    node_counts=(8,), models=("jev",))
            self.assertEqual(result["budgets"]["instances_per_model"], 10)
            self.assertEqual(len(calls[0]["prior_instances"]), 1)
            pool.unlink()
            _, manifest, loaded, _ = suite.load_plan(root)
            self.assertTrue(manifest["environment"]["packages"]["networkx"])
            self.assertEqual(datetime.fromisoformat(manifest["created_at_utc"]).tzinfo,
                             timezone.utc)
            self.assertEqual(set(manifest["input_sha256"]),
                             {"instances.jsonl", "design_metadata.json", "prior_graph_pool.jsonl"})
            self.assertEqual(loaded, instances)
            (root / "design_metadata.json").write_text("{}")
            with self.assertRaisesRegex(suite.Refusal, "input hash"):
                suite.load_plan(root)

    def test_new_task_replay_export_and_explicit_clustering(self):
        instances, metadata = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with patch.dict("sys.modules", {"src.benchmark.structural_tasks": SimpleNamespace(
                    build_structural_bank=lambda **_: (instances, metadata))}):
                suite.plan(root, design="structural_dev", node_counts=(8,), models=("jev",))

            async def preflight(_):
                return None

            with patch.object(suite.shared.tempfile, "gettempdir", return_value=directory):
                result = asyncio.run(suite.run_live(
                    root, client_factory=lambda _, **kw: FakeClient(kw["model"]), preflight=preflight))
            self.assertEqual(result["models"]["jev"]["status"], "completed")
            self.assertEqual(len(result["models"]["jev"]["by_task"]), 10)
            status = suite.read_json(root / "runs/jev/status.json")
            self.assertGreaterEqual(datetime.fromisoformat(status["finished_at_utc"]),
                                    datetime.fromisoformat(status["started_at_utc"]))
            self.assertGreater(status["wall_seconds"], 0)
            scores = suite.read_rows(root / "runs/jev/scores.jsonl")
            comparisons = paired_intervals(
                instances, scores, scores, repetitions=10,
                clusters={key: value["cluster_id"] for key, value in metadata["cases"].items()})
            self.assertEqual(comparisons["articulation_point"]["metric"], "macro-size accuracy difference")
            self.assertEqual(comparisons["adjacency"]["generation_clusters"], 1)
            export(root, root / "tables", ("jev",))
            self.assertTrue((root / "tables/size-rates.svg").exists())
            with (root / "tables/family-metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)
            with (root / "tables/degree-class-recall.csv").open() as stream:
                recalls = list(csv.DictReader(stream))
                self.assertEqual(len(recalls), 1)
                self.assertEqual(int(recalls[0]["instances"]), 1)
            with (root / "tables/optimization-quality.csv").open() as stream:
                quality = {row["task"]: row for row in csv.DictReader(stream)}
            self.assertEqual(len(quality), 4)
            self.assertEqual(quality["community_bipartition_construct"]["relative_gap"], "")
            tsp = quality["tsp_construct"]
            self.assertAlmostEqual(float(tsp["relative_gap"]),
                                   float(tsp["absolute_gap"]) / float(tsp["optimum"]))

    def test_required_prior_and_size_scope_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            for kwargs in ({"design": "structural_holdout"},
                           {"design": "structural_dev", "prior_pool": Path("unused")},
                           {"design": "structural_holdout", "prior_pool": Path("unused"),
                            "node_counts": (5,)}):
                with self.assertRaises(suite.Refusal):
                    suite.plan(root, models=("jev",), **kwargs)
                self.assertFalse(root.exists())
