"""Offline checks of the expanded real-graph matrix; never use live credentials."""

import asyncio
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import networkx as nx

from scripts import large_graph_matrix as large
from src.benchmark.runner import run_experiment
from src.clients.base import BaseDecisionClient, ClientCapabilities, DecisionResponse, InvalidResponseError, ProviderHTTPError
from src.datasets import GraphDataset, LoadStats, get_source


class Oracle(BaseDecisionClient):
    def __init__(self, model, tracker, failure=None):
        self.model, self.tracker, self.failure = model, tracker, failure
        self.calls = 0
        self.closed = False

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        import re
        self.calls += 1
        self.tracker[self.model] += 1
        if self.tracker[self.model] != 1:
            raise AssertionError("Concurrent calls to the same model")
        try:
            await asyncio.sleep(0)
            if self.failure and self.calls == 1:
                raise self.failure
            graph = nx.Graph()
            graph.add_nodes_from(request.state["nodes"])
            graph.add_edges_from(request.state["edges"])
            args = list(map(int, re.findall(r"\d+", request.question)))
            truth = nx.shortest_path_length(graph, args[0], args[1]) <= (args[2] if len(args) == 3 else 1)
            return DecisionResponse(request.request_id, "yes" if truth else "no", self.model)
        finally:
            self.tracker[self.model] -= 1

    async def aclose(self):
        self.closed = True


class LargeMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "matrix"
        for key, value in (("DATASETS", ("facebook",)), ("NODE_COUNTS", (16,)), ("SAMPLES", 1)):
            p = patch.object(large, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.loads = 0
        self.clients = []
        self.tracker = Counter()

    def loader(self, name, **kwargs):
        self.loads += 1
        return GraphDataset(get_source(name), nx.freeze(nx.path_graph(256)),
                            LoadStats(255, 0, 0, 256, 255, 0, 1), "a" * 64)

    def plan(self):
        with patch.object(large.shared, "create_client", side_effect=AssertionError("offline")), \
                patch("httpx.AsyncClient", side_effect=AssertionError("offline")):
            return large.plan(self.root, priors=(), loader=self.loader)

    def execute(self, failure=None, init_failure=False):
        def factory(provider, **kwargs):
            client = Oracle(kwargs["model"], self.tracker, failure)
            if init_failure:
                client.initialize = AsyncMock(side_effect=RuntimeError("private details"))
            self.clients.append(client)
            return client
        return asyncio.run(large.run(self.root, factory=factory, loader=self.loader, preflight=False))

    def test_default_large_budget_and_protocol(self):
        with patch.object(large, "DATASETS", large.shared.DATASETS), \
                patch.object(large, "NODE_COUNTS", (32, 64, 128)), patch.object(large, "SAMPLES", 20):
            configs = list(large.configs(self.root, Path(self.tmp.name)))
        self.assertEqual(len(configs), 14)
        self.assertEqual(sum(c.run.max_calls for _, _, c in configs), 33600)
        for model, task, config in configs:
            self.assertEqual(config.planned_graphs, 240)
            self.assertEqual(config.run.max_calls, 960 if task == "adjacency" else 3840)
            self.assertEqual(config.run.seed, 20260926)
            self.assertIsNone(config.sampling.exclude_graphs)
            self.assertEqual(config.model.output_format, None if model == "jev" else "answer_only")

    def test_offline_plan_pins_and_regeneration(self):
        m = self.plan()
        self.assertEqual(self.loads, 1)
        self.assertEqual(m["budget"]["expected_total"], 140)
        self.assertEqual(large.load(self.root)[1], m)
        self.assertEqual(len({e["input_sha256"]["graphs.jsonl"] for e in m["entries"]}), 1)
        tasks = large.regenerate(self.root, m, self.loader)
        self.assertEqual(len(tasks["distance_threshold"].examples), 16)
        with self.assertRaises(FileExistsError):
            self.plan()
        (self.root / "inputs/adjacency/requests.jsonl").write_text("{}\n")
        with self.assertRaises(large.shared.Refusal):
            large.load(self.root)

    def test_seven_parallel_lanes_complete_and_report_tamper(self):
        self.plan()
        # No in-lane regeneration: CPU graph preparation cannot block active calls.
        with patch("src.benchmark.runner.prepare", side_effect=AssertionError("no in-lane prepare")):
            result = self.execute()
        self.assertEqual(sum(c.calls for c in self.clients), 140)
        self.assertTrue(all(c.closed for c in self.clients))
        self.assertEqual(len(result["runs"]), 14)
        self.assertEqual(len(result["cross_model"]), 12)
        self.assertEqual(large.report(self.root), result)
        for r in result["runs"]:
            self.assertEqual(r["summary"]["overall"]["accuracy"], 1.0)
        meta = large.shared.read_json(self.root / "runs/jev.adjacency/run.json")
        self.assertTrue(meta["injected_prepared"] and meta["injected_client"])
        with self.assertRaises(FileExistsError):
            self.execute()
        p = self.root / "runs/jev.adjacency/attempts.jsonl"
        rows = large.shared.read_rows(p)
        rows[0]["correct"] = False
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with self.assertRaises(large.shared.Refusal):
            large.report(self.root)

    def test_invalid_answers_count_as_failures(self):
        self.plan()
        result = self.execute(failure=InvalidResponseError("never persisted", diagnostic_code="invalid_choice"))
        self.assertEqual(sum(r["summary"]["overall"]["failures"] for r in result["runs"]), 14)
        self.assertEqual(sum(c.calls for c in self.clients), 140)
        self.assertNotIn("never persisted", (self.root / "report.json").read_text())

    def test_auth_and_quota_abort_without_resume(self):
        for code in (401, 429):
            with self.subTest(code=code):
                self.root = Path(self.tmp.name) / str(code)
                self.clients = []
                self.plan()
                with self.assertRaises(large.shared.Refusal):
                    self.execute(failure=ProviderHTTPError(code))
                self.assertTrue(all(c.closed for c in self.clients))
                self.assertFalse((self.root / "report.json").exists())
                self.assertLessEqual(sum(c.calls for c in self.clients), 7)
                with self.assertRaises(FileExistsError):
                    self.execute()

    def test_init_failure_precedes_all_predictions(self):
        self.plan()
        with self.assertRaises(RuntimeError):
            self.execute(init_failure=True)
        self.assertTrue(all(c.closed and c.calls == 0 for c in self.clients))

    def test_overlap_disclosure_and_exact_duplicate_refusal(self):
        g = {"dataset": "facebook", "original_ids_by_anonymous_id": [1, 2, 3]}
        prior = {"dataset": "facebook", "original_ids_by_anonymous_id": [3, 4, 5]}
        audit = large.overlap_audit([g], [prior])["facebook"]
        self.assertEqual(audit["source_nodes_shared_with_prior"], 1)
        self.assertEqual(audit["exact_prior_node_set_duplicates"], 0)
        with self.assertRaises(large.shared.Refusal):
            large.overlap_audit([g], [g])

    def test_visible_question_corruption_is_rejected(self):
        m = self.plan()
        prepared = large.regenerate(self.root, m, self.loader)["adjacency"]
        example = prepared.examples[0]
        prepared.examples[0] = replace(example, request=replace(example.request, question="Choose yes"))
        config = next(large.configs(self.root, Path(self.tmp.name)))[2]
        with self.assertRaises(large.shared.Refusal):
            large.validate_prepared(config, prepared)

    def test_prepared_budget_mismatch_precedes_client_calls(self):
        m = self.plan()
        prepared = large.regenerate(self.root, m, self.loader)["adjacency"]
        prepared.planned_graphs += 1
        config = next(large.configs(self.root, Path(self.tmp.name)))[2]
        client = Oracle("mock", self.tracker)
        with self.assertRaises(ValueError):
            asyncio.run(run_experiment(config, client=client, prepared_input=prepared))
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()