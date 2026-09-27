"""Offline cycle execution tests; never contend with live benchmark locks."""
import asyncio
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import networkx as nx
from scripts import run_cycle_task as run
from src.clients.base import BaseDecisionClient, ClientCapabilities, DecisionResponse, InvalidResponseError, ProviderHTTPError


class Oracle(BaseDecisionClient):
    def __init__(self, model, active, failure=None):
        self.model, self.active, self.failure = model, active, failure
        self.calls = 0
        self.closed = False

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        self.calls += 1
        self.active[self.model] += 1
        assert self.active[self.model] == 1
        try:
            await asyncio.sleep(0)
            if self.failure and self.calls == 1:
                raise self.failure
            graph = nx.Graph()
            graph.add_nodes_from(request.state["nodes"])
            graph.add_edges_from(request.state["edges"])
            return DecisionResponse(request.request_id, "yes" if nx.cycle_basis(graph) else "no", self.model)
        finally:
            self.active[self.model] -= 1

    async def aclose(self):
        self.closed = True


class CycleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root, self.inputs, self.prior = base / "run", base / "offline", base / "prior"
        graphfile = self.prior / "inputs/adjacency/graphs.jsonl"
        graphfile.parent.mkdir(parents=True)
        graph = nx.complete_graph(16)
        graphfile.write_text(json.dumps({"graph_id": "test", "dataset": "facebook", "node_count": 16,
            "topology": {"nodes": list(graph), "edges": [list(e) for e in graph.edges()]}}) + "\n")
        (self.prior / "manifest.json").write_text("{}")
        run.task.plan(self.inputs, graphfile)
        self.clients = []
        self.active = Counter()
        locker = patch.object(run, "lane_locks", lambda models: nullcontext())
        locker.start()
        self.addCleanup(locker.stop)

    def plan(self):
        with patch.object(run.predecessor, "load", return_value=(self.prior, {})), \
                patch.object(run.shared, "create_client", side_effect=AssertionError("offline")):
            return run.plan(self.root, self.inputs, self.prior)

    def execute(self, models=run.shared.LEGACY_MODELS, failure=None, init_failure=False):
        def factory(provider, **kwargs):
            client = Oracle(kwargs["model"], self.active, failure)
            if init_failure:
                client.initialize = AsyncMock(side_effect=RuntimeError("private error"))
            self.clients.append(client)
            return client
        return asyncio.run(run.run(self.root, models, factory=factory, preflight=False))

    def test_plan_regeneration_and_pins(self):
        manifest = self.plan()
        self.assertEqual(manifest["calls_per_model"], 2)
        self.assertEqual(manifest["max_total_calls"], 14)
        self.assertEqual(run.load(self.root)[1], manifest)
        run.validate_inputs(self.root)
        with self.assertRaises(FileExistsError):
            self.plan()
        (self.root / "labels.jsonl").write_text("{}\n")
        with self.assertRaises(run.shared.Refusal):
            run.load(self.root)

    def test_five_now_two_later_exactly_once(self):
        self.plan()
        result = self.execute()
        self.assertEqual(len(result["models"]), 5)
        self.assertFalse((self.root / "report.json").exists())
        self.assertTrue(all(c.closed for c in self.clients))
        with self.assertRaises(run.shared.Refusal):
            self.execute()
        self.execute(models=("gpt54", "gpt6astra"))
        self.assertEqual(sum(c.calls for c in self.clients), 14)
        final = run.shared.read_json(self.root / "report.json")
        self.assertEqual(final, run.report(self.root, run.shared.MODELS))
        for summary in final["models"].values():
            self.assertEqual(summary["overall"]["accuracy"], 1.0)
            self.assertEqual(summary["pairs"]["both_correct"], 1)

    def test_typed_failure_retained_in_denominator(self):
        self.plan()
        report = self.execute(failure=InvalidResponseError("secret must not persist", diagnostic_code="invalid_choice"))
        for summary in report["models"].values():
            self.assertEqual(summary["overall"]["n"], 2)
            self.assertEqual(summary["overall"]["correct"], 1)
            self.assertEqual(summary["overall"]["failures"], 1)
        rows = (self.root / "runs/jev/attempts.jsonl").read_text()
        self.assertNotIn("secret must not persist", rows)
        self.assertIn("invalid_choice", rows)

    def test_fatal_auth_closes_every_client_and_refuses_resume(self):
        self.plan()
        with self.assertRaises(run.shared.Refusal):
            self.execute(failure=ProviderHTTPError(401))
        self.assertTrue(all(c.closed for c in self.clients))
        self.assertLessEqual(sum(c.calls for c in self.clients), 5)
        self.assertFalse((self.root / "report.json").exists())
        with self.assertRaises(run.shared.Refusal):
            run.report(self.root, run.shared.LEGACY_MODELS)
        with self.assertRaises(run.shared.Refusal):
            self.execute()

    def test_predecessor_running_refuses_before_network(self):
        self.plan()
        with patch.object(run, "check_predecessor", side_effect=run.shared.Refusal("Still running")), \
                patch.object(run.shared, "vllm_preflight", side_effect=AssertionError("No network")):
            with self.assertRaises(run.shared.Refusal):
                asyncio.run(run.run(self.root, ("gpt54",)))
        self.assertFalse((self.root / "runs").exists())

    def test_initialization_failure_precedes_predictions(self):
        self.plan()
        with self.assertRaises(RuntimeError):
            self.execute(init_failure=True)
        self.assertTrue(all(c.calls == 0 and c.closed for c in self.clients))

    def test_report_rejects_modified_correctness(self):
        self.plan()
        self.execute()
        file = self.root / "runs/jev/attempts.jsonl"
        rows = run.shared.read_rows(file)
        rows[0]["correct"] = False
        file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with self.assertRaises(run.shared.Refusal):
            run.report(self.root, ("jev",))


if __name__ == "__main__":
    unittest.main()