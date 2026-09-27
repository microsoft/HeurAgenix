"""Offline tests of the predeclared diagnostic and live orchestration boundaries."""

import asyncio
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import networkx as nx

from scripts import graph_sanity_diagnostic as d
from src.clients.base import BaseDecisionClient, ClientCapabilities, DecisionResponse, InvalidResponseError, ProviderHTTPError


class Oracle(BaseDecisionClient):
    def __init__(self, model, *, failure=None):
        self.model = model
        self.failure = failure
        self.calls = 0
        self.closed = False

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        self.calls += 1
        await asyncio.sleep(0)  # Yield, without delaying, to exercise cancellation/concurrency.
        if self.failure and self.calls == 1:
            raise self.failure
        from dataclasses import asdict
        return DecisionResponse(request.request_id, d.visible_truth(asdict(request)), self.model)

    async def aclose(self):
        self.closed = True


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "diagnostic"

    def test_authored_answers_and_transformations(self):
        rows = d.examples()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r["request"]["request_id"] for r in rows}), 48)
        self.assertEqual(Counter((r["label"]["task"], r["label"]["variant"], r["label"]["answer"]) for r in rows),
                         Counter({(t, v, a): 3 for t in ("adjacency", "distance_threshold")
                                  for v in d.VARIANTS for a in ("yes", "no")}))
        cases = {}
        for row in rows:
            cases.setdefault(row["label"]["case_id"], {})[row["label"]["variant"]] = row["request"]
            # Third checker uses NetworkX Floyd-Warshall instead of author's expected key or deque BFS.
            r = row["request"]
            g = nx.Graph()
            g.add_nodes_from(r["state"]["nodes"])
            g.add_edges_from(r["state"]["edges"])
            import re
            args = list(map(int, re.findall(r"\d+", r["question"])))
            u, v = args[:2]
            limit = args[2] if len(args) == 3 else 1
            truth = "yes" if nx.floyd_warshall(g)[u][v] <= limit else "no"
            self.assertEqual(truth, row["label"]["answer"])
        for variants in cases.values():
            baseline = variants["baseline"]
            option = variants["option_order"]
            self.assertEqual(option["state"], baseline["state"])
            self.assertEqual(option["question"], baseline["question"])
            self.assertEqual(option["options"], tuple(reversed(baseline["options"])))
            edge = variants["edge_order"]
            self.assertEqual(edge["question"], baseline["question"])
            self.assertEqual(edge["options"], baseline["options"])
            self.assertCountEqual(edge["state"]["edges"], baseline["state"]["edges"])
            self.assertNotEqual(edge["state"]["edges"], baseline["state"]["edges"])

    def test_plan_is_offline_immutable_and_budgeted(self):
        with patch.object(d.shared, "create_client", side_effect=AssertionError("offline")):
            manifest = d.plan(self.root)
            self.assertEqual(d.load(self.root)[1], manifest)
        self.assertEqual(manifest["max_total_calls"], 336)
        self.assertEqual(manifest["max_jev_calls"], 48)
        with self.assertRaises(FileExistsError):
            d.plan(self.root)
        with (self.root / "requests.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaises(d.shared.Refusal):
            d.load(self.root)

    def execute(self, *, failure=None):
        clients = []
        def factory(provider, **kwargs):
            client = Oracle(kwargs["model"], failure=failure)
            clients.append(client)
            return client
        result = asyncio.run(d.run(self.root, factory=factory, preflight=False))
        return result, clients

    def test_parallel_end_to_end_and_no_resume(self):
        d.plan(self.root)
        result, clients = self.execute()
        self.assertEqual(sum(c.calls for c in clients), 336)
        self.assertTrue(all(c.closed for c in clients))
        self.assertEqual(d.report(self.root), result)
        self.assertEqual(len(result["models"]), 7)
        for summary in result["models"].values():
            self.assertEqual(summary["overall"]["correct"], 48)
            for values in summary["invariance"].values():
                self.assertEqual(values["valid_pairs"], 12)
                self.assertEqual(values["answer_changed"], 0)
        with self.assertRaises(FileExistsError):
            self.execute()
        output = self.root / "runs" / "jev" / "summary.json"
        summary = json.loads(output.read_text())
        summary["overall"]["correct"] -= 1
        output.write_text(json.dumps(summary))
        with self.assertRaises(d.shared.Refusal):
            d.report(self.root)

    def test_failures_preserve_denominator_and_diagnostics(self):
        d.plan(self.root)
        result, clients = self.execute(failure=InvalidResponseError("not persisted", diagnostic_code="invalid_choice"))
        for summary in result["models"].values():
            self.assertEqual(summary["overall"]["n"], 48)
            self.assertEqual(summary["overall"]["correct"], 47)
            self.assertEqual(summary["overall"]["failures"], 1)
        first = d.json_rows(self.root / "runs" / "jev" / "attempts.jsonl")[0]
        self.assertEqual(first["diagnostic_code"], "invalid_choice")
        self.assertNotIn("not persisted", json.dumps(first))

    def test_auth_failure_stops_and_records_attempts(self):
        d.plan(self.root)
        with self.assertRaises(d.shared.AuthenticationAbort):
            self.execute(failure=ProviderHTTPError(401))
        for file in (self.root / "runs").glob("*/status.json"):
            status = json.loads(file.read_text())
            self.assertEqual(status["status"], "aborted")
            self.assertLessEqual(status["completed_calls"], 1)
        with self.assertRaises(d.shared.Refusal):
            d.report(self.root)

    def test_initialization_fails_before_any_predictions(self):
        d.plan(self.root)
        clients = []
        def factory(provider, **kwargs):
            client = Oracle(kwargs["model"])
            client.initialize = AsyncMock(side_effect=RuntimeError("do not print"))
            clients.append(client)
            return client
        with self.assertRaises(RuntimeError):
            asyncio.run(d.run(self.root, factory=factory, preflight=False))
        self.assertTrue(all(c.calls == 0 and c.closed for c in clients))

    def test_invariance_does_not_confuse_consistency_with_correctness(self):
        rows = [{**row["label"], "status": "success", "selected_option_id": "no",
                 "correct": row["label"]["answer"] == "no", "error_type": None} for row in d.examples()]
        summary = d.summarize(rows)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertTrue(all(v["answer_changed"] == 0 for v in summary["invariance"].values()))


if __name__ == "__main__":
    unittest.main()