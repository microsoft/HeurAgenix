"""Offline-only scale diagnostic checks, no contention with active model locks."""
import asyncio
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import networkx as nx
from scripts import cycle_scale_diagnostic as scale
from src.clients.typesafe import TypeSafeClient


class ScaleTests(unittest.TestCase):
    def test_pair_balance_and_independent_labels(self):
        cases = scale.build_cases()
        self.assertEqual(cases, scale.build_cases())
        self.assertEqual(len(cases), 550)
        counts = Counter((c["label"]["mode"], c["label"]["node_count"], c["label"]["answer"]) for c in cases)
        pairs = defaultdict(list)
        for c in cases:
            state, label = c["request"]["state"], c["label"]
            g = nx.Graph()
            g.add_nodes_from(state["nodes"])
            g.add_edges_from(state["edges"])
            self.assertEqual(nx.is_forest(g), label["answer"] == "no")
            pairs[label["pair_id"]].append(c)
        for n in scale.SIZES:
            for answer in ("yes", "no"):
                self.assertEqual(counts["branches", n, answer], 20)
                self.assertEqual(counts["isolates", n, answer], 5)
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            states = [c["request"]["state"] for c in pair]
            self.assertEqual(states[0]["nodes"], states[1]["nodes"])
            sets = [{tuple(sorted(e)) for e in s["edges"]} for s in states]
            self.assertEqual(len(sets[0]), len(sets[1]))
            self.assertEqual(len(sets[0] ^ sets[1]), 2)
        sizes = [c["label"]["node_count"] for c in cases]
        self.assertNotEqual(sizes, sorted(sizes))

    def test_mock_wire_scoring_and_no_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(scale.cycle, "lane_locks", lambda _: nullcontext()):
            root = Path(tmp) / "run"
            scale.plan(root)
            calls = []
            def handler(request):
                payload = json.loads(request.content)
                self.assertEqual(set(payload), {"model", "state", "questions"})
                self.assertNotIn("request_id", payload)
                calls.append(payload)
                g = nx.Graph()
                g.add_nodes_from(payload["state"]["nodes"])
                g.add_edges_from(payload["state"]["edges"])
                answer = "no" if nx.is_forest(g) else "yes"
                return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"decision": {
                    "type": "choice", "choice": answer, "probabilities": {"yes": float(answer == "yes"), "no": float(answer == "no")},
                    "confidence": 1.0}}, "usage": {"input_tokens": 1, "output_tokens": 1}})
            def factory(provider, **kwargs):
                return TypeSafeClient(**kwargs, api_key="offline-only", transport=httpx.MockTransport(handler))
            result = asyncio.run(scale.run(root, factory=factory, preflight=False))
            self.assertEqual(len(calls), 550)
            self.assertEqual(result["overall"]["correct"], 550)
            self.assertEqual(scale.report(root), result)
            with self.assertRaises(FileExistsError):
                asyncio.run(scale.run(root, factory=factory, preflight=False))
            self.assertEqual(len(calls), 550)
            with self.assertRaises(FileExistsError):
                scale.plan(root)

    def test_fatal_http_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(scale.cycle, "lane_locks", lambda _: nullcontext()):
            root = Path(tmp) / "run"
            scale.plan(root)
            def factory(provider, **kwargs):
                return TypeSafeClient(**kwargs, api_key="offline-only", transport=httpx.MockTransport(lambda _: httpx.Response(401)))
            with self.assertRaises(scale.DecisionClientError):
                asyncio.run(scale.run(root, factory=factory, preflight=False))
            self.assertEqual(scale.shared.read_json(root / "status.json")["status"], "aborted")
            self.assertEqual(len(scale.shared.read_rows(root / "attempts.jsonl")), 1)
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()