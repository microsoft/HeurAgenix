"""Offline renderer and real-adapter tests; never acquire production lane locks."""
import asyncio
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import httpx
import networkx as nx
from scripts import cycle_zero_shot as s
from src.clients.typesafe import TypeSafeClient
from src.clients.base import ProviderHTTPError


@unittest.skipUnless((s.DENSE / "manifest.json").exists() and (s.REAL / "manifest.json").exists(),
                     "Historical local diagnostic artifacts required")
class ZeroShotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.bank = s.source_bank()
        except s.shared.Refusal as exc:
            if str(exc) != "Source drift.":
                raise
            raise unittest.SkipTest(
                "Historical fixture pins a different source revision; live validation remains strict"
            ) from exc

    def test_renderers_and_schedule(self):
        records = s.schedule(self.bank)
        self.assertEqual(records, s.schedule(self.bank))
        self.assertEqual(len(records), 480)
        self.assertEqual(Counter((r['label']['arm'], r['label']['node_count'], r['label']['answer']) for r in records),
                         {(a, n, answer): 10 for a in s.ARMS for n in s.SIZES for answer in ('yes', 'no')})
        for case in self.bank:
            original = case['request']
            expected = s.dense.wire(original, 'jev-1.13.0')
            baseline = s.render(case, 'baseline')
            self.assertEqual(s.dense.wire(baseline, 'jev-1.13.0'), expected)
            for arm in s.ARMS:
                rendered = s.render(case, arm)
                self.assertEqual(set(rendered), set(original))
                changed = {k for k in original if k != 'request_id' and original[k] != rendered[k]}
                wanted = set() if arm == 'baseline' else {'options'} if arm == 'criteria' else {'question'} if arm in ('semantics', 'algorithm') else {'state'}
                self.assertEqual(changed, wanted)
                self.assertEqual(self.independent_answer(rendered['state']), case['answer'])
                self.assertNotIn('examples', rendered['state'])
        bad = deepcopy(s.render(self.bank[0], 'adjacency')['state'])
        a = next(k for k, v in bad['adjacency'].items() if v)
        bad['adjacency'][a].pop()
        with self.assertRaises(AssertionError):
            s.graph_of(bad)

    @staticmethod
    def independent_answer(state):
        graph = nx.Graph()
        graph.add_nodes_from(state['nodes'])
        if 'adjacency' in state:
            graph.add_edges_from((int(a), b) for a, neighbors in state['adjacency'].items() for b in neighbors)
        elif isinstance(state['edges'], str):
            graph.add_edges_from(tuple(map(int, re.findall(r'\d+', line))) for line in state['edges'].splitlines())
        else:
            graph.add_edges_from(state['edges'])
        return 'yes' if nx.cycle_basis(graph) else 'no'

    def execute_mock(self, root, codes=None):
        calls = []
        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            self.assertEqual(set(body), {'state', 'model', 'questions'})
            self.assertEqual(set(body['questions']), {'decision'})
            if codes and len(calls) in codes:
                return httpx.Response(codes[len(calls)])
            answer = self.independent_answer(body['state'])
            return httpx.Response(200, json={'model': 'jev-1.13.0', 'answers': {'decision': {
                'type': 'choice', 'choice': answer, 'confidence': 1.0,
                'probabilities': {'yes': float(answer == 'yes'), 'no': float(answer == 'no')}}},
                'usage': {'input_tokens': 100, 'output_tokens': 20}})
        def factory(provider, **kwargs):
            return TypeSafeClient(**kwargs, api_key='offline-only', transport=httpx.MockTransport(handler))
        with patch.object(s, 'source_bank', return_value=self.bank), patch.object(s.cycle, 'lane_locks', lambda _: nullcontext()):
            s.plan(root)
            if codes and 401 in codes.values():
                with self.assertRaises(ProviderHTTPError):
                    asyncio.run(s.run(root, factory=factory, preflight=False))
                self.assertEqual(len(calls), 1)
                self.assertEqual(s.shared.read_json(root / 'status.json')['status'], 'aborted')
                self.assertEqual(len(s.shared.read_rows(root / 'attempts.jsonl')), 1)
            else:
                result = asyncio.run(s.run(root, factory=factory, preflight=False))
                self.assertEqual(len(calls), 480)
                self.assertEqual(result, s.report(root))
                self.assertEqual(sum(v['overall']['correct'] for v in result['arms'].values()), 480 - len(codes or {}))
                for v in result['arms'].values():
                    self.assertEqual(v['overall']['n'], 80)
            count = len(calls)
            with self.assertRaises(FileExistsError):
                asyncio.run(s.run(root, factory=factory, preflight=False))
            self.assertEqual(len(calls), count)

    def test_all_calls_and_report_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            self.execute_mock(root)
            rows = s.shared.read_rows(root / 'attempts.jsonl')
            rows[0]['raw_choice'] = 'tampered'
            (root / 'attempts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            with self.assertRaises(AssertionError):
                s.report(root)

    def test_nonfatal_failure_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.execute_mock(Path(tmp) / 'run', {1: 503})

    def test_fatal_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.execute_mock(Path(tmp) / 'run', {1: 401})

    def test_plan_tampering(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(s, 'source_bank', return_value=self.bank):
            root = Path(tmp) / 'run'
            s.plan(root)
            with (root / 'requests.jsonl').open('a') as stream:
                stream.write('{}\n')
            with self.assertRaises(AssertionError):
                s.load(root)


if __name__ == '__main__':
    unittest.main()