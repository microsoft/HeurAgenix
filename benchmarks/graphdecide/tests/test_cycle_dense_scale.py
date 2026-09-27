"""No live calls or shared production lock acquisition."""
import asyncio
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import networkx as nx
from scripts import cycle_dense_scale as s
from src.clients.typesafe import TypeSafeClient


class DenseScaleTests(unittest.TestCase):
    def test_labels_balance_and_stable_nesting(self):
        records = s.cases()
        self.assertEqual(records, s.cases())
        self.assertEqual(len(records), 320)
        counts = Counter((r['label']['node_count'], r['label']['answer']) for r in records)
        lookup = {(r['label']['family'], r['label']['answer'], r['label']['node_count']): r['request']['state'] for r in records}
        for n in s.SIZES:
            for answer in ('yes', 'no'):
                self.assertEqual(counts[n, answer], 20)
                for f in range(20):
                    state = lookup[f, answer, n]
                    g = nx.Graph()
                    g.add_nodes_from(state['nodes'])
                    g.add_edges_from(state['edges'])
                    self.assertEqual(nx.is_forest(g), answer == 'no')
                    if n > 5:
                        old = lookup[f, answer, n - 1]
                        self.assertEqual([e for e in state['edges'] if n - 1 not in e], old['edges'])
                        self.assertEqual(sum(n - 1 in e for e in state['edges']), 1)
            for f in range(20):
                a, b = [set(tuple(sorted(e)) for e in lookup[f, ans, n]['edges']) for ans in ('no', 'yes')]
                self.assertEqual(len(a ^ b), 2)

    def test_full_http_mock_and_no_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(s.cycle, 'lane_locks', lambda _: nullcontext()):
            root = Path(tmp) / 'run'
            s.plan(root)
            calls = []
            def handler(request):
                body = json.loads(request.content)
                calls.append(body)
                g = nx.Graph()
                g.add_nodes_from(body['state']['nodes'])
                g.add_edges_from(body['state']['edges'])
                answer = 'no' if nx.is_forest(g) else 'yes'
                return httpx.Response(200, json={'model': 'jev-1.13.0', 'answers': {'decision': {
                    'type': 'choice', 'choice': answer, 'confidence': 1.0,
                    'probabilities': {'yes': float(answer == 'yes'), 'no': float(answer == 'no')}}}, 'usage': {}})
            def factory(provider, **kwargs):
                return TypeSafeClient(**kwargs, api_key='offline-only', transport=httpx.MockTransport(handler))
            result = asyncio.run(s.run(root, factory=factory, preflight=False))
            self.assertEqual(result['overall']['correct'], 320)
            self.assertEqual(len(calls), 320)
            self.assertEqual(result, s.report(root))
            self.assertEqual(result['adjacent_transitions']['6'], {'correct_to_correct': 40})
            with self.assertRaises(FileExistsError):
                asyncio.run(s.run(root, factory=factory, preflight=False))
            self.assertEqual(len(calls), 320)
            rows = s.shared.read_rows(root / 'attempts.jsonl')
            rows[0]['raw_choice'] = 'bad'
            (root / 'attempts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            with self.assertRaises(AssertionError):
                s.report(root)


if __name__ == '__main__':
    unittest.main()