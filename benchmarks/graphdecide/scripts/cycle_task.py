#!/usr/bin/env python3
"""Offline matched-size cycle task from real graph samples; no model calls."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import networkx as nx
from src.clients.base import DecisionOption, DecisionRequest

SEED = 20260929
QUESTION = (
    "Does this simple undirected graph contain a cycle of at least three "
    "distinct vertices anywhere? Traversing one edge back and forth is not "
    "a cycle. Return only yes or no."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def has_cycle(state):
    """Iterative DFS independent of the generation library's cycle basis."""
    nodes, edges = state["nodes"], state["edges"]
    if nodes != list(range(len(nodes))):
        raise ValueError("Expected contiguous anonymous nodes")
    neighbors = {v: set() for v in nodes}
    for a, b in edges:
        if a not in neighbors or b not in neighbors or a == b or b in neighbors[a]:
            raise ValueError("Expected simple undirected edges")
        neighbors[a].add(b)
        neighbors[b].add(a)
    visited = set()
    cyclic = False
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, None, iter(neighbors[start]))]
        while stack:
            v, parent, adjacent = stack[-1]
            w = next(adjacent, None)
            if w is None:
                stack.pop()
            elif w != parent:
                if w in visited:
                    cyclic = True
                else:
                    visited.add(w)
                    stack.append((w, v, iter(neighbors[w])))
    return cyclic


def build_pair(source, seed=SEED):
    """Return forest/unicyclic edge-subgraphs with n vertices and n-2 edges.

    All edges come from the real source sample. No rejection based on model
    outputs: construction uses only topology. Missing positive cases are explicit.
    """
    graph = nx.Graph()
    graph.add_nodes_from(source["topology"]["nodes"])
    graph.add_edges_from(source["topology"]["edges"])
    has_cycle(source["topology"])  # Validate simplicity before NetworkX deduplicates.
    n = len(graph)
    if n < 4 or not nx.is_connected(graph):
        return None
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{source['graph_id']}".encode()).digest(), "big"))
    edges = sorted(tuple(sorted(e)) for e in graph.edges())
    rng.shuffle(edges)
    # Random-priority Kruskal tree; no graph attributes enter a request.
    tree = nx.Graph()
    tree.add_nodes_from(range(n))
    components = nx.utils.UnionFind(range(n))
    for a, b in edges:
        if components[a] != components[b]:
            components.union(a, b)
            tree.add_edge(a, b)
    chords = [e for e in edges if not tree.has_edge(*e)]
    rng.shuffle(chords)
    for a, b in chords:
        path = nx.shortest_path(tree, a, b)
        cycle_edges = {tuple(sorted(e)) for e in zip(path, path[1:])}
        branches = [e for e in sorted(tuple(sorted(e)) for e in tree.edges()) if e not in cycle_edges]
        if len(branches) < 2:
            continue
        rng.shuffle(branches)
        forest = tree.copy()
        forest.remove_edge(*branches[0])
        positive = forest.copy()
        positive.add_edge(a, b)
        positive.remove_edge(*branches[1])
        permutation = list(range(n))
        rng.shuffle(permutation)
        records = []
        for answer, derived in (("no", forest), ("yes", positive)):
            visible_edges = [[permutation[u], permutation[v]] for u, v in derived.edges()]
            rng.shuffle(visible_edges)
            state = {"nodes": list(range(n)), "edges": visible_edges}
            assert derived.number_of_edges() == n - 2
            assert has_cycle(state) == bool(nx.cycle_basis(derived)) == (answer == "yes")
            request = DecisionRequest(f"cycle-{source['graph_id']}-{answer}", state, QUESTION,
                                      (DecisionOption("yes"), DecisionOption("no")))
            records.append({"request": asdict(request), "label": {
                "request_id": request.request_id, "task": "cycle_detection", "answer": answer,
                "pair_id": source["graph_id"], "dataset": source["dataset"],
                "node_count": n, "edge_count": n - 2,
                "components": nx.number_connected_components(derived),
                "cycle_rank": derived.number_of_edges() - n + nx.number_connected_components(derived),
            }})
        return records
    return None


def plan(root, prior):
    root, prior = Path(root).resolve(), Path(prior).resolve()
    root.mkdir(parents=True, exist_ok=False)
    body = prior.read_bytes()
    prior_hash = hashlib.sha256(body).hexdigest()
    graphs = [json.loads(line) for line in body.splitlines() if line]
    if len({g["graph_id"] for g in graphs}) != len(graphs):
        raise ValueError("Duplicate source graph IDs")
    records, skipped = [], []
    for source in graphs:
        pair = build_pair(source)
        if pair is None:
            skipped.append({"graph_id": source["graph_id"], "dataset": source["dataset"],
                            "node_count": source["node_count"], "reason": "no_cycle_with_two_off_cycle_tree_edges"})
        else:
            records.extend(pair)
    if not records:
        raise ValueError("No valid paired cases; no model calls made")
    random.Random(SEED).shuffle(records)
    for key, filename in (("request", "requests.jsonl"), ("label", "labels.jsonl")):
        with (root / filename).open("x") as stream:
            for row in records:
                stream.write(json.dumps(row[key], sort_keys=True) + "\n")
    counts = Counter((r["label"]["dataset"], r["label"]["node_count"], r["label"]["answer"]) for r in records)
    manifest = {"kind": "cycle-matched-edge-subgraphs-v1", "status": "offline-prepared-not-evaluated",
                "seed": SEED, "source_graphs": len(graphs), "generated_pairs": len(records) // 2,
                "questions_per_model": len(records), "seven_model_call_budget": len(records) * 7,
                "prior": {"path": str(prior), "sha256": prior_hash}, "skipped": skipped,
                "coverage": [{"dataset": d, "node_count": n, "answer": a, "questions": count}
                             for (d, n, a), count in sorted(counts.items())],
                "code_sha256": {"scripts/cycle_task.py": digest(__file__),
                                "src/clients/base.py": digest(REPO / "src/clients/base.py")},
                "artifact_sha256": {name: digest(root / name) for name in ("requests.jsonl", "labels.jsonl")},
                "notice": "Real-derived edge-subgraphs, not untouched induced samples or independent held-out data. Each pair has equal n and m=n-2. Source selection favors graphs supporting this construction. Component count and cycle rank are private labels only. Offline plan only, no new runner or provider calls."}
    with (root / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    manifest = plan(args.root, args.prior)
    print(json.dumps({k: manifest[k] for k in ("status", "source_graphs", "generated_pairs", "questions_per_model", "seven_model_call_budget")}))