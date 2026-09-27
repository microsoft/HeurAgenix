"""Freeze a graph-only exclusion pool without importing benchmark/model code."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


MIN_NODES = 8
MAX_NODES = 12
NATURAL_EDGE = re.compile(
    r"Node (\S+) and node (\S+) are joined by an undirected edge\."
)


class UnsupportedGraph(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scalar_key(value: Any) -> str:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise UnsupportedGraph("non_scalar_node")
    if isinstance(value, float) and not math.isfinite(value):
        raise UnsupportedGraph("nonfinite_node")
    return canonical_json(value)


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    nodes = state.get("nodes")
    if not isinstance(nodes, list):
        raise UnsupportedGraph("nodes_not_list")
    keys = [scalar_key(node) for node in nodes]
    if len(set(keys)) != len(keys):
        raise UnsupportedGraph("duplicate_node")
    mapping = {key: index for index, key in enumerate(sorted(keys))}
    aliases: dict[str, list[str]] = {}
    for node, key in zip(nodes, keys):
        aliases.setdefault(str(node), []).append(key)

    def endpoint(value: Any) -> int:
        key = scalar_key(value)
        if key not in mapping:
            raise UnsupportedGraph("unknown_endpoint")
        return mapping[key]

    def text_endpoint(value: str) -> int:
        candidates = aliases.get(value, [])
        if len(candidates) != 1:
            raise UnsupportedGraph("unknown_or_ambiguous_text_endpoint")
        return mapping[candidates[0]]

    directed = state.get("directed", False)
    if not isinstance(directed, bool):
        raise UnsupportedGraph("directed_not_boolean")
    if "adjacency" in state and "edges" in state:
        raise UnsupportedGraph("ambiguous_edge_encoding")
    raw_edges = state.get("edges")
    edges: list[list[Any]] = []
    if "adjacency" in state:
        adjacency = state["adjacency"]
        if not isinstance(adjacency, dict):
            raise UnsupportedGraph("adjacency_not_dictionary")
        arcs = set()
        covered = set()
        for raw_u, neighbors in adjacency.items():
            u = text_endpoint(raw_u)
            if u in covered or not isinstance(neighbors, list):
                raise UnsupportedGraph("invalid_adjacency_entry")
            covered.add(u)
            for neighbor in neighbors:
                v = endpoint(neighbor)
                if (u, v) in arcs:
                    raise UnsupportedGraph("duplicate_adjacency_neighbor")
                arcs.add((u, v))
        if covered != set(range(len(nodes))):
            raise UnsupportedGraph("incomplete_adjacency")
        if not directed and any((v, u) not in arcs for u, v in arcs):
            raise UnsupportedGraph("asymmetric_undirected_adjacency")
        edges = [
            [u, v] for u, v in sorted(arcs) if directed or u <= v
        ]
    elif isinstance(raw_edges, str):
        if directed:
            raise UnsupportedGraph("directed_natural_edges")
        for line in raw_edges.splitlines():
            if not line.strip():
                continue
            match = NATURAL_EDGE.fullmatch(line.strip())
            if match is None:
                raise UnsupportedGraph("unsupported_natural_edge_sentence")
            edges.append([text_endpoint(value) for value in match.groups()])
    elif isinstance(raw_edges, list):
        arities = set()
        for edge in raw_edges:
            if not isinstance(edge, list) or len(edge) not in (2, 3):
                raise UnsupportedGraph("invalid_edge_record")
            arities.add(len(edge))
            converted = [endpoint(edge[0]), endpoint(edge[1])]
            if len(edge) == 3:
                weight = edge[2]
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    raise UnsupportedGraph("nonnumeric_edge_weight")
                if isinstance(weight, float) and not math.isfinite(weight):
                    raise UnsupportedGraph("nonfinite_edge_weight")
                converted.append(weight)
            edges.append(converted)
        if len(arities) > 1:
            raise UnsupportedGraph("mixed_weighted_unweighted_edges")
    else:
        raise UnsupportedGraph("unsupported_edge_encoding")

    seen = set()
    for edge in edges:
        u, v = edge[:2]
        if u == v:
            raise UnsupportedGraph("self_loop")
        if not directed and u > v:
            edge[0], edge[1] = v, u
        pair = tuple(edge[:2])
        if pair in seen:
            raise UnsupportedGraph("duplicate_edge")
        seen.add(pair)
    return {
        "nodes": list(range(len(nodes))),
        "edges": sorted(edges),
        "directed": directed,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_state(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        raise UnsupportedGraph("row_not_object")
    state = row.get("state")
    nested = row.get("request")
    if state is None and isinstance(nested, dict):
        state = nested.get("state")
    if state is None:
        return None
    if not isinstance(state, dict):
        raise UnsupportedGraph("state_not_object")
    return state


def candidate_size(state: dict[str, Any]) -> int:
    nodes = state.get("nodes")
    if isinstance(nodes, (list, dict, tuple)):
        return len(nodes)
    # Missing node lists cannot safely be inferred from edges: isolates disappear.
    raise UnsupportedGraph("missing_or_invalid_nodes")


def collect_prior_graphs(scan_root: Path, output_root: Path) -> dict[str, Any]:
    scan_root = scan_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError("output root must not exist")
    if not scan_root.is_dir():
        raise NotADirectoryError("scan root must be an existing directory")
    sources = sorted(
        path
        for path in scan_root.rglob("*.jsonl")
        if path.name in {"requests.jsonl", "instances.jsonl"}
        and not path.resolve().is_relative_to(output_root)
    )
    graphs: dict[str, dict[str, Any]] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    unsupported = []
    source_pins = []
    total = Counter()
    candidate_by_size = Counter()
    for source in sources:
        relative = source.relative_to(scan_root).as_posix()
        before = source.stat()
        digest = hashlib.sha256()
        coverage = Counter()
        with source.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                coverage["rows"] += 1
                try:
                    row = json.loads(line)
                    state = row_state(row)
                    if state is None or not any(
                        key in state for key in ("nodes", "edges", "adjacency")
                    ):
                        coverage["non_graph_rows"] += 1
                        continue
                    size = candidate_size(state)
                    if not MIN_NODES <= size <= MAX_NODES:
                        coverage["outside_size_rows"] += 1
                        continue
                    coverage["candidate_rows"] += 1
                    candidate_by_size[str(size)] += 1
                    normalized = normalize_state(state)
                    serialized = canonical_json(normalized)
                    identifier = hashlib.sha256(serialized.encode()).hexdigest()
                    task = (
                        "lt_influence_construct"
                        if normalized["directed"]
                        else "tsp_construct"
                        if any(len(edge) == 3 for edge in normalized["edges"])
                        else "degree_exact"
                    )
                    record = {
                        "id": identifier,
                        "task": task,
                        "state": normalized,
                    }
                    if identifier in graphs and graphs[identifier] != record:
                        raise RuntimeError("graph identifier collision")
                    graphs[identifier] = record
                    provenance.setdefault(identifier, []).append(
                        {"source": relative, "line": line_number}
                    )
                    coverage["accepted_rows"] += 1
                except (UnsupportedGraph, json.JSONDecodeError, UnicodeDecodeError) as exc:
                    code = (
                        str(exc)
                        if isinstance(exc, UnsupportedGraph)
                        else "invalid_json_or_encoding"
                    )
                    unsupported.append(
                        {"source": relative, "line": line_number, "reason": code}
                    )
                    coverage["unsupported_rows"] += 1
        after = source.stat()
        pin = digest.hexdigest()
        if (
            (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or file_sha256(source) != pin
        ):
            raise RuntimeError(f"source changed during read: {relative}")
        source_pins.append(
            {
                "path": relative,
                "sha256": pin,
                "bytes": after.st_size,
                "coverage": dict(sorted(coverage.items())),
            }
        )
        total.update(coverage)

    manifest = {
        "schema_version": 1,
        "status": "rejected_unsupported_rows" if unsupported else "completed",
        "scan_root": str(scan_root),
        "node_counts": list(range(MIN_NODES, MAX_NODES + 1)),
        "policy": {
            "source_names": ["requests.jsonl", "instances.jsonl"],
            "deduplication": "exact labeled normalized graph state, not isomorphism",
            "node_remapping": "scalar JSON identities sorted before zero-based remapping",
            "retained": "nodes, edges including inline weights, directed",
            "dropped": (
                "queries, histories, labels, points/metric, and diffusion parameters "
                "including separate LT weight dictionaries; LT topology is retained"
            ),
            "unsupported": "fail closed; manifest only, no usable graph pool",
            "scope": (
                "Only named JSONL files present at scan start; no manifests, "
                "attempt logs, bank JSON, or later-created files are discovered."
            ),
        },
        "coverage": dict(sorted(total.items())),
        "candidate_rows_by_size": dict(sorted(candidate_by_size.items())),
        "unique_graphs": len(graphs),
        "unique_by_task": dict(sorted(Counter(x["task"] for x in graphs.values()).items())),
        "unique_by_size": dict(
            sorted(Counter(str(len(x["state"]["nodes"])) for x in graphs.values()).items())
        ),
        "sources": source_pins,
        "provenance": dict(sorted(provenance.items())),
        "unsupported": unsupported,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    if not unsupported:
        pool = output_root / "graph-pool.jsonl"
        with pool.open("x", encoding="utf-8") as stream:
            for identifier in sorted(graphs):
                stream.write(canonical_json(graphs[identifier]) + "\n")
        manifest["graph_pool_sha256"] = file_sha256(pool)
    with (output_root / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, default=Path("output"))
    args = parser.parse_args()
    try:
        manifest = collect_prior_graphs(args.scan_root, args.output_root)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("status", "coverage", "unique_graphs", "unique_by_task")
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
