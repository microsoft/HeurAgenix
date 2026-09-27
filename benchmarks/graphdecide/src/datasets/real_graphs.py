"""Read verified raw releases as deterministic simple topology graphs.

All four initial datasets are undirected and unweighted. Raw files are never
modified. Node IDs remain integers from the source, NOT anonymous 0..n-1 IDs.
The graph is structurally frozen; use graph.copy() for subsequent processing.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import networkx as nx

from .sources import DEFAULT_DATA_DIR, DatasetSource, get_source


@dataclass(frozen=True, slots=True)
class LoadStats:
    raw_edge_records: int
    duplicate_edges_removed: int
    self_loops_removed: int
    nodes: int
    edges: int
    isolated_nodes: int
    connected_components: int


@dataclass(frozen=True, slots=True)
class GraphDataset:
    """Separate graph topology from provenance; no labels or model payloads."""

    source: DatasetSource
    graph: nx.Graph
    stats: LoadStats
    raw_sha256: str


def verify_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset archive: {path}; run the explicit download command first")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path.name}; refusing changed or corrupt source data")
    return actual


def _read_edge_rows(path: Path, csv_format: bool) -> Iterable[tuple[int, int]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = next(csv.reader([line])) if csv_format else line.split()
            try:
                if len(row) != 2:
                    raise ValueError("expected exactly two node IDs")
                u, v = (int(x) for x in row)
                if u < 0 or v < 0:
                    raise ValueError("node IDs must be nonnegative")
            except ValueError as exc:
                raise ValueError(f"{path.name}:{lineno}: invalid edge row") from exc
            yield u, v


def _normalize(
    rows: Iterable[tuple[int, int]], *, nodes: Iterable[int] = (), directed: bool = False,
) -> tuple[nx.Graph, LoadStats]:
    node_set = set(nodes)
    edges: set[tuple[int, int]] = set()
    raw = duplicate = loops = 0
    for u, v in rows:
        raw += 1
        node_set.update((u, v))  # Retain even a node whose only edge is a self-loop.
        if u == v:
            loops += 1
            continue
        edge = (u, v) if directed or u < v else (v, u)
        if edge in edges:
            duplicate += 1
        else:
            edges.add(edge)
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(sorted(node_set))
    graph.add_edges_from(sorted(edges))
    components = (
        nx.number_weakly_connected_components(graph) if directed
        else nx.number_connected_components(graph)
    )
    stats = LoadStats(
        raw_edge_records=raw, duplicate_edges_removed=duplicate,
        self_loops_removed=loops, nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(), isolated_nodes=nx.number_of_isolates(graph),
        connected_components=components,
    )
    return nx.freeze(graph), stats


def _load_source(source: DatasetSource, path: Path) -> GraphDataset:
    """Internal entry point accepting a source spec, also used by offline fixtures."""
    digest = verify_file(path, source.sha256)
    if source.weighted:
        raise ValueError("This loader supports unweighted topology datasets only")
    if source.file_format in ("edgelist-gzip", "csv-gzip"):
        graph, stats = _normalize(
            _read_edge_rows(path, source.file_format == "csv-gzip"),
            directed=source.directed,
        )
    elif source.file_format == "power-gml-zip":
        # Read only the known member in memory. Never extract archive paths.
        with ZipFile(path) as archive:
            with archive.open("power.gml") as stream:
                original = nx.read_gml(stream, label=None)
        if original.is_directed() != source.directed:
            raise ValueError("GML direction does not match the declared dataset")
        if any("weight" in attrs for _, _, attrs in original.edges(data=True)):
            raise ValueError("Unexpected weights in an unweighted dataset")
        graph, stats = _normalize(
            original.edges(), nodes=original.nodes(), directed=source.directed,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {source.file_format}")
    return GraphDataset(source=source, graph=graph, stats=stats, raw_sha256=digest)


def load_graph(name: str, *, data_dir: str | Path = DEFAULT_DATA_DIR) -> GraphDataset:
    """Load locally with mandatory checksum verification; never download implicitly.

    data_dir is the collection root containing a raw/ subdirectory. For edge-list
    sources only observed endpoints are known; absent isolated nodes cannot be
    recovered. Explicit GML nodes, including isolates, are retained.
    """
    source = get_source(name)
    return _load_source(source, Path(data_dir) / "raw" / source.filename)