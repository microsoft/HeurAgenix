"""Deterministic induced subgraphs and balanced questions, with offline labels.

Only freshly constructed anonymous topology is ever placed in DecisionRequest.
The default sampler permits overlap. Pinned exclusions enable a source-node-
disjoint development pool, with mutually disjoint samples inside that pool.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import random
from typing import Callable

import networkx as nx

from src.clients.base import DecisionOption, DecisionRequest
from src.datasets import GraphDataset, load_graph
from .config import BenchmarkConfig
from .presentation import graph_from_state, question_for, state_for


@dataclass(frozen=True)
class Example:
    request: DecisionRequest
    graph_id: str
    dataset: str
    task: str
    node_count: int
    edge_count: int
    u: int
    v: int
    threshold: int | None
    distance: int
    answer: str

    def label_record(self) -> dict:
        record = asdict(self)
        record.pop("request")
        return {"request_id": self.request.request_id, **record}


@dataclass
class PreparedBenchmark:
    examples: list[Example]
    graphs: list[dict]
    issues: list[dict]
    sources: list[dict]
    planned_graphs: int
    planned_questions: int
    planned_strata: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        counts = Counter((e.dataset, e.node_count, e.task, e.threshold) for e in self.examples)
        graphs = Counter((g["dataset"], g["node_count"]) for g in self.graphs)
        strata = []
        for planned in self.planned_strata:
            count = counts[(planned["dataset"], planned["node_count"], planned["task"], planned["threshold"])]
            strata.append({**planned,
                           "generated_graphs": graphs[(planned["dataset"], planned["node_count"])],
                           "generated_questions": count,
                           "missing_questions": planned["planned_questions"] - count,
                           "question_coverage": count / planned["planned_questions"]})
        return {
            "planned_graphs": self.planned_graphs,
            "generated_graphs": len(self.graphs),
            "planned_questions": self.planned_questions,
            "generated_questions": len(self.examples),
            "question_coverage": len(self.examples) / self.planned_questions,
            "strata": strata,
            "issues": self.issues,
        }


def _rng(seed: int, *parts: object) -> random.Random:
    encoded = json.dumps([seed, *parts], separators=(",", ":"))
    return random.Random(int.from_bytes(hashlib.sha256(encoded.encode()).digest(), "big"))


def anonymous_induced_graph(original: nx.Graph, nodes: list, rng: random.Random) -> tuple[nx.Graph, list]:
    """Return a fresh graph with no graph/node/edge attributes and an offline map.

    Never use relabel_nodes(copy=True): that would retain source attributes.
    Mapping list entry i is the source ID of anonymous node i; never sent.
    """
    if original.is_directed() or original.is_multigraph() or nx.number_of_selfloops(original):
        raise ValueError("Expected a simple undirected loop-free source")
    mapping = sorted(nodes)
    rng.shuffle(mapping)
    inverse = {node: index for index, node in enumerate(mapping)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(mapping)))
    graph.add_edges_from(sorted(
        (min(inverse[u], inverse[v]), max(inverse[u], inverse[v]))
        for u, v in original.subgraph(nodes).edges()
    ))
    return nx.freeze(graph), mapping


def _sample_nodes(graph: nx.Graph, eligible: list[int], count: int, rng: random.Random) -> list[int]:
    start = rng.choice(eligible)
    selected = {start}
    frontier = set(graph.neighbors(start))
    while len(selected) < count:
        node = rng.choice(sorted(frontier))
        selected.add(node)
        frontier.remove(node)
        frontier.update(set(graph.neighbors(node)) - selected)
    return sorted(selected)


def _questions(graph: nx.Graph, graph_id: str, dataset: str, config: BenchmarkConfig) -> tuple[list[Example], list[dict]]:
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    pairs = [(u, v) for u in graph for v in graph if u < v]
    tasks = [("adjacency", None, config.tasks.adjacency_questions)] + [
        ("distance_threshold", k, config.tasks.distance_questions_per_threshold)
        for k in config.tasks.distance_thresholds
    ]
    examples, issues = [], []
    for task, threshold, count in tasks:
        if not count:
            continue
        positive = [(u, v) for u, v in pairs if distances[u][v] == (1 if threshold is None else threshold)]
        negative = [(u, v) for u, v in pairs if (
            distances[u][v] > 1 if threshold is None else distances[u][v] == threshold + 1
        )]
        if min(len(positive), len(negative)) < count // 2:
            issues.append({"graph_id": graph_id, "dataset": dataset, "task": task,
                           "threshold": threshold, "reason": "insufficient_balanced_pairs",
                           "positive_pairs": len(positive), "negative_pairs": len(negative),
                           "missing_questions": count})
            continue
        rng = _rng(config.run.seed, graph_id, task, threshold, "questions")
        chosen = [(u, v, "yes") for u, v in rng.sample(positive, count // 2)]
        chosen += [(u, v, "no") for u, v in rng.sample(negative, count // 2)]
        rng.shuffle(chosen)
        for index, (u, v, answer) in enumerate(chosen):
            request_id = f"{graph_id}-{task}-{threshold}-{index:04d}"
            question = question_for(u, v, threshold, config.presentation)
            # Explicit allowlist: never serialize a source graph or label record.
            request = DecisionRequest(
                request_id=request_id,
                state=state_for(graph, config.presentation),
                question=question,
                options=(DecisionOption("yes"), DecisionOption("no")),
            )
            examples.append(Example(request, graph_id, dataset, task, len(graph), graph.number_of_edges(),
                                    u, v, threshold, distances[u][v], answer))
    return examples, issues


def verify_examples(examples: list[Example]) -> None:
    """Independently check labels against Floyd-Warshall on the serialized graph.

    Generation uses BFS; verification reconstructs exactly the model-visible data.
    """
    cache = {}
    seen = set()
    for example in examples:
        state = example.request.state
        key = json.dumps(state, sort_keys=True)
        if key not in cache:
            graph = graph_from_state(state, example.node_count)
            cache[key] = (graph, dict(nx.floyd_warshall(graph)))
        graph, distances = cache[key]
        if len(graph) != example.node_count:
            raise ValueError("Invalid node count")
        distance = distances[example.u][example.v]
        truth = graph.has_edge(example.u, example.v) if example.task == "adjacency" else distance <= example.threshold
        if example.answer != ("yes" if truth else "no") or example.distance != distance:
            raise ValueError("Ground truth verification failed")
        if example.edge_count != graph.number_of_edges() or example.request.request_id in seen:
            raise ValueError("Invalid edge count or duplicate request ID")
        seen.add(example.request.request_id)


def excluded_source_nodes(config: BenchmarkConfig) -> dict[str, set[int]]:
    """Read only the pinned graph artifact; never send source IDs to a model."""
    exclusion = config.sampling.exclude_graphs
    if exclusion is None:
        return {}
    body = exclusion.path.read_bytes()
    if hashlib.sha256(body).hexdigest() != exclusion.sha256:
        raise ValueError("Prior graph artifact hash mismatch")
    excluded: dict[str, set[int]] = {}
    seen = set()
    try:
        for line in body.splitlines():
            row = json.loads(line)
            name, graph_id = row["dataset"], row["graph_id"]
            nodes = row["original_ids_by_anonymous_id"]
            if (not isinstance(name, str) or not isinstance(graph_id, str)
                    or not isinstance(nodes, list) or not nodes
                    or any(type(n) is not int for n in nodes) or len(set(nodes)) != len(nodes)
                    or type(row["node_count"]) is not int or row["node_count"] != len(nodes)
                    or (name, graph_id) in seen):
                raise ValueError()
            seen.add((name, graph_id))
            excluded.setdefault(name, set()).update(nodes)
        if not set(config.data.datasets) <= excluded.keys():
            raise ValueError()
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise ValueError("Invalid prior graph artifact") from None
    return excluded


def prepare(config: BenchmarkConfig, loader: Callable[..., GraphDataset] = load_graph) -> PreparedBenchmark:
    exclusions = excluded_source_nodes(config)
    result = PreparedBenchmark([], [], [], [], config.planned_graphs, config.planned_questions)
    tasks = [("adjacency", None, config.tasks.adjacency_questions)] + [
        ("distance_threshold", k, config.tasks.distance_questions_per_threshold)
        for k in config.tasks.distance_thresholds
    ]
    result.planned_strata = [
        {"dataset": dataset, "node_count": count, "task": task, "threshold": threshold,
         "planned_graphs": config.sampling.samples_per_size,
         "planned_questions": config.sampling.samples_per_size * questions}
        for dataset in config.data.datasets for count in config.sampling.node_counts
        for task, threshold, questions in tasks if questions
    ]
    for dataset in config.data.datasets:
        loaded = loader(dataset, data_dir=config.data.data_dir)
        source = loaded.graph
        blocked = set(exclusions.get(dataset, ()))
        if not blocked <= set(source):
            raise ValueError("Prior source nodes are absent from the loaded graph")
        result.sources.append({"source": asdict(loaded.source), "raw_sha256": loaded.raw_sha256,
                               "stats": asdict(loaded.stats)})
        components = list(nx.connected_components(source))
        for count in config.sampling.node_counts:
            eligible = sorted(n for component in components if len(component) >= count for n in component)
            used: set[tuple[int, ...]] = set()
            for index in range(config.sampling.samples_per_size):
                sampling_source = source
                if config.sampling.exclude_graphs is not None:
                    sampling_source = source.subgraph(set(source) - blocked)
                    remaining_components = nx.connected_components(sampling_source)
                    eligible = sorted(n for c in remaining_components if len(c) >= count for n in c)
                graph_id = f"{dataset}-n{count}-s{index:04d}"
                rng = _rng(config.run.seed, graph_id, "sampling")
                nodes = None
                attempts = 0
                if eligible:
                    for attempts in range(1, config.sampling.max_attempts + 1):
                        candidate = _sample_nodes(sampling_source, eligible, count, rng)
                        if tuple(candidate) not in used:
                            nodes = candidate
                            used.add(tuple(candidate))
                            if config.sampling.exclude_graphs is not None:
                                blocked.update(candidate)
                            break
                if nodes is None:
                    result.issues.append({"graph_id": graph_id, "dataset": dataset,
                                          "reason": "no_eligible_component" if not eligible else "duplicate_sample_limit",
                                          "attempts": attempts})
                    continue
                graph, mapping = anonymous_induced_graph(source, nodes, _rng(config.run.seed, graph_id, "ids"))
                state = {"nodes": list(graph), "edges": [list(e) for e in graph.edges()]}
                result.graphs.append({"graph_id": graph_id, "dataset": dataset,
                                      "node_count": count, "edge_count": graph.number_of_edges(),
                                      "sampling_attempts": attempts, "original_ids_by_anonymous_id": mapping,
                                      "topology": state})
                examples, issues = _questions(graph, graph_id, dataset, config)
                result.examples.extend(examples)
                result.issues.extend(issues)
    verify_examples(result.examples)
    return result