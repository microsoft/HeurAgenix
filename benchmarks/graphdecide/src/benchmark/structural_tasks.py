"""Bounded, offline structural graph banks with exact isomorphism exclusion.

Families are generation procedures, not disjoint mathematical graph classes.
Only seven-key task records are returned as instances; provenance, graph clusters,
family parameters, labels, rejection counts, and shortfalls live in metadata.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
import random

import networkx as nx

from . import extended_tasks as tasks

TASKS = (
    "adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
    "distance_threshold", "articulation_point", "maxcut_construct", "tsp_construct",
    "lt_influence_construct", "community_bipartition_construct",
)
BINARY_TASKS = {
    "adjacency", "cycle_detection", "pair_connectivity",
    "distance_threshold", "articulation_point",
}
MAX_ATTEMPTS_PER_CELL = 128

__all__ = ["build_structural_bank"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _state(graph):
    edges = list(graph.edges())
    if not graph.is_directed():
        edges = [tuple(sorted(edge)) for edge in edges]
    return {"nodes": sorted(graph.nodes()), "edges": [list(e) for e in sorted(edges)],
            "directed": graph.is_directed()}


def _graph(state, *, weighted=False):
    """Graph identity excludes questions, coordinates, labels, and node names."""
    _require(type(state) is dict and type(state.get("directed")) is bool, "Invalid graph state")
    nodes, edges = state.get("nodes"), state.get("edges")
    _require(type(nodes) is list and all(type(v) is int for v in nodes)
             and len(set(nodes)) == len(nodes) and type(edges) is list, "Invalid nodes/edges")
    _require(not (weighted and state["directed"]), "Weighted TSP graphs must be undirected")
    graph = nx.DiGraph() if state["directed"] else nx.Graph()
    graph.add_nodes_from(nodes)
    for edge in edges:
        _require(type(edge) in (list, tuple) and len(edge) >= 2
                 and all(type(v) is int and v in graph for v in edge[:2])
                 and edge[0] != edge[1], "Invalid graph edge")
        if weighted:
            _require(len(edge) == 3 and type(edge[2]) is int and edge[2] > 0,
                     "TSP distances must be positive integers")
            graph.add_edge(edge[0], edge[1], weight=edge[2])
        else:
            graph.add_edge(edge[0], edge[1])
    graph.graph["identity_kind"] = (
        "weighted_undirected" if weighted else "directed" if graph.is_directed() else "undirected")
    return graph


def _fingerprint(graph):
    """Relabeling-invariant bucket, NEVER an exact isomorphism/canonical-form claim."""
    kind = graph.graph["identity_kind"]
    work = graph.copy()
    degrees = sorted((work.in_degree(v), work.out_degree(v)) for v in work) if work.is_directed() \
        else sorted(d for _, d in work.degree())
    for vertex in work:
        work.nodes[vertex]["bucket_degree"] = (
            f"{work.in_degree(vertex)},{work.out_degree(vertex)}"
            if work.is_directed() else str(work.degree(vertex)))
    weighted = kind == "weighted_undirected"
    wl = nx.weisfeiler_lehman_graph_hash(
        work, node_attr="bucket_degree", edge_attr="weight" if weighted else None, iterations=3)
    invariant = {"kind": kind, "n": len(work), "m": work.number_of_edges(), "degrees": degrees,
                 "weights": sorted(d["weight"] for _, _, d in work.edges(data=True)) if weighted else None,
                 "wl": wl}
    return hashlib.sha256(json.dumps(invariant, sort_keys=True).encode()).hexdigest()


def _isomorphic(left, right):
    if left.graph["identity_kind"] != right.graph["identity_kind"]:
        return False
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("weight", None) \
        if left.graph["identity_kind"] == "weighted_undirected" else None
    matcher = nx.algorithms.isomorphism.DiGraphMatcher if left.is_directed() \
        else nx.algorithms.isomorphism.GraphMatcher
    return matcher(left, right, edge_match=edge_match).is_isomorphic()


class _IsoIndex:
    """Every bucket hit is checked by exact VF2, including WL/hash collisions."""

    def __init__(self):
        self.buckets = defaultdict(list)
        self.comparisons = 0
        self.size = 0

    def find(self, graph):
        for representative, token in self.buckets.get(_fingerprint(graph), ()):
            self.comparisons += 1
            if _isomorphic(representative, graph):
                return token
        return None

    def add(self, graph, token):
        existing = self.find(graph)
        if existing is not None:
            return existing
        self.buckets[_fingerprint(graph)].append((graph.copy(), token))
        self.size += 1
        return token


def _partition(rng, n, count=2, minimum=2):
    sizes = [minimum] * count
    for _ in range(n - minimum * count):
        sizes[rng.randrange(count)] += 1
    nodes = rng.sample(range(n), n)
    groups, start = [], 0
    for size in sizes:
        groups.append(nodes[start:start + size])
        start += size
    return groups


def _connected_subgraph(graph, nodes, rng, probability):
    order = rng.sample(nodes, len(nodes))
    for i in range(1, len(order)):
        graph.add_edge(order[i], rng.choice(order[:i]))
    for u, v in combinations(nodes, 2):
        if rng.random() < probability:
            graph.add_edge(u, v)


def _blocks(rng, n, *, disconnected=False, bridge=False):
    count = 3 if n >= 8 and rng.random() < 0.4 and not bridge else 2
    groups = _partition(rng, n, count)
    graph = nx.empty_graph(n)
    probabilities = [rng.uniform(0.15, 0.9) for _ in groups]
    for group, probability in zip(groups, probabilities):
        _connected_subgraph(graph, group, rng, probability)
    between = 0 if disconnected or bridge else rng.uniform(0.03, 0.25)
    if bridge:
        graph.add_edge(rng.choice(groups[0]), rng.choice(groups[1]))
    elif not disconnected:
        for left, right in combinations(groups, 2):
            for u in left:
                for v in right:
                    if rng.random() < between:
                        graph.add_edge(u, v)
    return graph, {"block_sizes": [len(g) for g in groups],
                   "within_probabilities": probabilities, "between_probability": between,
                   "single_bridge": bridge}


def _core_periphery(rng, n, *, disconnected=False):
    graph = nx.empty_graph(n)
    nodes = rng.sample(range(n), n)
    island = nodes[-2:] if disconnected else []
    active = nodes[:-2] if disconnected else nodes
    core_size = rng.randint(2, max(2, len(active) - 2))
    core = set(active[:core_size])
    core_p, spoke_p, outer_p = rng.uniform(0.65, 1), rng.uniform(0.15, 0.6), rng.uniform(0.01, 0.2)
    for u, v in combinations(active, 2):
        p = core_p if u in core and v in core else spoke_p if u in core or v in core else outer_p
        if rng.random() < p:
            graph.add_edge(u, v)
    if island:
        graph.add_edge(*island)
    return graph, {"core_size": core_size, "island_size": len(island),
                   "core_probability": core_p, "spoke_probability": spoke_p,
                   "periphery_probability": outer_p}


def _geometric(rng, n, *, disconnected=False):
    points = rng.sample([(x, y) for x in range(16) for y in range(16)], n)
    if disconnected:
        split = rng.randint(2, n - 2)
        points = [(x + (40 if i >= split else 0), y) for i, (x, y) in enumerate(points)]
    radius = rng.randint(5, 10)
    graph = nx.empty_graph(n)
    for u, v in combinations(range(n), 2):
        if sum((a - b) ** 2 for a, b in zip(points[u], points[v])) <= radius ** 2:
            graph.add_edge(u, v)
    return graph, {"radius": radius, "coordinates": [list(p) for p in points]}


def _query_base(rng, n, task):
    if task == "articulation_point":
        if n < 7 or rng.random() < 0.35:
            graph, parameters = _blocks(rng, n, bridge=True)
            parameters["block_base"] = "general_connected"
        else:
            internal = rng.randint(1, min(3, n - 6))
            groups = _partition(rng, n - internal, 2, minimum=3)
            graph = nx.empty_graph(n)
            probabilities = [rng.uniform(0.05, 0.4) for _ in groups]
            for group, probability in zip(groups, probabilities):
                order = rng.sample(group, len(group))
                graph.add_edges_from(zip(order, order[1:] + order[:1]))
                for u, v in combinations(group, 2):
                    if rng.random() < probability:
                        graph.add_edge(u, v)
            path = [rng.choice(groups[0]), *range(n - internal, n), rng.choice(groups[1])]
            graph.add_edges_from(zip(path, path[1:]))
            graph = nx.relabel_nodes(graph, dict(zip(range(n), rng.sample(range(n), n))))
            parameters = {"block_sizes": [len(group) for group in groups],
                          "within_chord_probabilities": probabilities,
                          "bridge_internal_vertices": internal,
                          "block_base": "cycles_with_chords"}
        return graph, "articulated_heterogeneous_blocks", parameters
    family = rng.choice(("heterogeneous_blocks", "core_periphery", "geometric"))
    disconnected = task == "pair_connectivity"
    if family == "heterogeneous_blocks":
        graph, parameters = _blocks(rng, n, disconnected=disconnected,
                                    bridge=task == "distance_threshold")
    elif family == "core_periphery":
        graph, parameters = _core_periphery(rng, n, disconnected=disconnected)
    else:
        graph, parameters = _geometric(rng, n, disconnected=disconnected)
    return graph, family, parameters


def _forest_switch(rng, n):
    forest = nx.from_prufer_sequence([rng.randrange(n) for _ in range(n - 2)])
    removed = rng.randint(0, min(2, n - 5))
    forest.remove_edges_from(rng.sample(sorted(forest.edges()), removed))
    paths = [path for source, paths in nx.all_pairs_shortest_path(forest)
             for target, path in paths.items() if source < target and len(path) >= 5]
    if not paths:
        return None
    path = rng.choice(paths)
    u, a, b, v = path[0], path[1], path[-2], path[-1]
    switched = forest.copy()
    switched.remove_edges_from(((u, a), (b, v)))
    switched.add_edges_from(((u, v), (a, b)))
    _require(nx.is_forest(forest) and not nx.is_forest(switched)
             and forest.number_of_edges() == switched.number_of_edges()
             and sorted(dict(forest.degree()).values()) == sorted(dict(switched.degree()).values()),
             "Invalid degree-preserving cycle switch")
    parameters = {"removed_forest_edges": removed, "switch_path_length": len(path) - 1,
                  "matched_sorted_degrees": sorted(dict(forest.degree()).values())}
    return [(_state(forest), "general_random_forest", parameters),
            (_state(switched), "degree_preserving_forest_switch", parameters)]


def _tsp(rng, n):
    family = rng.choice(("clustered_manhattan", "ring_manhattan"))
    if family == "clustered_manhattan":
        centres = [(20, 20), (80 + rng.randrange(20), 25), (40, 80 + rng.randrange(20))]
        count = rng.choice((2, 3))
        centres = centres[:count]
        groups = _partition(rng, n, count, minimum=1)
        points = [None] * n
        for group, (x, y) in zip(groups, centres):
            offsets = rng.sample([(a, b) for a in range(-5, 6) for b in range(-5, 6)], len(group))
            for node, (a, b) in zip(group, offsets):
                points[node] = [x + a, y + b]
        parameters = {"cluster_sizes": [len(g) for g in groups], "centres": [list(c) for c in centres]}
    else:
        radius, phase = rng.randint(20, 60), rng.random() * 2 * math.pi
        points = []
        for i in range(n):
            angle = phase + 2 * math.pi * (i + rng.uniform(-0.2, 0.2)) / n
            r = radius + rng.randint(-3, 3)
            points.append([100 + round(r * math.cos(angle)), 100 + round(r * math.sin(angle))])
        if len({tuple(p) for p in points}) != n:
            return None
        rng.shuffle(points)
        parameters = {"nominal_radius": radius}
    edges = [[u, v, sum(abs(a - b) for a, b in zip(points[u], points[v]))]
             for u, v in combinations(range(n), 2)]
    return [({"nodes": list(range(n)), "edges": edges, "directed": False,
              "points": points, "metric": "Manhattan"}, family, parameters)]


def _layered(rng, n):
    count = rng.choice((3, 4)) if n >= 8 else 3
    layers = _partition(rng, n, count, minimum=1)
    graph = nx.empty_graph(n, create_using=nx.DiGraph)
    adjacent_p, skip_p = rng.uniform(0.3, 0.85), rng.uniform(0.03, 0.3)
    for i, left in enumerate(layers):
        for j in range(i + 1, len(layers)):
            for u in left:
                for v in layers[j]:
                    if rng.random() < (adjacent_p if j == i + 1 else skip_p):
                        graph.add_edge(u, v)
    state = _state(graph)
    state.update(threshold={"numerator": 1, "denominator": 2}, seed_count=2,
                 weights=[{"source": u, "target": v, "numerator": 1,
                           "denominator": graph.in_degree(v)} for u, v in state["edges"]])
    return [(state, "layered_directed", {"layer_sizes": [len(g) for g in layers],
                                       "adjacent_probability": adjacent_p,
                                       "skip_probability": skip_p})]


def _choose_query_pair(graph, yes, no, rng, *, vertex_query=False):
    def signature(item):
        return graph.degree(item) if vertex_query else tuple(sorted(graph.degree(v) for v in item))

    positive, negative = defaultdict(list), defaultdict(list)
    for item in yes:
        positive[signature(item)].append(item)
    for item in no:
        negative[signature(item)].append(item)
    shared = sorted(set(positive) & set(negative))
    if not shared:
        return None
    degree = rng.choice(shared)
    pair = rng.choice(positive[degree]), rng.choice(negative[degree])
    signatures = [signature(item) for item in pair]
    control = {
        "criterion": "vertex_degree" if vertex_query else "unordered_endpoint_degrees",
        "matched": True,
        "reason": "matched",
        "positive_signature": list(signatures[0]) if not vertex_query else signatures[0],
        "negative_signature": list(signatures[1]) if not vertex_query else signatures[1],
    }
    return pair, control


def _candidate(task, n, rng, target_degree=None):
    if task == "cycle_detection":
        return _forest_switch(rng, n)
    if task == "tsp_construct":
        return _tsp(rng, n)
    if task == "lt_influence_construct":
        return _layered(rng, n)
    if task == "maxcut_construct":
        if rng.random() < 0.5:
            k = rng.choice([k for k in (2, 4, 6) if k < n])
            probability = rng.uniform(0.15, 0.8)
            graph = nx.circulant_graph(n, range(1, k // 2 + 1))
            for offset in range(1, k // 2 + 1):
                for u in range(n):
                    v = (u + offset) % n
                    available = [w for w in graph if w != u and not graph.has_edge(u, w)]
                    if available and rng.random() < probability:
                        graph.remove_edge(u, v)
                        graph.add_edge(u, rng.choice(available))
            family, parameters = "small_world", {"k": k, "rewiring_probability": probability}
        else:
            degree = rng.choice([d for d in range(2, min(6, n - 2) + 1) if n * d % 2 == 0])
            # NetworkX random_regular_graph may internally retry without a bound.
            # A cyclic regular base plus bounded degree-preserving switches avoids that.
            graph = nx.circulant_graph(n, range(1, degree // 2 + 1))
            if degree % 2:
                graph.add_edges_from((u, u + n // 2) for u in range(n // 2))
            for _ in range(8 * n):
                (u, v), (a, b) = rng.sample(sorted(graph.edges()), 2)
                if len({u, v, a, b}) == 4 and not graph.has_edge(u, b) and not graph.has_edge(a, v):
                    graph.remove_edges_from(((u, v), (a, b)))
                    graph.add_edges_from(((u, b), (a, v)))
            family, parameters = "switched_regular", {"degree": degree, "switch_attempts": 8 * n}
        return [(_state(graph), family, parameters)]
    if task == "community_bipartition_construct":
        graph, parameters = _blocks(rng, n)
        if graph.number_of_edges() == 0:
            return None
        return [(_state(graph), "heterogeneous_blocks", parameters)]
    graph, family, parameters = _query_base(rng, n, task)
    if task == "degree_exact":
        vertex = rng.randrange(n)
        graph.remove_edges_from(list(graph.edges(vertex)))
        graph.add_edges_from((vertex, v) for v in rng.sample(
            [v for v in range(n) if v != vertex], target_degree))
        return [({**_state(graph), "vertex": vertex}, "degree_conditioned_" + family,
                 {**parameters, "target_degree": target_degree})]
    pairs = list(combinations(range(n), 2))
    if task == "adjacency":
        yes = [pair for pair in pairs if graph.has_edge(*pair)]
        no = [pair for pair in pairs if not graph.has_edge(*pair)]
        fields = lambda item: {"pair": list(item)}
    elif task == "pair_connectivity":
        yes = [pair for pair in pairs if nx.has_path(graph, *pair)]
        no = [pair for pair in pairs if not nx.has_path(graph, *pair)]
        fields = lambda item: {"pair": list(item)}
    elif task == "distance_threshold":
        distances = dict(nx.all_pairs_shortest_path_length(graph))
        finite = [(pair, distances[pair[0]][pair[1]]) for pair in pairs
                  if pair[1] in distances[pair[0]]]
        by_distance = defaultdict(list)
        for pair, distance in finite:
            by_distance[distance].append(pair)
        signatures = {
            distance: {tuple(sorted(graph.degree(v) for v in pair)) for pair in pairs}
            for distance, pairs in by_distance.items()}
        thresholds = [k for k in (2, 3, 4)
                      if signatures.get(k, set()) & signatures.get(k + 1, set())]
        if not thresholds:
            return None
        threshold = rng.choice(thresholds)
        yes, no = by_distance[threshold], by_distance[threshold + 1]
        fields = lambda item: {"pair": list(item), "threshold": threshold}
    elif task == "articulation_point":
        yes = sorted(nx.articulation_points(graph))
        no = sorted(set(graph) - set(yes))
        fields = lambda item: {"vertex": item}
    else:
        raise ValueError("Unsupported structural task")
    if not yes or not no:
        return None
    selected = _choose_query_pair(graph, yes, no, rng, vertex_query=task == "articulation_point")
    if selected is None:
        return None
    pair, control = selected
    return [({**_state(graph), **fields(item)}, family,
             {**parameters, "_degree_matching": control}) for item in pair]


def _difficulty(task, state):
    graph = _graph(state, weighted=task == "tsp_construct")
    result = {"component_count": nx.number_weakly_connected_components(graph)
              if graph.is_directed() else nx.number_connected_components(graph)}
    if graph.is_directed():
        result.update(zero_indegree=sum(graph.in_degree(v) == 0 for v in graph),
                      max_indegree=max(dict(graph.in_degree()).values()), seed_count=state["seed_count"])
    else:
        result.update(min_degree=min(dict(graph.degree()).values()),
                      max_degree=max(dict(graph.degree()).values()))
    if "pair" in state:
        source, target = state["pair"]
        connected = nx.has_path(graph, source, target)
        result.update(pair=state["pair"], endpoint_degrees=[graph.degree(source), graph.degree(target)],
                      shortest_distance=nx.shortest_path_length(graph, source, target)
                      if connected else None, distance_kind="finite" if connected else "disconnected")
    if "threshold" in state and task == "distance_threshold":
        result["threshold"] = state["threshold"]
        result["threshold_relation"] = (
            "at_threshold" if result["shortest_distance"] == state["threshold"] else
            "one_beyond_threshold" if result["shortest_distance"] == state["threshold"] + 1
            else "other")
    if "vertex" in state:
        result.update(vertex=state["vertex"], vertex_degree=graph.degree(state["vertex"]))
        if task == "articulation_point":
            after = graph.copy()
            after.remove_node(state["vertex"])
            result["components_after_removal"] = nx.number_connected_components(after)
    if task == "cycle_detection":
        result.update(cycle_rank=graph.number_of_edges() - len(graph) + result["component_count"],
                      sorted_degrees=sorted(dict(graph.degree()).values()))
    if task == "tsp_construct":
        distances = [data["weight"] for _, _, data in graph.edges(data=True)]
        result.update(min_distance=min(distances), max_distance=max(distances))
    return result


def build_structural_bank(seed=20260930, node_counts=(8, 9, 10, 11, 12),
                          repetitions=2, prior_instances=()):
    """Return validated seven-key records and JSON-safe generation provenance.

    Each cell gets at most MAX_ATTEMPTS_PER_CELL independent proposals. Binary
    cells are atomic pairs; infeasible quotas remain explicit shortfalls. Same
    graph/query pairs intentionally share a cluster. Within-task graph reuse
    across cells and prior isomorphs are rejected before oracle computation.
    """
    _require(type(seed) is int, "seed must be an integer")
    _require(type(repetitions) is int and repetitions > 0, "repetitions must be positive")
    sizes = tuple(node_counts)
    _require(bool(sizes) and all(type(n) is int and 5 <= n <= 12 for n in sizes)
             and len(set(sizes)) == len(sizes), "node_counts must be distinct integers from 5 to 12")
    _require(set(TASKS) <= set(tasks._QUESTIONS), "extended_tasks lacks structural-task support")
    rng, id_rng = random.Random(seed), random.Random(f"structural-ids:{seed}")
    prior, task_indexes, clusters = _IsoIndex(), {task: _IsoIndex() for task in TASKS}, _IsoIndex()
    prior_records = list(prior_instances)
    for i, record in enumerate(prior_records):
        _require(type(record) is dict and "state" in record and "task" in record,
                 "prior_instances must contain task records")
        # Every undirected prior contributes its topology, including a TSP's K_n.
        prior.add(_graph(record["state"]), f"prior-{i}-topology")
        if record["task"] == "tsp_construct":
            prior.add(_graph(record["state"], weighted=True), f"prior-{i}-weighted")
    instances, details, coverage, cluster_details = [], {}, [], {}
    all_rejects = Counter()
    for n in sizes:
        for repetition in range(repetitions):
            for task in TASKS:
                for target in range(n) if task == "degree_exact" else (None,):
                    requested = 2 if task in BINARY_TASKS else 1
                    cell = {"task": task, "n": n, "repetition": repetition,
                            "target_degree": target, "requested": requested, "generated": 0,
                            "attempts": 0, "rejections": {}}
                    rejects = Counter()
                    for attempt in range(MAX_ATTEMPTS_PER_CELL):
                        cell["attempts"] = attempt + 1
                        candidate = _candidate(task, n, rng, target)
                        if candidate is None:
                            rejects["unavailable_query_or_graph"] += 1
                            continue
                        _require(len(candidate) == requested, "Invalid candidate cell cardinality")
                        graphs = [_graph(state, weighted=task == "tsp_construct")
                                  for state, _, _ in candidate]
                        if any(prior.find(graph) is not None for graph in graphs):
                            rejects["prior_isomorphism"] += 1
                            continue
                        if any(task_indexes[task].find(graph) is not None for graph in graphs):
                            rejects["within_task_isomorphism"] += 1
                            continue
                        records = [tasks._record(task, 0, deepcopy(state)) for state, _, _ in candidate]
                        if task in BINARY_TASKS:
                            _require(sorted(r["private"]["answer"] for r in records) == ["no", "yes"],
                                     "Candidate binary cell is not balanced")
                        elif task == "degree_exact":
                            _require(records[0]["private"]["answer"] == target,
                                     "Candidate degree quota mismatch")
                        match_id = f"p-{id_rng.getrandbits(128):032x}" if requested == 2 else None
                        for record, (state, family, parameters), graph in zip(records, candidate, graphs):
                            record["id"] = f"s-{id_rng.getrandbits(128):032x}"
                            cluster = clusters.add(graph, f"c-{id_rng.getrandbits(128):032x}")
                            task_indexes[task].add(graph, cluster)
                            info = cluster_details.setdefault(cluster, {
                                "cluster_id": cluster, "fingerprint": _fingerprint(graph),
                                "identity_kind": graph.graph["identity_kind"],
                                "instance_ids": [], "tasks": []})
                            info["instance_ids"].append(record["id"])
                            if task not in info["tasks"]:
                                info["tasks"].append(task)
                            details[record["id"]] = {
                                "instance_id": record["id"], "cluster_id": cluster, "pair_id": match_id,
                                "task": task, "family": family, "n": n, "m": graph.number_of_edges(),
                                "node_count": n, "edge_count": graph.number_of_edges(),
                                "density": nx.density(graph), "repetition": repetition,
                                "family_parameters": deepcopy({
                                    key: value for key, value in parameters.items() if not key.startswith("_")}),
                                "task_parameters": _difficulty(task, state),
                                "degree_matching": deepcopy(parameters.get("_degree_matching")),
                                "label": record["private"].get("answer"),
                            }
                            instances.append(record)
                        cell["generated"] = requested
                        break
                    cell.update(rejections=dict(rejects), shortfall=requested - cell["generated"],
                                emitted=cell["generated"],
                                reason="complete" if cell["generated"] == requested
                                else "bounded_attempts_exhausted")
                    all_rejects.update(rejects)
                    coverage.append(cell)
    rng.shuffle(instances)
    validation = tasks.validate_instances(instances)
    metadata = {
        "schema_version": 1, "design": "structural", "seed": seed,
        "query_pair_protocol": "strict_degree_matched_v2",
        "evaluation_scope": "prior_pool_relative" if prior_records else "development_no_prior_pool",
        "prior_exclusion_applied": bool(prior_records),
        "node_counts": list(sizes), "repetitions": repetitions,
        "max_attempts_per_cell": MAX_ATTEMPTS_PER_CELL,
        "requested_instances": sum(c["requested"] for c in coverage),
        "generated_instances": len(instances),
        "shortfall_instances": sum(c["shortfall"] for c in coverage),
        "planned_queries_per_model": sum(i["query_budget"] for i in instances),
        "requested_queries_per_model": repetitions * sum(4 * n + 8 for n in sizes),
        "cases": {i["id"]: details[i["id"]] for i in instances},
        "records": [details[i["id"]] for i in instances],
        "clusters": list(cluster_details.values()), "coverage": coverage,
        "rejections": dict(all_rejects), "validation": validation,
        "isomorphism": {
            "method": "invariant/WL buckets followed by exact NetworkX VF2; hash equality is not identity",
            "networkx_version": nx.__version__,
            "prior_records": len(prior_records), "prior_graph_views": prior.size,
            "prior_vf2_comparisons": prior.comparisons,
            "within_task_vf2_comparisons": sum(index.comparisons for index in task_indexes.values()),
            "cluster_vf2_comparisons": clusters.comparisons,
            "scope": "All prior undirected topologies (including TSP), directed graphs, and weighted TSP "
                     "views; per-task new-bank exclusion; exact cross-task cluster sharing.",
        },
        "family_scope": "Generation procedures overlap in mathematical support; this is not a "
                        "proof of out-of-distribution generalization.",
    }
    json.dumps(metadata, allow_nan=False)
    return instances, metadata
