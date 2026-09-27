"""Offline, bounded graph task generation; no client or network operations.

Public API: build_instances(seed=20260926), build_curriculum(...),
next_request(instance, decisions), score(instance, decisions),
validate_instances(instances). Records are JSON-safe;
ONLY requests may be sent to models. The top-level ``private`` field contains
answers or an optimum/reference and a classical baseline, never prompt material.

The matrix has 80 instances, 256 decisions/model (1,792 for seven models):
16 each degree/cycle/connectivity and 8 each cut/TSP/LT/two-community modularity.
Decision IDs are strings, including decimal integer IDs for degree/vertices.
Histories must be lists of exact IDs: no stripping, coercion, or answer repair.
next_request raises ValueError on invalid histories; score instead reports them
as infeasible. Incomplete optimization histories have no objective or solution.
The TSP's last vertex and return edge are forced by the declared construction,
not a repair. All complete feasible solutions are evaluated, not reference-matched.

All optimization quality uses 1/(1+absolute_gap), a bounded additive-gap score in
the task's objective units, NOT a ratio to the optimum (modularity may be negative).
It is not intended for comparing objective units across tasks. Standard modularity
is optimized over AT MOST TWO communities, including the all-A partition.
Private references break ties deterministically; any optimal solution is accepted.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import combinations, permutations, product
import json
import math
import random

from src.clients.base import DecisionOption, DecisionRequest

__all__ = ["build_instances", "build_curriculum", "next_request", "score", "validate_instances"]

_EXACT = ("degree_exact", "cycle_detection", "pair_connectivity")
_BINARY_EXACT = ("cycle_detection", "pair_connectivity", "adjacency",
                 "distance_threshold", "articulation_point")
_ALL_EXACT = ("degree_exact", *_BINARY_EXACT)
_OPTIMIZATION = (
    "maxcut_construct", "tsp_construct", "lt_influence_construct",
    "community_bipartition_construct",
)
_QUESTIONS = {
    "adjacency": (
        "Are the two distinct vertices in state.pair joined by a direct edge "
        "in this simple undirected graph? Paths through other vertices do not "
        "count as direct edges. Return only yes or no."
    ),
    "distance_threshold": (
        "Is the shortest undirected path between the vertices in state.pair "
        "at most state.threshold edges long? Count edges, not vertices. "
        "If no path exists, answer no. Return only yes or no."
    ),
    "articulation_point": (
        "Is state.vertex an articulation point of this simple undirected graph? "
        "Remove that vertex and all its incident edges. Answer yes exactly when "
        "the number of connected components of the remaining graph is greater "
        "than the original graph's component count. Isolated vertices count as "
        "components. Return only yes or no."
    ),
    "degree_exact": (
        "In this simple undirected graph, what is the degree of state.vertex? "
        "Return only its decimal integer option ID."
    ),
    "cycle_detection": (
        "Does this simple undirected graph contain a cycle of at least three "
        "distinct vertices anywhere? Traversing one edge back and forth is not "
        "a cycle. Return only yes or no."
    ),
    "pair_connectivity": (
        "Is there an undirected path between the two vertices in state.pair? "
        "Return only yes or no."
    ),
    "maxcut_construct": (
        "Construct a maximum cut of the full unweighted undirected graph: "
        "maximize the number of edges whose endpoints have different sides. "
        "Vertex 0 is fixed to A. Assign state.current_vertex to A or B, "
        "preserving state.partial_partition; future vertices are assigned in "
        "increasing order. Return only A or B."
    ),
    "tsp_construct": (
        "Construct a minimum-length Hamiltonian tour of this complete undirected "
        "integer Manhattan graph. Each edge is [u,v,distance]. Start at 0; "
        "extend state.partial_tour by one unvisited vertex. Once one vertex "
        "remains it and the return to 0 are appended automatically, without "
        "another query. Return only the chosen vertex's decimal option ID."
    ),
    "lt_influence_construct": (
        "Choose two distinct seeds to maximize final active vertices under "
        "deterministic normalized linear-threshold diffusion, not random or "
        "independent-cascade influence. Each directed u->v edge has weight "
        "1/indegree(v), disclosed in state.weights; all thresholds are 1/2. "
        "Both seeds start active simultaneously. In synchronous irreversible "
        "rounds, an inactive vertex with positive indegree activates when at "
        "least half its incoming neighbors were active in the previous round "
        "(equality activates). Unseeded indegree-zero vertices stay inactive. "
        "Iterate to a fixpoint. Preserve state.selected_seeds and return only "
        "one remaining vertex's decimal option ID."
    ),
    "community_bipartition_construct": (
        "Maximize STANDARD unweighted undirected modularity over AT MOST TWO "
        "communities A/B only, NOT unconstrained community detection: "
        "Q=sum_C [internal_edges(C)/m - (sum_degrees(C)/(2*m))^2], where m is "
        "the full graph's edge count. One community (all A) is allowed. "
        "Vertex 0 is fixed to A. Assign state.current_vertex, preserving "
        "state.partial_partition; future vertices are assigned in increasing "
        "order. Return only A or B."
    ),
}
_STATE_KEYS = {
    task: {"nodes", "edges", "directed"} | extra
    for task, extra in (
        ("degree_exact", {"vertex"}), ("cycle_detection", set()),
        ("pair_connectivity", {"pair"}), ("maxcut_construct", set()),
        ("adjacency", {"pair"}), ("distance_threshold", {"pair", "threshold"}),
        ("articulation_point", {"vertex"}),
        ("tsp_construct", {"points", "metric"}),
        ("lt_influence_construct", {"weights", "threshold", "seed_count"}),
        ("community_bipartition_construct", set()),
    )
}
_BASELINES = {
    "maxcut_construct": "sequential-greedy-cut",
    "tsp_construct": "nearest-neighbor",
    "lt_influence_construct": "marginal-greedy",
    "community_bipartition_construct": "single-sweep-greedy-modularity",
}
_QUALITY = "1/(1+absolute_gap); additive gap in objective units"


def _budget(task: str, n: int) -> int:
    if task in _ALL_EXACT:
        return 1
    if task == "tsp_construct":
        return n - 2
    if task == "lt_influence_construct":
        return 2
    return n - 1


def _number(value: int | Fraction) -> int | float:
    return float(value) if isinstance(value, Fraction) else value


def _neighbors(state: dict) -> list[set[int]]:
    neighbors = [set() for _ in state["nodes"]]
    for u, v in state["edges"]:
        neighbors[u].add(v)
        neighbors[v].add(u)
    return neighbors


def _exact_answer(task: str, state: dict) -> str | int:
    neighbors = _neighbors(state)
    if task == "degree_exact":
        return len(neighbors[state["vertex"]])
    if task == "adjacency":
        source, target = state["pair"]
        return "yes" if target in neighbors[source] else "no"
    if task == "distance_threshold":
        source, target = state["pair"]
        seen, frontier = {source}, {source}
        for _ in range(state["threshold"]):
            frontier = {v for u in frontier for v in neighbors[u]} - seen
            if target in frontier:
                return "yes"
            seen |= frontier
        return "no"
    if task == "articulation_point":
        def components(removed):
            unseen = set(state["nodes"]) - removed
            count = 0
            while unseen:
                count += 1
                pending = [unseen.pop()]
                while pending:
                    reached = neighbors[pending.pop()] & unseen
                    unseen -= reached
                    pending.extend(reached)
            return count

        return "yes" if components({state["vertex"]}) > components(set()) else "no"
    if task == "pair_connectivity":
        source, target = state["pair"]
        seen, pending = {source}, [source]
        while pending:
            for v in sorted(neighbors[pending.pop()] - seen):
                seen.add(v)
                pending.append(v)
        return "yes" if target in seen else "no"
    # Union-find handles undirected edges once, so a single edge is not a cycle.
    parent = list(state["nodes"])

    def root(u):
        while u != parent[u]:
            u = parent[u]
        return u

    for u, v in state["edges"]:
        a, b = root(u), root(v)
        if a == b:
            return "yes"
        parent[a] = b
    return "no"


def _distances(state: dict) -> list[list[int]]:
    n = len(state["nodes"])
    matrix = [[0] * n for _ in range(n)]
    for u, v, distance in state["edges"]:
        matrix[u][v] = matrix[v][u] = distance
    return matrix


def _lt_active(state: dict, seeds: list[int] | tuple[int, ...]) -> set[int]:
    incoming = [set() for _ in state["nodes"]]
    for u, v in state["edges"]:
        incoming[v].add(u)
    active = set(seeds)
    while True:
        added = {v for v in state["nodes"] if v not in active and incoming[v]
                 and 2 * len(incoming[v] & active) >= len(incoming[v])}
        if not added:
            return active
        active |= added


def _objective(task: str, state: dict, solution: list) -> int | Fraction:
    if task == "maxcut_construct":
        return sum(solution[u] != solution[v] for u, v in state["edges"])
    if task == "tsp_construct":
        distances = _distances(state)
        return sum(distances[u][v] for u, v in zip(solution, solution[1:]))
    if task == "lt_influence_construct":
        return len(_lt_active(state, solution))
    m = len(state["edges"])
    degrees = [0] * len(state["nodes"])
    internal = {"A": 0, "B": 0}
    for u, v in state["edges"]:
        degrees[u] += 1
        degrees[v] += 1
        if solution[u] == solution[v]:
            internal[solution[u]] += 1
    # Exact rational standard modularity, not a cut or modularity surrogate.
    return sum((Fraction(internal[c], m) - Fraction(
        sum(d for v, d in enumerate(degrees) if solution[v] == c), 2 * m
    ) ** 2 for c in ("A", "B")), Fraction(0))


def _held_karp(state: dict) -> tuple[int, list[int]]:
    distances = _distances(state)
    n = len(distances)
    # (visited nonzero-node bitmask, endpoint) -> (length, path), lexical ties.
    dp = {(1 << (v - 1), v): (distances[0][v], (0, v)) for v in range(1, n)}
    for mask in range(1, 1 << (n - 1)):
        for v in range(1, n):
            bit = 1 << (v - 1)
            if not mask & bit or mask == bit:
                continue
            previous = mask ^ bit
            dp[mask, v] = min(
                (dp[previous, u][0] + distances[u][v], dp[previous, u][1] + (v,))
                for u in range(1, n) if previous & (1 << (u - 1))
            )
    length, tour = min(
        (dp[(1 << (n - 1)) - 1, v][0] + distances[v][0],
         dp[(1 << (n - 1)) - 1, v][1] + (0,)) for v in range(1, n)
    )
    return length, list(tour)


def _optimize(task: str, state: dict) -> tuple[int | Fraction, list]:
    if task == "tsp_construct":
        return _held_karp(state)
    n = len(state["nodes"])
    candidates = (combinations(range(n), 2) if task == "lt_influence_construct"
                  else (("A",) + tail for tail in product(("A", "B"), repeat=n - 1)))
    best_value, best_solution = None, None
    for candidate in candidates:
        solution = list(candidate)
        value = _objective(task, state, solution)
        if best_value is None or value > best_value:
            best_value, best_solution = value, solution
    return best_value, best_solution


def _baseline(task: str, state: dict) -> list:
    n = len(state["nodes"])
    if task == "tsp_construct":
        distances, tour, unvisited = _distances(state), [0], set(range(1, n))
        while unvisited:
            v = min(unvisited, key=lambda v: (distances[tour[-1]][v], v))
            tour.append(v)
            unvisited.remove(v)
        return tour + [0]
    if task == "lt_influence_construct":
        seeds = []
        for _ in range(2):
            seeds.append(min((v for v in range(n) if v not in seeds),
                             key=lambda v: (-len(_lt_active(state, seeds + [v])), v)))
        return seeds
    if task == "maxcut_construct":
        neighbors, sides = _neighbors(state), ["A"]
        for v in range(1, n):
            sides.append(min(("A", "B"), key=lambda side: (
                -sum(sides[u] != side for u in neighbors[v] if u < v), side)))
        return sides
    sides = ["A"] * n
    for v in range(1, n):
        candidate = sides.copy()
        candidate[v] = "B"
        if _objective(task, state, candidate) > _objective(task, state, sides):
            sides = candidate
    return sides


def _record(task: str, index: int, state: dict) -> dict:
    if task in _ALL_EXACT:
        private = {"answer": _exact_answer(task, state)}
    else:
        optimum, reference = _optimize(task, state)
        baseline = _baseline(task, state)
        private = {
            "objective": _number(optimum), "reference": reference,
            "baseline": {"method": _BASELINES[task], "solution": baseline,
                         "objective": _number(_objective(task, state, baseline))},
        }
    return {"id": f"{task}-{index:02d}", "task": task,
            "kind": "exact" if task in _ALL_EXACT else "optimization", "state": state,
            "query_budget": _budget(task, len(state["nodes"])), "private": private}


def _graph(n: int, edges, *, directed: bool = False) -> dict:
    return {"nodes": list(range(n)), "edges": [list(e) for e in sorted(set(edges))],
            "directed": directed}


def build_instances(seed: int = 20260926) -> list[dict]:
    """Build the fixed 80-instance pilot with a local RNG and bounded generation."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng, instances = random.Random(seed), []
    for task in _EXACT:
        for i in range(16):
            n = 8 if i < 8 else 12
            order = rng.sample(range(n), n)
            if task == "degree_exact":
                vertex = order[0]
                degree = i if i < 8 else (0, 1, 2, 4, 6, 8, 10, 11)[i - 8]
                edges = {tuple(sorted((vertex, v))) for v in order[1:degree + 1]}
                edges.update((u, v) for u, v in combinations(range(n), 2)
                             if vertex not in (u, v) and rng.random() < 0.25)
                state = {**_graph(n, edges), "vertex": vertex}
            elif task == "cycle_detection":
                edges = {tuple(sorted((order[j], rng.choice(order[:j]))))
                         for j in range(1, n) if rng.random() < 0.7}
                if i % 2:
                    edges.update(tuple(sorted(e)) for e in combinations(order[:3], 2))
                state = _graph(n, edges)
            else:
                split = rng.randrange(3, n - 2)
                groups, edges = (order[:split], order[split:]), set()
                for group in groups:
                    edges.update(tuple(sorted((group[j], rng.choice(group[:j]))))
                                 for j in range(1, len(group)))
                    edges.update(tuple(sorted(e)) for e in combinations(group, 2)
                                 if rng.random() < 0.2)
                pair = rng.sample(groups[0], 2) if i % 2 else [groups[0][0], groups[1][0]]
                if i % 4 == 1:
                    edges.add(tuple(sorted((groups[0][0], groups[1][0]))))
                state = {**_graph(n, edges), "pair": pair}
            instances.append(_record(task, i, state))
    for task in _OPTIMIZATION:
        for i in range(8):
            n = 8 if task in ("tsp_construct", "lt_influence_construct") else 10
            if task == "tsp_construct":
                points = [[p % 20, p // 20] for p in rng.sample(range(400), n)]
                edges = [(u, v, sum(abs(a - b) for a, b in zip(points[u], points[v])))
                         for u, v in combinations(range(n), 2)]
                state = {**_graph(n, edges), "points": points, "metric": "Manhattan"}
            elif task == "lt_influence_construct":
                edges = [(u, v) for u in range(n) for v in range(n)
                         if u != v and not (i % 4 == 0 and v == 0)
                         and rng.random() < (0.2 + 0.05 * (i % 4))]
                state = _graph(n, edges, directed=True)
                indegrees = [sum(v == target for _, v in edges) for target in range(n)]
                state.update(weights=[{"source": u, "target": v, "numerator": 1,
                                       "denominator": indegrees[v]} for u, v in state["edges"]],
                             threshold={"numerator": 1, "denominator": 2}, seed_count=2)
            else:
                group = set(rng.sample(range(n), n // 2))
                edges = [(u, v) for u, v in combinations(range(n), 2)
                         if rng.random() < (
                             0.3 + 0.05 * (i % 4) if task == "maxcut_construct"
                             else (0.65 if (u in group) == (v in group) else 0.12))]
                state = _graph(n, edges or [(0, 1)])
            instances.append(_record(task, i, state))
    return instances


def build_curriculum(
    seed: int = 20260928, node_counts: tuple[int, ...] = tuple(range(5, 13)),
    samples_per_size: int = 2,
) -> list[dict]:
    """Generate consecutive-size development strata, without model-based selection.

    Degree covers every possible answer at each size. Cycle pairs have identical
    degree multisets and edge counts (path versus cycle plus path). Connectivity
    pairs share a disconnected graph and query degree-one endpoints. Optimization
    uses two alternating structural regimes. LT rejects only constant-objective
    seed-selection instances, using an explicit bounded, oracle-only filter.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (not isinstance(node_counts, (list, tuple)) or not node_counts or
            any(type(n) is not int or not 5 <= n <= 12 for n in node_counts) or
            list(node_counts) != sorted(set(node_counts))):
        raise ValueError("node_counts must be distinct increasing integers in 5..12")
    if type(samples_per_size) is not int or not 1 <= samples_per_size <= 100:
        raise ValueError("samples_per_size must be an integer in 1..100")
    instances = []

    def add(task, n, repeat, variant, state):
        record = _record(task, len(instances), state)
        suffix = hashlib.sha256(f"{seed}:{task}:{n}:{repeat}:{variant}".encode()).hexdigest()[:12]
        record["id"] = f"curriculum-{task}-n{n:02d}-r{repeat:03d}-{suffix}"
        instances.append(record)

    for n in node_counts:
        for task in (*_EXACT, *_OPTIMIZATION):
            for repeat in range(samples_per_size):
                rng = random.Random(f"{seed}:{task}:{n}:{repeat}")
                order = rng.sample(range(n), n)
                if task == "degree_exact":
                    for degree in range(n):
                        vertex = order[0]
                        neighbors = rng.sample(order[1:], degree)
                        edges = {tuple(sorted((vertex, v))) for v in neighbors}
                        edges.update((u, v) for u, v in combinations(range(n), 2)
                                     if vertex not in (u, v) and rng.random() < 0.3)
                        add(task, n, repeat, f"d{degree:02d}",
                            {**_graph(n, edges), "vertex": vertex})
                elif task == "cycle_detection":
                    cycle_size = 3 if repeat % 2 == 0 else n - 2
                    path = list(zip(order, order[1:]))
                    ring = order[:cycle_size]
                    cyclic = list(zip(ring, ring[1:] + ring[:1]))
                    cyclic.extend(zip(order[cycle_size:], order[cycle_size + 1:]))
                    for variant, edges in (("no", path), ("yes", cyclic)):
                        add(task, n, repeat, variant,
                            _graph(n, [tuple(sorted(e)) for e in edges]))
                elif task == "pair_connectivity":
                    split = n // 2
                    groups = (order[:split], order[split:])
                    edges = [tuple(sorted(e)) for group in groups
                             for e in zip(group, group[1:])]
                    group = groups[repeat % 2]
                    for variant, pair in (
                        ("yes", [group[0], group[-1]]),
                        ("no", [groups[0][0], groups[1][-1]]),
                    ):
                        add(task, n, repeat, variant, {**_graph(n, edges), "pair": pair})
                elif task == "tsp_construct":
                    points = [[p % 30, p // 30] for p in rng.sample(range(900), n)]
                    edges = [(u, v, sum(abs(a - b) for a, b in zip(points[u], points[v])))
                             for u, v in combinations(range(n), 2)]
                    add(task, n, repeat, "manhattan",
                        {**_graph(n, edges), "points": points, "metric": "Manhattan"})
                elif task == "lt_influence_construct":
                    for _ in range(200):
                        edges = [(u, v) for u in range(n) for v in range(n)
                                 if u != v and rng.random() < (0.12, 0.25)[repeat % 2]]
                        state = _graph(n, edges, directed=True)
                        indegrees = [sum(v == target for _, v in edges) for target in range(n)]
                        state.update(
                            weights=[{"source": u, "target": v, "numerator": 1,
                                      "denominator": indegrees[v]} for u, v in state["edges"]],
                            threshold={"numerator": 1, "denominator": 2}, seed_count=2,
                        )
                        values = [len(_lt_active(state, seeds))
                                  for seeds in combinations(range(n), 2)]
                        if min(values) < max(values) and max(values) > 2:
                            break
                    else:
                        raise ValueError("LT nonconstant-objective generation budget exhausted")
                    add(task, n, repeat, f"density{repeat % 2}", state)
                elif task == "maxcut_construct":
                    edges = {tuple(sorted((order[j], rng.choice(order[:j]))))
                             for j in range(1, n)}
                    edges.update((u, v) for u, v in combinations(range(n), 2)
                                 if rng.random() < (0.15, 0.4)[repeat % 2])
                    add(task, n, repeat, f"density{repeat % 2}", _graph(n, edges))
                else:
                    group = set(order[:n // 2])
                    within, between = ((0.8, 0.15), (0.55, 0.35))[repeat % 2]
                    for _ in range(200):
                        edges = [(u, v) for u, v in combinations(range(n), 2)
                                 if rng.random() < (within if (u in group) == (v in group)
                                                    else between)]
                        if edges:
                            break
                    else:
                        raise ValueError("Community nonempty-graph generation budget exhausted")
                    add(task, n, repeat, f"mixing{repeat % 2}", _graph(n, edges))
    random.Random(f"{seed}:instance-order").shuffle(instances)
    return instances


def _option_ids(task: str, n: int, decisions: list[str]) -> list[str]:
    if task == "degree_exact":
        return [str(v) for v in range(n)]
    if task in _BINARY_EXACT:
        return ["yes", "no"]
    if task == "tsp_construct":
        return [str(v) for v in range(1, n) if str(v) not in decisions]
    if task == "lt_influence_construct":
        return [str(v) for v in range(n) if str(v) not in decisions]
    return ["A", "B"]


def _check_history(instance: dict, decisions: list[str]) -> None:
    if type(decisions) is not list or any(type(d) is not str for d in decisions):
        raise ValueError("decisions must be a list of exact string option IDs")
    task, n = instance["task"], len(instance["state"]["nodes"])
    if task not in _QUESTIONS:
        raise ValueError("unknown task")
    if len(decisions) > _budget(task, n):
        raise ValueError("decision history exceeds query budget")
    for i, decision in enumerate(decisions):
        if decision not in _option_ids(task, n, decisions[:i]):
            raise ValueError(f"invalid option at decision {i}")


def next_request(instance: dict, decisions: list[str]) -> DecisionRequest | None:
    """Return a fresh, allowlisted public request, or None for a complete history.

    Record integrity is checked separately by validate_instances; the history is
    always checked here. No oracle, reference, baseline, or scoring value is read.
    """
    _check_history(instance, decisions)
    task, n = instance["task"], len(instance["state"]["nodes"])
    budget = _budget(task, n)
    if len(decisions) == budget:
        return None
    state = {key: deepcopy(instance["state"][key]) for key in sorted(_STATE_KEYS[task])}
    state["remaining_queries"] = budget - len(decisions)
    if task in ("maxcut_construct", "community_bipartition_construct"):
        state.update(partial_partition=["A"] + decisions, current_vertex=len(decisions) + 1)
    elif task == "tsp_construct":
        tour = [0] + [int(d) for d in decisions]
        state.update(partial_tour=tour, current_vertex=tour[-1],
                     unvisited=[v for v in range(1, n) if v not in tour])
    elif task == "lt_influence_construct":
        state.update(selected_seeds=[int(d) for d in decisions],
                     remaining_seeds=2 - len(decisions))
    options = tuple(DecisionOption(id=d) for d in _option_ids(task, n, decisions))
    if not 2 <= len(options) <= 255:
        raise ValueError("request must have 2..255 options")
    return DecisionRequest(request_id=f"{instance['id']}:{len(decisions)}", state=state,
                           question=_QUESTIONS[task], options=options)


def _solution(task: str, state: dict, decisions: list[str]) -> list:
    if task == "tsp_construct":
        tour = [0] + [int(d) for d in decisions]
        return tour + [v for v in state["nodes"] if v not in tour] + [0]
    if task == "lt_influence_construct":
        return [int(d) for d in decisions]
    return ["A"] + decisions


def score(instance: dict, decisions: list[str]) -> dict:
    """JSON-safe metrics; invalid/incomplete input is infeasible, never repaired."""
    task, state = instance["task"], instance["state"]
    status = "complete"
    try:
        _check_history(instance, decisions)
    except ValueError:
        status = "invalid"
    else:
        if len(decisions) < _budget(task, len(state["nodes"])):
            status = "incomplete"
    feasible = status == "complete"
    if task in _ALL_EXACT:
        return {"feasible": feasible, "correct": feasible and (
            decisions[0] == str(instance["private"]["answer"])), "status": status}
    optimum = instance["private"]["objective"]
    solution = _solution(task, state, decisions) if feasible else None
    objective = _number(_objective(task, state, solution)) if feasible else None
    gap = abs(optimum - objective) if feasible else None
    return {"feasible": feasible, "status": status, "solution": solution,
            "objective": objective, "optimum": optimum,
            "direction": "minimize" if task == "tsp_construct" else "maximize",
            "absolute_gap": gap, "normalized_quality": 1 / (1 + gap) if feasible else None,
            "quality_definition": _QUALITY,
            "optimal": feasible and math.isclose(gap, 0, rel_tol=0, abs_tol=1e-12)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_state(task: str, state: dict) -> None:
    _require(type(state) is dict and set(state) == _STATE_KEYS[task], "invalid state keys")
    nodes = state["nodes"]
    _require(type(nodes) is list and all(type(v) is int for v in nodes), "invalid nodes")
    n = len(nodes)
    _require(5 <= n <= 12 and nodes == list(range(n)), "invalid bounded node count/IDs")
    directed = task == "lt_influence_construct"
    _require(type(state["directed"]) is bool and state["directed"] == directed,
             "invalid graph direction")
    edges = state["edges"]
    _require(type(edges) is list, "edges must be a list")
    for edge in edges:
        _require(type(edge) is list and len(edge) == (3 if task == "tsp_construct" else 2)
                 and all(type(v) is int for v in edge), "invalid edge")
        u, v = edge[:2]
        _require(u in nodes and v in nodes and u != v and (directed or u < v),
                 "invalid edge endpoints")
    _require(edges == sorted(edges) and len({tuple(e[:2]) for e in edges}) == len(edges),
             "edges must be sorted and simple")
    if task in ("degree_exact", "articulation_point"):
        _require(type(state["vertex"]) is int and state["vertex"] in nodes, "invalid vertex")
    if task in ("pair_connectivity", "adjacency", "distance_threshold"):
        pair = state["pair"]
        _require(type(pair) is list and len(pair) == 2 and
                 all(type(v) is int and v in nodes for v in pair) and pair[0] != pair[1],
                 "invalid pair")
    if task == "distance_threshold":
        _require(type(state["threshold"]) is int and 1 <= state["threshold"] < n,
                 "invalid distance threshold")
    if task == "community_bipartition_construct":
        _require(bool(edges), "modularity requires a nonempty graph")
    if task == "tsp_construct":
        points = state["points"]
        _require(type(points) is list and len(points) == n and all(
            type(p) is list and len(p) == 2 and all(type(c) is int for c in p)
            for p in points), "invalid Manhattan points")
        _require(len({tuple(p) for p in points}) == n and state["metric"] == "Manhattan",
                 "points must be distinct Manhattan coordinates")
        expected = [[u, v, abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])]
                    for u, v in combinations(nodes, 2)]
        _require(edges == expected, "invalid complete Manhattan distances")
    if directed:
        expected = [{"source": u, "target": v, "numerator": 1,
                     "denominator": sum(b == v for _, b in edges)} for u, v in edges]
        _require(state["weights"] == expected and state["threshold"] == {
            "numerator": 1, "denominator": 2} and type(state["seed_count"]) is int
            and state["seed_count"] == 2, "invalid normalized LT parameters")


def _validate_solution(task: str, state: dict, solution: list) -> None:
    n = len(state["nodes"])
    _require(type(solution) is list, "solution must be a list")
    if task == "tsp_construct":
        _require(all(type(v) is int for v in solution) and len(solution) == n + 1 and
                 solution[0] == solution[-1] == 0 and sorted(solution[:-1]) == list(range(n)),
                 "invalid reference/baseline tour")
    elif task == "lt_influence_construct":
        _require(len(solution) == 2 and all(type(v) is int and 0 <= v < n for v in solution)
                 and solution[0] != solution[1], "invalid reference/baseline seeds")
    else:
        _require(len(solution) == n and solution[0] == "A" and all(
            type(v) is str and v in ("A", "B") for v in solution), "invalid partition")


def _independent_lt(state: dict, seeds) -> set[int]:
    """Rational-weight simulation, separate from integer neighbor-count scoring."""
    active = set(seeds)
    threshold = Fraction(state["threshold"]["numerator"], state["threshold"]["denominator"])
    for _ in state["nodes"]:  # at most n synchronous growth rounds
        totals = [Fraction(0) for _ in state["nodes"]]
        for edge in state["weights"]:
            if edge["source"] in active:
                totals[edge["target"]] += Fraction(edge["numerator"], edge["denominator"])
        updated = active | {v for v, total in enumerate(totals) if total >= threshold}
        if updated == active:
            return active
        active = updated
    return active


def _independent_objectives(task: str, state: dict):
    """Enumerate using independent formulas/algorithms (never _objective)."""
    import networkx as nx

    nodes, edges = state["nodes"], state["edges"]
    if task == "tsp_construct":
        # Use coordinates rather than the production edge-distance helper/DP.
        points = state["points"]

        def evaluate(tour):
            return sum(abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])
                       for u, v in zip(tour, tour[1:]))

        candidates = ([0, *tail, 0] for tail in permutations(nodes[1:]))
    elif task == "lt_influence_construct":
        def evaluate(seeds):
            return len(_independent_lt(state, seeds))

        candidates = (list(pair) for pair in combinations(nodes, 2))
    else:
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        def evaluate(sides):
            groups = [{v for v in nodes if sides[v] == c} for c in ("A", "B")]
            if task == "maxcut_construct":
                return nx.cut_size(graph, groups[0], groups[1])
            return nx.community.modularity(graph, [g for g in groups if g], weight=None)

        # Subset enumeration instead of the production Cartesian-product search.
        candidates = (["A"] + ["B" if mask & (1 << (v - 1)) else "A" for v in nodes[1:]]
                      for mask in range(1 << (len(nodes) - 1)))
    return evaluate, candidates


def _independent_tsp_optimum(state: dict) -> int:
    """Forward subset DP from coordinates; no production distance/path tables."""
    points, n = state["points"], len(state["nodes"])
    costs = {(1, 0): 0}
    for mask in range(1, 1 << n, 2):
        for last in range(n):
            cost = costs.get((mask, last))
            if cost is None:
                continue
            for nxt in range(1, n):
                if mask & (1 << nxt):
                    continue
                key = mask | (1 << nxt), nxt
                distance = sum(abs(a - b) for a, b in zip(points[last], points[nxt]))
                candidate = cost + distance
                if key not in costs or candidate < costs[key]:
                    costs[key] = candidate
    return min(costs[((1 << n) - 1, v)] +
               sum(abs(a - b) for a, b in zip(points[v], points[0])) for v in range(1, n))


def validate_instances(instances: list[dict]) -> dict:
    """Raise ValueError on corrupt records; return JSON-safe count/budget totals.

    Recompute truth using NetworkX, TSP optimum by permutation search through n=9
    and independent forward DP above n=9, all LT seed sets with Fraction weights,
    and every fixed-0 partition with NetworkX cut/modularity objectives. Check
    every candidate against the production evaluator and validate both reference
    and deterministic baseline. Subsets are accepted; this is not a quota checker.
    """
    import networkx as nx

    _require(type(instances) is list, "instances must be a list")
    try:
        json.dumps(instances, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("records must be finite JSON") from exc
    seen, counts, queries = set(), {}, 0
    for instance in instances:
        _require(type(instance) is dict and set(instance) == {
            "id", "task", "kind", "state", "query_budget", "private"}, "invalid record keys")
        task, state, private = instance["task"], instance["state"], instance["private"]
        _require(type(task) is str and task in _QUESTIONS, "unknown task")
        _require(type(instance["id"]) is str and bool(instance["id"]) and
                 instance["id"] not in seen, "invalid/duplicate instance ID")
        seen.add(instance["id"])
        _validate_state(task, state)
        _require(instance["kind"] == ("exact" if task in _ALL_EXACT else "optimization"),
                 "incorrect kind")
        _require(type(instance["query_budget"]) is int and instance["query_budget"] ==
                 _budget(task, len(state["nodes"])), "incorrect query budget")
        _require(type(private) is dict, "invalid private oracle")
        if task in _ALL_EXACT:
            _require(set(private) == {"answer"}, "invalid exact oracle keys")
            graph = nx.Graph()
            graph.add_nodes_from(state["nodes"])
            graph.add_edges_from(state["edges"])
            if task == "degree_exact":
                answer = graph.degree[state["vertex"]]
            else:
                if task == "cycle_detection":
                    positive = bool(nx.cycle_basis(graph))
                elif task == "articulation_point":
                    positive = state["vertex"] in set(nx.articulation_points(graph))
                elif task == "adjacency":
                    positive = graph.has_edge(*state["pair"])
                elif task == "distance_threshold":
                    distances = nx.single_source_shortest_path_length(
                        graph, state["pair"][0], cutoff=state["threshold"])
                    positive = state["pair"][1] in distances
                else:
                    positive = nx.has_path(graph, *state["pair"])
                answer = "yes" if positive else "no"
            _require(type(private["answer"]) is type(answer) and private["answer"] == answer
                     and _exact_answer(task, state) == answer, "incorrect exact answer")
        else:
            _require(set(private) == {"objective", "reference", "baseline"},
                     "invalid optimization oracle keys")
            baseline = private["baseline"]
            _require(type(baseline) is dict and set(baseline) == {"method", "solution", "objective"},
                     "invalid baseline keys")
            _validate_solution(task, state, private["reference"])
            _validate_solution(task, state, baseline["solution"])
            evaluate, candidates = _independent_objectives(task, state)
            values = []
            if task == "tsp_construct" and len(state["nodes"]) > 9:
                optimum = _independent_tsp_optimum(state)
                candidates = [private["reference"], private["baseline"]["solution"]]
            for candidate in candidates:
                value = evaluate(candidate)
                _require(math.isclose(value, _number(_objective(task, state, candidate)),
                                      rel_tol=0, abs_tol=1e-12), "objective implementation mismatch")
                values.append(value)
            if task != "tsp_construct" or len(state["nodes"]) <= 9:
                optimum = min(values) if task == "tsp_construct" else max(values)
            _require(baseline["method"] == _BASELINES[task], "incorrect baseline method")
            for solution, stored in ((private["reference"], private["objective"]),
                                     (baseline["solution"], baseline["objective"])):
                _require(type(stored) in (int, float) and math.isfinite(stored) and
                         math.isclose(stored, evaluate(solution), rel_tol=0, abs_tol=1e-12),
                         "incorrect stored objective")
            _require(math.isclose(private["objective"], optimum, rel_tol=0, abs_tol=1e-12),
                     "incorrect optimum/reference")
            _require(baseline["solution"] == _baseline(task, state), "incorrect deterministic baseline")
        counts[task] = counts.get(task, 0) + 1
        queries += instance["query_budget"]
    return {"instances": len(instances), "queries_per_model": queries, "by_task": counts}