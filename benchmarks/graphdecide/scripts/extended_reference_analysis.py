#!/usr/bin/env python3
"""Offline stronger heuristics and exact uniform-policy expectations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.benchmark import extended_tasks as tasks
from scripts.paired_graph_ablation import digest, read_rows, write_json


def uniform_objective(instance: dict) -> float:
    """Expected objective of uniform legal sequential decisions, not a sample."""
    task, state = instance["task"], instance["state"]
    n, edges = len(state["nodes"]), state["edges"]
    if task == "maxcut_construct":
        return len(edges) / 2
    if task == "tsp_construct":
        return 2 * sum(e[2] for e in edges) / (n - 1)
    if task == "lt_influence_construct":
        values = [len(tasks._lt_active(state, seeds)) for seeds in combinations(range(n), 2)]
        return sum(values) / len(values)
    if task == "community_bipartition_construct":
        degrees = [sum(v in e for e in edges) for v in state["nodes"]]
        return -sum(d * d for d in degrees) / (8 * len(edges) ** 2)
    raise ValueError("Uniform objective requires an optimization task")


def improved_solution(instance: dict) -> tuple[str, list]:
    """Deterministic full-solution local search, explicitly not online decisions."""
    task, state = instance["task"], instance["state"]
    n = len(state["nodes"])
    solution = instance["private"]["baseline"]["solution"].copy()
    if task == "lt_influence_construct":
        return "marginal-greedy", solution
    if task == "tsp_construct":
        method = "nearest-neighbor-plus-best-improvement-2opt"
    elif task == "maxcut_construct":
        method = "greedy-plus-best-single-vertex-flip"
    elif task == "community_bipartition_construct":
        method = "deterministic-multistart-best-single-vertex-flip"
    else:
        raise ValueError("Local search requires an optimization task")
    starts = [solution]
    if task == "community_bipartition_construct":
        # All-A is a one-flip local maximum; singleton starts avoid that trap.
        starts.extend(["B" if u == v else "A" for u in range(n)] for v in range(1, n))

    def improve(current):
        value = tasks._objective(task, state, current)
        while True:
            best_value, best = value, current
            if task == "tsp_construct":
                candidates = (current[:a] + current[a:b + 1][::-1] + current[b + 1:]
                              for a in range(1, n - 1) for b in range(a + 1, n))
            else:
                candidates = (current[:v] + ["B" if current[v] == "A" else "A"] + current[v + 1:]
                              for v in range(1, n))
            for candidate in candidates:
                objective = tasks._objective(task, state, candidate)
                better = objective < best_value if task == "tsp_construct" else objective > best_value
                if better:
                    best_value, best = objective, candidate
            if best is current:
                return value, current
            value, current = best_value, best

    candidates = [improve(start) for start in starts]
    best = min(candidates) if task == "tsp_construct" else max(candidates, key=lambda item: item[0])
    return method, best[1]


def analyze(instances: list[dict]) -> dict:
    rows, grouped = [], defaultdict(list)
    for instance in instances:
        if instance["kind"] != "optimization":
            continue
        task, state, private = instance["task"], instance["state"], instance["private"]
        method, solution = improved_solution(instance)
        objective = tasks._number(tasks._objective(task, state, solution))
        uniform = uniform_objective(instance)
        row = {
            "instance_id": instance["id"], "task": task, "n": len(state["nodes"]),
            "optimum": private["objective"],
            "uniform_expected_objective": uniform,
            "uniform_expected_gap": abs(uniform - private["objective"]),
            "greedy_objective": private["baseline"]["objective"],
            "improved_method": method, "improved_solution": solution,
            "improved_objective": objective,
            "improved_gap": abs(objective - private["objective"]),
        }
        rows.append(row)
        grouped[task].append(row)
    summary = {
        task: {
            "instances": len(group),
            "uniform_mean_expected_gap": sum(r["uniform_expected_gap"] for r in group) / len(group),
            "greedy_mean_gap": sum(abs(r["greedy_objective"] - r["optimum"]) for r in group) / len(group),
            "improved_mean_gap": sum(r["improved_gap"] for r in group) / len(group),
            "improved_optimal": sum(r["improved_gap"] < 1e-12 for r in group),
        } for task, group in grouped.items()
    }
    return {
        "protocol": "offline-full-solution-heuristics-v1",
        "note": "Local search can revise decisions; not compute/action-budget matched to model construction.",
        "uniform_policy": "Independent uniform legal actions; exact objective expectation, not sampled.",
        "summary": summary, "instances": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    instances = read_rows(args.root / "instances.jsonl")
    tasks.validate_instances(instances)
    report = analyze(instances)
    report["provenance"] = {
        "instances_sha256": digest(args.root / "instances.jsonl"),
        "code_sha256": {str(path.relative_to(REPO)): digest(path) for path in (
            Path(__file__).resolve(), REPO / "src/benchmark/extended_tasks.py")},
    }
    write_json(args.root / "classical-references.json", report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
