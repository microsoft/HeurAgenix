#!/usr/bin/env python3
"""Replay and paired-graph analysis of sealed TSP wording experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import tsp_state_rollout as rollout
from scripts.extended_graph_suite import digest, json_value, read_json, read_rows, require, write_json
from src.benchmark import extended_tasks as tasks


def interval(values, *, repetitions=5000):
    if not values:
        return {"paired_instances": 0, "mean": None, "percentile_95": None}
    groups = defaultdict(list)
    for size, value in values:
        groups[size].append(value)
    rng = random.Random(20261007)
    samples = sorted(sum(sum(rng.choice(cell) for _ in cell) for cell in groups.values())
                     / len(values) for _ in range(repetitions))
    return {"paired_instances": len(values), "by_size": {n: len(v) for n, v in groups.items()},
            "mean": sum(v for _, v in values)/len(values),
            "percentile_95": [samples[int(.025*repetitions)], samples[int(.975*repetitions)]]}


def analyze(root):
    manifest = read_json(root / "manifest.json")
    require(digest(root / "manifest.json") == read_json(root / "manifest.sha256"), "Manifest drift.")
    instances = {i["id"]: i for i in manifest["instances"]}
    results = {}
    for model in manifest["models"]:
        status = read_json(root / f"{model}-status.json")
        require(status["status"].startswith("completed"), "Incomplete rollout.")
        require(all(digest(root / name) == sha for name, sha in status["artifact_sha256"].items()),
                "Artifact drift.")
        histories, closed, strict_histories, strict_closed = defaultdict(list), set(), defaultdict(list), set()
        attempts = read_rows(root / f"{model}-attempts.jsonl")
        require(len(attempts) == status["calls"], "Call count differs.")
        for row in attempts:
            key = row["instance_id"], row["arm"]
            require(key not in closed, "Call continued after an episode failure.")
            history = histories[key]
            request = tasks.next_request(instances[key[0]], history)
            require(request is not None, "Call exceeds episode budget.")
            if key[1] == "state_explicit":
                request = rollout.controls.state_explicit_tsp_request(request)
            request = replace(request, request_id=f"{request.request_id}:{key[1]}")
            require(json_value(asdict(request)) == row["request"], "Request/history mismatch.")
            require(row["step"] == len(history), "Step mismatch.")
            if row["status"] == "success":
                require(row["selected_option_id"] in {o.id for o in request.options}, "Illegal action.")
                history.append(row["selected_option_id"])
            else:
                closed.add(key)
            if key not in strict_closed:
                if row.get("strict_status", row["status"]) == "success":
                    strict_histories[key].append(row["selected_option_id"])
                else:
                    strict_closed.add(key)
        rows = read_rows(root / f"{model}-scores.jsonl")
        expected = {(i, arm) for i in instances for arm in rollout.ARMS}
        require(set(histories) == expected, "Completed lane skipped a scheduled episode.")
        require(len(rows) == len(expected) and {(r["instance_id"], r["arm"]) for r in rows} == expected,
                "Scheduled episodes differ.")
        by_arm = {arm: {} for arm in rollout.ARMS}
        projections = defaultdict(list)
        for row in rows:
            key = row["instance_id"], row["arm"]
            instance = instances[key[0]]
            score = tasks.score(instance, histories[key])
            require(row["decisions"] == histories[key] and all(row[k] == v for k, v in score.items()),
                    "Saved score differs from replay.")
            if score["feasible"]:
                tour = score["solution"]
                distances = {(min(u, v), max(u, v)): d for u, v, d in instance["state"]["edges"]}
                require(tour[0] == tour[-1] == 0
                        and sorted(tour[:-1]) == instance["state"]["nodes"], "Invalid tour.")
                require(sum(distances[min(u, v), max(u, v)] for u, v in zip(tour, tour[1:])) ==
                        score["objective"], "Independent tour-length mismatch.")
            by_arm[key[1]][key[0]] = row
            projections[key[1]].append(tasks.score(instance, strict_histories[key]))
        optimum, completion, gaps = [], [], []
        for ident, instance in instances.items():
            a, b = by_arm["baseline"][ident], by_arm["state_explicit"][ident]
            size = len(instance["state"]["nodes"])
            optimum.append((size, int(b["optimal"])-int(a["optimal"])))
            completion.append((size, int(b["feasible"])-int(a["feasible"])))
            if a["feasible"] and b["feasible"]:
                gaps.append((size, (a["absolute_gap"]-b["absolute_gap"])/a["optimum"]))
        results[model] = {
            "attempts": len(attempts), "replay_and_independent_tour_mismatches": 0,
            "optimum_hit_improvement": interval(optimum),
            "completion_improvement": interval(completion),
            "relative_gap_reduction_common_feasible": interval(gaps),
            "strict_projection_of_observed_traces": {
                arm: {"scheduled": len(cell), "feasible": sum(r["feasible"] for r in cell),
                      "optimal": sum(r["optimal"] for r in cell)}
                for arm, cell in projections.items()},
        }
    return {
        "manifest_sha256": digest(root / "manifest.json"), "analysis_sha256": digest(__file__),
        "warning": "Pointwise paired-graph, size-stratified empirical intervals. Conditional gaps "
                   "do not impute failed episodes. A strict projection truncates these NEW observed "
                   "traces at the first strict failure; it never replaces earlier experimental scores.",
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    write_json(args.root / "paired-analysis.json", result)
    print(json.dumps(result["models"], sort_keys=True))


if __name__ == "__main__":
    main()
