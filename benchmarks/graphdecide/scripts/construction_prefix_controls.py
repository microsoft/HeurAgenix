#!/usr/bin/env python3
"""Prospective action-interface controls, not free-running construction scores."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from functools import lru_cache
from itertools import product
import math
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_action_controls as runner
from scripts import extended_graph_suite as suite
from scripts.summarize_extended_paper import verified_inputs
from scripts.graph_task_prompt import state_explicit_tsp_request
from src.benchmark import extended_tasks as tasks
from src.clients.base import DecisionOption

ARMS = ("baseline", "repeat_baseline", "reversed_options", "described_options")


def continuation_values(instance, history):
    request = tasks.next_request(instance, history)
    suite.require(request is not None and instance["kind"] == "optimization",
                  "Control needs an unfinished construction episode.")
    task, state = instance["task"], instance["state"]
    if task == "tsp_construct":
        distances = tasks._distances(state)
        path = [0, *map(int, history)]
        remaining = tuple(v for v in state["nodes"] if v not in path)
        prefix_cost = sum(distances[u][v] for u, v in zip(path, path[1:]))

        @lru_cache(None)
        def finish(last, unvisited):
            if not unvisited:
                return distances[last][0]
            return min(distances[last][v] + finish(
                v, tuple(w for w in unvisited if w != v)) for v in unvisited)

        return {str(v): prefix_cost + distances[path[-1]][v] + finish(
            v, tuple(w for w in remaining if w != v)) for v in remaining}
    values = {}
    for option in request.options:
        if task == "lt_influence_construct":
            selected = [*map(int, history), int(option.id)]
            solutions = ([selected] if len(selected) == 2 else
                         [selected + [v] for v in state["nodes"] if v not in selected])
        else:
            length = instance["query_budget"] - len(history) - 1
            solutions = (["A", *history, option.id, *tail]
                         for tail in product(("A", "B"), repeat=length))
        values[option.id] = max(tasks._objective(task, state, solution) for solution in solutions)
    return values


def optimal_history(instance):
    reference = instance["private"]["reference"]
    if instance["task"] == "tsp_construct":
        return [str(v) for v in reference[1:-2]]
    if instance["task"] == "lt_influence_construct":
        return [str(v) for v in reference]
    return reference[1:]


def build_controls(instances):
    controls, coverage = [], []
    for task in tasks._OPTIMIZATION:
        for size in (5, 8, 12):
            candidates = sorted((i for i in instances if i["task"] == task
                                 and len(i["state"]["nodes"]) == size), key=lambda i: i["id"])
            suite.require(len(candidates) >= 2, "Need two development instances per task/size.")
            for instance in candidates[:2]:
                full_history = optimal_history(instance)
                suite.require(tasks.score(instance, full_history)["optimal"],
                              "Reference trajectory is not optimal.")
                informative = []
                for step in range(instance["query_budget"]):
                    history = full_history[:step]
                    values = continuation_values(instance, history)
                    best = min(values.values()) if task == "tsp_construct" else max(values.values())
                    suite.require(math.isclose(float(best), instance["private"]["objective"],
                                               rel_tol=0, abs_tol=1e-12),
                                  "Continuation oracle disagrees with frozen optimum.")
                    if len(set(values.values())) > 1:
                        informative.append((step, history, values, best))
                coverage.append({"instance_id": instance["id"], "task": task, "node_count": size,
                                 "informative_prefixes": len(informative),
                                 "reason": "covered" if informative else "all_actions_tied"})
                if not informative:
                    continue
                selected = [informative[0]]
                if informative[-1][0] != informative[0][0]:
                    selected.append(informative[-1])
                for index, (step, history, values, best) in enumerate(selected):
                    stage = ("early_and_late" if len(selected) == 1 else
                             ("early" if index == 0 else "late"))
                    request = tasks.next_request(instance, history)
                    for arm in ARMS:
                        options = request.options
                        if arm == "reversed_options":
                            options = options[::-1]
                        elif arm == "described_options":
                            if task in ("maxcut_construct", "community_bipartition_construct"):
                                descriptions = {o.id: f"Assign vertex {request.state['current_vertex']} "
                                                f"to side {o.id}." for o in options}
                            elif task == "tsp_construct":
                                descriptions = {o.id: f"Visit vertex {o.id} next." for o in options}
                            else:
                                descriptions = {o.id: f"Select vertex {o.id} as a seed." for o in options}
                            options = tuple(DecisionOption(o.id, descriptions[o.id]) for o in options)
                        controls.append((replace(
                            request, request_id=f"{request.request_id}:{arm}", options=options), {
                                "task": task, "case": instance["id"], "node_count": size,
                                "stage": stage, "step": step, "arm": arm,
                                "optimal_option_ids": sorted(k for k, v in values.items() if v == best),
                                "continuation_values": {k: tasks._number(v) for k, v in values.items()},
                                "optimal_continuation": tasks._number(best),
                            }))
    random.Random(20261003).shuffle(controls)
    return controls, coverage


def build_tsp_state_controls(instances):
    controls, coverage = [], []
    for size in (6, 9, 11):
        selected = sorted((i for i in instances if i["task"] == "tsp_construct"
                           and len(i["state"]["nodes"]) == size), key=lambda i: i["id"])[:2]
        suite.require(len(selected) == 2, "Need two fresh TSP instances per size.")
        for instance in selected:
            reference = optimal_history(instance)
            for step in reversed(range(instance["query_budget"])):
                history = reference[:step]
                values = continuation_values(instance, history)
                if len(set(values.values())) > 1:
                    break
            else:
                raise suite.Refusal("No informative TSP prefix in the fixed selection.")
            request = tasks.next_request(instance, history)
            best = min(values.values())
            suite.require(math.isclose(best, instance["private"]["objective"],
                                       rel_tol=0, abs_tol=1e-12), "TSP continuation mismatch.")
            coverage.append({"instance_id": instance["id"], "node_count": size, "step": step})
            for arm in ("baseline", "repeat_baseline", "state_explicit"):
                variant = state_explicit_tsp_request(request) if arm == "state_explicit" else request
                controls.append((replace(variant,
                                         request_id=f"{request.request_id}:{arm}"), {
                    "task": "tsp_construct", "case": instance["id"], "node_count": size,
                    "stage": "late", "step": step, "arm": arm,
                    "optimal_option_ids": sorted(k for k, v in values.items() if v == best),
                    "continuation_values": values, "optimal_continuation": best,
                }))
    random.Random(20261004).shuffle(controls)
    return controls, coverage


def summarize(root):
    manifest = suite.read_json(root / "manifest.json")
    arms = sorted({item["label"]["arm"] for item in manifest["cases"]})
    results = {}
    for model in manifest["models"]:
        rows = suite.read_rows(root / f"{model}.jsonl")
        suite.require(len(rows) == len(manifest["cases"]), "Incomplete diagnostic lane.")
        cells = {}
        for task in tasks._OPTIMIZATION:
            for stage in ("early", "late", "early_and_late"):
                for arm in arms:
                    cell = [r for r in rows if (r["task"], r["stage"], r["arm"]) ==
                            (task, stage, arm)]
                    if cell:
                        cells[f"{task}/{stage}/{arm}"] = {
                            "planned": len(cell), "correct": sum(r["correct"] for r in cell),
                            "failures": sum(r["status"] == "failure" for r in cell)}
        paired = {}
        for row in rows:
            paired.setdefault((row["case"], row["step"]), {})[row["arm"]] = row
        flips = {}
        for arm in (a for a in arms if a != "baseline"):
            eligible = [(p["baseline"], p[arm]) for p in paired.values()
                        if p["baseline"]["status"] == p[arm]["status"] == "success"]
            flips[arm] = {"both_valid": len(eligible), "changed_choice": sum(
                a["selected_option_id"] != b["selected_option_id"] for a, b in eligible)}
        results[model] = {"by_task_stage_arm": cells, "choice_changes": flips,
                          "attempts": len(rows), "correct": sum(r["correct"] for r in rows),
                          "failures": sum(r["status"] == "failure" for r in rows)}
    suite.write_json(root / "detailed-summary.json", {
        "purpose": "Teacher-forced diagnostic only. Never substitute for free-running scores.",
        "models": results})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tsp-state-controls", action="store_true",
                        help="Fresh n6,9,11 TSP current-endpoint clarification diagnostic.")
    args = parser.parse_args()
    source, manifest, instances, _ = verified_inputs(args.source, ("jev", "qwen9constrained"))
    controls, coverage = (build_tsp_state_controls(instances) if args.tsp_state_controls else
                          build_controls(instances))
    provenance = {"source_manifest_sha256": suite.digest(source / "manifest.json"),
                  "builder_sha256": suite.digest(__file__), "coverage": coverage,
                  "selection": ("Two opaque-ID-sorted TSP instances at n6,9,11; latest informative "
                                "prefix; disjoint instances from initial prefix controls."
                                if args.tsp_state_controls else
                                "Two opaque-ID-sorted instances per task at n5,8,12; "
                                "earliest/latest informative optimal-reference prefixes; "
                                "never consult model predictions."),
                  "call_ceiling": 36 if args.tsp_state_controls else 384}
    suite.require(len(controls) * 2 <= provenance["call_ceiling"], "Diagnostic budget exceeded.")
    asyncio.run(runner.run(
        args.root, ("jev", "qwen9constrained"), controls=controls,
        purpose=("Post-hoc TSP anchoring hypothesis tested on disjoint development instances; "
                 "no oracle objective/gain is exposed." if args.tsp_state_controls else
                 "Frozen early/late optimal-prefix controls with repeated, reordered and described options."),
        provenance=provenance))
    results = summarize(args.root)
    print({name: {k: v for k, v in result.items() if k != "by_task_stage_arm"}
           for name, result in results.items()})


if __name__ == "__main__":
    main()
