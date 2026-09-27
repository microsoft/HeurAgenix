#!/usr/bin/env python3
"""Descriptive, post-run Laya summary and independent native-score checks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import laya_cpu_pilot as pilot


def metrics(rows):
    exact = [r for r in rows if r["kind"] == "exact"]
    optimization = [r for r in rows if r["kind"] != "exact"]
    result = {"scheduled": len(rows), "feasible": sum(r["feasible"] for r in rows),
              "exact_correct": sum(bool(r["correct"]) for r in exact),
              "exact_scheduled": len(exact),
              "optimization_optimal": sum(bool(r["optimal"]) for r in optimization),
              "optimization_scheduled": len(optimization),
              "optimization_feasible": sum(r["feasible"] for r in optimization)}
    return result


def summarize(root):
    pilot.verify_registration(root)
    assert json.loads((root / "run/offline-replay.json").read_text())["verified"]
    rows = json.loads((root / "run/scores.json").read_text())
    reference = json.loads((root / "reused-jev.json").read_text())["rows"]
    by_id = {r["instance_id"]: r for r in reference}
    attempts = pilot.read_rows(root / "run/attempts.jsonl")
    runtime = json.loads((root / "run/runtime.json").read_text())
    successful = [r for r in attempts if r["status"] == "success"]
    score_checks = []
    for row in successful:
        native = row["backend"]
        pilot.validate_answer(row["request"], native["answer"], native["raw_logits"])
        ids = [o["id"] for o in row["request"]["options"]]
        k = len(ids)
        bucket = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        temperature = runtime["temperature_by_options_applied"].get(
            f"choice:{bucket}", runtime["temperature"][0])
        logits = [z / temperature for z in native["raw_logits"]]
        weights = [math.exp(z - max(logits)) for z in logits]
        probabilities = [v / sum(weights) for v in weights]
        error = max(abs(p - native["answer"]["probabilities"][i])
                    for p, i in zip(probabilities, ids))
        assert error <= .000051
        assert abs(max(probabilities) - native["answer"]["answer_confidence"]) <= .000051
        concentration = (1 + sum(p * math.log(max(p, 1e-12)) for p in probabilities)
                         / math.log(k)) if k > 1 else 1.
        assert abs(concentration - native["answer"]["confidence"]) <= .000051
        assert native["coverage"]["compatible"]
        assert native["coverage"]["full_tokens"] == native["usage"]["input_tokens"]
        assert native["cuda_initialized"] is False
        score_checks.append(error)
    groups = []
    for task in sorted({r["task"] for r in rows}):
        for size in (8, 12):
            selected = [r for r in rows if r["task"] == task and r["node_count"] == size]
            eligible = [r for r in selected if r["initial_context_compatible"]]
            if selected:
                groups.append({"task": task, "node_count": size,
                               "laya_whole": metrics(selected),
                               "laya_initial_compatible": metrics(eligible),
                               "jev_whole": metrics([by_id[r["instance_id"]] for r in selected]),
                               "jev_initial_compatible": metrics(
                                   [by_id[r["instance_id"]] for r in eligible])})
    eligible = [r for r in rows if r["initial_context_compatible"]]
    result = {
        "analysis_scope": "descriptive_post_run_no_fitting",
        "source_script_sha256": pilot.digest(Path(__file__)),
        "scores_sha256": pilot.digest(root / "run/scores.json"),
        "attempts_sha256": pilot.digest(root / "run/attempts.jsonl"),
        "scheduled": len(rows),
        "execution_categories": dict(Counter(r["execution_category"] for r in rows)),
        "native_probability_semantics_verified_actions": len(score_checks),
        "largest_rounding_error": max(score_checks, default=None),
        "laya_whole": metrics(rows),
        "laya_initial_compatible": metrics(eligible),
        "jev_whole": metrics(reference),
        "jev_initial_compatible": metrics([by_id[r["instance_id"]] for r in eligible]),
        "by_task_size": groups,
        "cpu_only_descriptive_runtime_not_cross_model_ranking": {
            "successful_calls": len(successful),
            "sum_worker_seconds": sum(r["backend"]["worker_seconds"] for r in successful),
            "median_worker_seconds": statistics.median(
                [r["backend"]["worker_seconds"] for r in successful]) if successful else None,
            "maximum_worker_seconds": max(
                [r["backend"]["worker_seconds"] for r in successful], default=None),
            "sum_input_tokens": sum(r["backend"]["usage"]["input_tokens"] for r in successful),
            "max_input_tokens": max(
                [r["backend"]["usage"]["input_tokens"] for r in successful], default=None)}}
    pilot.save(root / "run/descriptive-analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=pilot.ROOT)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps({k: result[k] for k in (
        "execution_categories", "native_probability_semantics_verified_actions",
        "laya_whole", "laya_initial_compatible", "jev_whole", "jev_initial_compatible")}, indent=2))
