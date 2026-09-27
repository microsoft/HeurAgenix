#!/usr/bin/env python3
"""Replay-verified tables for matched explicit-action graph backends."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import evaluate_graph_release as evaluator
from scripts import extended_graph_suite as suite
from scripts import graph_release_api as release
from scripts.summarize_extended_paper import detail_tables, paired_intervals, size_plot, structural_clusters
from src.benchmark.config import ModelConfig


def same_aggregates(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_aggregates(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_aggregates(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=0, abs_tol=1e-12)
    return left == right


def verify_token_scores(attempt):
    data = attempt["backend_accounting"]
    ids = [option["id"] for option in attempt["request"]["options"]]
    values = data["candidate_logprobs"]
    suite.require(data["option_ids"] == ids and len(values) == len(ids)
                  and all(type(v) in (int, float) and math.isfinite(v) for v in values),
                  "Candidate token score identity mismatch.")
    weights = [math.exp(v-max(values)) for v in values]
    probabilities = {key: value/sum(weights) for key, value in zip(ids, weights)}
    suite.require(set(data["probabilities"]) == set(ids) and all(
        math.isclose(data["probabilities"][key], value, rel_tol=0, abs_tol=1e-12)
        for key, value in probabilities.items()), "Conditional probability mismatch.")
    suite.require(attempt["selected_option_id"] == max(ids, key=probabilities.__getitem__),
                  "Selected action differs from conditional argmax.")
    suite.require(data["candidate_count"] == len(ids) and data["candidate_prompts"] == 1
                  and data["prefill_tokens"] == data["shared_prefix_tokens"] ==
                  attempt["usage"]["input_tokens"] and data["generated_tokens"] ==
                  attempt["usage"]["output_tokens"] == 1
                  and data["generated_tokens_used_for_decision"] is False,
                  "Token-score compute accounting mismatch.")


def verified_run(root, bundle, split):
    manifest = suite.read_json(root / "manifest.json")
    suite.require(suite.digest(root / "manifest.json") == suite.read_json(root / "manifest.sha256"),
                  "Evaluation manifest drift.")
    suite.require(manifest["protocol"] == "explicit_actions" and manifest["split"] == split
                  and manifest["bundle_manifest_sha256"] == suite.digest(bundle / "manifest.json"),
                  "Public protocol/split/bundle mismatch.")
    suite.require(not manifest["injected_client"], "Mock/injected evaluations are not paper results.")
    status = suite.read_json(root / "status.json")
    suite.require(status["status"] in ("completed", "completed_with_failures"), "Unsealed lane.")
    suite.require(all(suite.digest(root / name) == sha for name, sha in status["artifact_sha256"].items()),
                  "Sealed artifact drift.")
    instances = release.labeled_instances(bundle, split, release.manifest_at(bundle))
    attempts = suite.read_rows(root / "attempts.jsonl")
    suite.require(len(attempts) == status["calls"], "Attempt count mismatch.")
    if manifest["profile"] == "qwen4_token_scores":
        for attempt in attempts:
            if attempt["status"] == "success":
                verify_token_scores(attempt)
    rows = evaluator.replay(instances, attempts, ModelConfig(**manifest["model"]),
                            native_action=manifest["native_action_observation"])
    suite.require(rows == suite.read_rows(root / "scores.jsonl"), "Recomputed per-instance scores differ.")
    suite.require(same_aggregates(suite.summarize(rows), suite.read_json(root / "summary.json")),
                  "Recomputed aggregate summary differs beyond numerical roundoff.")
    return manifest, status, attempts, rows


def export(matrix, bundle, split, profiles, output):
    suite.require(not output.exists(), "Analysis output exists.")
    release.verify(bundle)
    release_manifest = release.manifest_at(bundle)
    instances = release.labeled_instances(bundle, split, release_manifest)
    metadata = suite.read_json(bundle / f"analysis/{split}.json")
    clusters = structural_clusters(metadata)
    rows, summaries, timing, statuses, initial = {}, {}, {}, {}, {}
    for profile in profiles:
        manifest, status, attempts, own = verified_run(matrix / f"{split}.{profile}", bundle, split)
        suite.require(manifest["profile"] == profile, "Profile mismatch.")
        rows[profile], summaries[profile], statuses[profile] = own, suite.summarize(own), status
        first = {a["instance_id"]: a["request"] for a in attempts if a["step"] == 0}
        suite.require(len(first) == len(instances), "Incomplete execution coverage.")
        if initial:
            suite.require(first == initial, "Profiles did not receive identical initial public requests.")
        initial = first
        latency = sorted(a["latency_seconds"] for a in attempts)
        extras = [a["backend_accounting"] for a in attempts if "backend_accounting" in a]
        timing[profile] = {
            "calls": len(attempts), "wall_seconds": status["wall_seconds"],
            "median_seconds": latency[len(latency)//2], "p95_seconds": latency[int(.95*len(latency))],
            "request_seconds_sum": sum(latency),
            "token_totals": {k: sum(a["usage"][k] for a in attempts if a["usage"][k] is not None)
                             if any(a["usage"][k] is not None for a in attempts) else None
                             for k in suite.USAGE_KEYS},
            "token_observation_counts": {k: sum(a["usage"][k] is not None for a in attempts)
                                         for k in suite.USAGE_KEYS},
            "candidate_score_count": sum(e["candidate_count"] for e in extras) if extras else None,
            "prefill_sequences": sum(e["candidate_prompts"] for e in extras) if extras else None,
            "token_score_generated_tokens_ignored": sum(e["generated_tokens"] for e in extras) if extras else None,
        }
    pairs = {f"{a}-minus-{b}": paired_intervals(instances, rows[a], rows[b], clusters=clusters)
             for a, b in itertools.combinations(profiles, 2)}
    output.mkdir(parents=True, exist_ok=False)
    suite.write_json(output / "verified-summary.json", {
        "split": split, "profiles": profiles, "models": summaries, "statuses": statuses,
        "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "analysis_code_sha256": suite.digest(__file__),
        "analysis_environment": suite.environment_versions(),
        "verification_tolerance": {"per_instance": "exact",
                                   "aggregate_float_absolute": 1e-12,
                                   "reason": "Python3.11/3.12 float-sum aggregation differs at roundoff scale."},
        "initial_requests_identical": True,
        "scope": "Protocol replication on inspected graph banks; same information but different "
                 "backend rendering/scoring. Token-score backend is not certified CPU SemIf reproduction.",
    })
    suite.write_json(output / "paired-intervals.json", pairs)
    suite.write_json(output / "timing.json", timing)
    suite.write_json(output / "analysis-clusters.json", clusters)
    (output / "size-rates.svg").write_text(size_plot(summaries))
    detail_tables(output, rows, metadata)
    fields = ("profile", "task", "node_count", "instances", "feasible", "failure_count",
              "accuracy", "macro_size_accuracy", "optimality_rate", "mean_absolute_gap_feasible")
    with (output / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for profile, summary in summaries.items():
            for task, total in summary["by_task"].items():
                for size, cell in [("all", total), *summary["by_task_and_size"][task].items()]:
                    writer.writerow({"profile": profile, "task": task, "node_count": size,
                                     **{k: cell.get(k) for k in fields[3:]}})
    return {"split": split, "profiles": profiles, "instances_per_profile": len(instances),
            "actual_calls": sum(v["calls"] for v in timing.values()), "paired_comparisons": len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.matrix, args.bundle, args.split, args.profiles, args.output)))


if __name__ == "__main__":
    main()
