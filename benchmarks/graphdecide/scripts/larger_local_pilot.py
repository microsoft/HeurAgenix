#!/usr/bin/env python3
"""Offline, score-blind registration for the bounded larger-local-model pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import evaluate_graph_release as canonical
from scripts import extended_graph_suite as suite
from scripts import graph_invariance_controls as invariance
from scripts import graph_release_api as release

SIZES = (8, 12)
CONSTRUCTION = ("maxcut_construct", "tsp_construct", "lt_influence_construct",
                "community_bipartition_construct")
DEFAULT_ROOT = REPO / "output/larger-local-pilot-20260924-v1"


def select_cases(public, metadata):
    selected = [i for i in invariance.select_cases(public, metadata)
                if len(i["state"]["nodes"]) in SIZES]
    for task in CONSTRUCTION:
        for n in SIZES:
            pool = sorted((i for i in public if i["task"] == task
                           and i["kind"] == "optimization"
                           and len(i["state"]["nodes"]) == n), key=lambda i: i["id"])
            suite.require(len(pool) >= 2, "Missing construction cases.")
            selected.extend(pool[:2])
    suite.require(len(selected) == 64 and len({i["id"] for i in selected}) == 64,
                  "Pilot requires exactly 64 unique episodes.")
    return selected


def configurations():
    base = canonical.configuration("qwen9_grammar")[0]
    large = replace(base, model="Qwen3.5-27B", base_url="http://127.0.0.1:8010/v1",
                    timeout_seconds=180.0, max_tokens=64)
    return {
        "qwen27_grammar": large,
        "qwen27_thinking": replace(large, think=True, max_tokens=8192, constrain_choices=False),
        "qwen9_thinking": replace(base, think=True, max_tokens=8192, constrain_choices=False,
                                  timeout_seconds=180.0),
    }


def failure_denominators(instances, attempts=()):
    """Canonical replay retains every scheduled instance even with zero attempts."""
    return canonical.replay(instances, list(attempts), configurations()["qwen27_grammar"],
                            native_action=False)


def prepare(root, bundle):
    suite.require(not root.exists(), "Registration exists; no overwrite or resume.")
    hashes = invariance.source_hashes()
    hashes["scripts/larger_local_pilot.py"] = suite.digest(__file__)
    manifest = release.manifest_at(bundle)
    suite.require("explicit_actions" in manifest["protocols"], "Canonical protocol missing.")
    public = release.load_public(bundle, invariance.SPLIT, manifest)
    metadata = suite.read_json(release.checked_file(
        bundle, f"analysis/{invariance.SPLIT}.json", manifest))
    selected = select_cases(public, metadata)
    labeled = {i["id"]: i for i in release.labeled_instances(bundle, invariance.SPLIT, manifest)}
    instances = [labeled[i["id"]] for i in selected]
    validation = canonical.tasks.validate_instances(instances)
    for instance in instances:
        if instance["kind"] == "exact":
            suite.require(invariance.truth(instance["task"], instance["state"]) ==
                          str(instance["private"]["answer"]), "Independent reference mismatch.")
    existing = REPO / "output/graph-invariance-controls-20260924-v1/cases.jsonl"
    expected = {i["instance_id"] for i in suite.read_rows(existing) if i["node_count"] in SIZES}
    suite.require(expected == {i["id"] for i in selected if i["kind"] == "exact"},
                  "Exact selection differs from preregistered invariance cases.")
    root.mkdir(parents=True)
    invariance.write_rows(root / "pilot-cases.jsonl", [
        {"index": index, "instance_id": i["id"], "task": i["task"], "kind": i["kind"],
         "node_count": len(i["state"]["nodes"]), "query_budget": i["query_budget"],
         "cluster_id": metadata["cases"][i["id"]]["cluster_id"],
         "pair_id": metadata["cases"][i["id"]].get("pair_id")}
        for index, i in enumerate(selected)])
    invariance.write_rows(root / "initial-requests.jsonl", [
        {"instance_id": i["id"], "request": asdict(canonical.public_request(i, []))}
        for i in selected])
    invariance.freeze_sources(root, hashes)
    maximum = sum(i["query_budget"] for i in selected)
    registration = {
        "schema_version": 1, "registered_at_utc": invariance.now(),
        "registered_before_inference": True, "model_calls": 0,
        "bundle": str(bundle.resolve()), "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "split": invariance.SPLIT, "protocol": "explicit_actions",
        "selection": "Existing graph_invariance_controls score-blind exact selection filtered "
                     "to n8,n12 (two complete opaque-ID binary pairs; four degree strata). "
                     "Construction: first two opaque IDs per task/size. No scores consulted.",
        "selection_sha256": suite.digest(root / "pilot-cases.jsonl"),
        "initial_requests_sha256": suite.digest(root / "initial-requests.jsonl"),
        "invariance_cases_sha256": suite.digest(existing),
        "source_sha256": hashes, "independent_validation": validation,
        "sizes": list(SIZES), "episodes_per_profile": 64,
        "maximum_calls_per_profile": maximum, "maximum_calls_all_three_profiles": maximum * 3,
        "configurations": {name: asdict(config) for name, config in configurations().items()},
        "policy": {"retry": False, "repair": False, "resample": False,
                   "history": "Full own-history trajectories, fixed query budgets.",
                   "denominator": "All 64 planned episodes, including failures/unattempted.",
                   "initial_requests": "Identical canonical requests across profiles.",
                   "comparison": "Separate model size from inference-budget effect; descriptive "
                                 "pilot only; reused inspected references labeled separately."},
        "execution_status": "blocked_before_download_and_inference",
        "blocker": "Existing standard lane locks use the default /tmp directory. Runtime "
                   "prohibits all temporary-directory file operations; redirecting TMPDIR "
                   "would bypass shared locks and is forbidden. No lock attempted.",
    }
    suite.write_json(root / "registration.json", registration)
    suite.write_json(root / "registration.sha256", suite.digest(root / "registration.json"))
    suite.write_json(root / "dry-run.json", {
        "model_calls": 0, "planned_episodes": len(instances), "maximum_calls": maximum,
        "zero_attempt_failure_denominator": len(failure_denominators(instances)),
        "independent_validation": validation, "source_snapshot_verified": all(
            suite.digest(root / "source-snapshot" / name) == digest
            for name, digest in hashes.items())})
    for name in ("pilot-cases.jsonl", "initial-requests.jsonl", "registration.json",
                 "registration.sha256", "dry-run.json"):
        (root / name).chmod(0o444)
    return registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bundle", type=Path, default=invariance.DEFAULT_BUNDLE)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.bundle), indent=2))


if __name__ == "__main__":
    main()
