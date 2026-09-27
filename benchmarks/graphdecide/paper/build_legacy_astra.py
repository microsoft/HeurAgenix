#!/usr/bin/env python3
"""Export a separately labelled, frozen-verified legacy Astra challenge table."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts import graph_release_api as release
from scripts.build_paper_evidence import EXACT, TASK_LABELS, latex_table
from scripts.graph_task_prompt import canonical_graph_request


def table_rows(summary):
    rows = []
    for task in (*EXACT, "maxcut_construct", "tsp_construct",
                 "lt_influence_construct", "community_bipartition_construct"):
        cell = summary["by_task"][task]
        hits = cell["correct"] if task in EXACT else cell["optimal_count"]
        rows.append([TASK_LABELS[task], f'{hits}/{cell["instances"]}',
                     f'{cell["feasible"]}/{cell["instances"]}',
                     "--" if task in EXACT else f'{cell["mean_absolute_gap_feasible"]:.6f}'])
    return rows


def export(root, bundle, output):
    suite.require(not output.exists(), "Legacy Astra evidence exists; choose a new output.")
    frozen = root / "frozen-source/scripts/extended_graph_suite.py"
    checked = subprocess.run([sys.executable, str(frozen), "report", "--root", str(root),
                              "--models", "gpt6astra"], capture_output=True, text=True)
    suite.require(checked.returncode == 0,
                  f"Frozen legacy verification failed:\n{checked.stdout}\n{checked.stderr}")
    report = json.loads(checked.stdout)
    summary = report["models"]["gpt6astra"]
    suite.require(summary["status"] == "completed" and not summary["injected_client"]
                  and summary["attempts"] == 1920, "Legacy Astra challenge is incomplete or mock.")
    release.verify(bundle)
    manifest = release.manifest_at(bundle)
    suite.require(manifest["splits"]["structural_challenge"]["source_manifest_sha256"]
                  == suite.digest(root / "manifest.json"), "Different source bank.")
    instances = release.labeled_instances(bundle, "structural_challenge", manifest)
    suite.require(instances == suite.read_rows(root / "instances.jsonl")
                  and len(instances) == 960, "Graph/reference identity differs.")
    attempts = suite.read_rows(root / "runs/gpt6astra/attempts.jsonl")
    initial = {a["instance_id"]: a["request"] for a in attempts if a["step"] == 0}
    suite.require(len(initial) == len(instances), "Missing initial requests.")
    equal = sum(initial[i["id"]] == suite.json_value(asdict(canonical_graph_request(
        i["task"], release.tasks.next_request(i, [])))) for i in instances)
    output.mkdir(parents=True, exist_ok=False)
    (output / "challenge.tex").write_text(latex_table(
        ("Task", "Correct / optimal", "Feasible", "Mean gap"), table_rows(summary)))
    suite.write_json(output / "summary.json", summary)
    paths = [root / "manifest.json", bundle / "manifest.json",
             *(root / "runs/gpt6astra" / name for name in
               ("status.json", "attempts.jsonl", "scores.jsonl", "summary.json"))]
    suite.write_json(output / "provenance.json", {
        "source_sha256": {str(p): suite.digest(p) for p in paths},
        "exporter_sha256": suite.digest(__file__),
        "generated_sha256": {p.name: suite.digest(p) for p in output.iterdir()},
        "frozen_report_verified": True, "same_graphs_and_references_as_canonical_challenge": True,
        "canonical_initial_requests_equal": equal, "initial_requests_compared": len(instances),
        "protocol": "legacy_frozen", "model_calls": 0,
        "scope": "Separate default-reasoning, 4096-token strong-model reference; "
                 "not an equal-protocol or equal-compute column of the primary matrix."})
    return {"status": "verified_and_exported", "episodes": 960, "recorded_calls": 1920,
            "new_calls": 0, "canonical_initial_requests_equal": equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(suite.canonical(export(args.root, args.bundle, args.output)))


if __name__ == "__main__":
    main()
