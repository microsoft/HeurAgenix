#!/usr/bin/env python3
"""Render the verified timeout-only control without crediting truncated responses."""
import argparse
from collections import Counter
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts.build_paper_evidence import OPTIMIZATION, TASK_LABELS, latex_table

SOURCE = REPO / "output/qwen9-deadline-finalization-20260924-v1/export"
RUN = REPO / "output/qwen9-deadline-control-20260924-v1"


def render(models, attempts):
    suite.require(len(attempts) == 16 and all(
        a["status"] == "failure" and a["diagnostic_code"] == "finish_length"
        and a["usage"]["output_tokens"] == 8192
        and 0 < a["latency_seconds"] < 600 for a in attempts),
        "Unexpected completed token-limit control.")
    rows = models["qwen9_600"]
    previous = models["qwen9_180"]
    suite.require(len(rows) == len(previous) == 16
                  and {r["instance_id"] for r in rows} == {r["instance_id"] for r in previous},
                  "Mismatched scheduled cases.")
    suite.require(len({a["instance_id"] for a in attempts}) == 16
                  and {a["instance_id"] for a in attempts} == {r["instance_id"] for r in rows},
                  "Incomplete or duplicate attempted cases.")
    table = []
    for task in OPTIMIZATION:
        own = [r for r in rows if r["task"] == task]
        old = [r for r in previous if r["task"] == task]
        suite.require(len(own) == len(old) == 4 and all(
            not r["feasible"] and r["failure_count"] == 1 and r["absolute_gap"] is None
            for r in own + old), "Unexpected coverage or conditional gap.")
        table.append([TASK_LABELS[task], "0/4", "0/4", str(len(own)), "--"])
    return latex_table(
        ("Task", "180s feasible", "600s feasible", "600s length stops", "600s gap"), table)


def export(source, output):
    suite.require(not output.exists(), "Output exists; no overwrite.")
    seal = suite.read_json(source / "export-seal.json")
    suite.require(seal["status"] == "verified_completed_analysis", "Unverified control export.")
    for name, digest in seal["artifact_sha256"].items():
        path = source / name
        suite.require(path.resolve().is_relative_to(source.resolve())
                      and not path.is_symlink() and suite.digest(path) == digest, "Artifact drift.")
    analysis = suite.read_json(source / "analysis.json")
    suite.require(analysis["status"] == "verified_completed_analysis"
                  and analysis["source_status"] == "completed_with_failures"
                  and analysis["configuration_delta"] == {"timeout_seconds": [180, 600]},
                  "Unexpected control contract.")
    hashes = analysis["input_sha256_before_and_after"]
    for name, digest in hashes.items():
        suite.require(suite.digest(name) == digest, "Original control evidence changed.")
    attempts_path = RUN / "attempts.jsonl"
    suite.require(str(attempts_path) in hashes, "Attempt ledger not bound.")
    attempts = suite.read_rows(attempts_path)
    text = render(suite.read_json(source / "episodes.json"), attempts)
    output.mkdir(parents=True, exist_ok=False)
    (output / "deadline.tex").write_text(text)
    suite.write_json(output / "provenance.json", {
        "status": "verified_and_rendered", "new_model_calls": 0,
        "source": str(source), "export_seal_sha256": suite.digest(source / "export-seal.json"),
        "exporter_sha256": suite.digest(__file__), "verified_source_hashes": len(hashes),
        "attempts_sha256": suite.digest(attempts_path),
        "failure_codes": dict(Counter(a["diagnostic_code"] for a in attempts)),
        "output_tokens": sorted({a["usage"]["output_tokens"] for a in attempts}),
        "latency_seconds_range": [min(a["latency_seconds"] for a in attempts),
                                  max(a["latency_seconds"] for a in attempts)],
        "generated_sha256": {"deadline.tex": suite.digest(output / "deadline.tex")},
        "scope": "Separate posthoc timeout-only control; original failures retained. "
                 "No truncated-answer salvage, conditional quality estimate or capacity claim."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
