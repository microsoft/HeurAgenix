#!/usr/bin/env python3
"""Publish completed thinking configurations without treating timeouts as wrong answers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts.build_paper_evidence import EXACT, TASK_LABELS, latex_table
from paper.build_pilot_table import CONSTRUCTIONS


def verified(source, status):
    seal = suite.read_json(source / "export-seal.json")
    suite.require(seal["status"] == "completed", "Unsealed export.")
    for name, digest in seal["artifact_sha256"].items():
        path = source / name
        suite.require(path.resolve().is_relative_to(source.resolve()) and not path.is_symlink()
                      and suite.digest(path) == digest, "Export artifact changed.")
    report = suite.read_json(source / "verified-summary.json")
    suite.require(report["status"] == status and report["episodes_per_profile"] == 64,
                  "Unexpected verified report.")
    for name, digest in report["source_sha256"].items():
        path = REPO / name
        suite.require(path.resolve().is_relative_to(REPO) and suite.digest(path) == digest,
                      "Source evidence changed.")
    return report


def tables(thinking, bounded):
    profiles = ("jev_action", "qwen27_grammar", "qwen27_thinking")
    cells = {}
    for report in (bounded, thinking):
        for row in report["metrics"]:
            if str(row["node_count"]) == "all" and row["profile"] in profiles:
                key = row["task"], row["profile"]
                if key in cells:
                    suite.require(all(cells[key][field] == row[field] for field in
                                      ("episodes", "hits", "feasible", "failure_calls")),
                                  "Historical reference mismatch.")
                cells[key] = row
    suite.require(len(cells) == 30, "Missing task/profile cells.")
    exact, construction = [], []
    for task in (*EXACT, *CONSTRUCTIONS):
        selected = [cells[task, profile] for profile in profiles]
        expected = 8 if task in EXACT else 4
        suite.require(all(r["episodes"] == expected for r in selected), "Wrong denominator.")
        candidate = selected[-1]
        if task in EXACT:
            exact.append([TASK_LABELS[task], *(
                f'{r["hits"]}/{r["episodes"]}' for r in selected),
                str(candidate["failure_calls"])])
        else:
            gap = candidate["mean_absolute_gap_feasible"]
            suite.require((gap is None) == (candidate["feasible"] == 0),
                          "Missing or invented conditional gap.")
            construction.append([
                TASK_LABELS[task], f'{candidate["feasible"]}/{expected}',
                str(candidate["failure_calls"]), "--" if gap is None else f"{gap:.3f}"])
    return {
        "exact.tex": latex_table(
            ("Task", "Jev", "27B grammar", "27B thinking", "Thinking failures"), exact),
        "construction.tex": latex_table(
            ("Task", "Thinking feasible", "Failed calls", "Conditional gap"), construction)}


def export(source, bounded_source, output):
    suite.require(not output.exists(), "Output exists.")
    thinking = verified(source, "verified_terminal_expanded_pilot_export")
    suite.require(thinking["profile"] == "qwen27_thinking"
                  and thinking["source_status"] in ("completed", "completed_with_failures"),
                  "Requires a terminal 27B thinking run.")
    bounded = verified(bounded_source, "verified_completed_bounded_pilot_export")
    rendered = tables(thinking, bounded)
    output.mkdir(parents=True, exist_ok=False)
    for name, text in rendered.items():
        (output / name).write_text(text)
    suite.write_json(output / "provenance.json", {
        "source": str(source), "bounded_source": str(bounded_source),
        "export_seals_sha256": {str(p / "export-seal.json"): suite.digest(p / "export-seal.json")
                               for p in (source, bounded_source)},
        "exporter_sha256": suite.digest(__file__), "model_calls": 0,
        "generated_sha256": {p.name: suite.digest(p) for p in output.iterdir()},
        "failure_categories": thinking["provenance"]["qwen27_thinking"]["failure_categories"],
        "scope": "Same exploratory panel; thinking changes token budget and grammar as well as "
                 "reasoning mode. All planned denominators retained. Missing conditional gaps "
                 "are not zero. Historical timing is not a controlled comparison."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bounded-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.bounded_source, args.output)
