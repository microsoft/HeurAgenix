#!/usr/bin/env python3
"""Build a descriptive larger-model table from hash-verified offline evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts.build_paper_evidence import EXACT, TASK_LABELS, latex_table

PROFILES = ("jev_action", "qwen9_grammar", "qwen27_grammar")
CONSTRUCTIONS = ("maxcut_construct", "tsp_construct", "lt_influence_construct",
                 "community_bipartition_construct")


def tables(metrics):
    all_rows = [r for r in metrics if str(r["node_count"]) == "all"
                and r["profile"] in PROFILES]
    cells = {(r["task"], r["profile"]): r for r in all_rows}
    suite.require(len(cells) == len(all_rows) == 30, "Missing or duplicate pilot task cells.")
    exact, construction = [], []
    for task in (*EXACT, *CONSTRUCTIONS):
        selected = [cells[task, profile] for profile in PROFILES]
        expected = 8 if task in EXACT else 4
        suite.require(all(r["episodes"] == expected and r["feasible"] == expected
                          and r["failed_episodes"] == r["failure_calls"] == 0 for r in selected),
                      "Table contract requires the fully feasible completed pilot.")
        hits = [f'{r["hits"]}/{r["episodes"]}' for r in selected]
        if task in EXACT:
            exact.append([TASK_LABELS[task], *hits])
        else:
            construction.append([TASK_LABELS[task], *hits, *(
                f'{r["mean_absolute_gap_feasible"]:.3f}' for r in selected)])
    return {"exact.tex": latex_table(("Task", "Jev", "9B grammar", "27B grammar"), exact),
            "construction.tex": latex_table(
                ("Task", "Jev hits", "9B hits", "27B hits", "Jev gap", "9B gap", "27B gap"),
                construction)}


def export(source, output):
    suite.require(not output.exists(), "Pilot table output exists.")
    seal = suite.read_json(source / "export-seal.json")
    suite.require(seal["status"] == "completed"
                  and "verified-summary.json" in seal["artifact_sha256"], "Unsealed pilot export.")
    for name, expected in seal["artifact_sha256"].items():
        path = source / name
        suite.require(not Path(name).is_absolute() and ".." not in Path(name).parts
                      and path.is_file() and not path.is_symlink()
                      and suite.digest(path) == expected, "Pilot export artifact changed.")
    summary = suite.read_json(source / "verified-summary.json")
    suite.require(summary["status"] == "verified_completed_bounded_pilot_export"
                  and summary["canonical_initial_requests_identical"]
                  and summary["episodes_per_profile"] == 64, "Unexpected pilot evidence.")
    for name, expected in summary["source_sha256"].items():
        path = REPO / name
        suite.require(path.resolve().is_relative_to(REPO) and suite.digest(path) == expected,
                      "Original pilot source changed.")
    rendered = tables(summary["metrics"])
    output.mkdir(parents=True, exist_ok=False)
    for name, text in rendered.items():
        (output / name).write_text(text)
    suite.write_json(output / "provenance.json", {
        "source": str(source), "export_seal_sha256": suite.digest(source / "export-seal.json"),
        "summary_sha256": suite.digest(source / "verified-summary.json"),
        "exporter_sha256": suite.digest(__file__),
        "generated_sha256": {p.name: suite.digest(p) for p in output.iterdir()},
        "scope": summary["scope"], "new_model_calls": 0})
    return {"status": "verified_and_exported", "output": str(output), "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(suite.canonical(export(args.source, args.output)))


if __name__ == "__main__":
    main()
