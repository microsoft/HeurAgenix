#!/usr/bin/env python3
"""Generate paper diagnostics from sealed invariance and prospective probability runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import graph_invariance_controls as invariance
from scripts import graph_probability_audit as probability
from scripts import verify_graph_probability_audit as recovery
from scripts.verify_graph_probability_audit import common_valid
from scripts import fixed_prefix_readout_control as readout
from scripts import extended_graph_suite as suite
from scripts.build_paper_evidence import EXACT, LABELS, PROFILES, TASK_LABELS, latex_table

def number(value):
    return "--" if value is None else f"{value:.3f}"


def loss(cell):
    return r"$+\infty$" if cell["mean_log_loss_is_infinite"] else number(cell["mean_log_loss"])


def export(invariance_root, probability_root, probability_analysis, readout_root, output):
    suite.require(not output.exists(), "Robustness evidence exists; use a new path.")
    _, cases, requests = invariance.verify(invariance_root)
    invariant, sources = {}, {}
    for profile in PROFILES:
        status, attempts, scores, summary = invariance.verify_run(invariance_root, profile)
        suite.require(status["status"].startswith("completed"), "Invariance profile is incomplete.")
        invariant[profile] = summary
        for name in ("manifest.json", "status.json", "attempts.jsonl", "scores.jsonl"):
            path = invariance_root / profile / name
            sources[str(path)] = suite.digest(path)

    reports, attempts = {}, {}
    analysis_manifest = suite.read_json(probability_analysis / "manifest.json")
    suite.require(Path(analysis_manifest["source"]).resolve() == probability_root.resolve(),
                  "Recovered probability analysis uses a different source audit.")
    for profile in probability.PROFILES:
        reports[profile] = recovery.report(probability_analysis, profile)
        status = reports[profile]["status"]
        expected_status = ("offline_recovered_analysis" if profile == "jev_action"
                           else "offline_verified_analysis")
        suite.require(status["status"] == expected_status
                      and status["original_collector_calls"] == status["planned_calls"] == probability.CALLS,
                      "Recovered probability collection is incomplete.")
        attempts[profile] = {row["instance_id"]: row for row in
                             suite.read_rows(probability_root / profile / "attempts.jsonl")}
        sources[str(probability_root / profile / "attempts.jsonl")] = suite.digest(
            probability_root / profile / "attempts.jsonl")
        for name in ("status.json", "scores.jsonl", "summary.json"):
            path = probability_analysis / profile / name
            sources[str(path)] = suite.digest(path)
    probability_manifest, probability_cases = probability.load_plan(probability_root)
    common = {task: common_valid([c for c in probability_cases if c["task"] == task],
                                 attempts, probability_manifest["profiles"])
              for task in EXACT}
    subprocess.run([sys.executable, str(REPO / "scripts/fixed_prefix_readout_control.py"),
                    "verify", "--output", str(readout_root)], check=True, capture_output=True, text=True)
    readout_summary = readout.summarize(readout_root)
    sources[str(readout_root / "summary.json")] = suite.digest(readout_root / "summary.json")
    sources[str(readout_root / "manifest.json")] = suite.digest(readout_root / "manifest.json")

    robust, flips, accuracies, briers, losses = [], [], [], [], []
    for task in EXACT:
        robust.append([TASK_LABELS[task], *(
            f'{row["all_five_correct"]}/{row["scheduled_cases"]}'
            for profile in PROFILES for row in invariant[profile]["five_arm_patterns"]
            if row["task"] == task)])
        for profile, label in zip(PROFILES, LABELS):
            selected = {row["arm"]: row for row in invariant[profile]["paired_choice_changes"]
                        if row["task"] == task and row["actual_input_changed"] == "all"}
            flips.append([TASK_LABELS[task], label, *(
                f'{selected[arm]["choice_flips"]}/{selected[arm]["both_valid"]}'
                for arm in ("identical_repeat", "node_relabel", "edge_shuffle", "reversed_options"))])
            selected = {row["arm"]: row for row in invariant[profile]["accuracy"]
                        if row["task"] == task and row["node_count"] == "all"}
            accuracies.append([TASK_LABELS[task], label, *(
                f'{selected[arm]["correct"]}/{selected[arm]["scheduled"]}'
                for arm in invariance.ARMS)])
        own = [reports[p]["summary"]["by_task"][task] for p in probability.PROFILES]
        joined = common[task]
        sizes = "/".join(str(cell["vector_denominator"]) for cell in own)
        if joined["profiles"] is None:
            briers.append([TASK_LABELS[task], sizes, "0", "--", "--", "--"])
            losses.append([TASK_LABELS[task], "0", "--", "--", "--"])
        else:
            cells = [joined["profiles"][p] for p in probability.PROFILES]
            briers.append([TASK_LABELS[task], sizes, str(joined["denominator"]), *(
                number(cell["mean_summed_multiclass_brier"]) for cell in cells),
                number(cells[0]["uniform_reference_same_rows"]["mean_summed_multiclass_brier"])])
            losses.append([TASK_LABELS[task], str(joined["denominator"]), *(loss(cell) for cell in cells),
                           number(cells[0]["uniform_reference_same_rows"]["mean_log_loss"])])
    tables = {
        "robust-correct.tex": latex_table(("Task", *LABELS), robust),
        "choice-flips.tex": latex_table(("Task", "Method", "Repeat", "Relabel", "Edges", "Options"), flips),
        "arm-accuracy.tex": latex_table(("Task", "Method", "Original", "Repeat", "Relabel", "Edges", "Options"), accuracies),
        "brier.tex": latex_table(("Task", "Valid J/T", "Common", "Jev", "4B scores", "Uniform"), briers),
        "log-loss.tex": latex_table(("Task", "Common", "Jev", "4B scores", "Uniform"), losses),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, text in tables.items():
        (output / name).write_text(text)
    suite.write_json(output / "probability-comparison.json", {
        "own_valid_reports": reports, "common_valid": common,
        "analysis_environment": suite.environment_versions(),
        "verification": "Original producer outcomes preserved: native CLI exit2 after792 responses, "
                        "token CLI exit0. Versioned offline recovery validates original digests and "
                        "unchanged vectors, permits1e-12 derived-sum bookkeeping only, and preserves "
                        "the1e-6 vector gate. Stable common-valid arithmetic is declared separately.",
        "scope": probability.NOTICE,
    })
    reliability = []
    for task, joined in common.items():
        if joined["profiles"]:
            for profile, summary in joined["profiles"].items():
                reliability.extend({"task": task, "profile": profile, **row}
                                   for row in summary["reliability_bins"])
    with (output / "common-reliability-bins.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "task", "profile", "lower", "upper", "upper_inclusive", "count",
            "mean_selected_probability", "actual_action_accuracy"))
        writer.writeheader()
        writer.writerows(reliability)
    suite.write_json(output / "readout-summary.json", readout_summary)
    for root, name in ((invariance_root, "registration.json"),
                       (probability_root, "manifest.json"),
                       (probability_analysis, "manifest.json"),
                       (probability_analysis, "seal.json")):
        sources[str(root / name)] = suite.digest(root / name)
    suite.write_json(output / "provenance.json", {
        "source_artifact_sha256": sources,
        "analysis_code_sha256": {str(path.relative_to(REPO)): suite.digest(path) for path in (
            Path(__file__), Path(invariance.__file__), Path(probability.__file__), Path(readout.__file__),
            Path(recovery.__file__),
            REPO / "scripts/build_paper_evidence.py")},
        "generated_sha256": {p.name: suite.digest(p) for p in sorted(output.iterdir())},
        "inference_calls": 0, "invariance_cases": len(cases),
        "invariance_calls": len(requests) * len(PROFILES),
        "probability_calls": sum(len(rows) for rows in attempts.values()),
        "fixed_prefix_calls": readout_summary["calls"],
        "analysis_environment": suite.environment_versions(),
    })
    return {"status": "verified_and_exported", "output": str(output), "inference_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invariance", type=Path, required=True)
    parser.add_argument("--probability", type=Path, required=True)
    parser.add_argument("--probability-analysis", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.invariance, args.probability, args.probability_analysis,
                            args.readout, args.output), indent=2))


if __name__ == "__main__":
    main()
