#!/usr/bin/env python3
"""Publish a canonical GPT appendix only after frozen verification; never infer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts import graph_release_api as release
from scripts import summarize_release_matrix as matrices
from scripts.build_paper_evidence import EXACT, OPTIMIZATION, TASK_LABELS, latex_table
from scripts.summarize_extended_paper import paired_intervals, structural_clusters
from src.benchmark.config import ModelConfig

COMPLETE = {"completed", "completed_with_failures"}
ACTIVE = {"registered", "waiting", "running"}


def execution_hashes():
    return {
        **matrices.evaluator.code_hashes(),
        "paper/finalize_gpt.py": suite.digest(__file__),
        **{name: suite.digest(REPO / name) for name in (
            "scripts/build_paper_evidence.py", "scripts/summarize_release_matrix.py",
            "scripts/summarize_extended_paper.py")},
    }


def verify_execution_hashes(expected):
    suite.require(execution_hashes() == expected,
                  "Paper analysis source changed while waiting; preserve the run and restart verification.")


def ready(status, sealed):
    name = status["status"]
    if name in COMPLETE:
        return sealed
    suite.require(name in ACTIVE, f"GPT anchor terminated without usable completion: {name}")
    return False


def render(summary, baseline, paired, attempts):
    exact_rows, optimization_rows = [], []
    for task in EXACT:
        left, right = summary["by_task"][task], baseline["by_task"][task]
        low, high = paired[task]["percentile_95"]
        exact_rows.append([
            TASK_LABELS[task], f'{100 * left["macro_size_accuracy"]:.2f}',
            f'{100 * right["macro_size_accuracy"]:.2f}',
            f'[{100 * low:.2f}, {100 * high:.2f}]',
        ])
    for task in OPTIMIZATION:
        left, right = summary["by_task"][task], baseline["by_task"][task]
        gap = left["mean_absolute_gap_feasible"]
        optimization_rows.append([
            TASK_LABELS[task], f'{left["feasible"]}/{left["instances"]}',
            f'{left["optimal_count"]}/{left["instances"]}',
            f'{right["optimal_count"]}/{right["instances"]}',
            "--" if gap is None else f"{gap:.3f}",
        ])
    latency = sorted(row["latency_seconds"] for row in attempts)
    failures = sum(row["status"] == "failure" for row in attempts)
    return "\n".join([
        r"\subsection{Completed canonical GPT-5.4 anchor}",
        "This separately registered tool-free, default-reasoning GPT-5.4 run uses",
        "the identical canonical structural-challenge requests and own histories.",
        f"It made {len(attempts):,} actual calls, with {failures} failed attempts.",
        "Failures remain in scheduled denominators; no retry or repair was performed.",
        r"\begin{table}[ht]\centering\small",
        latex_table((r"Exact task", "GPT-5.4 (\\%)", "Jev (\\%)", r"95\% interval (pp)"), exact_rows),
        r"\caption{Canonical exact macro-size accuracy. Intervals are paired,",
        r"size-stratified pointwise GPT-minus-Jev differences, not family-wise tests.}",
        r"\end{table}",
        r"\begin{table}[ht]\centering\small",
        latex_table(("Task", "GPT feasible", "GPT optima", "Jev optima", "GPT gap"),
                    optimization_rows),
        r"\caption{Canonical construction results. Gaps condition on GPT feasible",
        r"completion and have different units across rows; they are not a pooled score.}",
        r"\end{table}",
        f"Observed median/p95 GPT request latency was {1000 * latency[len(latency)//2]:.2f}/"
        f"{1000 * latency[int(.95 * len(latency))]:.2f} ms.",
        "Reasoning budget and hosting differ from the small no-thinking local baselines.",
        "These are inspected-bank common-protocol anchors, not the prospective new-seed",
        "confirmation or a matched-compute architectural comparison.",
        "",
    ])


def finalize(anchor, matrix, bundle, output, *, expected_hashes=None):
    expected_hashes = execution_hashes() if expected_hashes is None else expected_hashes
    verify_execution_hashes(expected_hashes)
    suite.require(not output.exists(), "GPT paper evidence already exists.")
    manifest = suite.read_json(anchor / "manifest.json")
    status = suite.read_json(anchor / "status.json")
    suite.require(ready(status, (anchor / "seal.json").exists()), "GPT result is not sealed.")
    suite.require(not manifest["injected_client"] and not manifest["test_only"]
                  and manifest["protocol"] == "explicit_actions"
                  and manifest["profile"] == "gpt54_default_reasoning"
                  and manifest["split"] == "structural_challenge"
                  and manifest["model"]["model"] == "gpt-5.4"
                  and manifest["bundle_manifest_sha256"] == suite.digest(bundle / "manifest.json"),
                  "GPT source is not the registered real common-protocol anchor.")
    frozen = anchor / manifest["source_snapshot"] / "scripts/run_paper_gpt_anchor.py"
    subprocess.run([sys.executable, str(frozen), "verify", "--output", str(anchor)], check=True)
    release.verify(bundle)
    instances = release.labeled_instances(bundle, "structural_challenge", release.manifest_at(bundle))
    attempts = suite.read_rows(anchor / "attempts.jsonl")
    rows = matrices.evaluator.replay(instances, attempts, ModelConfig(**manifest["model"]),
                                    native_action=False)
    suite.require(rows == suite.read_rows(anchor / "episodes.jsonl"), "GPT episode replay mismatch.")
    for row in attempts:
        suite.require(math.isfinite(row["latency_seconds"]) and row["latency_seconds"] >= 0,
                      "Invalid measured latency.")
    _, _, baseline_attempts, baseline_rows = matrices.verified_run(
        matrix / "structural_challenge.jev_action", bundle, "structural_challenge")
    initial = {row["instance_id"]: row["request"] for row in attempts if row["step"] == 0}
    reference = {row["instance_id"]: row["request"] for row in baseline_attempts if row["step"] == 0}
    suite.require(len(initial) == len(instances) and initial == reference,
                  "Canonical GPT/primary requests or coverage differ.")
    summary, baseline = suite.summarize(rows), suite.summarize(baseline_rows)
    clusters = structural_clusters(suite.read_json(bundle / "analysis/structural_challenge.json"))
    paired = paired_intervals(instances, rows, baseline_rows, clusters=clusters)
    text = render(summary, baseline, paired, attempts)
    alerts = [task for task in EXACT if summary["by_task"][task]["correct"] == 0]
    if alerts:
        text += ("\n\\paragraph{Diagnostic review flag.} Zero exact answers were correct "
                 "for: " + ", ".join(TASK_LABELS[task] for task in alerts) +
                 ". These adverse results are retained, but should be investigated before "
                 "attributing them to intrinsic model limitations.\n")
    verify_execution_hashes(expected_hashes)
    output.mkdir(parents=True, exist_ok=False)
    suite.write_json(output / "summary.json", summary)
    suite.write_json(output / "paired-intervals.json", paired)
    suite.write_json(output / "review-alerts.json", {
        "zero_correct_exact_tasks": alerts, "requires_diagnostic_review": bool(alerts),
        "policy": "Flag suspicious outcomes without censoring scores or declaring a proven bug.",
    })
    suite.write_json(output / "provenance.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status["status"], "initial_requests_identical": True,
        "instance_evaluations": len(rows), "calls": len(attempts), "inference_calls": 0,
        "source_sha256": {str(path): suite.digest(path) for path in (
            anchor / "manifest.json", anchor / "seal.json", anchor / "attempts.jsonl",
            anchor / "episodes.jsonl", matrix / "structural_challenge.jev_action/status.json",
            bundle / "manifest.json", Path(__file__))},
        "render_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "analysis_source_sha256": expected_hashes,
    })
    temporary = output / "anchor.tex.pending"
    temporary.write_text(text)
    temporary.rename(output / "anchor.tex")
    return {"status": "verified_and_rendered", "calls": len(attempts),
            "instances": len(rows), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=0)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    suite.require(math.isfinite(args.wait_hours) and args.wait_hours >= 0, "Invalid waiting budget.")
    hashes = execution_hashes()
    suite.write_json(args.anchor / f"paper-finalizer-launch-{os.getpid()}.json", {
        "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "maximum_wait_hours": args.wait_hours, "analysis_source_sha256": hashes,
        "inference_calls": 0,
    })
    deadline = time.monotonic() + 3600 * args.wait_hours
    while not ready(suite.read_json(args.anchor / "status.json"), (args.anchor / "seal.json").exists()):
        suite.require(time.monotonic() < deadline, "GPT anchor still pending at finalizer deadline.")
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    result = finalize(args.anchor, args.matrix, args.bundle, args.output, expected_hashes=hashes)
    if args.build:
        subprocess.run(["latexmk", "-cd", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "-outdir=build", "paper/main.tex"], cwd=REPO, check=True)
        result["pdf_sha256"] = suite.digest(REPO / "paper/build/main.pdf")
    suite.write_json(args.output / "completion.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
