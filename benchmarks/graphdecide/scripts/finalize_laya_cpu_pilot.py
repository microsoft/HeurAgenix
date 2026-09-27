#!/usr/bin/env python3
"""Replay the durable Laya ledger after the frozen in-memory finalizer failed.

The original registration/source and run status remain untouched. No inference
is performed. Canonical JSON round-tripping removes dataclass tuple/list
representation differences, not differences in public request content.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import laya_cpu_pilot as pilot


def finalize(root):
    pilot.verify_registration(root)
    from scripts import evaluate_graph_release as evaluation
    output = root / "run"
    if (output / "completion.json").exists():
        raise RuntimeError("Finalization already completed; use replay to reverify")
    attempts = pilot.read_rows(output / "attempts.jsonl")
    instances, _ = pilot.panel()
    replayed = evaluation.replay(
        instances, attempts, SimpleNamespace(model=pilot.MODEL), native_action=False)
    fatal_rows = [r for r in attempts if r["fatal"]]
    fatal = fatal_rows[0]["diagnostic_code"] if fatal_rows else None
    # A prefix-only ledger has not completed just because all observed rows passed.
    if any(not r["feasible"] and r["failure_count"] == 0 for r in replayed):
        fatal = fatal or "incomplete_durable_ledger"
    forwards = sum(r.get("forward_count", 0) for r in attempts)
    assert forwards <= 160
    if (output / "scores.json").exists() or (output / "summary.json").exists():
        raise RuntimeError("Refusing to overwrite earlier scoring outputs")
    summary = pilot.summarize(root, instances, attempts, fatal, forwards, None)
    pilot.replay(root)
    observed = ((output / "attempts.jsonl").stat().st_mtime
                - (output / "worker-process.json").stat().st_mtime)
    completion = {
        "status": "completed_with_initial_unsupported" if not fatal else "aborted",
        "inference_retried": False, "source_or_registration_modified": False,
        "postprocessing_recovery": "Frozen runner retained asdict option tuples in its "
            "in-memory ledger; canonical replay compares JSON lists. All persisted "
            "JSON requests/actions replay exactly. Finalized from the durable ledger.",
        "source_script_sha256": pilot.digest(Path(__file__)),
        "original_status_preserved": str(output / "status.json"),
        "original_error_log": str(root / "run.log"),
        "scheduled_episodes": len(replayed), "official_forwards": forwards,
        "attempt_statuses": dict(Counter(r["status"] for r in attempts)),
        "execution_categories": summary["execution_categories"],
        "wall_seconds": None,
        "worker_creation_to_last_ledger_write_seconds_from_filesystem": observed,
        "timing_caveat": "Exact supervisor wall timer was not persisted before finalizer "
            "failure; filesystem interval is observational, not benchmark latency.",
        "artifacts_sha256": {name: pilot.digest(output / name) for name in (
            "attempts.jsonl", "scores.json", "summary.json", "offline-replay.json",
            "runtime.json", "worker-process.json")}}
    pilot.save(output / "completion.json", completion)
    return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=pilot.ROOT)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root), indent=2))
