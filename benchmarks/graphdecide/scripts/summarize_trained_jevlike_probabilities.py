#!/usr/bin/env python3
"""Derive temperature-labelled traces from immutable trained-pilot outputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.prepare_trained_jevlike import digest, save


def trace(attempt, runtime):
    metadata = runtime["runtime"]
    temperature = metadata["temperature"]
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid recorded upstream temperature")
    probabilities = attempt.get("probabilities")
    selected = attempt.get("selected_option_id")
    success = attempt["status"] == "success"
    if success and (not isinstance(probabilities, dict) or selected not in probabilities):
        raise ValueError("Successful attempt lacks its recorded native distribution")
    return {
        "instance_id": attempt["instance_id"], "step": attempt["step"],
        "request_id": attempt["request"]["request_id"], "status": attempt["status"],
        "candidate": metadata["candidate"], "dtype": metadata["dtype"],
        "selected_option_id": selected, "probabilities": probabilities,
        "probability_kind": attempt.get("probability_kind"),
        "native_temperature": temperature, "temperature_applied_by": "pinned_upstream_inference",
        "confidence": probabilities[selected] if success else None,
        "confidence_source": "derived_from_returned_selected_option_probability",
        "confidence_semantics": "uncalibrated_on_this_graph_panel_not_a_correctness_guarantee",
        "logits": attempt.get("backend", {}).get("result", {}).get("logits"),
        "logits_semantics": "post_upstream_temperature_if_available",
        "strict_status": attempt.get("strict_status"),
        "action_observation": attempt.get("action_observation"),
    }


def summarize(root):
    target = root / "probability-traces.jsonl"
    if target.exists():
        raise RuntimeError("Derived trace already exists; refuse overwrite")
    inputs, rows = {}, []
    for candidate in ("decider", "kev"):
        run = root / f"run-{candidate}"
        status = json.loads((run / "status.json").read_text())
        if status["status"] not in {"completed", "completed_with_failures", "aborted"}:
            raise RuntimeError("Model run is not terminal")
        inputs[str(run / "status.json")] = digest(run / "status.json")
        runtime = run / "runtime.json"
        attempts = run / "attempts.jsonl"
        if not attempts.exists():
            if status["calls"] != 0:
                raise RuntimeError("Missing nonempty attempt stream")
            continue
        inputs[str(attempts)] = digest(attempts)
        inputs[str(runtime)] = digest(runtime)
        metadata = json.loads(runtime.read_text())
        for line in attempts.read_text().splitlines():
            rows.append(trace(json.loads(line), metadata))
    with target.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    save(root / "probability-trace-provenance.json", {
        "derived_reporting_only": True, "inference_outputs_modified": False,
        "source_sha256": digest(__file__), "input_sha256": inputs,
        "output_sha256": digest(target), "rows": len(rows),
        "runtime_scope": "compatible_runtime_upstream_inference_pilot_not_full_environment_reproduction",
        "calibration_claim": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.root.resolve())
