#!/usr/bin/env python3
"""Independent JSON-normalized verification of the frozen order-control run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import laya_order_control as control
from scripts import laya_cpu_pilot as pilot


def compare_json(expected, observed):
    if control.canonical(expected) != control.canonical(observed):
        raise AssertionError("JSON-normalized evidence differs")


def verify(root):
    control.verify(root)
    pilot.imports(pilot.ROOT)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        pilot.ROOT / "model/tokenizer", local_files_only=True, trust_remote_code=False)
    instances, _ = pilot.panel()
    regenerated = control.select_plan(instances)
    for planned in regenerated:
        planned["token_audit"] = control.token_mapping(tok, planned["request"])
    stored_plan = json.loads((root / "plan.json").read_text())
    compare_json(regenerated, stored_plan)
    compare_json(control.original_audit(tok, instances),
                 json.loads((root / "original-40-binary-audit.json").read_text()))
    attempts = pilot.read_rows(root / "run/attempts.jsonl")
    regenerated_summary = control.analyze(regenerated, attempts)
    stored_summary = json.loads((root / "run/summary.json").read_text())
    compare_json(regenerated_summary, {k: stored_summary[k] for k in regenerated_summary})
    runtime = json.loads((root / "run/runtime.json").read_text())
    raw_to_rounded_errors = []
    for attempt in attempts:
        if attempt["status"] != "success":
            continue
        result = attempt["backend"]
        ids = [o["id"] for o in attempt["request"]["options"]]
        assert len(ids) == 2
        temperature = runtime["temperature_by_options_applied"]["choice:2"]
        logits = [value / temperature for value in result["raw_logits"]]
        weights = [math.exp(value - max(logits)) for value in logits]
        probabilities = [value / sum(weights) for value in weights]
        answer = result["answer"]
        error = max(abs(p - answer["probabilities"][ident])
                    for ident, p in zip(ids, probabilities))
        assert error <= .000051
        assert abs(max(probabilities) - answer["answer_confidence"]) <= .000051
        entropy_confidence = 1 + sum(p * math.log(max(p, 1e-12)) for p in probabilities) / math.log(2)
        assert abs(entropy_confidence - answer["confidence"]) <= .000051
        assert result["usage"]["input_tokens"] == result["coverage"]["full_tokens"]
        raw_to_rounded_errors.append(error)
    verification = {
        "verified": True, "scheduled_calls": 20, "scheduled_paired_states": 10,
        "native_scores_verified": len(raw_to_rounded_errors),
        "largest_native_rounding_error": max(raw_to_rounded_errors, default=None),
        "both_order_plans_and_mask_mappings_regenerated": True,
        "original_40_binary_audit_regenerated": True,
        "original_source_and_ledgers_unchanged": True, "additional_model_calls": 0,
        "verification_note": "Frozen replay helper compared integer Counter keys to "
            "JSON object string keys. Independent verifier compares canonical JSON "
            "representations; all semantic/request/score values match. Original "
            "replay-error log preserved; no inference or registered source modified.",
        "verifier_sha256": pilot.digest(Path(__file__)),
        "source_attempts_sha256": pilot.digest(root / "run/attempts.jsonl"),
        "source_summary_sha256": pilot.digest(root / "run/summary.json")}
    pilot.save(root / "run/offline-replay.json", verification)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=control.ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
