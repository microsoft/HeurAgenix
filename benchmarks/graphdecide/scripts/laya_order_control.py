#!/usr/bin/env python3
"""Posthoc, label-balanced, 20-call CPU option-order diagnostic for pinned Laya."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import laya_cpu_pilot as pilot

ROOT = REPO / "output/laya-order-control-20260924-v1"
TASKS = ("adjacency", "articulation_point", "cycle_detection",
         "distance_threshold", "pair_connectivity")
LIMITS = {"max_forwards": 20, "forward_seconds": 30, "load_seconds": 300,
          "total_seconds": 600, "torch_threads": 4, "interop_threads": 1}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_plan(instances):
    from scripts import evaluate_graph_release as evaluation
    plan = []
    for task in TASKS:
        for answer in ("yes", "no"):
            pool = sorted((i for i in instances if i["task"] == task
                           and len(i["state"]["nodes"]) == 8
                           and str(i["private"]["answer"]) == answer), key=lambda i: i["id"])
            if not pool:
                raise ValueError(f"Missing fixed-panel diagnostic stratum {task}/{answer}")
            instance = pool[0]
            original = canonical(asdict(evaluation.public_request(instance, [])))
            for order in ("original", "reversed"):
                request = deepcopy(original)
                if order == "reversed":
                    request["options"].reverse()
                plan.append({"index": len(plan), "instance_id": instance["id"], "task": task,
                             "node_count": 8, "reference_answer": answer, "order": order,
                             "request": request, "request_sha256": hash_value(request)})
    assert len(plan) == 20
    return plan


def token_mapping(tok, request):
    from laya.common import build_sequence, render_options
    check = pilot.coverage(tok, request)
    definition = pilot.question(request)
    q = {"t": "choice", "ins": definition["instructions"], "crit": definition["criteria"]}
    ids, positions = build_sequence(tok, request["state"], q, 1024, 256)
    mappings = []
    for option, text, position in zip(request["options"], render_options(q), positions):
        expected = tok(" " + text, add_special_tokens=False, truncation=False)["input_ids"]
        matches = ids[position] == tok.mask_token_id and ids[
            position + 1:position + 1 + len(expected)] == expected
        mappings.append({"semantic_id": option["id"], "marker_position": position,
                         "option_token_count": len(expected), "full_option_tokens_match": matches})
    verified = check["compatible"] and len(mappings) == len(request["options"]) and all(
        m["full_option_tokens_match"] for m in mappings)
    return {**check, "mask_mapping": mappings, "lossless_mask_mapping_verified": verified}


def original_audit(tok, instances):
    by_id = {i["id"]: i for i in instances}
    original = [r for r in pilot.read_rows(pilot.ROOT / "run/attempts.jsonl")
                if by_id[r["instance_id"]]["task"] in TASKS]
    rows = []
    for row in original:
        result = row["backend"]
        pilot.validate_answer(row["request"], result["answer"], result["raw_logits"])
        mapping = token_mapping(tok, row["request"])
        assert mapping["lossless_mask_mapping_verified"]
        assert mapping["native_sequence_sha256"] == result["coverage"]["native_sequence_sha256"]
        ids = [o["id"] for o in row["request"]["options"]]
        index = max(range(len(ids)), key=lambda i: result["raw_logits"][i])
        assert ids[index] == row["selected_option_id"] == result["answer"]["choice"]
        assert result["forward_count"] == 1
        rows.append({"instance_id": row["instance_id"], "task": by_id[row["instance_id"]]["task"],
                     "state_sha256": hash_value(row["request"]["state"]),
                     "request_sha256": hash_value(row["request"]),
                     "raw_logits": result["raw_logits"], "native_argmax_index": index,
                     "mapped_semantic_id": ids[index], "mask_and_token_audit": mapping})
    assert len(rows) == 40
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    return {
        "source_attempts_sha256": pilot.digest(pilot.ROOT / "run/attempts.jsonl"),
        "binary_decisions": len(rows),
        "semantic_choices": dict(Counter(r["mapped_semantic_id"] for r in rows)),
        "argmax_indices": dict(Counter(r["native_argmax_index"] for r in rows)),
        "unique_logits_vectors": len({tuple(r["raw_logits"]) for r in rows}),
        "unique_public_states": len({r["state_sha256"] for r in rows}),
        "all_full_mask_mappings_verified": True,
        "cached_response_evidence": "No identical logits/state hashes among 40 requests. "
            "Audited upstream system_one -> predict_batch -> _forward -> _infer -> "
            "DecisionModel.forward runs once per request; native forward hook captured once. "
            "Tokenizer caching is not response caching. This is implementation evidence, "
            "not an inference from constant choices alone.",
        "by_task": {task: {"requests": len(items),
                           "distinct_logits": len({tuple(r["raw_logits"]) for r in items}),
                           "minimum_yes_minus_no_logit": min(
                               r["raw_logits"][0] - r["raw_logits"][1] for r in items),
                           "maximum_yes_minus_no_logit": max(
                               r["raw_logits"][0] - r["raw_logits"][1] for r in items)}
                    for task, items in groups.items()},
        "rows": rows}


def prepare(root):
    if root.exists():
        raise RuntimeError("New output root required")
    pilot.verify_registration(pilot.ROOT)
    pilot.imports(pilot.ROOT)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        pilot.ROOT / "model/tokenizer", local_files_only=True, trust_remote_code=False)
    instances, _ = pilot.panel()
    plan = select_plan(instances)
    root.mkdir()
    pilot.save(root / "original-40-binary-audit.json", original_audit(tok, instances))
    for row in plan:
        row["token_audit"] = token_mapping(tok, row["request"])
    pilot.save(root / "plan.json", plan)
    source_paths = [Path(__file__), REPO / "tests/test_laya_order_control.py"]
    hashes = {str(p.relative_to(REPO)): pilot.digest(p) for p in source_paths}
    for path in source_paths:
        snapshot = root / "source-snapshot" / path.relative_to(REPO)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        snapshot.chmod(0o444)
    # Preserve the pre-diagnostic document before its explicitly requested update.
    doc = REPO / "docs/laya-cpu-pilot-20260924.md"
    (root / "prior-laya-document.md").write_bytes(doc.read_bytes())
    registration = {
        "registered_at_unix": time.time(), "registered_before_inference": True,
        "classification": "posthoc_adaptive_label_balanced_diagnostic_not_confirmatory",
        "selection": "Five binary tasks; n8; lexicographically first yes and first no "
            "instance from the already fixed 64-panel; each original then reversed order.",
        "scheduled_cases": 10, "scheduled_calls": 20, "limits": LIMITS,
        "base_registration_sha256": pilot.digest(pilot.ROOT / "registration.json"),
        "source_revision": pilot.SOURCE, "checkpoint_revision": pilot.REVISION,
        "checkpoint_sha256": pilot.WEIGHTS_SHA, "selection_sha256": pilot.SELECTION_SHA,
        "plan_sha256": pilot.digest(root / "plan.json"),
        "original_audit_sha256": pilot.digest(root / "original-40-binary-audit.json"),
        "code_sha256": hashes,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "initial_compatible_calls": sum(r["token_audit"]["compatible"] for r in plan),
        "policy": {"cpu_only": True, "retry": False, "repair": False, "truncate": False,
                   "recalibration": False, "renormalize_native_probabilities": False,
                   "all_20_scheduled_in_denominator": True,
                   "oracle_values_sent_to_worker": False, "new_confirmatory_test": False,
                   "main_pilot_rerun_or_repair": False},
        "probability_validation": "Retain four-decimal native values. Sum deviation is "
            "diagnostic; frozen native validation allows k*0.000051 rounding tolerance. "
            "Native entropy confidence and pmax confidence are distinct, uncalibrated.",
        "baseline_note": "No pending-model or other-model scores read."}
    pilot.save(root / "registration.json", registration)
    pilot.save(root / "registration.sha256", pilot.digest(root / "registration.json"))
    for name in ("plan.json", "registration.json", "registration.sha256",
                 "original-40-binary-audit.json"):
        (root / name).chmod(0o444)
    return registration


def verify(root):
    pilot.verify_registration(pilot.ROOT)
    reg = json.loads((root / "registration.json").read_text())
    assert pilot.digest(root / "registration.json") == json.loads(
        (root / "registration.sha256").read_text())
    assert pilot.digest(pilot.ROOT / "registration.json") == reg["base_registration_sha256"]
    assert pilot.digest(root / "plan.json") == reg["plan_sha256"]
    assert pilot.digest(root / "original-40-binary-audit.json") == reg["original_audit_sha256"]
    for name, expected in reg["code_sha256"].items():
        assert pilot.digest(REPO / name) == expected
    assert {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()} == reg["packages"]
    return reg


def analyze(plan, attempts):
    assert len(plan) == len(attempts) == 20
    for planned, actual in zip(plan, attempts):
        assert actual["index"] == planned["index"]
        assert actual["request"] == planned["request"]
        assert hash_value(actual["request"]) == planned["request_sha256"]
        if actual["status"] == "success":
            result = actual["backend"]
            pilot.validate_answer(actual["request"], result["answer"], result["raw_logits"])
            assert result["coverage"]["compatible"]
            assert result["coverage"]["native_sequence_sha256"] == planned[
                "token_audit"]["native_sequence_sha256"]
            assert result["forward_count"] == 1 and not result["cuda_initialized"]
            assert result["answer"]["choice"] == actual["choice"]
    groups = []
    for pos in range(0, 20, 2):
        original, reversed_row = attempts[pos:pos + 2]
        expected, reversal = plan[pos:pos + 2]
        assert expected["instance_id"] == reversal["instance_id"]
        assert reversal["request"]["options"] == list(reversed(expected["request"]["options"]))
        assert {k: v for k, v in expected["request"].items() if k != "options"} == {
            k: v for k, v in reversal["request"].items() if k != "options"}
        complete = original["status"] == reversed_row["status"] == "success"
        row = {"instance_id": expected["instance_id"], "task": expected["task"],
               "reference_answer": expected["reference_answer"], "paired_complete": complete,
               "original_choice": original.get("choice"), "reversed_choice": reversed_row.get("choice"),
               "semantic_choice_preserved": original.get("choice") == reversed_row.get("choice")
                   if complete else None}
        if complete:
            row["original_argmax_index"] = [o["id"] for o in original["request"]["options"]].index(
                original["choice"])
            row["reversed_argmax_index"] = [o["id"] for o in reversed_row["request"]["options"]].index(
                reversed_row["choice"])
            row["position_preserved"] = row["original_argmax_index"] == row["reversed_argmax_index"]
            row["raw_logits_by_semantic_id"] = {
                order: dict(zip([o["id"] for o in actual["request"]["options"]],
                                actual["backend"]["raw_logits"]))
                for order, actual in (("original", original), ("reversed", reversed_row))}
        groups.append(row)
    by_order = {}
    for order in ("original", "reversed"):
        selected = [(p, a) for p, a in zip(plan, attempts) if p["order"] == order]
        success = [(p, a) for p, a in selected if a["status"] == "success"]
        by_order[order] = {
            "scheduled": len(selected), "successful": len(success),
            "semantic_choices": dict(Counter(a["choice"] for _, a in success)),
            "argmax_indices": dict(Counter([o["id"] for o in a["request"]["options"]].index(
                a["choice"]) for _, a in success)),
            "correct": sum(a["choice"] == p["reference_answer"] for p, a in success),
            "probability_sum_errors": [
                sum(a["backend"]["answer"]["probabilities"].values()) - 1 for _, a in success]}
    old = {r["instance_id"]: r for r in pilot.read_rows(pilot.ROOT / "run/attempts.jsonl")
           if r["step"] == 0}
    repeat_differences = []
    for planned, actual in zip(plan, attempts):
        if planned["order"] == "original" and actual["status"] == "success":
            prior = old[planned["instance_id"]]
            assert prior["request"] == actual["request"]
            repeat_differences.append(max(abs(a - b) for a, b in zip(
                prior["backend"]["raw_logits"], actual["backend"]["raw_logits"])))
    return {"scheduled_calls": 20, "scheduled_paired_cases": 10,
            "successful_calls": sum(a["status"] == "success" for a in attempts),
            "statuses": dict(Counter(a["status"] for a in attempts)),
            "completed_pairs": sum(g["paired_complete"] for g in groups),
            "semantic_preserved_pairs": sum(g["semantic_choice_preserved"] is True for g in groups),
            "position_preserved_pairs": sum(g.get("position_preserved") is True for g in groups),
            "by_order": by_order, "paired_cases": groups,
            "original_order_repeat_max_abs_logit_difference": max(repeat_differences, default=None),
            "interpretation_scope": "Posthoc selection, 10 distinct states/20 calls, paired "
                "positions not independent cases. Not a ranking, calibration, or confirmatory test."}


def run(root):
    started = time.monotonic()
    deadline = started + 600
    verify(root)
    plan = json.loads((root / "plan.json").read_text())
    output = root / "run"
    output.mkdir()
    worker, attempts, fatal, reserved = None, [], None, 0
    try:
        worker = pilot.Worker(pilot.ROOT, output)
        ready = worker.receive(min(300, max(0, deadline - time.monotonic())))
        pilot.save(output / "runtime.json", ready)
        assert ready["status"] == "ready" and not ready["cuda_initialized"]
        assert ready["threads"] == 4 and ready["interop_threads"] == 1
        with (output / "attempts.jsonl").open("x") as stream:
            for planned in plan:
                row = {"index": planned["index"], "request": planned["request"],
                       "status": "unattempted", "choice": None, "forward_count": 0}
                if not fatal:
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or reserved >= 20:
                            raise TimeoutError("Registered total/call budget exhausted")
                        pilot.save(output / "status.json", {
                            "status": "running", "index": planned["index"],
                            "forward_budget_reserved": reserved})
                        if not planned["token_audit"]["compatible"]:
                            row["status"] = "unsupported"
                        else:
                            reserved += 1
                            row["forward_count"] = 1
                            tick = time.monotonic()
                            result = worker.predict(planned["request"], min(30, remaining))
                            row.update(backend=result, status=result["status"],
                                       supervisor_seconds=time.monotonic() - tick)
                            if result["status"] == "success":
                                pilot.validate_answer(planned["request"], result["answer"],
                                                      result["raw_logits"])
                                row["choice"] = result["answer"]["choice"]
                            elif result["status"] == "unsupported":
                                row["forward_count"] = 0
                                reserved -= 1
                            else:
                                fatal = result.get("error", result["status"])
                    except Exception as exc:
                        fatal = f"{type(exc).__name__}: {str(exc)[:1000]}"
                        row.update(status="failure", error=fatal)
                pilot.append(stream, row)
                attempts.append(row)
    except BaseException as exc:
        fatal = f"{type(exc).__name__}: {str(exc)[:1000]}"
    finally:
        if worker:
            worker.close()
        if len(attempts) < 20:
            with (output / "attempts.jsonl").open("a") as stream:
                for planned in plan[len(attempts):]:
                    row = {"index": planned["index"], "request": planned["request"],
                           "status": "unattempted", "choice": None, "forward_count": 0}
                    pilot.append(stream, row)
        attempts = pilot.read_rows(output / "attempts.jsonl")
        summary = analyze(plan, attempts)
        summary.update(fatal=fatal, wall_seconds=time.monotonic() - started,
                       forward_budget_reserved=reserved)
        pilot.save(output / "summary.json", summary)
        pilot.save(output / "status.json", {
            "status": "aborted" if fatal else "completed", "fatal": fatal,
            "scheduled_calls": 20, "successful_calls": summary["successful_calls"],
            "wall_seconds": summary["wall_seconds"], "worker_released": True})
    verify(root)
    return summary


def replay(root):
    verify(root)
    pilot.imports(pilot.ROOT)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        pilot.ROOT / "model/tokenizer", local_files_only=True, trust_remote_code=False)
    instances, _ = pilot.panel()
    planned = json.loads((root / "plan.json").read_text())
    regenerated = select_plan(instances)
    for original, fresh in zip(planned, regenerated):
        fresh["token_audit"] = token_mapping(tok, fresh["request"])
        assert original == fresh
    attempts = pilot.read_rows(root / "run/attempts.jsonl")
    summary = analyze(planned, attempts)
    stored = json.loads((root / "run/summary.json").read_text())
    assert all(stored[k] == value for k, value in summary.items())
    result = {"verified": True, "scheduled_calls": 20, "scheduled_cases": 10,
              "regenerated_selection_and_both_orders": True,
              "lossless_marker_mapping_reverified": True,
              "attempts_sha256": pilot.digest(root / "run/attempts.jsonl"),
              "summary_sha256": pilot.digest(root / "run/summary.json")}
    pilot.save(root / "run/offline-replay.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "replay"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    value = {"prepare": prepare, "run": run, "replay": replay}[args.mode](args.root)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
