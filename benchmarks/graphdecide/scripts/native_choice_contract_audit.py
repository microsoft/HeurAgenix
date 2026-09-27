#!/usr/bin/env python3
"""Prospective paired native-choice audit; never used by production scoring."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts.summarize_extended_paper import verified_inputs
from src.benchmark import extended_tasks as tasks
from src.benchmark.config import ModelConfig
from src.clients.base import InvalidResponseError
from src.clients.registry import create_client

TASKS = tuple(sorted(tasks._ALL_EXACT))
SIZES = tuple(range(8, 13))
CAP = 120
SELECTION_SEED = "native-choice-contract-audit-20260924-v1"
RELAXED_CODES = frozenset({"invalid_probability_sum", "choice_probability_mismatch"})
DISCLAIMER = (
    "This is a prospective paired-protocol diagnostic, not replacement/recovery of earlier "
    "scores or evidence vendor violated unspecified tolerance."
)


def rank(instance):
    return hashlib.sha256(f"{SELECTION_SEED}:{instance['id']}".encode()).hexdigest(), instance["id"]


def select_cases(instances):
    """Only frozen task/state/labels/IDs participate; no run artifacts are inputs."""
    selected = []
    suite.require(len({i["id"] for i in instances}) == len(instances), "Duplicate instance ID.")
    for task in TASKS:
        for size in SIZES:
            pool = [i for i in instances if i["task"] == task and i["kind"] == "exact"
                    and len(i["state"]["nodes"]) == size]
            labels = ([str(k * (size - 1) // 3) for k in range(4)]
                      if task == "degree_exact" else ["yes", "no"])
            quota = 1 if task == "degree_exact" else 2
            for label in labels:
                candidates = sorted(
                    (i for i in pool if str(i["private"]["answer"]) == label), key=rank)
                suite.require(len(candidates) >= quota, "Insufficient exact-task stratum.")
                selected.extend(candidates[:quota])
    suite.require(len(selected) == CAP and len({i["id"] for i in selected}) == CAP,
                  "Selection must contain exactly 120 distinct instances.")
    return selected


def finite_probability(value):
    return type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value)


def inspect_native(request, body, expected_model):
    """Allowlisted facts only; ignore no contract condition except sum and argmax."""
    issues = []
    body_object = type(body) is dict
    model_matches = body_object and type(body.get("model")) is str and body["model"] == expected_model
    if not body_object:
        issues.append("missing_response" if body is None else "malformed_response")
    if not model_matches:
        issues.append("invalid_model")
    answers = body.get("answers") if body_object else None
    shape = type(answers) is dict and set(answers) == {"decision"}
    answer = answers["decision"] if shape else None
    shape = shape and type(answer) is dict and answer.get("type") == "choice"
    if not shape:
        issues.append("invalid_native_answer")
    answer = answer if type(answer) is dict else {}
    ids = {o.id for o in request.options}
    choice = answer.get("choice")
    supplied = type(choice) is str and choice in ids
    if not supplied:
        issues.append("invalid_choice")
    probabilities = answer.get("probabilities")
    keys_match = type(probabilities) is dict and set(probabilities) == ids
    valid_probabilities = keys_match and all(finite_probability(p) for p in probabilities.values())
    if not valid_probabilities:
        issues.append("invalid_probabilities")
    total = sum(probabilities.values()) if valid_probabilities else None
    maximum = max(probabilities.values()) if valid_probabilities else None
    selected = probabilities[choice] if valid_probabilities and supplied else None
    sum_valid = total is not None and math.isclose(total, 1, rel_tol=0, abs_tol=1e-6)
    argmax_valid = selected is not None and selected + 1e-6 >= maximum
    if valid_probabilities and not sum_valid:
        issues.append("invalid_probability_sum")
    if valid_probabilities and supplied and not argmax_valid:
        issues.append("choice_probability_mismatch")
    if not finite_probability(answer.get("confidence")):
        issues.append("invalid_confidence")
    usage = body.get("usage") if body_object else None
    if type(usage) is not dict or any(
            usage.get(k) is not None and (type(usage[k]) is not int or usage[k] < 0)
            for k in ("input_tokens", "output_tokens")):
        issues.append("invalid_usage")
    return {
        "model_matches": model_matches, "answer_shape_valid": shape,
        "choice_is_string": type(choice) is str, "choice_is_supplied": supplied,
        "probability_keys_match": keys_match, "probability_values_valid": valid_probabilities,
        "probability_sum": total, "sum_within_strict_tolerance": sum_valid,
        "selected_probability": selected, "max_probability": maximum,
        "selected_is_strict_argmax": argmax_valid, "contract_issues": issues,
    }, choice if supplied else None


def action_observation(request, body, expected_model, label, strict):
    diagnostics, choice = inspect_native(request, body, expected_model)
    reasons = [code for code in diagnostics["contract_issues"] if code not in RELAXED_CODES]
    if strict["status"] != "success" and strict["diagnostic_code"] not in RELAXED_CODES:
        reasons.append("strict_failure_not_probability_contract")
    eligible = not reasons
    return {
        "eligible": eligible, "ineligibility_reasons": reasons,
        "selected_option_id": choice if eligible else None,
        "correct": choice == label if eligible else None,
        "native_diagnostics": diagnostics,
    }


def action_execution_record(request, body, expected_model, strict):
    """Prospective controller record; keep strict conformance as a separate observation."""
    observation = action_observation(request, body, expected_model, None, strict)
    observation.pop("correct")
    record = dict(strict)
    if observation["eligible"]:
        record.update(
            status="success", fatal=False, diagnostic_code=None, http_status=None,
            selected_option_id=observation["selected_option_id"], resolved_model=expected_model,
            usage={**{key: body["usage"].get(key) for key in ("input_tokens", "output_tokens")},
                   "reasoning_tokens": None})
    elif strict["status"] == "success":
        record.update(status="failure", fatal=True, diagnostic_code="invalid_response",
                      selected_option_id=None, resolved_model=None)
    record.update(strict_status=strict["status"],
                  strict_diagnostic_code=strict["diagnostic_code"], action_observation=observation)
    return record


async def observe(client, request, case, config):
    started, body = time.perf_counter(), None
    try:
        response = await client.predict(request)
        body = response.raw_output
        strict = suite.response_record(response, request, config)
    except BaseException as exc:
        if isinstance(exc, InvalidResponseError) and exc.raw_output is not None:
            body = exc.raw_output
        strict = {"status": "failure", "selected_option_id": None, **suite.failure(exc)}
    strict["correct"] = strict["status"] == "success" and strict["selected_option_id"] == case["label"]
    return {
        "instance_id": case["instance_id"], "request_id": request.request_id,
        "task": case["task"], "node_count": case["node_count"],
        "strict": strict,
        "action_observation": action_observation(request, body, config.model, case["label"], strict),
        "latency_seconds": time.perf_counter() - started,
    }


def metrics(cases, rows):
    planned = len(cases)
    strict_success = sum(r["strict"]["status"] == "success" for r in rows)
    strict_correct = sum(r["strict"]["correct"] for r in rows)
    eligible = sum(r["action_observation"]["eligible"] for r in rows)
    action_correct = sum(r["action_observation"]["correct"] is True for r in rows)
    return {
        "planned": planned, "attempted": len(rows), "unattempted": planned - len(rows),
        "strict_success": strict_success, "strict_failures": len(rows) - strict_success,
        "strict_correct": strict_correct, "strict_accuracy_planned": strict_correct / planned,
        "strict_accuracy_successful": strict_correct / strict_success if strict_success else None,
        "strict_diagnostics": dict(Counter(r["strict"]["diagnostic_code"] for r in rows
                                           if r["strict"]["status"] != "success")),
        "action_observation_eligible": eligible,
        "action_observation_ineligible": len(rows) - eligible,
        "action_observation_correct": action_correct,
        "action_observation_accuracy_planned": action_correct / planned,
        "action_observation_accuracy_eligible": action_correct / eligible if eligible else None,
        "action_observation_ineligibility_reasons": dict(Counter(
            reason for r in rows for reason in r["action_observation"]["ineligibility_reasons"])),
        "strict_failure_action_eligible": sum(
            r["strict"]["status"] == "failure" and r["action_observation"]["eligible"] for r in rows),
        "strict_failure_action_correct": sum(
            r["strict"]["status"] == "failure" and r["action_observation"]["correct"] is True
            for r in rows),
    }


def summarize(cases, rows, status, diagnostic_code):
    return {
        "disclaimer": DISCLAIMER, "status": status, "diagnostic_code": diagnostic_code,
        "denominators": "Planned rates include all 120 cases, even if aborted; conditional "
                        "rates use strict successes or action-observation eligible responses only. "
                        "No inference means no measured model accuracy.",
        "overall": metrics(cases, rows),
        "by_task": {task: metrics([c for c in cases if c["task"] == task],
                                 [r for r in rows if r["task"] == task]) for task in TASKS},
        "by_task_and_size": {
            task: {str(n): metrics(
                [c for c in cases if c["task"] == task and c["node_count"] == n],
                [r for r in rows if r["task"] == task and r["node_count"] == n])
                   for n in SIZES} for task in TASKS},
    }


def source_stable(root, manifest):
    suite.require(suite.digest(root / "manifest.json") ==
                  suite.read_json(root / "manifest.sha256"), "Source manifest changed.")
    suite.require(all(suite.digest(root / name) == digest
                      for name, digest in manifest["input_sha256"].items()), "Source inputs changed.")
    suite.require(suite.code_hashes() == manifest["code_sha256"], "Current source differs from frozen source.")
    suite.verify_source_snapshot(root, manifest["code_sha256"])


async def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    suite.require(not output.exists(), "Audit output exists; no resume or overwrite.")
    suite.require(output.is_relative_to(REPO), "Audit output must be within the repository.")
    # Freeze the selection before even verification reads primary run artifacts.
    selected = select_cases(suite.read_rows(root / "instances.jsonl"))
    cases = [{
        "instance_id": i["id"], "task": i["task"], "node_count": len(i["state"]["nodes"]),
        "label": str(i["private"]["answer"]), "request": asdict(tasks.next_request(i, [])),
    } for i in selected]
    verified_root, source, verified_instances, _ = verified_inputs(root, ("jev",))
    suite.require(root == verified_root and selected == select_cases(verified_instances),
                  "Frozen selection changed during verification.")
    source_stable(root, source)
    config = ModelConfig(**source["models"]["jev"])
    suite.require(config.provider == "typesafe", "Native Typesafe provider required.")
    audit_hash = suite.digest(Path(__file__))
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER, "source_root": str(root.relative_to(REPO)),
        "source_manifest_sha256": suite.digest(root / "manifest.json"),
        "source_input_sha256": source["input_sha256"], "source_code_sha256": source["code_sha256"],
        "audit_code_sha256": audit_hash, "model_config": asdict(config),
        "planned_calls": CAP, "hard_call_cap": CAP, "retries": 0, "resume": False,
        "selection": {
            "seed": SELECTION_SEED, "ranking": "ascending SHA256(seed + ':' + instance ID), then ID",
            "tasks": TASKS, "sizes": SIZES, "per_task_size": 4,
            "binary_labels": {"yes": 2, "no": 2},
            "degree_targets": "floor(k*(n-1)/3) for k=0,1,2,3; one instance per target",
            "inputs": "frozen instances only; selection fixed before verification reads run artifacts",
        },
        "protocols": {
            "strict": "Unmodified registry TypeSafeClient plus existing suite.response_record validation.",
            "action_observation": "Same response only. Require exact configured model identity, exactly "
            "answers.decision with type choice, supplied string choice, valid probability keys/values, "
            "confidence and usage. Ignore ONLY probability sum and selected-argmax constraints. "
            "Non-probability failures remain ineligible; inspect all fields even after strict failure.",
            "prohibitions": "No probability repair, choice inference, renormalization, retry, prompt "
            "change, primary-score substitution or earlier-failure recovery.",
            "errors": "Existing suite.failure HTTP/timeout/transport/fatal conventions; terminate "
            "on fatal error, preserve planned denominator. Never persist raw provider text.",
        },
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=False)
    suite.write_json(output / "manifest.json", manifest)
    suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
    for name in ("manifest.json", "manifest.sha256"):
        (output / name).chmod(0o444)
    rows, client, status, code = [], None, "aborted", None
    with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
        try:
            with suite.lane_locks(("jev",)):
                suite.shared.credential_preflight(("jev",))
                client = create_client(config.provider, **config.client_kwargs())
                await client.initialize()
                status = "running"
                for instance, case in zip(selected, cases, strict=True):
                    source_stable(root, source)
                    suite.require(suite.digest(Path(__file__)) == audit_hash
                                  and suite.read_json(output / "manifest.sha256") ==
                                  suite.digest(output / "manifest.json"), "Audit plan/source changed.")
                    suite.require(len(rows) < CAP, "Hard call cap reached.")
                    request = tasks.next_request(instance, [])
                    suite.require(suite.json_value(asdict(request)) == suite.json_value(case["request"]),
                                  "Request changed after planning.")
                    row = await observe(client, request, case, config)
                    suite.append_row(stream, row)
                    rows.append(row)
                    if row["strict"]["fatal"]:
                        code = row["strict"]["diagnostic_code"]
                        status = "interrupted" if code == "interrupted" else "aborted"
                        break
                else:
                    status = ("completed_with_failures" if any(
                        r["strict"]["status"] == "failure" for r in rows) else "completed")
                if client is not None:
                    await client.aclose()
                    client = None
        except suite.Refusal:
            status, code = "aborted", "preflight_or_integrity_refusal"
        except BaseException as exc:
            failure = suite.failure(exc)
            code = failure["diagnostic_code"]
            status = "interrupted" if code == "interrupted" else "aborted"
        finally:
            if client is not None:
                try:
                    await client.aclose()
                except BaseException:
                    status, code = "aborted", "shutdown_failed"
    result = summarize(cases, rows, status, code)
    suite.write_json(output / "summary.json", result)
    suite.write_json(output / "status.json", {
        "status": status, "diagnostic_code": code, "planned_calls": CAP, "attempted_calls": len(rows),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {name: suite.digest(output / name)
                            for name in ("manifest.json", "manifest.sha256", "attempts.jsonl", "summary.json")},
    })
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return result


async def run_cli(root, output):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run(root, output)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(run_cli(args.root, args.output))
        print(suite.canonical({"status": result["status"], "diagnostic_code": result["diagnostic_code"],
                               "overall": result["overall"]}))
        return 0 if result["status"].startswith("completed") else 2
    except BaseException:
        print('{"status":"refused","diagnostic_code":"verification_or_execution_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
