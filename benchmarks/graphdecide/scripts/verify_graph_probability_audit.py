#!/usr/bin/env python3
"""Version 1 offline recovery of probability-audit arithmetic; never makes model calls."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import graph_probability_audit as original

suite = original.suite
VERSION = "graph-probability-offline-verifier-v1"
BOOKKEEPING_ATOL = 1e-12
VECTOR_ATOL = original.TOLERANCE
PRODUCERS = {
    "jev_action": "/home/xianya/miniconda3/envs/jev/bin/python",
    "qwen4_token_scores": "/fastdata/xianya/vllm-venv/bin/python",
}
ORIGINAL_EXITS = {"jev_action": 2, "qwen4_token_scores": 0}
ROW_KEYS = {
    "instance_id", "request_sha256", "strict", "native_diagnostics", "action_eligible",
    "action_ineligibility_reasons", "selected_option_id", "confidence", "probabilities",
    "probability_sum", "vector_omission_reasons", "latency_seconds",
}
VECTOR_REASONS = {"action_ineligible", "missing_probabilities", "invalid_probability_keys",
                  "invalid_probability_values", "invalid_probability_sum"}
ACTION_REASONS = original.native.InvalidResponseError._DIAGNOSTIC_CODES | {
    "missing_response", "malformed_response", "strict_failure_not_probability_contract",
    "strict_failure", "invalid_score_vector",
}
NATIVE_FLAGS = {
    "model_matches", "answer_shape_valid", "choice_is_string", "choice_is_supplied",
    "probability_keys_match", "probability_values_valid", "sum_within_strict_tolerance",
    "selected_is_strict_argmax",
}
NATIVE_KEYS = NATIVE_FLAGS | {
    "probability_sum", "selected_probability", "max_probability", "contract_issues",
}


class VerificationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise VerificationError(code, message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def probability(value):
    return finite(value) and 0 <= value <= 1


def checked_reasons(value, allowed):
    return (type(value) is list and all(type(v) is str and v in allowed for v in value)
            and len(set(value)) == len(value))


def check_sum_bookkeeping(derived, stored):
    require(finite(stored) and math.isclose(derived, stored, rel_tol=0, abs_tol=BOOKKEEPING_ATOL),
            "derived_sum_bookkeeping_mismatch",
            "Derived sum differs from stored sum by more than 1e-12; vector gate remains 1e-6.")
    return abs(derived - stored)


def validate_row(case, row, profile, config):
    require(type(row) is dict and set(row) == ROW_KEYS, "row_schema", "Unexpected attempt fields.")
    require(row["instance_id"] == case["instance_id"]
            and row["request_sha256"] == case["request_sha256"] == original.value_hash(case["request"]),
            "request_identity", "Request hash, instance identity, or order changed.")
    ids = [o["id"] for o in case["request"]["options"]]
    require(case["label"] in ids, "label_identity", "Label is not a supplied choice.")
    require(finite(row["latency_seconds"]) and row["latency_seconds"] >= 0,
            "latency_schema", "Invalid latency.")
    require(row["confidence"] is None or probability(row["confidence"]),
            "confidence_schema", "Invalid saved confidence.")
    require(checked_reasons(row["vector_omission_reasons"], VECTOR_REASONS)
            and checked_reasons(row["action_ineligibility_reasons"], ACTION_REASONS),
            "reason_schema", "Unknown or malformed omission/ineligibility reasons.")
    strict = row["strict"]
    require(type(strict) is dict and set(strict) == {
        "status", "selected_option_id", "resolved_model", "usage",
        "diagnostic_code", "http_status", "fatal"},
        "strict_schema", "Unexpected strict-record fields.")
    require(type(strict["fatal"]) is bool and strict["status"] in ("success", "failure"),
            "strict_schema", "Invalid strict status/fatality.")
    usage = strict["usage"]
    require(type(usage) is dict and set(usage) == set(suite.USAGE_KEYS)
            and all(v is None or type(v) is int and v >= 0 for v in usage.values()),
            "usage_schema", "Invalid token usage.")
    if strict["status"] == "success":
        require(strict["resolved_model"] == config["model"] and strict["selected_option_id"] in ids
                and strict["diagnostic_code"] is None and strict["http_status"] is None
                and not strict["fatal"], "strict_schema", "Invalid strict success.")
    else:
        require(strict["selected_option_id"] is None and strict["resolved_model"] is None
                and strict["diagnostic_code"] in suite.DIAGNOSTICS,
                "strict_schema", "Invalid strict failure.")
        http = strict["http_status"]
        require(http is None or type(http) is int and 100 <= http <= 599,
                "strict_schema", "Invalid HTTP diagnostic.")
    eligible, selected = row["action_eligible"], row["selected_option_id"]
    require(type(eligible) is bool and
            ((eligible and type(selected) is str and selected in ids
              and not row["action_ineligibility_reasons"])
             or (not eligible and selected is None and row["action_ineligibility_reasons"])),
            "action_schema", "Invalid action eligibility/choice.")
    diagnostics = row["native_diagnostics"]
    if profile == "jev_action":
        require(type(diagnostics) is dict and set(diagnostics) == NATIVE_KEYS
                and all(type(diagnostics[k]) is bool for k in NATIVE_FLAGS)
                and checked_reasons(diagnostics["contract_issues"], ACTION_REASONS),
                "native_schema", "Invalid allowlisted native diagnostics.")
        reasons = [c for c in diagnostics["contract_issues"] if c not in original.native.RELAXED_CODES]
        if strict["status"] != "success" and strict["diagnostic_code"] not in original.native.RELAXED_CODES:
            reasons.append("strict_failure_not_probability_contract")
        require(row["action_ineligibility_reasons"] == reasons and eligible == (not reasons),
                "native_eligibility", "Conservative native eligibility changed.")
        if eligible:
            require(all(diagnostics[k] for k in NATIVE_FLAGS - {
                "sum_within_strict_tolerance", "selected_is_strict_argmax"})
                    and probability(row["confidence"]), "native_eligibility", "Missing native contract checks.")
    else:
        require(profile == "qwen4_token_scores" and diagnostics is None and row["confidence"] is None,
                "token_schema", "Unexpected token-score profile/diagnostics/confidence.")
        require(not eligible or strict["status"] == "success",
                "token_eligibility", "Token-score failure was rescued.")
    if eligible and strict["status"] == "success":
        require(selected == strict["selected_option_id"], "choice_changed", "Stored action differs from actual choice.")

    vector, stored_sum = row["probabilities"], row["probability_sum"]
    if vector is None:
        require(bool(row["vector_omission_reasons"]), "omission_schema", "Omitted vector lacks a reason.")
        if "invalid_probability_sum" in row["vector_omission_reasons"]:
            require(finite(stored_sum) and 0 <= stored_sum <= len(ids)
                    and not math.isclose(stored_sum, 1, rel_tol=0, abs_tol=VECTOR_ATOL),
                    "omission_gate", "Omitted sum does not fail the unchanged 1e-6 gate.")
            if diagnostics is not None:
                check_sum_bookkeeping(stored_sum, diagnostics["probability_sum"])
        return {"stable_sum": None, "bookkeeping_abs_difference": None}

    require(eligible and not row["vector_omission_reasons"] and type(vector) is dict
            and set(vector) == set(ids) and all(probability(v) for v in vector.values()),
            "vector_schema", "Retained vector violates keys, values, or action eligibility.")
    stable_sum = math.fsum(vector[key] for key in ids)
    delta = check_sum_bookkeeping(stable_sum, stored_sum)
    # These are eligibility checks, not bookkeeping comparisons. No added tolerance.
    sums = (stable_sum, stored_sum, sum(vector.values()), sum(vector[key] for key in ids))
    require(all(math.isclose(total, 1, rel_tol=0, abs_tol=VECTOR_ATOL) for total in sums),
            "vector_gate", "Retained vector fails or changes the original absolute 1e-6 gate.")
    if diagnostics is not None:
        check_sum_bookkeeping(stable_sum, diagnostics["probability_sum"])
        require(diagnostics["selected_probability"] == vector[selected]
                and diagnostics["max_probability"] == max(vector.values())
                and diagnostics["sum_within_strict_tolerance"],
                "native_vector_consistency", "Native vector diagnostics disagree with actual numbers.")
        require(diagnostics["selected_is_strict_argmax"] ==
                (vector[selected] + 1e-6 >= max(vector.values())),
                "native_argmax_consistency", "Native selected-argmax diagnostic changed.")
    return {"stable_sum": stable_sum, "bookkeeping_abs_difference": delta}


def score_row(case, row, profile, config):
    validation = validate_row(case, row, profile, config)
    vector, selected = row["probabilities"], row["selected_option_id"]
    score = {
        "instance_id": case["instance_id"], "task": case["task"],
        "action_correct": row["action_eligible"] and selected == case["label"],
        "valid_vector": vector is not None, "choice_argmax_disagreement": None,
        "brier": None, "log_loss": None, "log_loss_infinite": False,
        "uniform_brier": None, "uniform_log_loss": None, "selected_probability": None,
        "reliability_bin": None,
    }
    if vector is not None:
        true_probability, chosen_probability = vector[case["label"]], vector[selected]
        score.update(
            brier=math.fsum((vector[o["id"]] - int(o["id"] == case["label"])) ** 2
                            for o in case["request"]["options"]),
            log_loss=-math.log(true_probability) if true_probability > 0 else None,
            log_loss_infinite=true_probability == 0,
            uniform_brier=1 - 1 / len(vector), uniform_log_loss=math.log(len(vector)),
            selected_probability=chosen_probability,
            choice_argmax_disagreement=chosen_probability < max(vector.values()),
            reliability_bin=min(bisect_right(original.BIN_EDGES, chosen_probability) - 1, 4))
    return score, validation


def mean(values):
    return math.fsum(values) / len(values) if values else None


def task_summary(cases, rows, scores):
    valid = [s for s in scores if s["valid_vector"]]
    n = len(valid)
    infinite = sum(s["log_loss_infinite"] for s in valid)
    finite_loss = math.fsum(s["log_loss"] for s in valid if not s["log_loss_infinite"])
    bins = []
    for index in range(5):
        members = [s for s in valid if s["reliability_bin"] == index]
        bins.append({
            "lower": original.BIN_EDGES[index], "upper": original.BIN_EDGES[index + 1],
            "upper_inclusive": index == 4, "count": len(members),
            "mean_selected_probability": mean([s["selected_probability"] for s in members]),
            "actual_action_accuracy": mean([int(s["action_correct"]) for s in members]),
        })
    return {
        "scheduled": len(cases), "attempted": len(rows), "unattempted": len(cases) - len(rows),
        "action_eligible": sum(r["action_eligible"] for r in rows),
        "action_ineligibility_reasons": dict(Counter(
            reason for r in rows for reason in r["action_ineligibility_reasons"])),
        "action_correct": sum(s["action_correct"] for s in scores),
        "action_accuracy_all_scheduled": sum(s["action_correct"] for s in scores) / len(cases),
        "strict_failures": dict(Counter(r["strict"]["diagnostic_code"] for r in rows
                                       if r["strict"]["status"] == "failure")),
        "vector_denominator": n, "omitted_vectors_attempted": len(rows) - n,
        "vector_omission_reasons": dict(Counter(reason for r in rows for reason in r["vector_omission_reasons"])),
        "choice_argmax_disagreements_valid_vectors": sum(s["choice_argmax_disagreement"] for s in valid),
        "brier_sum_over_rows": math.fsum(s["brier"] for s in valid),
        "mean_summed_multiclass_brier": mean([s["brier"] for s in valid]),
        "log_loss_infinite_count": infinite, "finite_log_loss_sum": finite_loss,
        "finite_log_loss_count": n - infinite,
        "mean_log_loss": finite_loss / n if n and not infinite else None,
        "mean_log_loss_is_infinite": infinite > 0,
        "uniform_reference_same_rows": {
            "denominator": n, "mean_summed_multiclass_brier": mean([s["uniform_brier"] for s in valid]),
            "mean_log_loss": mean([s["uniform_log_loss"] for s in valid])},
        "reliability_bins": bins,
    }


def replay(cases, rows, profile, config):
    require(len(rows) <= len(cases), "call_cap", "More attempts than planned cases.")
    scores, bookkeeping, fatal = [], [], False
    for case, row in zip(cases, rows):
        require(not fatal, "fatal_continuation", "Attempt follows a fatal error.")
        score, validation = score_row(case, row, profile, config)
        scores.append(score)
        bookkeeping.append(validation)
        fatal = row["strict"]["fatal"]
    by_task = {}
    for task in sorted({c["task"] for c in cases}):
        own = [c for c in cases if c["task"] == task]
        ids = {c["instance_id"] for c in own}
        by_task[task] = task_summary(own, [r for r in rows if r["instance_id"] in ids],
                                     [s for s in scores if s["task"] == task])
    differences = [b["bookkeeping_abs_difference"] for b in bookkeeping
                   if b["bookkeeping_abs_difference"] is not None]
    return {
        "scores": scores,
        "summary": {"evidence_limit": original.EVIDENCE_LIMIT, "notice": original.NOTICE,
                    "scheduled": len(cases), "attempted": len(rows),
                    "unattempted": len(cases) - len(rows), "by_task": by_task},
        "bookkeeping": {"stable_sum_differs_from_stored_count": sum(d != 0 for d in differences),
                        "max_abs_stable_sum_difference": max(differences, default=0),
                        "vector_eligibility_changes": 0},
    }


def common_valid(task_cases, attempts, configs):
    """Same-task common-valid metrics, retaining all-scheduled metrics in separate summaries."""
    require(len({c["task"] for c in task_cases}) <= 1, "pooled_tasks", "Common-valid analysis must be per task.")
    by_id = {}
    for profile in original.PROFILES:
        rows = attempts[profile]
        rows = list(rows.values()) if isinstance(rows, dict) else rows
        by_id[profile] = {r["instance_id"]: r for r in rows}
        require(len(by_id[profile]) == len(rows), "duplicate_attempt", "Duplicate common-valid instance.")
    common = [c for c in task_cases if all(
        c["instance_id"] in by_id[p] and by_id[p][c["instance_id"]]["probabilities"] is not None
        for p in original.PROFILES)]
    result = {"denominator": len(common), "status": "defined" if common else "undefined_no_common_valid_rows",
              "profiles": None}
    if not common:
        return result
    result["profiles"] = {}
    fields = {
        "vector_denominator", "mean_summed_multiclass_brier", "brier_sum_over_rows",
        "mean_log_loss", "mean_log_loss_is_infinite", "log_loss_infinite_count",
        "finite_log_loss_sum", "finite_log_loss_count", "uniform_reference_same_rows",
        "reliability_bins", "choice_argmax_disagreements_valid_vectors",
    }
    for profile in original.PROFILES:
        rows = [by_id[profile][c["instance_id"]] for c in common]
        scores = [score_row(c, r, profile, configs[profile])[0] for c, r in zip(common, rows, strict=True)]
        summary = task_summary(common, rows, scores)
        result["profiles"][profile] = {key: summary[key] for key in sorted(fields)}
    return result


def original_paths(source):
    paths = [source / name for name in ("manifest.json", "manifest.sha256", "cases.jsonl")]
    for profile in original.PROFILES:
        lane = source / profile
        require(not lane.is_symlink(), "source_symlink", "Symlinked original lane refused.")
        expected = ({"attempts.jsonl"} if profile == "jev_action"
                    else {"attempts.jsonl", "scores.jsonl", "summary.json", "status.json"})
        require({p.name for p in lane.iterdir()} == expected, "original_layout",
                "Original profile layout changed; Jev must remain unsealed.")
        paths.extend(lane / name for name in sorted(expected))
    return paths


def check_source_seal(source, seal):
    paths = original_paths(source)
    require(set(seal["artifact_sha256"]) == {str(p.relative_to(source)) for p in paths},
            "source_seal_files", "Original artifact inventory changed.")
    for path in paths:
        require(not path.is_symlink() and path.stat().st_mode & 0o222 == 0,
                "original_writable", "Original data artifact is writable or symlinked.")
        require(suite.digest(path) == seal["artifact_sha256"][str(path.relative_to(source))],
                "original_digest_changed", "Original data bytes changed after recovery input seal.")


def seal_original(source, output):
    manifest, cases = original.load_plan(source)
    paths = original_paths(source)
    hashes = {str(p.relative_to(source)): suite.digest(p) for p in paths}
    changes = []
    for path in paths:
        require(not path.is_symlink(), "source_symlink", "Symlinked original artifact refused.")
        if path.stat().st_mode & 0o222:
            require(path == source / "jev_action/attempts.jsonl", "unexpected_writable",
                    "Only the unsealed Jev raw-attempt file may need write-bit removal.")
            path.chmod(path.stat().st_mode & ~0o222)
            changes.append(str(path.relative_to(source)))
    seal = {
        "version": VERSION, "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "artifact_sha256": hashes, "targeted_write_bits_removed": changes,
        "logs_excluded_and_untouched": True,
        "original_cli_exit_codes": ORIGINAL_EXITS,
        "exit_code_evidence": "Parent completion notification; not inferred from recovered scores.",
        "original_producer_code_sha256": manifest["code_sha256"]["scripts/graph_probability_audit.py"],
        "verifier_code_sha256": suite.digest(Path(__file__)),
    }
    check_source_seal(source, seal)
    suite.write_json(output / "original-seal.json", seal)
    (output / "original-seal.json").chmod(0o444)
    return manifest, cases, seal


def producer_check(source, profile):
    manifest, cases = original.load_plan(source)
    rows = suite.read_rows(source / profile / "attempts.jsonl")
    differences = [abs(sum(r["probabilities"].values()) - r["probability_sum"])
                   for r in rows if r["probabilities"] is not None]
    result = {
        "profile": profile, "python": platform.python_version(), "executable": sys.executable,
        "producer_code_sha256": manifest["code_sha256"]["scripts/graph_probability_audit.py"],
        "collector_calls": len(rows),
        "canonical_json_sum_differs_from_stored_count": sum(d != 0 for d in differences),
        "max_abs_canonical_json_sum_difference": max(differences, default=0),
    }
    if profile == "jev_action":
        try:
            original.replay(cases, rows)
        except suite.Refusal as exc:
            require(str(exc) == "Invalid retained vector.", "unexpected_original_failure",
                    "Original Jev replay failed for a different reason.")
            result.update(original_replay="failed_as_recorded", original_error="Invalid retained vector.")
        else:
            raise VerificationError("failure_not_reproduced", "Original Jev seal failure did not reproduce.")
    else:
        original.report(source, profile)
        result.update(original_replay="passed_exact_producer_runtime")
    return result


def run_producer_check(source, profile):
    completed = subprocess.run(
        [PRODUCERS[profile], "-B", str(Path(__file__).resolve()), "producer-check",
         "--source", str(source), "--profile", profile],
        stdout=subprocess.PIPE, text=True, check=False)
    require(completed.returncode == 0, "producer_runtime_check_failed",
            f"Offline original replay failed unexpectedly in producing runtime for {profile}.")
    result = json.loads(completed.stdout)
    expected = "3.11." if profile == "jev_action" else "3.12."
    require(result["python"].startswith(expected), "producer_runtime_mismatch", "Unexpected producing Python.")
    return result


def numeric_differences(old, new):
    """Quantify arithmetic changes; old artifacts must first pass exact producer-runtime replay."""
    differences = []
    if type(old) is dict and type(new) is dict:
        require(set(old) == set(new), "legacy_shape", "Legacy and corrected metric fields differ.")
        for key in old:
            differences.extend(numeric_differences(old[key], new[key]))
    elif type(old) is list and type(new) is list:
        require(len(old) == len(new), "legacy_shape", "Legacy and corrected row counts differ.")
        for left, right in zip(old, new, strict=True):
            differences.extend(numeric_differences(left, right))
    elif type(old) is float and type(new) is float:
        if old != new:
            differences.append(abs(old - new))
    else:
        require(type(old) is type(new) and old == new, "legacy_discrete_change",
                "Corrected analysis changed a discrete legacy score or decision.")
    return differences


def verify_profile(source, profile, *, source_seal=None, producer_evidence=None):
    """Read-only API: return {'status', 'summary', 'scores'} for the corrected analysis."""
    source = Path(source).resolve()
    require(profile in original.PROFILES, "unknown_profile", "Only the two frozen profiles are supported.")
    manifest, cases = original.load_plan(source)
    if source_seal is not None:
        check_source_seal(source, source_seal)
    evidence = producer_evidence if producer_evidence is not None else run_producer_check(source, profile)
    expected_replay = ("failed_as_recorded" if profile == "jev_action"
                       else "passed_exact_producer_runtime")
    require(evidence["profile"] == profile and evidence["collector_calls"] == 792
            and evidence["producer_code_sha256"] ==
            manifest["code_sha256"]["scripts/graph_probability_audit.py"]
            and evidence["original_replay"] == expected_replay,
            "producer_evidence", "Producer-runtime evidence does not match the original source/lane.")
    rows = suite.read_rows(source / profile / "attempts.jsonl")
    require(len(rows) == len(cases) == 792 and not any(r["strict"]["fatal"] for r in rows),
            "collection_incomplete", "Recovery requires the completed 792-call, nonfatal collectors.")
    result = replay(cases, rows, profile, manifest["profiles"][profile])
    legacy_diff = []
    if profile == "qwen4_token_scores":
        status = suite.read_json(source / profile / "status.json")
        require(status["status"] == "completed" and status["attempted_calls"] == 792,
                "original_token_status", "Original token lane is not completed.")
        for name, expected in status["artifact_sha256"].items():
            require(suite.digest(source / profile / name) == expected,
                    "original_token_digest", "Original sealed token artifact changed.")
        legacy_diff.extend(numeric_differences(
            suite.read_rows(source / profile / "scores.jsonl"), result["scores"]))
        legacy_diff.extend(numeric_differences(
            suite.read_json(source / profile / "summary.json"), result["summary"]))
    else:
        require(not (source / profile / "status.json").exists(), "original_jev_status",
                "Do not retrofit a status into the original unsealed Jev lane.")
    result["status"] = {
        "status": "offline_recovered_analysis" if profile == "jev_action" else "offline_verified_analysis",
        "verifier_version": VERSION, "profile": profile,
        "original_cli_exit_code": ORIGINAL_EXITS[profile], "original_lane_sealed": profile != "jev_action",
        "planned_calls": len(cases), "original_collector_calls": len(rows), "new_model_calls": 0,
        "original_attempts_sha256": suite.digest(source / profile / "attempts.jsonl"),
        "producer_runtime_check": evidence, "derived_sum_bookkeeping": result.pop("bookkeeping"),
        "existing_legacy_float_fields_changed_by_stable_arithmetic": len(legacy_diff),
        "max_abs_legacy_derived_float_difference": max(legacy_diff, default=0),
        "arithmetic": "math.fsum for derived vector/class/row sums; no probability transformation",
        "bookkeeping_absolute_tolerance": BOOKKEEPING_ATOL, "vector_absolute_tolerance": VECTOR_ATOL,
    }
    return result


def comparison(cases, attempts, results, configs):
    return {"verifier_version": VERSION, "notice": original.NOTICE,
            "by_task": {
                task: {
                    "per_method_all_scheduled_and_own_valid": {
                        p: results[p]["summary"]["by_task"][task] for p in original.PROFILES},
                    "common_valid": common_valid([c for c in cases if c["task"] == task], attempts, configs),
                } for task in sorted(original.COUNTS)}}


def recover(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(source), "output_exists",
            "Recovery needs an exclusive new sidecar outside the original audit root.")
    output.mkdir(parents=True, exist_ok=False)
    manifest, cases, source_seal = seal_original(source, output)
    # Finalized raw digests above precede all scoring/producer-runtime analysis below.
    checks = {p: run_producer_check(source, p) for p in original.PROFILES}
    suite.write_json(output / "producer-checks.json", checks)
    results = {}
    for profile in original.PROFILES:
        result = verify_profile(source, profile, source_seal=source_seal, producer_evidence=checks[profile])
        results[profile] = result
        lane = output / profile
        lane.mkdir(exist_ok=False)
        with (lane / "scores.jsonl").open("x", encoding="utf-8") as stream:
            for row in result["scores"]:
                suite.append_row(stream, row)
        suite.write_json(lane / "summary.json", result["summary"])
        suite.write_json(lane / "status.json", result["status"])
    attempts = {p: suite.read_rows(source / p / "attempts.jsonl") for p in original.PROFILES}
    suite.write_json(output / "common-valid.json", comparison(cases, attempts, results, manifest["profiles"]))
    suite.write_json(output / "manifest.json", {
        "verifier_version": VERSION, "source": str(source),
        "runtime": {"python": platform.python_version(), "executable": sys.executable},
        "verifier_code_sha256": suite.digest(Path(__file__)),
        "original_producer_code_sha256": source_seal["original_producer_code_sha256"],
        "original_seal_sha256": suite.digest(output / "original-seal.json"),
        "original_cli_exit_codes": ORIGINAL_EXITS, "new_model_calls": 0,
        "policy": {"bookkeeping_atol": BOOKKEEPING_ATOL, "vector_gate_atol": VECTOR_ATOL,
                   "stable_arithmetic": "math.fsum", "probabilities_mutated": False,
                   "original_status_retrofitted": False,
                   "cpu_gate": "The separate historical CPU 0.01 reproduction gate remains failed."},
    })
    check_source_seal(source, source_seal)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    suite.write_json(output / "seal.json", {
        "verifier_version": VERSION,
        "artifact_sha256": {str(p.relative_to(output)): suite.digest(p) for p in files}})
    for path in [*files, output / "seal.json"]:
        path.chmod(0o444)
    for profile in original.PROFILES:
        (output / profile).chmod(0o555)
    output.chmod(0o555)
    return check(output)


def check(output):
    output = Path(output).resolve()
    seal = suite.read_json(output / "seal.json")
    files = {str(p.relative_to(output)): p for p in output.rglob("*") if p.is_file()
             and p != output / "seal.json"}
    require(set(files) == set(seal["artifact_sha256"]), "sidecar_inventory", "Sidecar files changed.")
    for name, path in files.items():
        require(not path.is_symlink() and path.stat().st_mode & 0o222 == 0
                and suite.digest(path) == seal["artifact_sha256"][name],
                "sidecar_digest", "Sidecar artifact changed or is writable.")
    require(all(p.stat().st_mode & 0o222 == 0 for p in
                (output, output / "seal.json", *(output / p for p in original.PROFILES))),
            "sidecar_writable", "Sidecar must remain read-only.")
    manifest = suite.read_json(output / "manifest.json")
    require(manifest["verifier_version"] == VERSION
            and manifest["verifier_code_sha256"] == suite.digest(Path(__file__)),
            "verifier_changed", "Corrected verifier version/source changed.")
    source = Path(manifest["source"])
    source_seal = suite.read_json(output / "original-seal.json")
    check_source_seal(source, source_seal)
    _, cases = original.load_plan(source)
    checks = suite.read_json(output / "producer-checks.json")
    results = {}
    for profile in original.PROFILES:
        result = verify_profile(source, profile, source_seal=source_seal, producer_evidence=checks[profile])
        require(result["scores"] == suite.read_rows(output / profile / "scores.jsonl")
                and result["summary"] == suite.read_json(output / profile / "summary.json")
                and result["status"] == suite.read_json(output / profile / "status.json"),
                "sidecar_replay", "Stable sidecar replay differs.")
        results[profile] = result
    attempts = {p: suite.read_rows(source / p / "attempts.jsonl") for p in original.PROFILES}
    original_manifest, _ = original.load_plan(source)
    require(comparison(cases, attempts, results, original_manifest["profiles"]) ==
            suite.read_json(output / "common-valid.json"), "common_valid_replay", "Common-valid replay differs.")
    return {"status": "verified_offline_recovery", "verifier_version": VERSION, "new_model_calls": 0,
            "original_collector_calls": {p: len(attempts[p]) for p in original.PROFILES},
            "original_cli_exit_codes": ORIGINAL_EXITS, "runtime": manifest["runtime"]}


def report(output, profile):
    """Exporter API for the sealed sidecar: status, summary, scores; no original report() call."""
    require(profile in original.PROFILES, "unknown_profile", "Only the two frozen profiles are supported.")
    output = Path(output)
    check(output)
    return {"status": suite.read_json(output / profile / "status.json"),
            "summary": suite.read_json(output / profile / "summary.json"),
            "scores": suite.read_rows(output / profile / "scores.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("recover", "check", "producer-check"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=original.PROFILES)
    args = parser.parse_args()
    try:
        if args.command == "recover":
            require(args.source is not None and args.output is not None and args.profile is None,
                    "arguments", "recover needs --source and --output.")
            result = recover(args.source, args.output)
        elif args.command == "check":
            require(args.output is not None and args.source is None and args.profile is None,
                    "arguments", "check needs only --output.")
            result = check(args.output)
        else:
            require(args.source is not None and args.profile is not None and args.output is None,
                    "arguments", "producer-check needs --source and --profile.")
            result = producer_check(args.source, args.profile)
        print(suite.canonical(result))
        return 0
    except VerificationError as exc:
        print(suite.canonical({"status": "verification_failed", "diagnostic_code": exc.code,
                               "detail": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
