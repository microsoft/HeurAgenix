#!/usr/bin/env python3
"""Preregistered, task-specific prospective probability audit; no primary scoring."""

from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_right
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import evaluate_graph_release as evaluation
from scripts import native_choice_contract_audit as native
from scripts import semif_vllm_bridge as bridge
from src.benchmark.config import ModelConfig
from src.clients.base import DecisionOption, DecisionRequest, InvalidResponseError, TokenUsage
from src.clients.registry import create_client

suite, release = evaluation.suite, evaluation.release
PROFILES = ("jev_action", "qwen4_token_scores")
COUNTS = {"adjacency": 80, "articulation_point": 80, "cycle_detection": 72,
          "degree_exact": 400, "distance_threshold": 80, "pair_connectivity": 80}
CALLS = sum(COUNTS.values())
SPLIT = "structural_challenge"
TOLERANCE = 1e-6
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
EVIDENCE_LIMIT = (
    "Canonical Jev logs omit complete probability vectors; their calibration cannot be "
    "retrospectively scored. These are new prospective calls, not recovered primary scores."
)
NOTICE = (
    "Task-specific descriptive reliability only; no training, threshold tuning, pooled "
    "cross-task ranking or calibration-superiority claim. Qwen4 vectors are conditional "
    "option-logit scores, not calibrated posteriors."
)


def value_hash(value):
    return hashlib.sha256(suite.canonical(value).encode()).hexdigest()


def code_hashes():
    return {**evaluation.code_hashes(),
            "scripts/graph_probability_audit.py": suite.digest(Path(__file__))}


def request_from(case):
    public = case["request"]
    return DecisionRequest(**{**public, "options": tuple(DecisionOption(**o) for o in public["options"])})


def cases_for(instances):
    cases = []
    for instance in instances:
        if instance["kind"] != "exact":
            continue
        suite.require(instance["task"] in COUNTS and instance["query_budget"] == 1,
                      "Unexpected exact task or query budget.")
        public = {key: instance[key] for key in release.PUBLIC_KEYS}
        request = suite.json_value(asdict(evaluation.public_request(public, [])))
        label = str(instance["private"]["answer"])
        suite.require(label in {o["id"] for o in request["options"]}, "Label outside supplied options.")
        cases.append({"instance_id": instance["id"], "task": instance["task"],
                      "node_count": len(instance["state"]["nodes"]), "label": label,
                      "request": request, "request_sha256": value_hash(request)})
    suite.require(dict(Counter(c["task"] for c in cases)) == COUNTS
                  and len({c["instance_id"] for c in cases}) == CALLS,
                  "Require all 792 distinct confirmation exact cases.")
    return cases


def backend_pins():
    report_path = bridge.DEFAULT_OUTPUT / "report.json"
    check_path = REPO / "output/semif-vllm-methodcheck-20260924-v1/independent-parser-check.json"
    upstream_path = bridge.DEFAULT_CPU / "upstream/source-manifest.json"
    provenance_path = bridge.DEFAULT_CPU / "provenance.json"
    report, check = suite.read_json(report_path), suite.read_json(check_path)
    upstream, provenance = suite.read_json(upstream_path), suite.read_json(provenance_path)
    suite.require(report["valid"] == report["matching_choices"] == 12
                  and check["independent_token_index_or_probability_mismatches"] == 0,
                  "Bridge action mapping verification failed.")
    suite.require(upstream["revision"] == bridge.SOURCE_REVISION
                  and provenance["declared_model_revision"] == bridge.MODEL_REVISION,
                  "Bridge reference revisions changed.")
    paths = [report_path, check_path, upstream_path, provenance_path]
    for item in upstream["files"]:
        path = bridge.DEFAULT_CPU / "upstream" / item["path"]
        suite.require(path.resolve().is_relative_to(bridge.DEFAULT_CPU / "upstream")
                      and suite.digest(path) == item["sha256"], "Upstream source changed.")
        paths.append(path)
    tokenizer_files = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    for item in provenance["model_files"]:
        if item["name"] in tokenizer_files:
            path = bridge.DEFAULT_MODEL / item["name"]
            suite.require(suite.digest(path) == item["sha256"], "Tokenizer/config changed.")
            paths.append(path)
            tokenizer_files.remove(item["name"])
    suite.require(not tokenizer_files, "Missing tokenizer provenance.")
    return {
        "file_sha256": {str(p): suite.digest(p) for p in paths},
        "upstream_revision": bridge.SOURCE_REVISION, "model_revision": bridge.MODEL_REVISION,
        "protocol": "semif-vllm-requested-token-logprobs-v1", "scoring_mode": "requested",
        "max_prefix_tokens": 4096, "timeout_seconds": 120.0,
        "sampling_seed_used": False, "generated_token_used_for_decision": False,
        "cpu_numerical_gate_passed": report["passes_predeclared_gate"],
        "reference_caveat": "Distinct GPU scoring reference; CPU 0.01 reproduction gate failed.",
    }


def plan(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    suite.require(not output.exists(), "Output exists; no overwrite.")
    verified = release.verify(bundle)
    source = release.manifest_at(bundle)
    suite.require("explicit_actions" in source["protocols"], "Canonical action protocol missing.")
    cases = cases_for(release.labeled_instances(bundle, SPLIT, source))
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle), "split": SPLIT, "protocol": "explicit_actions",
        "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "bundle_file_sha256": source["sha256"], "verified_bundle_instances": verified["instances"],
        "code_sha256": code_hashes(), "preparation_environment": suite.environment_versions(),
        "profiles": {p: asdict(evaluation.configuration(p)[0]) for p in PROFILES},
        "backend": backend_pins(), "calls_per_profile": CALLS, "total_call_cap": CALLS * 2,
        "task_counts": COUNTS, "selection": "All exact cases, in frozen public-bundle order.",
        "policy": {"retry": False, "resume": False, "training": False, "threshold_tuning": False,
                   "probability_repair": False, "private_labels_sent": False,
                   "vector_sum_absolute_tolerance": TOLERANCE, "reliability_bin_edges": BIN_EDGES,
                   "bin_intervals": "left closed, right open, except final bin includes 1",
                   "brier": "sum over all supplied classes; no division by class count",
                   "log_loss": "natural logarithm; zero true probability is explicitly infinite",
                   "uniform_reference": "1/K on identical valid-vector rows, without sampled actions",
                   "vector_conditioning": "Eligible actual action AND complete finite [0,1] vector "
                   "whose sum is within tolerance. Retain original values; omit all other vectors.",
                   "choice_argmax_disagreement": "selected probability strictly below maximum; "
                   "diagnostic only, no choice substitution or vector exclusion",
                   "denominators": "Action accuracy: all scheduled cases per task. Probability "
                   "scores and reliability: explicitly conditioned on valid-vector rows per task.",
                   "failure": "Existing suite.failure conventions; fatal errors terminate a profile."},
        "evidence_limit": EVIDENCE_LIMIT, "notice": NOTICE,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            suite.append_row(stream, case)
    manifest["cases_sha256"] = suite.digest(output / "cases.jsonl")
    suite.write_json(output / "manifest.json", manifest)
    suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
    for name in ("cases.jsonl", "manifest.json", "manifest.sha256"):
        (output / name).chmod(0o444)
    return {"status": "planned", "calls_per_profile": CALLS, "total_call_cap": CALLS * 2,
            "task_counts": COUNTS, "manifest_sha256": suite.digest(output / "manifest.json")}


def load_plan(output, *, check_sources=True):
    output = Path(output)
    suite.require(not output.is_symlink(), "Symlinked plan refused.")
    for name in ("manifest.json", "manifest.sha256", "cases.jsonl"):
        suite.require(not (output / name).is_symlink() and (output / name).stat().st_mode & 0o222 == 0,
                      "Plan files must remain read-only and non-symlinked.")
    suite.require(suite.digest(output / "manifest.json") == suite.read_json(output / "manifest.sha256"),
                  "Plan manifest changed.")
    manifest = suite.read_json(output / "manifest.json")
    suite.require(suite.digest(output / "cases.jsonl") == manifest["cases_sha256"],
                  "Frozen cases changed.")
    cases = suite.read_rows(output / "cases.jsonl")
    suite.require(manifest["calls_per_profile"] == len(cases) == CALLS
                  and manifest["total_call_cap"] == CALLS * 2
                  and set(manifest["profiles"]) == set(PROFILES), "Invalid planned scope.")
    if check_sources:
        suite.require(code_hashes() == manifest["code_sha256"], "Audit source changed.")
        suite.require({p: asdict(evaluation.configuration(p)[0]) for p in PROFILES}
                      == manifest["profiles"], "Profile configuration changed.")
        bundle = Path(manifest["bundle"])
        suite.require(suite.digest(bundle / "manifest.json") == manifest["bundle_manifest_sha256"],
                      "Bundle manifest changed.")
        source = release.manifest_at(bundle)
        suite.require(source["sha256"] == manifest["bundle_file_sha256"], "Bundle file map changed.")
        for relative in source["sha256"]:
            release.checked_file(bundle, relative, source)
        suite.require(cases_for(release.labeled_instances(bundle, SPLIT, source)) == cases,
                      "Canonical requests/labels changed.")
        suite.require(backend_pins() == manifest["backend"], "Backend inputs changed.")
    return manifest, cases


def vector_record(request, candidate, eligible):
    ids = {o.id for o in request.options}
    reasons = []
    if not eligible:
        reasons.append("action_ineligible")
        if candidate is None:
            return {"probabilities": None, "probability_sum": None,
                    "vector_omission_reasons": reasons}
    if candidate is None:
        reasons.append("missing_probabilities")
    elif type(candidate) is not dict or set(candidate) != ids:
        reasons.append("invalid_probability_keys")
    elif not all(native.finite_probability(p) for p in candidate.values()):
        reasons.append("invalid_probability_values")
    total = (sum(candidate.values()) if type(candidate) is dict and set(candidate) == ids
             and all(native.finite_probability(p) for p in candidate.values()) else None)
    if total is not None and not math.isclose(total, 1.0, rel_tol=0, abs_tol=TOLERANCE):
        reasons.append("invalid_probability_sum")
    return {"probabilities": dict(candidate) if not reasons else None,
            "probability_sum": total, "vector_omission_reasons": reasons}


async def observe(client, request, case, config, profile):
    started, response, body = time.perf_counter(), None, None
    try:
        response = await client.predict(request)
        body = response.raw_output
        strict = suite.response_record(response, request, config)
        expected_kind = "native" if profile == "jev_action" else "option_logits"
        if response.probability_kind != expected_kind:
            raise InvalidResponseError("Unexpected probability kind", diagnostic_code="invalid_probabilities")
        if profile == "qwen4_token_scores":
            evaluation.bridge_metadata(response, request)
    except BaseException as exc:
        if isinstance(exc, InvalidResponseError) and exc.raw_output is not None:
            body = exc.raw_output
        strict = {"status": "failure", "selected_option_id": None, "resolved_model": None,
                  "usage": suite.safe_usage(getattr(exc, "usage", None)), **suite.failure(exc)}
    candidate, confidence = None, None
    if profile == "jev_action":
        observation = native.action_observation(request, body, config.model, case["label"], strict)
        eligible, choice = observation["eligible"], observation["selected_option_id"]
        reasons = observation["ineligibility_reasons"]
        diagnostics = observation["native_diagnostics"]
        if eligible:
            answer = body["answers"]["decision"]
            candidate = (response.probabilities if strict["status"] == "success"
                         else answer["probabilities"])
            confidence = response.confidence if strict["status"] == "success" else answer["confidence"]
            strict["usage"] = suite.safe_usage(TokenUsage(
                body["usage"].get("input_tokens"), body["usage"].get("output_tokens")))
    else:
        eligible = strict["status"] == "success"
        choice = strict["selected_option_id"] if eligible else None
        reasons, diagnostics = ([] if eligible else ["strict_failure"]), None
        if eligible:
            candidate, confidence = response.probabilities, response.confidence
    vector = vector_record(request, candidate, eligible)
    # The bridge has no failure-relaxation protocol; malformed scores invalidate its action.
    if profile == "qwen4_token_scores" and eligible and vector["vector_omission_reasons"]:
        eligible, choice, reasons = False, None, ["invalid_score_vector"]
        vector = vector_record(request, candidate, False)
    return {
        "instance_id": case["instance_id"], "request_sha256": case["request_sha256"],
        "strict": strict, "native_diagnostics": diagnostics,
        "action_eligible": eligible, "action_ineligibility_reasons": reasons,
        "selected_option_id": choice, "confidence": confidence if native.finite_probability(confidence) else None,
        **vector, "latency_seconds": time.perf_counter() - started,
    }


def score_row(case, row):
    request = request_from(case)
    suite.require(row["instance_id"] == case["instance_id"]
                  and row["request_sha256"] == case["request_sha256"] == value_hash(case["request"]),
                  "Attempt/request order mismatch.")
    selected, eligible = row["selected_option_id"], row["action_eligible"]
    suite.require(type(eligible) is bool and
                  ((eligible and type(selected) is str and selected in {o.id for o in request.options}
                    and not row["action_ineligibility_reasons"])
                   or (not eligible and selected is None and row["action_ineligibility_reasons"])),
                  "Invalid action observation.")
    vector = row["probabilities"]
    if vector is not None:
        checked = vector_record(request, vector, eligible)
        suite.require(not checked["vector_omission_reasons"] and not row["vector_omission_reasons"]
                      and checked["probability_sum"] == row["probability_sum"], "Invalid retained vector.")
    else:
        suite.require(bool(row["vector_omission_reasons"]), "Missing vector omission reason.")
    correct = eligible and selected == case["label"]
    scores = {"instance_id": case["instance_id"], "task": case["task"], "action_correct": correct,
              "valid_vector": vector is not None, "choice_argmax_disagreement": None,
              "brier": None, "log_loss": None, "log_loss_infinite": False,
              "uniform_brier": None, "uniform_log_loss": None, "selected_probability": None,
              "reliability_bin": None}
    if vector is not None:
        true_probability = vector[case["label"]]
        selected_probability = vector[selected]
        scores.update(
            brier=sum((p - int(key == case["label"])) ** 2 for key, p in vector.items()),
            log_loss=-math.log(true_probability) if true_probability > 0 else None,
            log_loss_infinite=true_probability == 0,
            uniform_brier=1 - 1 / len(vector), uniform_log_loss=math.log(len(vector)),
            selected_probability=selected_probability,
            choice_argmax_disagreement=selected_probability < max(vector.values()),
            reliability_bin=min(bisect_right(BIN_EDGES, selected_probability) - 1, 4))
    return scores


def task_summary(cases, attempts, scores):
    valid = [s for s in scores if s["valid_vector"]]
    n, infinite = len(valid), sum(s["log_loss_infinite"] for s in valid)
    finite_loss = sum(s["log_loss"] for s in valid if not s["log_loss_infinite"])
    bins = []
    for index in range(5):
        members = [s for s in valid if s["reliability_bin"] == index]
        bins.append({"lower": BIN_EDGES[index], "upper": BIN_EDGES[index + 1],
                     "upper_inclusive": index == 4, "count": len(members),
                     "mean_selected_probability": suite.mean([s["selected_probability"] for s in members]),
                     "actual_action_accuracy": suite.mean([int(s["action_correct"]) for s in members])})
    return {
        "scheduled": len(cases), "attempted": len(attempts), "unattempted": len(cases) - len(attempts),
        "action_eligible": sum(r["action_eligible"] for r in attempts),
        "action_ineligibility_reasons": dict(Counter(
            reason for r in attempts for reason in r["action_ineligibility_reasons"])),
        "action_correct": sum(s["action_correct"] for s in scores),
        "action_accuracy_all_scheduled": sum(s["action_correct"] for s in scores) / len(cases),
        "strict_failures": dict(Counter(r["strict"]["diagnostic_code"] for r in attempts
                                       if r["strict"]["status"] == "failure")),
        "vector_denominator": n, "omitted_vectors_attempted": len(attempts) - n,
        "vector_omission_reasons": dict(Counter(
            reason for r in attempts for reason in r["vector_omission_reasons"])),
        "choice_argmax_disagreements_valid_vectors": sum(s["choice_argmax_disagreement"] for s in valid),
        "brier_sum_over_rows": sum(s["brier"] for s in valid),
        "mean_summed_multiclass_brier": suite.mean([s["brier"] for s in valid]),
        "log_loss_infinite_count": infinite, "finite_log_loss_sum": finite_loss,
        "finite_log_loss_count": n - infinite,
        "mean_log_loss": finite_loss / n if n and not infinite else None,
        "mean_log_loss_is_infinite": infinite > 0,
        "uniform_reference_same_rows": {
            "denominator": n, "mean_summed_multiclass_brier": suite.mean([s["uniform_brier"] for s in valid]),
            "mean_log_loss": suite.mean([s["uniform_log_loss"] for s in valid])},
        "reliability_bins": bins,
    }


def replay(cases, attempts):
    suite.require(len(attempts) <= len(cases), "Call cap exceeded.")
    scores = []
    fatal = False
    for case, row in zip(cases, attempts):
        suite.require(not fatal, "Attempt after fatal failure.")
        scores.append(score_row(case, row))
        fatal = row["strict"]["fatal"]
    by_task = {}
    for task in sorted({c["task"] for c in cases}):
        task_cases = [c for c in cases if c["task"] == task]
        ids = {c["instance_id"] for c in task_cases}
        by_task[task] = task_summary(task_cases, [r for r in attempts if r["instance_id"] in ids],
                                     [s for s in scores if s["task"] == task])
    return scores, {"evidence_limit": EVIDENCE_LIMIT, "notice": NOTICE,
                    "scheduled": len(cases), "attempted": len(attempts),
                    "unattempted": len(cases) - len(attempts), "by_task": by_task}


def outer_lock(profile):
    suite.require(profile in PROFILES, "Unsupported profile.")
    return suite.lane_locks(("jev",)) if profile == "jev_action" else nullcontext()


def make_client(profile, config):
    if profile == "jev_action":
        suite.shared.credential_preflight(("jev",))
        return create_client(config.provider, **config.client_kwargs())
    tokenizer, encoder = bridge.upstream_functions(bridge.DEFAULT_CPU, bridge.DEFAULT_MODEL)
    return bridge.SemIfVLLMBridge(tokenizer, encoder, max_prefix_tokens=4096,
                                 scoring_mode="requested", timeout=120.0)


async def collect(client, cases, config, profile, stream):
    rows = []
    for case in cases:
        suite.require(len(rows) < CALLS, "Call cap exceeded.")
        row = await observe(client, request_from(case), case, config, profile)
        suite.append_row(stream, row)
        rows.append(row)
        if row["strict"]["fatal"]:
            break
    return rows


def seal_lane(output, profile, cases, status, code):
    lane = output / profile
    attempts = suite.read_rows(lane / "attempts.jsonl")
    scores, summary = replay(cases, attempts)
    with (lane / "scores.jsonl").open("x", encoding="utf-8") as stream:
        for row in scores:
            suite.append_row(stream, row)
    suite.write_json(lane / "summary.json", summary)
    status_row = {
        "status": status, "diagnostic_code": code, "profile": profile,
        "planned_calls": len(cases), "attempted_calls": len(attempts),
        "runtime_environment": suite.environment_versions(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": suite.digest(output / "manifest.json"),
        "artifact_sha256": {name: suite.digest(lane / name)
                            for name in ("attempts.jsonl", "scores.jsonl", "summary.json")},
    }
    suite.write_json(lane / "status.json", status_row)
    for path in lane.iterdir():
        path.chmod(0o444)
    lane.chmod(0o555)
    return status_row


def report(output, profile):
    suite.require(profile in PROFILES, "Unsupported profile.")
    _, cases = load_plan(output)
    lane = output / profile
    status = suite.read_json(lane / "status.json")
    suite.require(status["profile"] == profile and status["status"] in {
        "completed", "completed_with_failures", "aborted", "interrupted"},
        "Unsealed profile.")
    suite.require(status["manifest_sha256"] == suite.digest(output / "manifest.json")
                  and set(status["artifact_sha256"]) == {"attempts.jsonl", "scores.jsonl", "summary.json"},
                  "Lane provenance mismatch.")
    for name, expected in status["artifact_sha256"].items():
        suite.require(not (lane / name).is_symlink() and suite.digest(lane / name) == expected,
                      "Lane artifact changed.")
    suite.require(all(p.stat().st_mode & 0o222 == 0 for p in (lane, *lane.iterdir())),
                  "Lane must remain read-only.")
    attempts = suite.read_rows(lane / "attempts.jsonl")
    scores, summary = replay(cases, attempts)
    suite.require(scores == suite.read_rows(lane / "scores.jsonl")
                  and summary == suite.read_json(lane / "summary.json")
                  and status["planned_calls"] == len(cases)
                  and status["attempted_calls"] == len(attempts), "Scoring replay differs.")
    if status["status"].startswith("completed"):
        suite.require(len(attempts) == len(cases) and not any(r["strict"]["fatal"] for r in attempts),
                      "Incomplete run labeled completed.")
        suite.require((status["status"] == "completed") ==
                      all(r["strict"]["status"] == "success" for r in attempts),
                      "Strict-failure completion status mismatch.")
    seal_path = output / "seal.json"
    if seal_path.exists():
        suite.require(not seal_path.is_symlink()
                      and all(p.stat().st_mode & 0o222 == 0 for p in (output, seal_path)),
                      "Root seal must remain read-only and non-symlinked.")
        suite.require(suite.read_json(seal_path) == {
            "manifest_sha256": suite.digest(output / "manifest.json"),
            "profile_status_sha256": {p: suite.digest(output / p / "status.json") for p in PROFILES}},
            "Root seal differs from profile artifacts.")
    return {"status": status, "summary": summary}


async def run(output, profile):
    output = Path(output).resolve()
    suite.require(profile in PROFILES, "Unsupported profile.")
    manifest, cases = load_plan(output)
    lane = output / profile
    suite.require(not lane.exists(), "Profile already attempted; no resume or overwrite.")
    lane.mkdir(exist_ok=False)
    status, code = "aborted", None
    config = ModelConfig(**manifest["profiles"][profile])
    with (lane / "attempts.jsonl").open("x", encoding="utf-8") as stream:
        try:
            with outer_lock(profile):
                client = None
                try:
                    client = make_client(profile, config)
                    await client.initialize()
                    load_plan(output)
                    rows = await collect(client, cases, config, profile, stream)
                    if rows and rows[-1]["strict"]["fatal"]:
                        code = rows[-1]["strict"]["diagnostic_code"]
                        status = "interrupted" if code == "interrupted" else "aborted"
                    else:
                        status = ("completed_with_failures" if any(
                            r["strict"]["status"] == "failure" for r in rows) else "completed")
                    load_plan(output)
                finally:
                    if client is not None:
                        try:
                            await client.aclose()
                        except BaseException:
                            status, code = "aborted", "shutdown_failed"
        except BaseException as exc:
            code = "preflight_or_integrity_refusal" if isinstance(exc, suite.Refusal) else suite.failure(exc)["diagnostic_code"]
            status = "interrupted" if code == "interrupted" else "aborted"
    result = seal_lane(output, profile, cases, status, code)
    report(output, profile)
    if all((output / p / "status.json").is_file() for p in PROFILES):
        for p in PROFILES:
            report(output, p)
        suite.write_json(output / "seal.json", {
            "manifest_sha256": suite.digest(output / "manifest.json"),
            "profile_status_sha256": {p: suite.digest(output / p / "status.json") for p in PROFILES}})
        (output / "seal.json").chmod(0o444)
        output.chmod(0o555)
    return result


async def run_cli(output, profile):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run(output, profile)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--profile", choices=PROFILES)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            suite.require(args.bundle is not None and args.profile is None, "Plan needs a bundle, not a profile.")
            result = plan(args.bundle, args.output)
        else:
            suite.require(args.profile is not None and args.bundle is None, "Run/report needs a frozen profile.")
            result = (asyncio.run(run_cli(args.output, args.profile)) if args.command == "run"
                      else report(args.output, args.profile))
        print(suite.canonical(result))
        return int(args.command == "run" and result["status"] in {"aborted", "interrupted"})
    except BaseException:
        print('{"status":"refused","diagnostic_code":"verification_or_execution_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
