#!/usr/bin/env python3
"""Frozen own-history evaluation of the portable, explicit-action graph protocol."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts import graph_release_api as release
from scripts import graph_task_prompt as prompts
from scripts import native_choice_contract_audit as native
from src.benchmark import extended_tasks as tasks
from src.clients.base import InvalidResponseError
from src.clients.registry import create_client

PROFILES = {"jev_action": ("jev", "jev"),
            "qwen4_grammar": ("qwen4", "qwen4"),
            "qwen9_grammar": ("qwen9constrained", "qwen9"),
            "qwen4_token_scores": ("qwen4", "qwen4")}


def configuration(profile):
    name, resource = PROFILES[profile]
    config = suite.model_configs((name,), seed=20261008, no_think_max_tokens=64)[name]
    if config.provider == "vllm" and profile != "qwen4_token_scores":
        config = replace(config, constrain_choices=True)
    if profile == "qwen4_token_scores":
        config = replace(config, temperature=1.0, max_tokens=1, top_p=1.0,
                         top_k=-1, presence_penalty=0.0, timeout_seconds=120.0)
    return config, resource


def code_hashes():
    return {**suite.code_hashes(), **{
        str(Path(module.__file__).relative_to(REPO)): suite.digest(module.__file__)
        for module in (release, prompts, native)},
        "scripts/evaluate_graph_release.py": suite.digest(__file__),
        "scripts/semif_vllm_bridge.py": suite.digest(REPO / "scripts/semif_vllm_bridge.py")}


def bridge_metadata(response, request):
    suite.require(isinstance(response.raw_output, dict)
                  and isinstance(response.raw_output.get("bridge"), dict), "Missing bridge accounting.")
    source = response.raw_output["bridge"]
    suite.require(source["selected_option_id"] == response.selected_option_id
                  and source["candidate_count"] == len(request.options)
                  and source["candidate_prompts"] == 1 and source["generated_tokens"] == 1
                  and source["generated_tokens_used_for_decision"] is False
                  and source["scoring_mode"] == "requested", "Unexpected bridge scoring method.")
    keys = ("candidate_count", "candidate_prompts", "shared_prefix_tokens", "prefill_tokens",
            "generated_tokens", "requested_max_tokens", "scoring_rule", "scoring_mode",
            "generated_tokens_used_for_decision", "prompt_sha256", "candidate_logprobs",
            "probabilities", "option_ids")
    return {key: source[key] for key in keys}


def public_request(instance, history):
    request = tasks.next_request(instance, history)
    return None if request is None else prompts.canonical_graph_request(instance["task"], request)


def replay(instances, attempts, config, *, native_action):
    position, rows, aborted = 0, [], False
    for instance in instances:
        history, own = [], []
        while position < len(attempts) and attempts[position]["instance_id"] == instance["id"]:
            row = attempts[position]
            suite.require(not aborted and not (own and own[-1]["status"] == "failure"),
                          "Attempt continued after failure.")
            request = public_request(instance, history)
            suite.require(request is not None and row["step"] == len(history)
                          and row["request"] == suite.json_value(asdict(request)),
                          "Request or history replay mismatch.")
            if row["status"] == "success":
                suite.require(row["selected_option_id"] in {o.id for o in request.options}
                              and row["resolved_model"] == config.model and not row["fatal"],
                              "Invalid successful action.")
                if native_action:
                    suite.require(row["action_observation"]["eligible"]
                                  and row["selected_option_id"] ==
                                  row["action_observation"]["selected_option_id"],
                                  "Native action eligibility mismatch.")
                history.append(row["selected_option_id"])
            else:
                suite.require(row["status"] == "failure" and row["selected_option_id"] is None,
                              "Invalid failure record.")
                aborted = row["fatal"]
            own.append(row)
            position += 1
        score = tasks.score(instance, history)
        if position < len(attempts):
            suite.require(score["feasible"] or own and own[-1]["status"] == "failure",
                          "Unfinished instance skipped.")
        row = {"instance_id": instance["id"], "task": instance["task"], "kind": instance["kind"],
               "node_count": len(instance["state"]["nodes"]),
               "edge_count": len(instance["state"]["edges"]), "query_budget": instance["query_budget"],
               "query_count": len(own), "successful_queries": len(history),
               "failure_count": sum(r["status"] == "failure" for r in own),
               "decisions": history, **score}
        if instance["kind"] == "exact":
            row.update(answer=str(instance["private"]["answer"]),
                       prediction=history[0] if history else None)
        else:
            baseline = instance["private"]["baseline"]
            gap = abs(baseline["objective"]-instance["private"]["objective"])
            row.update(baseline_method=baseline["method"], baseline_objective=baseline["objective"],
                       baseline_absolute_gap=gap, baseline_optimal=gap <= 1e-12)
        rows.append(row)
    suite.require(position == len(attempts), "Unknown or out-of-order instance attempts.")
    return rows


async def run(bundle, split, output, profile, *, client_factory=create_client,
              preflight=suite.shared.vllm_preflight):
    suite.require(not output.exists(), "Output exists; no resume or overwrite.")
    verified = release.verify(bundle)
    bundle_manifest = release.manifest_at(bundle)
    suite.require("explicit_actions" in bundle_manifest["protocols"],
                  "This evaluation requires the explicit-actions release.")
    instances = release.labeled_instances(bundle, split, bundle_manifest)
    config, resource = configuration(profile)
    native_action = profile == "jev_action"
    token_scores = profile == "qwen4_token_scores"
    backend, actual_factory = {}, None
    if token_scores and client_factory is create_client:
        from scripts import semif_vllm_bridge as bridge
        calibration = suite.read_json(bridge.DEFAULT_OUTPUT / "report.json")
        checked = suite.read_json(REPO / "output/semif-vllm-methodcheck-20260924-v1/independent-parser-check.json")
        suite.require(calibration["valid"] == calibration["matching_choices"] == 12
                      and checked["independent_token_index_or_probability_mismatches"] == 0,
                      "GPU scoring action mapping or independent parser check failed.")
        tokenizer, encoder = bridge.upstream_functions(bridge.DEFAULT_CPU, bridge.DEFAULT_MODEL)
        initial_lengths = [len(encoder(tokenizer, bridge.row_for(public_request(i, [])), 4096)[0])
                           for i in instances]
        backend = {"protocol": "semif-vllm-requested-token-logprobs-v1",
                   "scoring_mode": "requested", "max_prefix_tokens": 4096,
                   "maximum_initial_prefix_tokens": max(initial_lengths),
                   "upstream_revision": bridge.SOURCE_REVISION,
                   "declared_model_revision": bridge.MODEL_REVISION,
                   "sampling_seed_used": False,
                   "generated_token_used_for_decision": False,
                   "timeout_seconds": 120.0,
                   "cpu_numerical_gate_passed": calibration["passes_predeclared_gate"],
                   "cpu_probability_max_abs_difference": calibration["max_abs_probability_error"],
                   "numerical_reproduction_gate": "failed CPU 0.01 gate; distinct GPU scoring reference",
                   "calibration_report_sha256": suite.digest(bridge.DEFAULT_OUTPUT / "report.json"),
                   "upstream_source_manifest_sha256": suite.digest(
                       bridge.DEFAULT_CPU / "upstream/source-manifest.json")}
        actual_factory = lambda: bridge.SemIfVLLMBridge(
            tokenizer, encoder, max_prefix_tokens=4096, scoring_mode="requested", timeout=120.0)
    before = code_hashes()
    # The bridge itself acquires the same qwen4 lease in initialize().
    with (nullcontext() if token_scores else suite.lane_locks((resource,))):
        await preflight([SimpleNamespace(model=config)])
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1, "profile": profile, "physical_resource": resource,
            "protocol": "explicit_actions", "split": split, "model": asdict(config),
            "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
            "bundle": str(bundle.resolve()), "verified_instances": verified["instances"],
            "planned_instances": len(instances), "maximum_calls": sum(i["query_budget"] for i in instances),
            "native_action_observation": native_action, "injected_client": client_factory is not create_client,
            "backend": backend,
            "code_sha256": before, "environment": suite.environment_versions(),
            "policy": {"retry": False, "repair": False, "resume": False, "private_labels_sent": False},
        }
        suite.write_json(output / "manifest.json", manifest)
        suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
        attempts, interrupted, client = [], None, None
        started = time.perf_counter()
        status = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "diagnostic_code": None}
        suite.set_status(output, status)
        with (output / "attempts.jsonl").open("x") as stream:
            try:
                client = (actual_factory() if actual_factory is not None else
                          client_factory(config.provider, **config.client_kwargs()))
                await client.initialize()
                suite.require(code_hashes() == before, "Code changed before inference.")
                for instance in instances:
                    history = []
                    while (request := public_request(instance, history)) is not None:
                        tick, body = time.perf_counter(), None
                        try:
                            response = await client.predict(request)
                            body = response.raw_output
                            record = suite.response_record(response, request, config)
                            if token_scores:
                                record["backend_accounting"] = bridge_metadata(response, request)
                        except BaseException as exc:
                            if isinstance(exc, InvalidResponseError):
                                body = exc.raw_output
                            record = {"status": "failure", "selected_option_id": None,
                                      "resolved_model": None, "usage": suite.safe_usage(getattr(exc, "usage", None)),
                                      **suite.failure(exc)}
                            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                                interrupted = exc
                        if native_action:
                            record = native.action_execution_record(request, body, config.model, record)
                        row = {"instance_id": instance["id"], "step": len(history),
                               "request": asdict(request), "latency_seconds": time.perf_counter()-tick, **record}
                        suite.append_row(stream, row)
                        attempts.append(suite.json_value(row))
                        if record["status"] == "failure":
                            if record["fatal"]:
                                status.update(status="aborted", diagnostic_code=record["diagnostic_code"])
                            break
                        history.append(record["selected_option_id"])
                    if status["status"] != "running":
                        break
                if status["status"] == "running":
                    suite.require(code_hashes() == before, "Code changed during inference.")
                    status["status"] = ("completed_with_failures" if any(
                        a["status"] == "failure" for a in attempts) else "completed")
            except BaseException as exc:
                status.update(status="aborted", diagnostic_code=suite.failure(exc)["diagnostic_code"])
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    interrupted = exc
            finally:
                if client is not None:
                    try:
                        await client.aclose()
                    except BaseException:
                        status.update(status="aborted", diagnostic_code="shutdown_failed")
                if interrupted is not None:
                    status["status"] = "interrupted"
                rows = replay(instances, attempts, config, native_action=native_action)
                with (output / "scores.jsonl").open("x") as scores:
                    for row in rows:
                        suite.append_row(scores, row)
                suite.write_json(output / "summary.json", suite.summarize(rows))
                status.update(
                    calls=len(attempts), finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=time.perf_counter()-started,
                    strict_validation_failures=sum(a.get("strict_status", a["status"]) == "failure"
                                                   for a in attempts),
                    artifact_sha256={name: suite.digest(output / name) for name in
                                     ("attempts.jsonl", "scores.jsonl", "summary.json")})
                suite.set_status(output, status)
        if interrupted is not None:
            raise interrupted
        return status


async def run_cli(args):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run(args.bundle, args.split, args.output, args.profile)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("development", "structural_challenge", "tsp_confirmation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    result = asyncio.run(run_cli(args))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
