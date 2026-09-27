#!/usr/bin/env python3
"""Matched full-episode TSP prompt ablation, with no teacher-forced actions."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import construction_prefix_controls as controls
from scripts import extended_graph_suite as suite
from scripts import native_choice_contract_audit as native_audit
from scripts.summarize_extended_paper import verified_inputs
from src.benchmark import extended_tasks as tasks
from src.clients.base import InvalidResponseError
from src.clients.registry import create_client

ARMS = ("baseline", "state_explicit")


async def run(root, instances, models, provenance, *, client_factory=create_client,
              preflight=suite.shared.vllm_preflight, native_action_observation=False):
    suite.require(not root.exists(), "Ablation root exists; no resume or overwrite.")
    suite.require(bool(instances) and all(i["task"] == "tsp_construct" for i in instances),
                  "Only nonempty TSP instance banks are supported.")
    suite.require(type(native_action_observation) is bool, "Select a boolean native protocol.")
    tasks.validate_instances(instances)
    configs = suite.model_configs(models, seed=20261005, no_think_max_tokens=64)
    schedule = [(i, arm) for i in instances for arm in ARMS]
    random.Random(20261005).shuffle(schedule)
    with suite.lane_locks(models):
        await preflight([SimpleNamespace(model=c) for c in configs.values()])
        root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "purpose": "Post-hoc matched development prompt ablation; actual own-history rollouts.",
            "provenance": provenance, "instances": instances, "arms": ARMS,
            "schedule": [{"instance_id": i["id"], "arm": arm} for i, arm in schedule],
            "models": {name: asdict(c) for name, c in configs.items()},
            "max_calls": len(models) * len(ARMS) * sum(i["query_budget"] for i in instances),
            "injected_client": client_factory is not create_client,
            "native_action_observation": native_action_observation,
            "native_protocol_definition": (
                "Prospective supplied-action eligibility; strict sum/argmax diagnostics remain "
                "recorded but do not gate continuation. All other native contract checks apply."
                if native_action_observation else "Existing strict native response contract."),
            "policy": {"repair": False, "retry": False, "oracle_history": False},
            "code_sha256": {**suite.code_hashes(), "scripts/tsp_state_rollout.py": suite.digest(__file__),
                            "scripts/construction_prefix_controls.py": suite.digest(controls.__file__),
                            "scripts/native_choice_contract_audit.py": suite.digest(native_audit.__file__)},
        }
        suite.write_json(root / "manifest.json", manifest)
        suite.write_json(root / "manifest.sha256", suite.digest(root / "manifest.json"))
        results = {}
        for model, config in configs.items():
            rows, histories, failures = [], {}, {}
            status, diagnostic, calls = "running", None, 0
            strict_failures = 0
            interrupted = None
            try:
                async with client_factory(config.provider, **config.client_kwargs()) as client:
                    await client.initialize()
                    with (root / f"{model}-attempts.jsonl").open("x") as stream:
                        for instance, arm in schedule:
                            history = histories.setdefault((instance["id"], arm), [])
                            while (request := tasks.next_request(instance, history)) is not None:
                                if arm == "state_explicit":
                                    request = controls.state_explicit_tsp_request(request)
                                request = replace(request, request_id=f"{request.request_id}:{arm}")
                                started, body = time.perf_counter(), None
                                try:
                                    response = await client.predict(request)
                                    body = response.raw_output
                                    record = suite.response_record(response, request, config)
                                except BaseException as exc:
                                    if isinstance(exc, InvalidResponseError):
                                        body = exc.raw_output
                                    record = {"status": "failure", "selected_option_id": None,
                                              **suite.failure(exc)}
                                    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                                        interrupted = exc
                                strict_failures += record["status"] == "failure"
                                if native_action_observation and config.provider == "typesafe":
                                    record = native_audit.action_execution_record(
                                        request, body, config.model, record)
                                calls += 1
                                suite.append_row(stream, {
                                    "instance_id": instance["id"], "arm": arm, "step": len(history),
                                    "request": asdict(request), "latency_seconds": time.perf_counter()-started,
                                    **record})
                                if record["status"] == "failure":
                                    failures[instance["id"], arm] = record["diagnostic_code"]
                                    if record["fatal"]:
                                        status, diagnostic = "aborted", record["diagnostic_code"]
                                    break
                                history.append(record["selected_option_id"])
                            if status == "aborted":
                                break
                    if status == "running":
                        status = "completed_with_failures" if failures else "completed"
            except BaseException as exc:
                status, diagnostic = "aborted", suite.failure(exc)["diagnostic_code"]
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    interrupted = exc
            if interrupted is not None:
                status = "interrupted"
            for instance, arm in schedule:
                history = histories.get((instance["id"], arm), [])
                score = tasks.score(instance, history)
                rows.append({"instance_id": instance["id"], "node_count": len(instance["state"]["nodes"]),
                             "arm": arm, "decisions": history, "attempted": (instance["id"], arm) in histories,
                             "failure": failures.get((instance["id"], arm)), **score})
            with (root / f"{model}-scores.jsonl").open("x") as stream:
                for row in rows:
                    suite.append_row(stream, row)
            summary = {}
            for arm in ARMS:
                cell = [r for r in rows if r["arm"] == arm]
                valid = [r for r in cell if r["feasible"]]
                summary[arm] = {
                    "scheduled": len(cell), "attempted": sum(r["attempted"] for r in cell),
                    "feasible": len(valid), "optimal": sum(r["optimal"] for r in cell),
                    "mean_relative_gap_feasible": suite.mean(
                        [r["absolute_gap"]/r["optimum"] for r in valid])}
            results[model] = {"status": status, "diagnostic_code": diagnostic, "calls": calls,
                              "strict_validation_failures": strict_failures,
                              "by_arm": summary, "artifact_sha256": {
                                  name: suite.digest(root / name) for name in
                                  (f"{model}-attempts.jsonl", f"{model}-scores.jsonl")
                                  if (root / name).exists()}}
            suite.write_json(root / f"{model}-status.json", results[model])
            if interrupted is not None:
                raise interrupted
        suite.write_json(root / "summary.json", results)
        return results


async def run_cli(root, instances, provenance, *, models, native_action_observation):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run(root, instances, models, provenance,
                         native_action_observation=native_action_observation)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("jev", "qwen9constrained"),
                        default=("jev", "qwen9constrained"))
    parser.add_argument("--native-action-observation", action="store_true",
                        help="Separate prospective action protocol; preserve strict diagnostics, "
                             "ignore only sum/argmax for native action eligibility.")
    args = parser.parse_args()
    source, _, instances, _ = verified_inputs(args.source, ("jev", "qwen9constrained"))
    instances = [i for i in instances if i["task"] == "tsp_construct"]
    results = asyncio.run(run_cli(args.root, instances, {
        "source_manifest_sha256": suite.digest(source / "manifest.json"),
        "selection": "Every TSP instance in the supplied frozen bank; no model-outcome filtering."},
        models=tuple(args.models), native_action_observation=args.native_action_observation))
    print(suite.canonical(results))
    return 0 if all(r["status"].startswith("completed") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
