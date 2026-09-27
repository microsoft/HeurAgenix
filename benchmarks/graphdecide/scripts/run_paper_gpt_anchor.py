#!/usr/bin/env python3
"""Preregistered, tool-free GPT-5.4 evaluation of canonical released graph actions."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import evaluate_graph_release as evaluation
from scripts import extended_graph_suite as suite
from scripts import graph_release_api as release
from src.clients.registry import create_client

SEED = 20261009
SPLIT = "structural_challenge"
LOCK_MODELS = ("gpt54",)
RELEASE_SHA256 = "0735d1422c35b4593f9621706425aa13d72b1b9ffdcae959d67a1dbef072365d"
V2_MANIFEST_SHA256 = "e7f15bc73e7331e87292680eb04e2dd638afccd7e0e8b50cf937241cbe29a689"
ARTIFACTS = ("requests.jsonl", "attempts.jsonl", "episodes.jsonl", "summary.json",
             "events.jsonl", "status.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def configuration():
    # Copilot does not expose a sampling seed: retain the suite's exact config.
    return suite.model_configs(("gpt54",), seed=SEED)["gpt54"]


def resource_locks(lock_factory=suite.lane_locks):
    return lock_factory(LOCK_MODELS)


def sources():
    paths = set(suite.code_hashes())
    paths.update("scripts/" + name for name in (
        "run_paper_gpt_anchor.py", "evaluate_graph_release.py", "graph_release_api.py",
        "graph_task_prompt.py", "native_choice_contract_audit.py", "summarize_extended_paper.py"))
    return {name: suite.digest(REPO / name) for name in sorted(paths)}


def resource_amendment(prior, registration):
    old = suite.read_json(prior / "manifest.json")
    old_digest = suite.digest(prior / "manifest.json")
    suite.require(old_digest == V2_MANIFEST_SHA256 ==
                  suite.read_json(prior / "manifest.sha256"), "Unexpected v2 registration.")
    status = suite.read_json(prior / "status.json")
    suite.require(status["status"] in {"registered", "waiting", "interrupted"}
                  and status["calls"] == 0 and all(not (prior / name).exists()
                  for name in ("attempts.jsonl", "requests.jsonl", "worker-process.json")),
                  "Resource amendment requires an unstarted zero-call original.")
    invariants = ("protocol", "profile", "split", "bundle", "bundle_manifest_sha256",
                  "planned_instances", "maximum_calls", "registration_seed",
                  "provider_sampling_seed_supported", "model", "config_sha256",
                  "shared_lock_directory", "test_only", "injected_client")
    suite.require(all(old[key] == registration[key] for key in invariants),
                  "Resource amendment changed benchmark configuration.")
    for name, digest in registration["code_sha256"].items():
        if name != "scripts/run_paper_gpt_anchor.py":
            suite.require(old["code_sha256"].get(name) == digest,
                          "Existing benchmark dependency changed during resource amendment.")
    suite.require({k: v for k, v in old["policy"].items() if k != "shared_cli_limit"} ==
                  {k: v for k, v in registration["policy"].items() if k != "shared_cli_limit"},
                  "Resource amendment changed non-resource policy.")
    return {
        "schema_version": 1, "kind": "pre_inference_resource_only", "at_utc": now(),
        "prior_root": str(prior.resolve()), "prior_manifest_sha256": old_digest,
        "prior_status_observation": status, "prior_calls": 0,
        "prior_attempts_exist": False, "prior_worker_started": False,
        "old_resource_models": old["resource_models"],
        "new_resource_models": registration["resource_models"],
        "unchanged_fields": list(invariants), "existing_benchmark_dependencies_unchanged": True,
        "reason": "User authorizes distinct-model concurrency; existing suite runs separate GPT "
                  "lanes concurrently, with independent adapter runtimes, directories and locks. "
                  "Both-model serialization was conservative, not a demonstrated SDK requirement.",
        "ownership": "Acquire the unchanged standard gpt54 lock only; never acquire, release, "
                     "rename, bypass or otherwise alter the gpt6astra lock or its owner.",
        "source_snapshot_scope": "Pin actual imported script dependencies plus suite source hashes; "
                                 "exclude unrelated concurrently developed scripts.",
        "provider_failure_policy": "Unchanged: no retry, repair, alternate-model fallback or "
                                   "automatic serialization fallback; classify and preserve failures.",
    }


def register(bundle, output, *, test_only=False, split=SPLIT, amends=None):
    bundle, output = bundle.resolve(), output.resolve()
    suite.require(not output.exists(), "Output exists; overwrite/resume is forbidden.")
    checked = release.verify(bundle)
    manifest = release.manifest_at(bundle)
    suite.require(test_only or suite.digest(bundle / "manifest.json") == RELEASE_SHA256,
                  "Production anchor requires the pinned portable v2 release.")
    suite.require("explicit_actions" in manifest["protocols"], "Canonical actions unavailable.")
    instances = release.labeled_instances(bundle, split, manifest)
    budget = sum(i["query_budget"] for i in instances)
    suite.require(test_only or split == SPLIT and len(instances) == 960 and budget == 1920,
                  "Production anchor requires the released challenge960/1920 budget.")
    config = configuration()
    suite.require(config.model == "gpt-5.4" and config.max_tokens == 4096
                  and config.think is None and config.reasoning_effort is None
                  and config.output_format == "answer_only", "Unexpected GPT baseline.")
    output.mkdir(parents=True)
    hashes = sources()
    for name, expected in hashes.items():
        target = output / suite.SOURCE_SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((REPO / name).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        suite.require(suite.digest(target) == expected, "Source changed while freezing.")
        target.chmod(0o444)
    suite.require(hashes == sources(), "Source changed during registration.")
    suite.write_json(output / "config.json", asdict(config))
    registration = {
        "schema_version": 1, "registered_at_utc": now(), "protocol": "explicit_actions",
        "profile": "gpt54_default_reasoning", "split": split, "bundle": str(bundle),
        "repository": str(REPO),
        "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "verified_release_instances": checked["instances"],
        "planned_instances": len(instances), "maximum_calls": budget,
        "registration_seed": SEED, "provider_sampling_seed_supported": False,
        "model": asdict(config), "config_sha256": suite.digest(output / "config.json"),
        "source_snapshot": suite.SOURCE_SNAPSHOT, "code_sha256": hashes,
        "environment": suite.environment_versions(), "injected_client": test_only,
        "test_only": test_only, "resource_models": list(LOCK_MODELS),
        "shared_lock_directory": os.environ.get("TMPDIR") or "/tmp",
        "policy": {"retry": False, "repair": False, "resume": False,
                   "tools": False, "cross_question_sessions": False,
                   "private_labels_sent": False, "own_history": True,
                   "sdk_internal_retries": "adapter rejects observed retry/multiple usage events",
                   "shared_cli_limit": "one gpt54 client under its standard model lock for full "
                                       "runtime lifetime; other models use independent runtimes"},
    }
    if amends is not None:
        amendment = resource_amendment(amends, registration)
        with (output / "prior-manifest.json").open("xb") as stream:
            stream.write((amends / "manifest.json").read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        suite.write_json(output / "resource-amendment.json", amendment)
        registration["resource_amendment_sha256"] = suite.digest(output / "resource-amendment.json")
    suite.write_json(output / "manifest.json", registration)
    suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
    suite.set_status(output, {"status": "registered", "calls": 0, "updated_at_utc": now()})
    return registration


def load(output, *, check_execution_sources=False):
    manifest = suite.read_json(output / "manifest.json")
    suite.require(suite.digest(output / "manifest.json") ==
                  suite.read_json(output / "manifest.sha256"), "Manifest hash mismatch.")
    suite.verify_source_snapshot(output, manifest["code_sha256"])
    suite.require(manifest["resource_models"] == list(LOCK_MODELS), "Resource ownership mismatch.")
    if "resource_amendment_sha256" in manifest:
        suite.require(suite.digest(output / "resource-amendment.json") ==
                      manifest["resource_amendment_sha256"], "Resource amendment changed.")
        amendment = suite.read_json(output / "resource-amendment.json")
        suite.require(suite.digest(output / "prior-manifest.json") ==
                      amendment["prior_manifest_sha256"] == V2_MANIFEST_SHA256,
                      "Original registration pin changed.")
    suite.require(suite.digest(output / "config.json") == manifest["config_sha256"]
                  and suite.read_json(output / "config.json") == manifest["model"]
                  and asdict(configuration()) == manifest["model"], "Configuration mismatch.")
    if check_execution_sources:
        suite.require(sources() == manifest["code_sha256"], "Executing source differs from registration.")
    bundle = Path(manifest["bundle"])
    suite.require(suite.digest(bundle / "manifest.json") == manifest["bundle_manifest_sha256"],
                  "Bundle identity changed.")
    release.verify(bundle)
    instances = release.labeled_instances(bundle, manifest["split"], release.manifest_at(bundle))
    suite.require(len(instances) == manifest["planned_instances"] and
                  sum(i["query_budget"] for i in instances) == manifest["maximum_calls"],
                  "Registered budget changed.")
    return manifest, instances, configuration()


def replay(instances, attempts, config):
    rows = evaluation.replay(instances, attempts, config, native_action=False)
    histories = {}
    by_id = {i["id"]: i for i in instances}
    for row in attempts:
        history = histories.setdefault(row["instance_id"], [])
        request = evaluation.public_request(by_id[row["instance_id"]], history)
        suite.validate_attempt(row, request, config)
        if row["status"] == "success":
            history.append(row["selected_option_id"])
    return rows


def summary_for(rows, attempts):
    return {**suite.summarize(rows),
            "calls": len(attempts),
            "latency_seconds": {
                "total": sum(r["latency_seconds"] for r in attempts),
                "maximum": max((r["latency_seconds"] for r in attempts), default=None)},
            "usage": {key: {"known_total": sum(r["usage"][key] or 0 for r in attempts),
                            "known_calls": sum(r["usage"][key] is not None for r in attempts),
                            "unknown_calls": sum(r["usage"][key] is None for r in attempts)}
                      for key in suite.USAGE_KEYS}}


def verify(output):
    manifest, instances, config = load(output)
    seal = suite.read_json(output / "seal.json")
    suite.require(set(seal["sha256"]) == set(ARTIFACTS) | {"manifest.json", "manifest.sha256",
                                                        "config.json"}, "Seal file set mismatch.")
    for name, expected in seal["sha256"].items():
        suite.require(not (output / name).is_symlink() and suite.digest(output / name) == expected,
                      "Sealed artifact changed.")
    attempts = suite.read_rows(output / "attempts.jsonl")
    suite.require(len(attempts) <= manifest["maximum_calls"], "Call budget exceeded.")
    requests = [{"instance_id": r["instance_id"], "step": r["step"], "request": r["request"]}
                for r in attempts]
    suite.require(requests == suite.read_rows(output / "requests.jsonl"), "Request ledger mismatch.")
    rows = replay(instances, attempts, config)
    suite.require(rows == suite.read_rows(output / "episodes.jsonl"), "Episode replay mismatch.")
    suite.require(summary_for(rows, attempts) == suite.read_json(output / "summary.json"),
                  "Summary replay mismatch.")
    status = suite.read_json(output / "status.json")
    suite.require(status["status"] in suite.TERMINAL and status["calls"] == len(attempts),
                  "Invalid final status.")
    if status["status"].startswith("completed"):
        suite.require(all(r["feasible"] or r["failure_count"] for r in rows),
                      "Completed run left unattempted episodes.")
        expected = "completed_with_failures" if any(r["status"] == "failure" for r in attempts) else "completed"
        suite.require(status["status"] == expected, "Completion/failure status mismatch.")
    return {"status": "verified", "run_status": status["status"], "calls": len(attempts),
            "planned_instances": len(rows), "injected_client": manifest["injected_client"],
            "manifest_sha256": suite.digest(output / "manifest.json"),
            "seal_sha256": suite.digest(output / "seal.json")}


async def execute(output, *, client_factory=create_client):
    manifest, instances, config = load(output, check_execution_sources=True)
    suite.require(manifest["test_only"] == (client_factory is not create_client),
                  "Fake clients require an explicitly test-only registration.")
    if client_factory is create_client:
        suite.require(suite.read_json(output / "dispatch-owner.json")["pid"] == os.getppid(),
                      "Production worker requires its lock-owning dispatcher.")
    suite.require(not (output / "attempts.jsonl").exists(), "No resume or second execution.")
    attempts, client, interrupted = [], None, None
    tick = time.perf_counter()
    status = {"status": "running", "calls": 0, "pid": os.getpid(), "started_at_utc": now(),
              "diagnostic_code": None, "injected_client": manifest["injected_client"]}
    suite.set_status(output, status)
    with ExitStack() as stack:
        logs = {name: stack.enter_context((output / name).open("x"))
                for name in ("requests.jsonl", "attempts.jsonl", "events.jsonl")}

        def event(phase, exc):
            suite.append_row(logs["events.jsonl"], {"phase": phase, "at_utc": now(),
                                                  **suite.failure(exc)})

        phase = "initialization"
        try:
            client = client_factory(config.provider, **config.client_kwargs())
            await client.initialize()
            load(output, check_execution_sources=True)
            phase = "prediction"
            for instance in instances:
                history = []
                while (request := evaluation.public_request(instance, history)) is not None:
                    public = {"instance_id": instance["id"], "step": len(history),
                              "request": suite.json_value(asdict(request))}
                    suite.append_row(logs["requests.jsonl"], public)
                    started = time.perf_counter()
                    try:
                        response = await client.predict(request)
                        record = suite.response_record(response, request, config)
                    except BaseException as exc:
                        record = {"status": "failure", "selected_option_id": None,
                                  "resolved_model": None,
                                  "usage": suite.safe_usage(getattr(exc, "usage", None)),
                                  **suite.failure(exc)}
                        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                            interrupted = exc
                    row = {**public, "latency_seconds": time.perf_counter() - started, **record}
                    suite.append_row(logs["attempts.jsonl"], row)
                    attempts.append(row)
                    status["calls"] = len(attempts)
                    suite.set_status(output, status)
                    if record["status"] == "failure":
                        if record["fatal"]:
                            status.update(status="aborted", diagnostic_code=record["diagnostic_code"])
                        break
                    history.append(record["selected_option_id"])
                if status["status"] != "running":
                    break
            if status["status"] == "running":
                load(output, check_execution_sources=True)
                status["status"] = ("completed_with_failures" if any(
                    a["status"] == "failure" for a in attempts) else "completed")
        except BaseException as exc:
            event(phase, exc)
            status.update(status="aborted", diagnostic_code=(
                "initialization_failed" if phase == "initialization" else suite.failure(exc)["diagnostic_code"]))
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                interrupted = exc
        finally:
            if client is not None:
                cleanup = asyncio.create_task(client.aclose())
                try:
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError as exc:
                            interrupted = exc
                    cleanup.result()
                except BaseException as exc:
                    event("shutdown", exc)
                    status.update(status="aborted", diagnostic_code="shutdown_failed")
                    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                        interrupted = exc
            if interrupted is not None:
                status.update(status="interrupted", diagnostic_code="interrupted")
            rows = replay(instances, attempts, config)
            with (output / "episodes.jsonl").open("x") as stream:
                for row in rows:
                    suite.append_row(stream, row)
            suite.write_json(output / "summary.json", summary_for(rows, attempts))
            status.update(calls=len(attempts), finished_at_utc=now(),
                          wall_seconds=time.perf_counter() - tick)
            suite.set_status(output, status)
    suite.write_json(output / "seal.json", {
        "schema_version": 1, "sealed_at_utc": now(),
        "sha256": {name: suite.digest(output / name)
                   for name in (*ARTIFACTS, "manifest.json", "manifest.sha256", "config.json")}})
    suite.write_json(output / "verification.json", verify(output))
    if interrupted is not None:
        raise interrupted
    return status


async def worker(output):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        return await execute(output)
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def dispatch(output, *, allow_system_shared_locks=False):
    """Own the standard GPT-5.4 lock before starting its isolated SDK worker."""
    manifest, _, _ = load(output, check_execution_sources=True)
    suite.require(not manifest["test_only"], "Cannot dispatch a fake-client registration.")
    suite.require(not (output / "attempts.jsonl").exists(), "No resume.")
    suite.require((os.environ.get("TMPDIR") or "/tmp") == manifest["shared_lock_directory"],
                  "Shared lock namespace changed.")
    with (output / "dispatcher.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite.write_json(output / "dispatch-authorization.json", {
            "at_utc": now(), "pid": os.getpid(),
            "allow_system_shared_locks": allow_system_shared_locks,
            "shared_lock_directory": manifest["shared_lock_directory"],
            "resource_models": list(LOCK_MODELS),
        })

        def cancel_wait(number, _):
            suite.set_status(output, {"status": "interrupted", "calls": 0, "pid": os.getpid(),
                                      "diagnostic_code": "interrupted", "phase": "resource_wait",
                                      "finished_at_utc": now(), "signal": number})
            raise SystemExit(128 + number)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, cancel_wait)
        while True:
            waiting = {"status": "waiting", "calls": 0, "pid": os.getpid(),
                       "updated_at_utc": now(), "resource_models": list(LOCK_MODELS)}
            lock_root = Path(os.path.abspath(manifest["shared_lock_directory"]))
            if not allow_system_shared_locks and not lock_root.is_relative_to(
                    Path(manifest["repository"])):
                # Never relocate shared locks: doing so would silently bypass active jobs.
                waiting["waiting_reason"] = "shared_lock_directory_outside_project_policy"
                suite.set_status(output, waiting)
                time.sleep(30)
                continue
            try:
                with resource_locks():
                    local_runtime = output / "sdk-runtime"
                    local_runtime.mkdir(exist_ok=False)
                    environment = dict(os.environ, TMPDIR=str(local_runtime),
                                       TMP=str(local_runtime), TEMP=str(local_runtime))
                    command = [sys.executable, "-u", str(Path(__file__).resolve()),
                               "_worker", "--output", str(output)]
                    suite.write_json(output / "dispatch-owner.json", {"pid": os.getpid()})
                    with (output / "worker.log").open("x") as log:
                        process = subprocess.Popen(command, env=environment, stdout=log,
                                                   stderr=subprocess.STDOUT)
                        suite.write_json(output / "worker-process.json",
                                         {"pid": process.pid, "dispatcher_pid": os.getpid(),
                                          "started_at_utc": now(), "command": command})
                        previous = {}
                        for sig in (signal.SIGTERM, signal.SIGINT):
                            previous[sig] = signal.signal(
                                sig, lambda number, _: process.send_signal(number))
                        try:
                            return process.wait()
                        finally:
                            for sig, handler in previous.items():
                                signal.signal(sig, handler)
            except suite.Refusal:
                waiting["waiting_reason"] = "shared_gpt_locks_busy"
                suite.set_status(output, waiting)
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "dispatch", "_worker", "verify"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--amends", type=Path, help="Zero-call v2 registration to preserve and amend.")
    parser.add_argument("--allow-system-shared-locks", action="store_true",
                        help="Parent-only launch authorization for the existing system lock namespace.")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        suite.require(args.bundle is not None, "--bundle is required for registration.")
        result = register(args.bundle, output, amends=args.amends)
    elif args.command == "dispatch":
        return dispatch(output, allow_system_shared_locks=args.allow_system_shared_locks)
    elif args.command == "_worker":
        # The only production worker entry must be a child of its lock-owning dispatcher.
        owner = suite.read_json(output / "dispatch-owner.json")
        suite.require(owner["pid"] == os.getppid(),
                      "Worker must be launched by the registered lock owner.")
        result = asyncio.run(worker(output))
    else:
        result = verify(output)
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("status") in {"aborted", "interrupted"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
