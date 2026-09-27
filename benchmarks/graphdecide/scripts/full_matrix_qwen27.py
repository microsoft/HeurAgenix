#!/usr/bin/env python3
"""Registered canonical Qwen27B direct-decision evaluation; no thinking lanes."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import larger_local_pilot_runtime as runtime
from src.clients.base import ClientTimeoutError

suite, canonical, release = runtime.suite, runtime.canonical, runtime.release
invariance = runtime.panel.invariance
ROOT = REPO / "output/full-matrix-qwen27-20260925-v1"
BUNDLE = REPO / "output/graphbench-portable-release-20260924-v2"
REFERENCE = REPO / "output/explicit-actions-matrix-20260924-v1/structural_challenge.jev_action"
SPLIT = "structural_challenge"
MAX_CALLS = 1920
MAX_SECONDS = 6 * 60 * 60
EXPECTED = {
    "adjacency": 80, "degree_exact": 400, "cycle_detection": 80,
    "pair_connectivity": 80, "distance_threshold": 80, "articulation_point": 80,
    "maxcut_construct": 40, "tsp_construct": 40,
    "lt_influence_construct": 40, "community_bipartition_construct": 40,
}


def configuration():
    return runtime.panel.configurations()["qwen27_grammar"]


def source_hashes():
    files = ("scripts/larger_local_pilot.py", "scripts/larger_local_pilot_runtime.py",
             "scripts/full_matrix_qwen27.py")
    return {**invariance.source_hashes(), **{
        name: suite.digest(REPO / name) for name in files}}


def check_instances(instances):
    suite.require(len(instances) == 960 and len({i["id"] for i in instances}) == 960,
                  "Full canonical challenge requires 960 unique instances.")
    suite.require(Counter(i["task"] for i in instances) == EXPECTED,
                  "Full canonical task counts differ.")
    suite.require(sum(i["query_budget"] for i in instances) == MAX_CALLS,
                  "Canonical call budget differs.")


def register(root, bundle=BUNDLE, artifacts=runtime.ARTIFACTS):
    suite.require(not root.exists(), "Registration exists; no overwrite or resume.")
    verified = release.verify(bundle)
    manifest = release.manifest_at(bundle)
    instances = release.labeled_instances(bundle, SPLIT, manifest)
    check_instances(instances)
    validation = canonical.tasks.validate_instances(instances)
    reference_status = suite.read_json(REFERENCE / "status.json")
    suite.require(reference_status["status"] == "completed", "Reference is not complete.")
    for name, expected in reference_status["artifact_sha256"].items():
        suite.require(suite.digest(REFERENCE / name) == expected, "Reference artifact drift.")
    reference = {a["instance_id"]: a["request"] for a in
                 suite.read_rows(REFERENCE / "attempts.jsonl") if a["step"] == 0}
    requests = [{"instance_id": i["id"], "request": suite.json_value(
        asdict(canonical.public_request(i, [])))} for i in instances]
    suite.require(len(reference) == 960 and all(
        reference[r["instance_id"]] == r["request"] for r in requests),
        "Initial requests differ from canonical Jev.")
    checkpoint = runtime.verify_artifacts(artifacts)
    hashes = source_hashes()
    root.mkdir(parents=True)
    invariance.write_rows(root / "initial-requests.jsonl", requests)
    invariance.freeze_sources(root, hashes)
    registration = {
        "registered_at_utc": runtime.now(), "model_calls": 0,
        "protocol": "explicit_actions", "split": SPLIT,
        "bundle": str(bundle.resolve()), "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "artifacts": str(artifacts.resolve()),
        "artifacts_sha256": suite.digest(artifacts / "artifacts.json"),
        "checkpoint_revision": checkpoint["revision"], "model": asdict(configuration()),
        "source_sha256": hashes, "episodes": 960, "maximum_calls": MAX_CALLS,
        "maximum_run_seconds": MAX_SECONDS, "task_counts": EXPECTED,
        "initial_requests_sha256": suite.digest(root / "initial-requests.jsonl"),
        "reference_status_sha256": suite.digest(REFERENCE / "status.json"),
        "reference_attempts_sha256": suite.digest(REFERENCE / "attempts.jsonl"),
        "verified_release": verified, "independent_validation": validation,
        "environment": suite.environment_versions(),
        "policy": {"retry": False, "repair": False, "resume": False, "thinking": False,
                   "history": "Each model's own complete trajectory.",
                   "selection": "Every canonical structural-challenge instance, unchanged order.",
                   "denominator": "All 960 scheduled cases, including failures and unattempted.",
                   "resource_owner": "Registered parent coordinator holds qwen08/qwen2 leases.",
                   "response_storage": "No response bodies or reasoning text.",
                   "comparison": "Inspected-bank full extension, not fresh confirmation."},
    }
    suite.write_json(root / "registration.json", registration)
    suite.write_json(root / "registration.sha256", suite.digest(root / "registration.json"))
    suite.write_json(root / "dry-run.json", {
        "model_calls": 0, "initial_requests_identical": 960,
        "zero_attempt_score_rows": len(canonical.replay(
            instances, [], configuration(), native_action=False))})
    runtime.status(root, status="registered", calls=0)
    return registration


def load(root):
    value = suite.read_json(root / "registration.json")
    suite.require(suite.digest(root / "registration.json") ==
                  suite.read_json(root / "registration.sha256"), "Registration drift.")
    suite.require(value["model"] == asdict(configuration()) and value["episodes"] == 960
                  and value["maximum_calls"] == MAX_CALLS
                  and value["maximum_run_seconds"] == MAX_SECONDS, "Configuration drift.")
    verify_sources(root, value)
    bundle = Path(value["bundle"])
    suite.require(suite.digest(bundle / "manifest.json") == value["bundle_manifest_sha256"],
                  "Bundle drift.")
    suite.require(suite.digest(Path(value["artifacts"]) / "artifacts.json") ==
                  value["artifacts_sha256"], "Checkpoint manifest drift.")
    instances = release.labeled_instances(bundle, SPLIT, release.manifest_at(bundle))
    check_instances(instances)
    suite.require(suite.digest(root / "initial-requests.jsonl") ==
                  value["initial_requests_sha256"], "Initial request file drift.")
    expected = [{"instance_id": i["id"], "request": suite.json_value(
        asdict(canonical.public_request(i, [])))} for i in instances]
    suite.require(suite.read_rows(root / "initial-requests.jsonl") == expected,
                  "Initial request reconstruction differs.")
    return value, instances


def verify_sources(root, registration):
    suite.require(source_hashes() == registration["source_sha256"], "Live source drift.")
    for name, digest in registration["source_sha256"].items():
        suite.require(suite.digest(root / "source-snapshot" / name) == digest,
                      "Source snapshot drift.")


async def run(root, *, client_factory=runtime.RecordingClient,
              preflight=suite.shared.vllm_preflight):
    registration, instances = load(root)
    output = root / "run"
    suite.require(not output.exists(), "Run already exists; no retry/resume.")
    output.mkdir()
    config = configuration()
    suite.write_json(output / "manifest.json", {
        "registration_sha256": suite.digest(root / "registration.json"),
        "model": asdict(config), "episodes": 960, "maximum_calls": MAX_CALLS,
        "injected_client": client_factory is not runtime.RecordingClient})
    attempts, client = [], None
    state = {"status": "running", "calls": 0, "pid": os.getpid()}
    started = time.monotonic()
    runtime.status(root, **state)
    with (output / "attempts.jsonl").open("x") as stream:
        try:
            async with asyncio.timeout(MAX_SECONDS):
                await preflight([SimpleNamespace(model=config)])
                client = client_factory(**config.client_kwargs())
                await asyncio.wait_for(client.initialize(), config.timeout_seconds)
                for instance in instances:
                    history = []
                    while (request := canonical.public_request(instance, history)) is not None:
                        suite.require(len(attempts) < MAX_CALLS and
                                      len(history) < instance["query_budget"], "Call budget exceeded.")
                        verify_sources(root, registration)
                        tick, diagnostics, error_type = time.perf_counter(), None, None
                        cancellation = None
                        try:
                            response = await asyncio.wait_for(
                                client.predict(request), config.timeout_seconds)
                            diagnostics = runtime.response_diagnostics(response.raw_output)
                            record = suite.response_record(response, request, config)
                        except BaseException as exc:
                            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                                cancellation = exc
                            if isinstance(exc, TimeoutError):
                                exc = ClientTimeoutError("Full-matrix request deadline exceeded.")
                            error_type = type(exc).__name__
                            diagnostics = getattr(exc, "response_diagnostics", None)
                            record = {"status": "failure", "selected_option_id": None,
                                      "resolved_model": None,
                                      "usage": suite.safe_usage(getattr(exc, "usage", None)),
                                      **suite.failure(exc)}
                            if cancellation is not None:
                                record["fatal"] = True
                        row = {"instance_id": instance["id"], "step": len(history),
                               "request": asdict(request), "response_diagnostics": diagnostics,
                               "exception_type": error_type,
                               "latency_seconds": time.perf_counter() - tick, **record}
                        suite.append_row(stream, row)
                        stream.flush()
                        os.fsync(stream.fileno())
                        attempts.append(suite.json_value(row))
                        runtime.status(root, **{**state, "calls": len(attempts),
                                               "last_instance_id": instance["id"]})
                        if cancellation is not None:
                            raise cancellation
                        if record["status"] == "failure":
                            if record["fatal"]:
                                state.update(status="aborted",
                                             diagnostic_code=record["diagnostic_code"])
                            break
                        history.append(record["selected_option_id"])
                    if state["status"] != "running":
                        break
                verify_sources(root, registration)
                if state["status"] == "running":
                    state["status"] = "completed_with_failures" if any(
                        a["status"] == "failure" for a in attempts) else "completed"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(
                exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)) else "aborted",
                diagnostic_code="run_deadline" if isinstance(exc, TimeoutError)
                else suite.failure(exc)["diagnostic_code"], exception_type=type(exc).__name__)
        finally:
            if client is not None:
                try:
                    await asyncio.wait_for(client.aclose(), 30)
                except BaseException as exc:
                    state.update(status="aborted", diagnostic_code="shutdown_failed",
                                 exception_type=type(exc).__name__)
            rows = canonical.replay(instances, attempts, config, native_action=False)
            invariance.write_rows(output / "scores.jsonl", rows)
            suite.write_json(output / "summary.json", suite.summarize(rows))
            state.update(calls=len(attempts), wall_seconds=time.monotonic() - started,
                         finished_at_utc=runtime.now(),
                         failure_counts=dict(Counter(a["diagnostic_code"] for a in attempts
                                                    if a["status"] == "failure")))
            suite.write_json(output / "terminal.json", state)
            suite.write_json(output / "seal.json", {
                name: suite.digest(output / name) for name in
                ("manifest.json", "attempts.jsonl", "scores.jsonl", "summary.json", "terminal.json")})
            runtime.status(root, **state)
    verify(root, allow_injected=client_factory is not runtime.RecordingClient)
    return state


def verify(root, *, allow_injected=False):
    registration, instances = load(root)
    output = root / "run"
    for name, digest in suite.read_json(output / "seal.json").items():
        suite.require(suite.digest(output / name) == digest, "Output seal mismatch.")
    manifest = suite.read_json(output / "manifest.json")
    suite.require(allow_injected or manifest["injected_client"] is False,
                  "Injected result cannot be accepted as model evidence.")
    suite.require(manifest["registration_sha256"] == suite.digest(root / "registration.json")
                  and manifest["model"] == registration["model"], "Run identity drift.")
    attempts = suite.read_rows(output / "attempts.jsonl")
    suite.require(len(attempts) <= MAX_CALLS, "Attempt budget exceeded.")
    rows = canonical.replay(instances, attempts, configuration(), native_action=False)
    suite.require(rows == suite.read_rows(output / "scores.jsonl"), "Score replay differs.")
    suite.require(suite.json_value(suite.summarize(rows)) ==
                  suite.read_json(output / "summary.json"), "Summary replay differs.")
    terminal = suite.read_json(output / "terminal.json")
    suite.require(terminal["calls"] == len(attempts), "Terminal attempt count differs.")
    if terminal["status"].startswith("completed"):
        suite.require(all(r["feasible"] or r["failure_count"] for r in rows),
                      "Completed status conceals unattempted instances.")
    return {"verified": True, "status": terminal["status"], "episodes": len(rows),
            "calls": len(attempts), "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = register(args.root) if args.command == "register" else verify(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
