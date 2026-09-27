#!/usr/bin/env python3
"""Post-hoc, preregistered Qwen9 construction-only deadline sensitivity control."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import larger_local_pilot_runtime as shared
from src.clients.base import ClientTimeoutError

suite, panel, canonical = shared.suite, shared.panel, shared.canonical
ROOT = REPO / "output/qwen9-deadline-control-20260924-v1"
PREVIOUS = REPO / "output/larger-local-pilot-execution-20260924-v1"
SERVICE_PID = 713079
WAIT_SECONDS = 24 * 60 * 60
RUN_SECONDS = 20 * 60 * 60
CALL_SECONDS = 600.0
MAX_CALLS = 112
TERMINAL = frozenset(("completed", "completed_with_failures", "aborted", "interrupted", "unattempted"))
WAITING = frozenset(("registered", "running"))


class AdmissionError(RuntimeError):
    pass


class WaitDeadline(RuntimeError):
    pass


class RunDeadline(RuntimeError):
    pass


def configuration():
    return replace(panel.configurations()["qwen9_thinking"], timeout_seconds=CALL_SECONDS)


def environment():
    result = suite.environment_versions()
    result["executable"] = sys.executable
    for name in ("vllm", "torch", "transformers"):
        try:
            result["packages"][name] = version(name)
        except PackageNotFoundError:
            result["packages"][name] = None
    return result


def select_instances(instances):
    result = [i for i in instances if i["kind"] == "optimization"]
    expected = {(task, size): 2 for task in panel.CONSTRUCTION for size in panel.SIZES}
    suite.require(Counter((i["task"], len(i["state"]["nodes"])) for i in result) == expected,
                  "Expected all sixteen construction episodes, two per task/size.")
    suite.require(len(result) == 16 and len({i["id"] for i in result}) == 16
                  and sum(i["query_budget"] for i in result) == MAX_CALLS,
                  "Construction panel cardinality/budget differs.")
    return result


def service_snapshot():
    identity = shared.process(SERVICE_PID)
    argv = identity["argv"]
    suite.require("serve" in argv and argv[argv.index("--port") + 1] == "8003"
                  and argv[argv.index("--host") + 1] == "127.0.0.1"
                  and argv[argv.index("--served-model-name") + 1] == "Qwen3.5-9B"
                  and "--trust-remote-code" not in argv,
                  "Unexpected existing Qwen9 service configuration.")
    return {key: identity[key] for key in
            ("pid", "ppid", "start_ticks", "argv", "cuda_environment", "cwd")}


def checkpoint_snapshot(service):
    argv = service["argv"]
    model = Path(argv[argv.index("serve") + 1]).resolve()
    files = {}
    for path in sorted(model.iterdir()):
        if path.name == ".cache":
            continue
        suite.require(path.is_file() and not path.is_symlink(), "Unexpected checkpoint entry.")
        files[path.name] = {"sha256": suite.digest(path), "bytes": path.stat().st_size}
    index = suite.read_json(model / "model.safetensors.index.json")
    suite.require(set(index["weight_map"].values()) <= set(files), "Missing checkpoint shard.")
    metadata = model / ".cache/huggingface/download/config.json.metadata"
    revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    suite.require(revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                  "Invalid local Hub revision metadata.")
    return {"model_directory": str(model), "local_hub_config_revision": revision,
            "files": files, "identity_basis": "SHA256 of every existing non-cache checkpoint file"}


def verify_service(expected):
    suite.require(service_snapshot() == expected, "Existing Qwen9 service identity/configuration changed.")


def verify_checkpoint(checkpoint):
    model = Path(checkpoint["model_directory"])
    names = {p.name for p in model.iterdir() if p.name != ".cache"}
    suite.require(names == set(checkpoint["files"]), "Checkpoint file inventory changed.")
    for name, entry in checkpoint["files"].items():
        path = model / name
        suite.require(path.is_file() and not path.is_symlink()
                      and path.stat().st_size == entry["bytes"]
                      and suite.digest(path) == entry["sha256"], "Checkpoint checksum changed.")


def sources(previous):
    result = dict(previous["code_sha256"])
    for name, digest in result.items():
        suite.require(suite.digest(REPO / name) == digest, "Existing frozen dependency drift.")
    result["scripts/qwen9_deadline_control.py"] = suite.digest(__file__)
    return result


def prepare(root=ROOT, previous_root=PREVIOUS):
    root = shared.local(root)
    suite.require(not root.exists(), "Fresh registration required; overwrite forbidden.")
    previous = shared.verify_manifest(previous_root, "execution.json")
    before = sources(previous)
    panel_root = Path(previous["panel"])
    panel_registration, all_instances = shared.load_panel(panel_root)
    instances = select_instances(all_instances)
    original, updated = previous["configurations"]["qwen9_thinking"], asdict(configuration())
    suite.require(original == asdict(panel.configurations()["qwen9_thinking"])
                  and {k for k in updated if updated[k] != original[k]} == {"timeout_seconds"}
                  and original["timeout_seconds"] == 180.0 and updated["timeout_seconds"] == 600.0,
                  "Only the 180-to-600-second deadline may change.")
    validation = canonical.tasks.validate_instances(instances)
    service = service_snapshot()
    checkpoint = checkpoint_snapshot(service)
    verify_service(service)
    suite.require(before == sources(previous), "Source changed during registration.")
    root.mkdir(parents=True)
    panel.invariance.write_rows(root / "cases.jsonl", [
        {"index": index, "instance_id": i["id"], "task": i["task"],
         "node_count": len(i["state"]["nodes"]), "query_budget": i["query_budget"]}
        for index, i in enumerate(instances)])
    panel.invariance.write_rows(root / "initial-requests.jsonl", [
        {"instance_id": i["id"], "request": asdict(canonical.public_request(i, []))}
        for i in instances])
    shared.write(root / "checkpoint.json", checkpoint)
    panel.invariance.freeze_sources(root, before)
    registration = {
        "schema_version": 1, "registered_at_utc": shared.now(), "model_calls": 0,
        "design": "Post-hoc registered deadline sensitivity; not repaired original results "
                  "and not confirmatory accuracy or an isolated planning-quality ranking.",
        "selection": "All sixteen construction episodes of the existing immutable64 panel "
                     "in its original order. No predecessor correctness read or failure-selected cases.",
        "previous_root": str(previous_root.resolve()),
        "previous_execution_sha256": suite.digest(previous_root / "execution.json"),
        "predecessor_status": str(previous_root / "qwen9_thinking/status.json"),
        "panel": str(panel_root), "panel_registration_sha256": suite.digest(panel_root / "registration.json"),
        "panel_selection_sha256": panel_registration["selection_sha256"],
        "cases_sha256": suite.digest(root / "cases.jsonl"),
        "initial_requests_sha256": suite.digest(root / "initial-requests.jsonl"),
        "checkpoint_sha256": suite.digest(root / "checkpoint.json"),
        "service_identity": service, "source_sha256": before,
        "environment": environment(),
        "original_configuration": original, "configuration": updated,
        "changed_configuration_fields": ["timeout_seconds"],
        "episodes": 16, "maximum_calls": MAX_CALLS, "maximum_call_seconds": 67200,
        "prestart_wait_limit_seconds": WAIT_SECONDS, "total_run_limit_seconds": RUN_SECONDS,
        "run_limit_scope": "Admission, checksum/source checks, startup/preflight, all requests, "
                           "and bounded client shutdown. Prestart waiting is separately bounded.",
        "independent_validation": validation,
        "policy": {"retry": False, "repair": False, "resume": False, "resample": False,
                   "old_scores_read": False, "own_history": True, "all_scheduled_denominator": 16,
                   "model_context_or_token_cap_changes": False,
                   "max_tokens": 8192, "length_cap_failures_retained": True,
                   "lock_api": "extended_graph_suite.lane_locks(('qwen9',))",
                   "lock_admission": "Once only, nonblocking, after predecessor terminal.",
                   "service_or_gpu_actions": False, "checkpoint_downloads": False,
                   "response_storage": "Existing redacted client and allowlisted diagnostics; "
                                       "no full response body, final answer text, or reasoning text."}}
    shared.write(root / "registration.json", registration)
    shared.write(root / "registration.sha256", suite.digest(root / "registration.json"))
    for name in ("registration.json", "registration.sha256", "cases.jsonl",
                 "initial-requests.jsonl", "checkpoint.json"):
        (root / name).chmod(0o444)
    return registration


def verified_inputs(root):
    registration = suite.read_json(root / "registration.json")
    suite.require(suite.digest(root / "registration.json") ==
                  suite.read_json(root / "registration.sha256"), "Registration digest changed.")
    suite.require(environment() == registration["environment"], "Registered runtime environment changed.")
    for name, expected in registration["source_sha256"].items():
        suite.require(suite.digest(REPO / name) == expected, "Frozen source changed.")
        suite.require(suite.digest(root / "source-snapshot" / name) == expected,
                      "Source snapshot changed.")
    for filename, field in (("cases.jsonl", "cases_sha256"),
                            ("initial-requests.jsonl", "initial_requests_sha256"),
                            ("checkpoint.json", "checkpoint_sha256")):
        suite.require(suite.digest(root / filename) == registration[field], "Registered input changed.")
    previous_root, panel_root = Path(registration["previous_root"]), Path(registration["panel"])
    suite.require(suite.digest(previous_root / "execution.json") ==
                  registration["previous_execution_sha256"], "Original configuration changed.")
    suite.require(suite.digest(panel_root / "registration.json") ==
                  registration["panel_registration_sha256"], "Original panel changed.")
    _, all_instances = shared.load_panel(panel_root)
    instances = select_instances(all_instances)
    cases = suite.read_rows(root / "cases.jsonl")
    suite.require([i["id"] for i in instances] == [i["instance_id"] for i in cases],
                  "Fixed episode order changed.")
    initial = [{"instance_id": i["id"], "request": suite.json_value(
                asdict(canonical.public_request(i, [])))} for i in instances]
    suite.require(initial == suite.read_rows(root / "initial-requests.jsonl"),
                  "Initial canonical requests changed.")
    suite.require(asdict(configuration()) == registration["configuration"], "Configuration changed.")
    return registration, instances


async def wait_terminal(path, *, deadline, clock=time.monotonic, sleep=asyncio.sleep,
                        reader=suite.read_json):
    """Read only the predecessor status record; never results, attempts, or scores."""
    while True:
        if clock() >= deadline:
            raise WaitDeadline("Predecessor did not become terminal within the 24-hour wait bound.")
        state = reader(path).get("status")
        if state in TERMINAL:
            return state
        if state not in WAITING:
            raise AdmissionError("Unexpected predecessor status; refusing admission.")
        await sleep(min(30.0, max(0.0, deadline - clock())))


def exception_record(exc):
    return {"status": "failure", "selected_option_id": None, "resolved_model": None,
            "usage": suite.safe_usage(getattr(exc, "usage", None)), **suite.failure(exc),
            "exception_type": type(exc).__name__[:80]}


async def calls(instances, config, client, stream, attempts, *, guard,
                deadline, clock=time.monotonic, on_attempt=lambda: None):
    for instance in instances:
        history = []
        while (request := canonical.public_request(instance, history)) is not None:
            if deadline - clock() < CALL_SECONDS:
                raise RunDeadline("Insufficient total-run budget for another full 600-second call.")
            guard()
            suite.require(len(attempts) < MAX_CALLS and len(history) < instance["query_budget"],
                          "Registered query budget exceeded.")
            diagnostics, started, cancelled = None, clock(), None
            try:
                response = await asyncio.wait_for(client.predict(request), CALL_SECONDS)
                diagnostics = shared.response_diagnostics(response.raw_output)
                record = {**suite.response_record(response, request, config), "exception_type": None}
            except BaseException as exc:
                if isinstance(exc, TimeoutError):
                    exc = ClientTimeoutError("Deadline-control request timeout")
                diagnostics = getattr(exc, "response_diagnostics", None) or diagnostics or (
                    shared.response_diagnostics(getattr(exc, "raw_output", None)))
                record = exception_record(exc)
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit, RunDeadline)):
                    record["fatal"] = True
                    cancelled = exc
                    if isinstance(exc, RunDeadline) or clock() >= deadline:
                        record["diagnostic_code"] = "run_deadline_exceeded"
            row = {"instance_id": instance["id"], "step": len(history),
                   "request": asdict(request), "response_diagnostics": diagnostics,
                   "latency_seconds": clock() - started, **record}
            suite.append_row(stream, row)
            stream.flush()
            os.fsync(stream.fileno())
            attempts.append(suite.json_value(row))
            on_attempt()
            if cancelled is not None:
                raise cancelled
            if record["status"] == "failure":
                if record["fatal"]:
                    raise AdmissionError("Fatal client failure; remaining episodes are unattempted.")
                break
            history.append(record["selected_option_id"])


def report(root, instances, attempts, config, state):
    rows = canonical.replay(instances, attempts, config, native_action=False)
    panel.invariance.write_rows(root / "scores.jsonl", rows)
    shared.write(root / "summary.json", suite.summarize(rows))
    shared.write(root / "diagnostics.json", {
        "planned_episodes": len(instances), "calls": len(attempts),
        "failures": dict(Counter(a["diagnostic_code"] for a in attempts if a["status"] == "failure")),
        "unattempted_episodes": sum(r["query_count"] == 0 for r in rows),
        "infeasible_episodes": sum(not r["feasible"] for r in rows),
        "length_cap_note": "8192-token cap unchanged; finish_length remains a failure, never retried."})
    shared.status(root, **state, calls=len(attempts), planned_episodes=len(instances))


async def controller(root, *, loader=verified_inputs, wait=wait_terminal,
                     lock_factory=suite.lane_locks, preflight=suite.shared.vllm_preflight,
                     service_guard=verify_service, checkpoint_guard=verify_checkpoint,
                     client_factory=shared.RecordingClient, clock=time.monotonic,
                     arm_deadline=lambda seconds, phase: None,
                     expired_deadline=lambda: None):
    registration, instances = loader(root)
    config = configuration()
    suite.require(not (root / "controller-start.json").exists(), "No controller resume or overwrite.")
    shared.write(root / "controller-start.json", {
        "pid": os.getpid(), "started_at_utc": shared.now(),
        "registration_sha256": suite.digest(root / "registration.json"),
        "wait_limit_seconds": WAIT_SECONDS, "run_limit_seconds": RUN_SECONDS})
    attempts, client, after_ok = [], None, False
    state = {"status": "waiting_predecessor", "stage": "waiting_predecessor", "model_calls_started": False}
    shared.status(root, **state)
    with (root / "attempts.jsonl").open("x") as stream:
        try:
            wait_deadline = clock() + WAIT_SECONDS
            arm_deadline(WAIT_SECONDS, "wait")
            async with asyncio.timeout(WAIT_SECONDS):
                terminal = await wait(Path(registration["predecessor_status"]),
                                      deadline=wait_deadline)
            shared.write(root / "predecessor-terminal.json",
                         {"status": terminal, "observed_at_utc": shared.now(),
                          "only_status_read": True})
            state.update(status="admission", stage="lock_admission")
            shared.status(root, **state)
            run_started = clock()
            deadline = run_started + RUN_SECONDS
            arm_deadline(RUN_SECONDS, "run")
            async with asyncio.timeout(RUN_SECONDS):
                # Nonblocking standard admission, once only; no lease polling or namespace changes.
                with lock_factory(("qwen9",)):
                    try:
                        state["stage"] = "source_verification_before"
                        loader(root)
                        state["stage"] = "service_verification_before"
                        service_guard(registration["service_identity"])
                        state["stage"] = "checkpoint_verification_before"
                        checkpoint_guard(suite.read_json(root / "checkpoint.json"))
                        service_guard(registration["service_identity"])
                        state["stage"] = "idle_endpoint_preflight"
                        await preflight([SimpleNamespace(model=config)])
                        if deadline - clock() < CALL_SECONDS:
                            raise RunDeadline("Startup consumed the total-run budget.")
                        shared.write(root / "admission.json", {
                            "at_utc": shared.now(), "resource": "qwen9", "idle_preflight": True,
                            "service_identity_verified": True, "checkpoint_verified": True,
                            "run_limit_seconds": RUN_SECONDS})
                        client = client_factory(**config.client_kwargs())
                        state["stage"] = "client_initialize"
                        await asyncio.wait_for(client.initialize(), min(30, deadline - clock()))
                        state.update(status="running", stage="inference", inference_admitted=True)
                        shared.status(root, **state)

                        def guard():
                            loader(root)
                            service_guard(registration["service_identity"])

                        await calls(instances, config, client, stream, attempts, guard=guard,
                                    deadline=deadline, clock=clock,
                                    on_attempt=lambda: shared.status(
                                        root, **{**state, "calls": len(attempts),
                                                 "model_calls_started": bool(attempts)}))
                        state["status"] = ("completed_with_failures" if any(
                            a["status"] == "failure" for a in attempts) else "completed")
                    except BaseException:
                        state["failure_stage"] = state["stage"]
                        raise
                    finally:
                        state["stage"] = "bounded_client_shutdown"
                        if client is not None:
                            await asyncio.wait_for(client.aclose(), 10)
                            client = None
                        state["stage"] = "source_service_checkpoint_verification_after"
                        loader(root)
                        service_guard(registration["service_identity"])
                        checkpoint_guard(suite.read_json(root / "checkpoint.json"))
                        after_ok = True
            state["run_wall_seconds"] = clock() - run_started
        except BaseException as exc:
            if isinstance(exc, (WaitDeadline, TimeoutError)) and state["status"] == "waiting_predecessor":
                code = "wait_deadline_exceeded"
            elif isinstance(exc, (RunDeadline, TimeoutError)):
                code = "run_deadline_exceeded"
            elif isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                code = "interrupted"
            else:
                code = "blocked_or_aborted"
            state.update(status=code, exception_type=type(exc).__name__[:80],
                         diagnostic_code=suite.failure(exc)["diagnostic_code"])
            state.setdefault("failure_stage", state["stage"])
        finally:
            arm_deadline(0, "done")
            expired = expired_deadline()
            if expired in ("wait", "run"):
                state.update(status=f"{expired}_deadline_exceeded", hard_deadline_observed=True)
            state["source_and_service_verified_after"] = after_ok
            state["model_calls_started"] = bool(attempts)
            report(root, instances, attempts, config, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dry-run", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = shared.local(args.root)
    if args.command == "prepare":
        result = prepare(root)
    elif args.command == "dry-run":
        registration, instances = verified_inputs(root)
        result = {"model_calls": 0, "episodes": len(instances),
                  "maximum_calls": sum(i["query_budget"] for i in instances),
                  "changed_configuration_fields": registration["changed_configuration_fields"],
                  "independent_validation": canonical.tasks.validate_instances(instances),
                  "predecessor_scores_read": False, "locks_acquired": False}
    else:
        async def run():
            loop, task = asyncio.get_running_loop(), asyncio.current_task()
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            deadline_state = {"expired": None}
            def arm(seconds, phase):
                def expired(signum, frame):
                    deadline_state["expired"] = phase
                    if phase == "wait":
                        raise WaitDeadline("Hard prestart waiting deadline reached.")
                    raise RunDeadline("Hard total-run deadline reached.")
                signal.signal(signal.SIGALRM, expired)
                signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                return await controller(root, arm_deadline=arm,
                                        expired_deadline=lambda: deadline_state["expired"])
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                loop.remove_signal_handler(signal.SIGTERM)
        result = asyncio.run(run())
    print(json.dumps(result, sort_keys=True))
    return int(args.command == "run" and not result["status"].startswith("completed"))


if __name__ == "__main__":
    raise SystemExit(main())
