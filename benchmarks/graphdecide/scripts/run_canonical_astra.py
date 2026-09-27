#!/usr/bin/env python3
"""New canonical Astra lane, reusing the unchanged GPT54 producer and SDK adapter."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import importlib.util
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
from scripts import run_paper_gpt_anchor as reference

# A separate module namespace keeps the pinned producer and other importers untouched.
spec = importlib.util.spec_from_file_location(
    "_canonical_astra_producer", REPO / "scripts/run_paper_gpt_anchor.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
suite, release, evaluation = core.suite, core.release, core.evaluation
LOCK_MODELS = ("gpt6astra",)
TOTAL_SECONDS = 48 * 60 * 60
CLEANUP_SECONDS = 240
SEED = 20261009
REFERENCE = "output/paper-gpt54-canonical-anchor-20260924-v3"


def configuration():
    return suite.model_configs(LOCK_MODELS, seed=SEED)["gpt6astra"]


def sources():
    return {**reference.sources(),
            "scripts/run_canonical_astra.py": suite.digest(Path(__file__).resolve())}


core.LOCK_MODELS = LOCK_MODELS
core.configuration = configuration
core.sources = sources
core.ARTIFACTS = (*core.ARTIFACTS, "protocol-check.json", "artifact-manifest.json")


def protocol_check(instances, reference_root):
    """Compare every initial public request, not just IDs, against the sealed anchor."""
    reference.verify(reference_root)
    prior = suite.read_json(reference_root / "manifest.json")
    initial = {row["instance_id"]: row["request"]
               for row in suite.read_rows(reference_root / "requests.jsonl") if row["step"] == 0}
    actual = {i["id"]: suite.json_value(asdict(evaluation.public_request(i, [])))
              for i in instances}
    suite.require(len(actual) == len(instances) == 960 and actual == initial,
                  "Full initial public request equality failed.")
    config = asdict(configuration())
    old_config = dict(prior["model"])
    old_config["model"] = config["model"]
    suite.require(config == old_config, "Astra differs from reference beyond model identity.")
    for name, digest in reference.sources().items():
        suite.require(prior["code_sha256"].get(name) == digest,
                      "Canonical reference dependency changed: " + name)
    return {"status": "verified", "initial_requests": len(actual),
            "equal_requests": len(actual), "different_requests": 0,
            "reference_root": str(reference_root),
            "reference_manifest_sha256": suite.digest(reference_root / "manifest.json"),
            "reference_seal_sha256": suite.digest(reference_root / "seal.json"),
            "comparison": "full serialized DecisionRequest equality at step zero",
            "configuration_difference": ["model"],
            "canonical_dependencies_identical": True}


def register(bundle, output, *, test_only=False, split=core.SPLIT, reference_root=None):
    bundle, output = bundle.resolve(), output.resolve()
    suite.require(not output.exists(), "Output exists; overwrite/resume is forbidden.")
    checked = release.verify(bundle)
    manifest = release.manifest_at(bundle)
    instances = release.labeled_instances(bundle, split, manifest)
    budget = sum(i["query_budget"] for i in instances)
    suite.require(test_only or (suite.digest(bundle / "manifest.json") == core.RELEASE_SHA256
                  and split == core.SPLIT and len(instances) == 960 and budget == 1920),
                  "Production requires pinned canonical challenge960/1920 release.")
    suite.require("explicit_actions" in manifest["protocols"], "Canonical actions unavailable.")
    config = configuration()
    suite.require(config.model == "gpt-6-astra" and config.max_tokens == 4096
                  and config.timeout_seconds == 180.0 and config.seed is None
                  and config.think is None and config.reasoning_effort is None
                  and config.output_format == "answer_only", "Unexpected Astra configuration.")
    check = ({"status": "test_only", "initial_requests": len(instances)} if test_only else
             protocol_check(instances, (reference_root or REPO / REFERENCE).resolve()))
    hashes = sources()
    output.mkdir(parents=True)
    for name, expected in hashes.items():
        target = output / suite.SOURCE_SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((REPO / name).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        suite.require(suite.digest(target) == expected, "Source changed during freezing.")
        target.chmod(0o444)
    suite.require(hashes == sources(), "Source changed during registration.")
    suite.write_json(output / "config.json", asdict(config))
    suite.write_json(output / "protocol-check.json", check)
    suite.write_json(output / "artifact-manifest.json", {
        "schema_version": 1, "artifacts": list(core.ARTIFACTS),
        "raw_response_bodies": False, "reasoning_content": False,
        "attempt_limit": budget, "scheduled_denominator": len(instances),
        "terminal_seal": "seal.json", "offline_verification": "verify",
        "deadline": "48 hours from dispatcher start, including resource wait and cleanup",
        "cleanup_reserve_seconds": CLEANUP_SECONDS,
    })
    registration = {
        "schema_version": 1, "registered_at_utc": core.now(), "protocol": "explicit_actions",
        "profile": "gpt6astra_default_reasoning", "split": split, "bundle": str(bundle),
        "repository": str(REPO), "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "verified_release_instances": checked["instances"],
        "planned_instances": len(instances), "maximum_calls": budget,
        "registration_seed": SEED, "provider_sampling_seed_supported": False,
        "maximum_wall_seconds": TOTAL_SECONDS,
        "model": asdict(config), "config_sha256": suite.digest(output / "config.json"),
        "source_snapshot": suite.SOURCE_SNAPSHOT, "code_sha256": hashes,
        "environment": suite.environment_versions(), "test_only": test_only,
        "injected_client": test_only, "resource_models": list(LOCK_MODELS),
        "shared_lock_directory": os.environ.get("TMPDIR") or "/tmp",
        "protocol_check_sha256": suite.digest(output / "protocol-check.json"),
        "artifact_manifest_sha256": suite.digest(output / "artifact-manifest.json"),
        "policy": {"retry": False, "repair": False, "resume": False, "fallback": False,
                   "tools": False, "cross_question_sessions": False,
                   "private_labels_sent": False, "own_history": True,
                   "sdk_internal_retries": "adapter rejects observed retry/multiple usage events",
                   "shared_cli_limit": "standard gpt6astra lock for entire SDK runtime lifetime"},
    }
    suite.write_json(output / "manifest.json", registration)
    suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
    for name in ("config.json", "manifest.json", "manifest.sha256",
                 "protocol-check.json", "artifact-manifest.json"):
        (output / name).chmod(0o444)
    suite.set_status(output, {"status": "registered", "calls": 0, "updated_at_utc": core.now()})
    return registration


original_load = core.load


def load(output, *, check_execution_sources=False):
    manifest, instances, config = original_load(output, check_execution_sources=check_execution_sources)
    for name, key in (("protocol-check.json", "protocol_check_sha256"),
                      ("artifact-manifest.json", "artifact_manifest_sha256")):
        suite.require(suite.digest(output / name) == manifest[key], "Registration artifact changed.")
    suite.require(manifest["maximum_wall_seconds"] == TOTAL_SECONDS, "Deadline changed.")
    return manifest, instances, config


core.load = load
execute, verify = core.execute, core.verify


def process_identity(pid):
    return {"pid": pid, "startticks": int(
        Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])}


async def worker(output):
    launch = suite.read_json(output / "dispatch-owner.json")
    remaining = max(0, launch["deadline_epoch"] - CLEANUP_SECONDS - time.time())
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    timer = loop.call_later(remaining, task.cancel)
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        return await execute(output)
    finally:
        timer.cancel()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def finalize_unstarted(output, diagnostic):
    manifest, instances, config = load(output)
    suite.require(not (output / "requests.jsonl").exists(), "Cannot finalize a started lane as unstarted.")
    for name in ("requests.jsonl", "attempts.jsonl", "events.jsonl"):
        (output / name).touch(exist_ok=False)
    rows = core.replay(instances, [], config)
    with (output / "episodes.jsonl").open("x") as stream:
        for row in rows:
            suite.append_row(stream, row)
    suite.write_json(output / "summary.json", core.summary_for(rows, []))
    suite.set_status(output, {"status": "aborted", "calls": 0, "diagnostic_code": diagnostic,
                              "finished_at_utc": core.now()})
    suite.write_json(output / "seal.json", {
        "schema_version": 1, "sealed_at_utc": core.now(),
        "sha256": {name: suite.digest(output / name) for name in (
            *core.ARTIFACTS, "manifest.json", "manifest.sha256", "config.json")}})
    suite.write_json(output / "verification.json", verify(output))


def dispatch(output, *, allow_system_shared_locks=False):
    manifest, _, _ = load(output, check_execution_sources=True)
    suite.require(not manifest["test_only"], "Cannot dispatch fake registration.")
    suite.require(not (output / "dispatch-owner.json").exists(), "No redispatch/resume.")
    suite.require((os.environ.get("TMPDIR") or "/tmp") == manifest["shared_lock_directory"],
                  "Shared lock namespace changed.")
    with (output / "dispatcher.lock").open("x") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        owner = {**process_identity(os.getpid()), "started_at_utc": core.now(),
                 "deadline_epoch": started + TOTAL_SECONDS,
                 "allow_system_shared_locks": allow_system_shared_locks}
        suite.write_json(output / "dispatch-owner.json", owner)
        stopped = False

        def stop_wait(*_):
            nonlocal stopped
            stopped = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop_wait)
        while not stopped and time.time() < owner["deadline_epoch"] - CLEANUP_SECONDS:
            waiting = {"status": "waiting", "calls": 0, **owner,
                       "updated_at_utc": core.now(), "resource_models": list(LOCK_MODELS)}
            lock_root = Path(os.path.abspath(manifest["shared_lock_directory"]))
            if not allow_system_shared_locks and not lock_root.is_relative_to(
                    Path(manifest["repository"])):
                waiting["waiting_reason"] = "shared_lock_directory_outside_project_policy"
                suite.set_status(output, waiting)
                time.sleep(5)
                continue
            with ExitStack() as stack:
                try:
                    stack.enter_context(suite.lane_locks(LOCK_MODELS))
                except suite.Refusal:
                    waiting["waiting_reason"] = "standard_gpt6astra_lock_busy"
                    suite.set_status(output, waiting)
                    time.sleep(5)
                    continue
                runtime = output / "sdk-runtime"
                runtime.mkdir(exist_ok=False)
                environment = dict(os.environ, TMPDIR=str(runtime), TMP=str(runtime),
                                   TEMP=str(runtime), PYTHONDONTWRITEBYTECODE="1")
                command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                           "_worker", "--output", str(output)]
                with (output / "worker.log").open("x") as log:
                    process = subprocess.Popen(command, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    suite.write_json(output / "worker-process.json", {
                        **process_identity(process.pid), "dispatcher_pid": os.getpid(),
                        "started_at_utc": core.now(), "command": command})
                    for sig in (signal.SIGTERM, signal.SIGINT):
                        signal.signal(sig, lambda number, _: process.send_signal(number))
                    try:
                        return process.wait(timeout=max(0, owner["deadline_epoch"] - time.time()))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        suite.write_json(output / "hard-stop.json", {
                            "at_utc": core.now(), "reason": "deadline_cleanup_grace_exceeded",
                            "offline_finalization_required": True})
                        offline_summary(output)
                        return 1
        finalize_unstarted(output, "interrupted" if stopped else "wall_budget_exhausted")
        return 1


def offline_summary(output):
    """Preserve hard-stop raw ledgers and materialize all planned denominators offline."""
    manifest, instances, config = load(output)
    for name in ("worker-process.json",):
        if (output / name).exists():
            identity = suite.read_json(output / name)
            try:
                alive = process_identity(identity["pid"])["startticks"] == identity["startticks"]
            except (FileNotFoundError, ProcessLookupError):
                alive = False
            suite.require(not alive, "Offline recovery refuses a live worker.")
    attempts = suite.read_rows(output / "attempts.jsonl") if (
        output / "attempts.jsonl").exists() else []
    requests = suite.read_rows(output / "requests.jsonl") if (
        output / "requests.jsonl").exists() else []
    completed_requests = [{k: row[k] for k in ("instance_id", "step", "request")}
                          for row in attempts]
    suite.require(requests[:len(attempts)] == completed_requests
                  and len(requests) - len(attempts) in (0, 1),
                  "Invalid interrupted ledger prefix.")
    suite.require(len(requests) <= manifest["maximum_calls"], "Request budget exceeded.")
    rows = core.replay(instances, attempts, config)
    target = output / "offline-finalized"
    target.mkdir(exist_ok=False)
    with (target / "episodes.jsonl").open("x") as stream:
        for row in rows:
            suite.append_row(stream, row)
    suite.write_json(target / "summary.json", {
        **core.summary_for(rows, attempts), "status": "interrupted_unsealed",
        "dispatched_requests": len(requests),
        "unresolved_inflight_requests": len(requests) - len(attempts),
        "interpretation": "incomplete requests are not scored as successful; all scheduled "
                          "instances retained; no reconstructed response or replayed inference",
    })
    suite.write_json(target / "provenance.json", {
        "created_at_utc": core.now(), "no_inference": True, "raw_ledgers_unchanged": True,
        "source_sha256": {name: suite.digest(output / name) for name in (
            "manifest.json", "requests.jsonl", "attempts.jsonl") if (output / name).exists()},
        "artifact_sha256": {name: suite.digest(target / name)
                            for name in ("episodes.jsonl", "summary.json")},
    })
    return suite.read_json(target / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "dispatch", "_worker", "verify",
                                           "offline-summary"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--allow-system-shared-locks", action="store_true",
                        help="Parent-only authorization for the unchanged standard lock namespace.")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "register":
        suite.require(args.bundle is not None, "--bundle is required.")
        result = register(args.bundle, output, reference_root=args.reference)
    elif args.command == "dispatch":
        return dispatch(output, allow_system_shared_locks=args.allow_system_shared_locks)
    elif args.command == "_worker":
        result = asyncio.run(worker(output))
    elif args.command == "offline-summary":
        result = offline_summary(output)
    else:
        result = verify(output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
