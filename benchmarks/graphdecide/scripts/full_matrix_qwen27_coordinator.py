#!/usr/bin/env python3
"""Durable, restoring resource coordinator for the full no-thinking Qwen27 run."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import full_matrix_gpu_resources as resources
from scripts import full_matrix_qwen27 as qwen

suite, runtime = qwen.suite, qwen.runtime


def hashes():
    return {name: suite.digest(REPO / name) for name in
            ("scripts/full_matrix_qwen27_coordinator.py",
             "scripts/full_matrix_gpu_resources.py")}


def register(root):
    qwen.load(root)
    out = root / "coordinator"
    suite.require(not out.exists(), "Coordinator registration exists; no overwrite.")
    out.mkdir()
    source = hashes()
    qwen.invariance.freeze_sources(out, source)
    registration = {
        "at_utc": runtime.now(), "qwen_registration_sha256": suite.digest(root / "registration.json"),
        "source_sha256": source, "resources": list(resources.LANES),
        "maximum_resource_wait_seconds": 24 * 60 * 60,
        "maximum_server_startup_seconds": 1200, "maximum_run_seconds": qwen.MAX_SECONDS,
        "run_only": "qwen27_grammar", "thinking": False,
        "restore_originals": True, "retries": False, "new_model_calls": 0,
        "server_command": runtime.server_command(runtime.ARTIFACTS / "snapshot"),
    }
    suite.write_json(out / "registration.json", registration)
    suite.write_json(out / "registration.sha256", suite.digest(out / "registration.json"))
    runtime.status(out, status="registered")
    return registration


def verify(root):
    out = root / "coordinator"
    registration = suite.read_json(out / "registration.json")
    suite.require(suite.digest(out / "registration.json") ==
                  suite.read_json(out / "registration.sha256"), "Coordinator registration drift.")
    suite.require(registration["source_sha256"] == hashes()
                  and registration["qwen_registration_sha256"] ==
                  suite.digest(root / "registration.json"), "Coordinator source/input drift.")
    for name, digest in registration["source_sha256"].items():
        suite.require(suite.digest(out / "source-snapshot" / name) == digest,
                      "Coordinator snapshot drift.")
    qwen.load(root)
    return registration


async def run_cancellable(root):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    previous = signal.getsignal(signal.SIGTERM)
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await qwen.run(root)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        signal.signal(signal.SIGTERM, previous)


def launch(root):
    registration = verify(root)
    out = root / "coordinator"
    suite.require(not (out / "start.json").exists(), "Coordinator cannot resume.")
    suite.write_json(out / "start.json", {
        "at_utc": runtime.now(), "identity": runtime.process(os.getpid())})
    runtime.status(out, status="verifying_checkpoint", pid=os.getpid())
    try:
        checkpoint = runtime.verify_artifacts(runtime.ARTIFACTS)
        with resources.acquire(root / "resources",
                               wait_seconds=registration["maximum_resource_wait_seconds"]) as lease:
            verify(root)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(lease.targets)}
            server = lease.start(runtime.server_command(checkpoint["snapshot"]), env, "qwen27-server")
            runtime.wait_ready(server, out,
                               timeout_seconds=registration["maximum_server_startup_seconds"])
            owned = {row["pid"] for row in runtime.descendants(server.pid)}
            occupants = [(int(pid), uuid) for pid, uuid, _ in runtime.gpu_inventory()["processes"]
                         if uuid in lease.targets]
            suite.require({uuid for _, uuid in occupants} == set(lease.targets)
                          and all(pid in owned for pid, _ in occupants),
                          "27B server GPU mapping differs from leased devices.")
            suite.write_json(out / "model-ready.json", {
                "at_utc": runtime.now(), "server": runtime.process(server.pid),
                "target_gpu_uuids": lease.targets, "checkpoint_revision": runtime.REVISION,
                "preflight_verified": True})
            runtime.status(out, status="inference_running", pid=os.getpid(), server_pid=server.pid)
            result = asyncio.run(run_cancellable(root))
            verification = qwen.verify(root)
            suite.write_json(out / "run-verification.json", verification)
            suite.require(result["status"] in ("completed", "completed_with_failures"),
                          "Qwen matrix stopped early; all planned scores preserved.")
            lease.stop(server)
            runtime.status(out, status="restoring_original_services", run_status=result["status"])
        runtime.status(out, status="completed", run_status=result["status"],
                       calls=result["calls"], original_services_restored=True)
        return result
    except BaseException as exc:
        runtime.status(out, status="failed", error_type=type(exc).__name__,
                       message=str(exc)[:1000])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "launch", "verify"))
    parser.add_argument("--root", type=Path, default=qwen.ROOT)
    args = parser.parse_args()
    if args.command == "launch":
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Coordinator signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        result = launch(args.root)
    elif args.command == "register":
        result = register(args.root)
    else:
        result = verify(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
