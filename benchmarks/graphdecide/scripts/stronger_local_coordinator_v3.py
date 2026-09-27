#!/usr/bin/env python3
"""V3: short real-tested IPC paths, strict standalone replay, verified weight reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import stronger_local_coordinator_v2 as v2

v1, run, runtime, suite = v2.v1, v2.run, v2.runtime, v2.suite
ROOT = REPO / "output/stronger-local-execution-20260926-v3"


def ipc_directory(root):
    suffix = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:8]
    return REPO / "output" / f"ipc-v3-{suffix}"


def validate_ipc_length(directory, limit=107):
    candidate = directory / "00000000-0000-0000-0000-000000000000"
    length = len(os.fsencode(candidate))
    suite.require(length < limit, f"IPC socket path needs {length} bytes; limit is {limit}.")
    return length


def prepare_ipc_directory(root):
    directory = ipc_directory(root)
    suite.require(not directory.is_symlink()
                  and directory.resolve().is_relative_to(REPO / "output"),
                  "IPC directory escapes the project.")
    validate_ipc_length(directory)
    owner = {"execution_root": str(root.resolve()), "uid": os.getuid(), "version": 3}
    if directory.exists():
        suite.require(suite.read_json(directory / "owner.json") == owner,
                      "Short IPC directory belongs to another execution.")
    else:
        directory.mkdir(mode=0o700)
        suite.write_json(directory / "owner.json", owner)
    return directory


def child_environment(root, **updates):
    environment = v1.child_environment(root, **updates)
    environment["TMPDIR"] = str(prepare_ipc_directory(root))
    return environment


def ipc_probe(root):
    import zmq
    before = {key: os.environ.get(key) for key in ("TMPDIR", "TEMP", "TMP")}
    directory = prepare_ipc_directory(root)
    limit = getattr(zmq, "IPC_PATH_MAX_LEN", 107)
    validate_ipc_length(directory, limit)
    endpoint = directory / str(uuid.uuid4())
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    try:
        socket.bind("ipc://" + str(endpoint))
        result = {
            "at_utc": runtime.now(), "bound": True, "path": str(endpoint),
            "path_bytes": len(os.fsencode(endpoint)), "ipc_limit": limit,
            "scratch_directory": str(directory), "gpu_calls": 0, "model_calls": 0}
    finally:
        socket.close(linger=0)
        context.term()
        # Only this freshly created test endpoint is removed, never service IPC.
        endpoint.unlink(missing_ok=True)
    suite.require(before == {key: os.environ.get(key) for key in before},
                  "Parent temporary-directory environment changed.")
    result["parent_tmp_environment_unchanged"] = True
    return result


def reuse_storage(free_bytes, verified_artifacts, headroom=v1.admission.HEADROOM_BYTES):
    suite.require(set(verified_artifacts) == set(run.SPECS),
                  "Both locally verified pinned artifacts are required; no implicit download.")
    retained = sum(sum(entry["bytes"] for entry in artifact["files"].values())
                   for artifact in verified_artifacts.values())
    result = {
        "available_bytes": free_bytes, "verified_existing_artifact_bytes": retained,
        "new_checkpoint_bytes_required": 0, "reserved_headroom_bytes": headroom,
        "required_available_bytes": headroom, "remaining_after_new_checkpoints_bytes": free_bytes,
        "sufficient": free_bytes >= headroom,
        "policy": "Existing payload hashes verified before credit; no redownload or second cache."}
    suite.require(result["sufficient"], "Insufficient remaining disk headroom after verified reuse.")
    return result


def verified_reuse(materials):
    # Full local LFS/blob/payload verification, not filename/size-only credit.
    artifacts = {lane: run.verify_artifacts(lane, materials) for lane in run.SPECS}
    storage = reuse_storage(shutil.disk_usage(REPO).free, artifacts)
    evidence = {
        lane: {"model": value["model"], "revision": value["revision"],
               "artifact_manifest_sha256": suite.digest(materials / lane / "artifacts.json"),
               "files": len(value["files"]), "payload_hashes_verified": True}
        for lane, value in artifacts.items()}
    return artifacts, {"at_utc": runtime.now(), "artifacts": evidence, "storage": storage,
                       "network_downloads": 0, "model_calls": 0}


def source_hashes():
    return {**v2.source_hashes(), **{
        name: suite.digest(REPO / name) for name in (
            "scripts/stronger_local_coordinator_v3.py",
            "scripts/stronger_local_replay_v3.py",
            "tests/test_stronger_local_coordinator_v3.py")}}


def register(root, materials):
    before = source_hashes()
    base = v2.register(root, materials)
    version = root / "coordinator-v3"
    version.mkdir()
    with (version / "safety-tests.log").open("xb") as log:
        tests = subprocess.run(
            [runtime.PYTHON, "-B", "-m", "unittest", "tests.test_stronger_local_coordinator_v3", "-v"],
            cwd=REPO, env=child_environment(root), stdout=log, stderr=subprocess.STDOUT,
            timeout=300, check=False)
    suite.require(tests.returncode == 0, "V3 standalone replay/IPC/storage tests failed.")
    suite.write_json(version / "ipc-bind-preflight.json", ipc_probe(root))
    _, evidence = verified_reuse(materials)
    suite.write_json(version / "verified-artifact-reuse.json", evidence)
    suite.require(before == source_hashes(), "Source drift during v3 preparation.")
    run.base.invariance.freeze_sources(version, before)
    value = {
        "registered_at_utc": runtime.now(), "coordinator_version": 3, "source_sha256": before,
        "v2_registration_sha256": suite.digest(root / "coordinator-v2/registration.json"),
        "safety_tests_sha256": suite.digest(version / "safety-tests.log"),
        "ipc_probe_sha256": suite.digest(version / "ipc-bind-preflight.json"),
        "artifact_reuse_sha256": suite.digest(version / "verified-artifact-reuse.json"),
        "short_child_ipc_directory": str(ipc_directory(root)),
        "reuse_verified_weights_only": True, "network_downloads": 0,
        "model_calls": 0, "state": "prepared_not_launched"}
    suite.write_json(version / "registration.json", value)
    suite.write_json(version / "registration.sha256", suite.digest(version / "registration.json"))
    runtime.status(root, status="prepared_not_launched", coordinator_version=3, model_calls=0)
    return {**base, "coordinator_v3": value}


def verify_registration(root):
    base = v2.verify_registration(root)
    version = root / "coordinator-v3"
    value = suite.read_json(version / "registration.json")
    suite.require(suite.digest(version / "registration.json") ==
                  suite.read_json(version / "registration.sha256"), "V3 registration drift.")
    suite.require(value["source_sha256"] == source_hashes()
                  and value["v2_registration_sha256"] ==
                  suite.digest(root / "coordinator-v2/registration.json"),
                  "V3 source/input drift.")
    for key, filename in (("safety_tests_sha256", "safety-tests.log"),
                          ("ipc_probe_sha256", "ipc-bind-preflight.json"),
                          ("artifact_reuse_sha256", "verified-artifact-reuse.json")):
        suite.require(value[key] == suite.digest(version / filename), "V3 admission evidence drift.")
    for name, digest in value["source_sha256"].items():
        suite.require(suite.digest(version / "source-snapshot" / name) == digest,
                      "V3 source snapshot drift.")
    suite.require(value["short_child_ipc_directory"] == str(ipc_directory(root))
                  and value["reuse_verified_weights_only"] is True, "V3 configuration drift.")
    return {**base, "coordinator_v3": value}


def replay_command(lane_root, output):
    return [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_replay_v3.py"),
            "--root", str(lane_root), "--bundle", str(run.base.BUNDLE), "--output", str(output)]


def audit_result(lane_root, output, verification):
    for name, digest in suite.read_json(output / "seal.json").items():
        suite.require(suite.digest(output / name) == digest, "Independent v3 report seal differs.")
    independent = suite.read_json(output / "report.json")
    suite.require(independent["verified"] and independent["episodes"] == 960
                  and independent["provider_modules_imported"] == []
                  and independent["input_registration_sha256"] ==
                  suite.digest(lane_root / "registration.json")
                  and independent["input_run_seal_sha256"] == suite.digest(lane_root / "run/seal.json"),
                  "Independent v3 replay identity differs.")
    result = {
        "runner": verification, "independent_replay_path": str(output / "report.json"),
        "independent_replay_sha256": suite.digest(output / "report.json"),
        "construction_gaps_path": str(output / "construction-gaps.csv"),
        "construction_gaps_sha256": suite.digest(output / "construction-gaps.csv")}
    if not (lane_root / "verification.json").exists():
        suite.write_json(lane_root / "verification.json", result)
    return result


def terminal_audits(root, result, failure):
    for lane in run.SPECS:
        lane_root = root / lane
        if not (lane_root / "registration.json").exists():
            continue
        try:
            if not (lane_root / "run").exists():
                run.preserve_unattempted(lane_root, "Coordinator stopped before this lane started.")
            verification = run.verify(lane_root)
            output = lane_root / "independent-v3"
            if not (output / "seal.json").exists():
                if output.exists():
                    output = lane_root / "terminal-independent-v3"
                with (root / f"{lane}-terminal-replay-v3.log").open("xb") as log:
                    audit = subprocess.run(
                        replay_command(lane_root, output), cwd=REPO,
                        env=child_environment(root, CUDA_VISIBLE_DEVICES=""),
                        stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
                suite.require(audit.returncode == 0, "Terminal independent v3 replay failed.")
            audit_result(lane_root, output, verification)
            result[lane] = verification
        except BaseException as exc:
            failure = {**(failure or {}), f"{lane}_terminal_audit_error": str(exc)[:1000]}
    return failure


def launch(root, *, allow_standard_lock_access=False):
    v1.require_standard_lock_access(allow_standard_lock_access)
    registration = verify_registration(root)
    materials = Path(registration["materials"])
    with (root / "start.json").open("x") as stream:
        json.dump({"at_utc": runtime.now(), "identity": runtime.process(os.getpid()),
                   "coordinator_version": 3}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    children, result, failure = v1.Children(root), {}, None
    try:
        v1.snapshot_originals(root, "resources-before-artifact-reuse.json")
        # Exercise an actual AF_UNIX/ZMQ bind before any original shutdown.
        suite.write_json(root / "ipc-bind-launch-preflight.json", ipc_probe(root))
        artifacts = {}
        for lane in run.SPECS:
            verify_registration(root)
            runtime.status(root, status="verifying_existing_artifacts", lane=lane,
                           pid=os.getpid(), network_downloads=0, model_calls=0)
            child = children.start(
                [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_runtime_v1.py"),
                 "verify-artifacts", "--lane", lane, "--materials", str(materials)],
                child_environment(root, CUDA_VISIBLE_DEVICES=""), f"{lane}-verify-artifacts")
            children.wait(child, registration["maximum_download_seconds_per_model"])
            artifacts[lane] = run.verify_artifacts(lane, materials, payload=False)
        storage = reuse_storage(shutil.disk_usage(REPO).free, artifacts)
        suite.write_json(root / "storage-before-lease.json", storage)
        for lane in run.SPECS:
            run.register_lane(root / lane, lane, materials)
        with v2.acquire(root / "resources",
                        wait_seconds=registration["maximum_lease_wait_seconds"]) as lease:
            for lane, spec in run.SPECS.items():
                verify_registration(root)
                run.load(root / lane)
                checkpoint = run.verify_artifacts(lane, materials)
                expected = lease.targets[:spec["tensor_parallel_size"]]
                lease.require_empty()
                server = lease.start(
                    v1.server_command(lane, checkpoint["snapshot"]),
                    child_environment(root, CUDA_VISIBLE_DEVICES=",".join(expected)),
                    f"{lane}-server")
                v1.ready(lease, server, lane, materials, expected,
                         timeout=registration["maximum_startup_seconds_per_model"])
                runtime.status(root, status="inference", lane=lane, pid=os.getpid(),
                               server_pid=server["identity"]["pid"], coordinator_version=3)
                worker = lease.start(
                    [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_runtime_v1.py"),
                     "run", "--root", str(root / lane)],
                    child_environment(root, CUDA_VISIBLE_DEVICES=""), f"{lane}-runner")
                lease.wait(worker, registration["maximum_run_seconds_per_lane"] + 120,
                           heartbeat=lambda: lease.check_server(server, expected), allow_failure=True)
                verification = run.verify(root / lane)
                output = root / lane / "independent-v3"
                audit = lease.start(
                    replay_command(root / lane, output),
                    child_environment(root, CUDA_VISIBLE_DEVICES=""), f"{lane}-replay-v3")
                lease.wait(audit, 300)
                audit_result(root / lane, output, verification)
                result[lane] = verification
                lease.stop(server)
                lease.require_empty()
                suite.require(verification["status"].startswith("completed"),
                              "Lane stopped early; preserved denominator is not a completed lane.")
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        try:
            children.cleanup()
        except BaseException as exc:
            failure = {**(failure or {}), "artifact_verifier_cleanup_error": str(exc)[:1000]}
        failure = terminal_audits(root, result, failure)
        path = root / "resources/restoration.json"
        restoration = suite.read_json(path) if path.exists() else None
        terminal = {
            "at_utc": runtime.now(), "status": "failed" if failure else "completed",
            "coordinator_version": 3, "lanes": result, "failure": failure,
            "original_services_restored": restoration["original_services_restored"]
            if restoration else None,
            "resources_touched": (root / "resources/shutdown-intent.json").exists(),
            "restoration": restoration, "network_downloads": 0}
        suite.write_json(root / "terminal.json", terminal)
        runtime.status(root, **terminal)
    return terminal


def recover(root, *, allow_standard_lock_access=False):
    verify_registration(root)
    # V2 recovery already binds the corrected post-exec lease; it does not
    # launch inference servers, so no short child scratch override is needed.
    return v2.recover(root, allow_standard_lock_access=allow_standard_lock_access)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "verify", "launch", "recover"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--materials", type=Path, default=run.MATERIALS)
    parser.add_argument("--allow-standard-lock-access", action="store_true")
    args = parser.parse_args()
    root = runtime.local(args.root)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Coordinator received signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    if args.command == "register":
        result = register(root, runtime.local(args.materials))
    elif args.command == "verify":
        result = verify_registration(root)
    elif args.command == "launch":
        result = launch(root, allow_standard_lock_access=args.allow_standard_lock_access)
    else:
        result = recover(root, allow_standard_lock_access=args.allow_standard_lock_access)
    print(json.dumps(result, sort_keys=True))
    return 2 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
