#!/usr/bin/env python3
"""V2 restoration admission: pin birth identity, then confirm post-exec settings.

V1 sources and registrations remain unchanged. The runner/artifacts/lease protocol
are reused; only the restoration identity handshake and its coordinator binding
are versioned here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

from scripts import stronger_local_coordinator_v1 as v1

run, runtime, suite = v1.run, v1.runtime, v1.suite
resources, admission, LANES = v1.resources, v1.admission, v1.LANES
ROOT = REPO / "output/stronger-local-execution-20260926-v2"
IDENTITY_SECONDS = 30
IDENTITY_POLL_SECONDS = .1


def read_live_identity(pid):
    value = runtime.process(pid)
    value["restore_environment"] = v1.restoration_environment(value)
    return value


def initial_identity(pid):
    # Read birth identity independently of cmdline/cwd: those may still be in
    # transition even when Popen has returned.
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    anchor = {"pid": pid, "start_ticks": stat[19]}
    try:
        metadata = read_live_identity(pid)
        suite.require(metadata["pid"] == pid and metadata["start_ticks"] == anchor["start_ticks"],
                      "Restoration PID changed during initial capture.")
        observation = {"metadata": metadata}
    except (FileNotFoundError, ProcessLookupError, PermissionError) as exc:
        observation = {"metadata": None, "read_error_type": type(exc).__name__}
    return {"at_utc": runtime.now(), "anchor": anchor, **observation,
            "definitive_identity": False}


def wait_post_exec(anchor, expected, *, timeout=IDENTITY_SECONDS,
                   reader=read_live_identity, is_alive=runtime.alive,
                   clock=time.monotonic, sleep=time.sleep):
    started = clock()
    deadline = started + timeout
    count, transient_errors, mismatches, last_error = 0, 0, [], None
    fields = ("argv", "cwd", "cuda_environment", "restore_environment")
    while True:
        suite.require(is_alive(anchor), "Restoration process exited before post-exec confirmation.")
        count += 1
        try:
            current = reader(anchor["pid"])
            suite.require(current["pid"] == anchor["pid"]
                          and current["start_ticks"] == anchor["start_ticks"],
                          "Restoration PID/start-tick identity changed.")
            suite.require(current.get("state") != "Z",
                          "Restoration process exited before post-exec confirmation.")
            mismatches = [field for field in fields if current.get(field) != expected.get(field)]
            if not mismatches:
                return current, {
                    "at_utc": runtime.now(), "observations": count,
                    "transient_read_errors": transient_errors,
                    "elapsed_seconds": clock() - started,
                    "maximum_wait_seconds": timeout,
                    "matched_fields": list(fields), "same_pid_and_start_ticks": True,
                    "definitive_identity": True,
                }
        except (FileNotFoundError, ProcessLookupError, PermissionError) as exc:
            transient_errors += 1
            last_error = type(exc).__name__
        remaining = deadline - clock()
        suite.require(remaining > 0, "Restoration post-exec identity deadline exceeded; "
                      f"mismatched_fields={mismatches}; last_read_error={last_error}; "
                      f"observations={count}; transient_read_errors={transient_errors}.")
        sleep(min(IDENTITY_POLL_SECONDS, remaining))


class Lease(v1.Lease):
    def confirmed_restoration(self, lane, old, initial):
        identity, evidence = wait_post_exec(initial["anchor"], old)
        evidence["initial_capture_sha256"] = suite.digest(
            self.root / f"restore-{lane}-initial-process.json")
        evidence["confirmed_anchor"] = initial["anchor"]
        confirmation = self.root / f"restore-{lane}-post-exec.json"
        if confirmation.exists():
            previous = suite.read_json(confirmation)
            suite.require(previous["initial_capture_sha256"] == evidence["initial_capture_sha256"]
                          and previous["confirmed_anchor"] == initial["anchor"]
                          and previous["definitive_identity"] is True,
                          "Previous post-exec confirmation differs.")
        else:
            suite.write_json(confirmation, evidence)
        # Only a confirmed capture is eligible for the original v1 readiness
        # identity check. The potentially empty/pre-exec capture remains separate.
        suite.write_json(self.root / f"restore-{lane}-process.json", identity)
        return identity

    def restore(self):
        restoration, blocked = {}, {}
        for lane, (gpu, port, _) in LANES.items():
            old = self.services[lane]
            try:
                if runtime.alive(old):
                    suite.require(resources.same_identity(old, resources.service(lane)),
                                  "Original PID identity changed.")
                    restoration[lane] = old
                    continue
                previous = self.root / f"restore-{lane}-process.json"
                initial_path = self.root / f"restore-{lane}-initial-process.json"
                if previous.exists():
                    captured = suite.read_json(previous)
                    suite.require(runtime.alive(captured)
                                  and resources.same_identity(captured, resources.service(lane)),
                                  "Previous confirmed restoration is not a matching live service.")
                    restoration[lane] = captured
                    continue
                if initial_path.exists():
                    restoration[lane] = self.confirmed_restoration(
                        lane, old, suite.read_json(initial_path))
                    continue
                self.require_empty([self.targets[gpu]])
                resources.port_free(port)
                environment = {k: v for k, v in os.environ.items() if not v1.relevant_env(k)}
                environment.update(old["restore_environment"])
                intent = self.root / f"restore-{lane}-intent.json"
                suite.require(not intent.exists(), "Unresolved prior restoration launch intent.")
                suite.write_json(intent, {
                    "at_utc": runtime.now(), "argv": old["argv"], "cwd": old["cwd"],
                    "environment": old["restore_environment"],
                    "post_exec_identity_deadline_seconds": IDENTITY_SECONDS})
                with (self.root / f"restore-{lane}.log").open("xb") as log:
                    child = subprocess.Popen(
                        old["argv"], cwd=old["cwd"], env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                initial = initial_identity(child.pid)
                suite.write_json(initial_path, initial)
                restoration[lane] = self.confirmed_restoration(lane, old, initial)
            except BaseException as exc:
                blocked[lane] = {"error_type": type(exc).__name__, "message": str(exc)[:1000]}
        deadline, ready = time.monotonic() + v1.RESTORE_SECONDS, False
        while not blocked:
            try:
                suite.require(all(runtime.alive(identity) for identity in restoration.values()),
                              "Restored service exited before readiness.")
                v1.idle_originals()
                current = v1.originals()
                suite.require(all(resources.same_identity(restoration[lane], current[lane])
                                  for lane in LANES), "Restored original identity differs.")
                admission.four_gpu_mapping(current, runtime.gpu_inventory())
                restoration, ready = current, True
                break
            except Exception as exc:
                runtime.status(self.root, status="restoring", last_error=str(exc)[:500],
                               pid=os.getpid(), coordinator_version=2)
                if time.monotonic() >= deadline:
                    blocked["readiness"] = {"message": str(exc)[:1000]}
                    break
                time.sleep(v1.POLL_SECONDS)
        run.atomic_json(self.root / "restoration.json", {
            "at_utc": runtime.now(), "original_services_restored": ready,
            "services": restoration, "blocked": blocked, "coordinator_version": 2,
            "post_exec_identity_deadline_seconds": IDENTITY_SECONDS,
            "unknown_occupants_never_stopped": True})
        runtime.status(self.root, status="restored" if ready else "restoration_blocked",
                       original_services_restored=ready, blocked=blocked, pid=os.getpid(),
                       coordinator_version=2)
        suite.require(ready, f"Restoration incomplete; explicit blockers: {blocked}")


@contextmanager
def acquire(root, *, wait_seconds=v1.LEASE_WAIT_SECONDS):
    suite.require(not root.exists(), "Resource acquisition exists; no inference resume.")
    root.mkdir()
    deadline = time.monotonic() + wait_seconds
    while True:
        manager = suite.lane_locks(tuple(LANES))
        try:
            manager.__enter__()
            break
        except suite.shared.Refusal as exc:
            runtime.status(root, status="waiting_for_standard_leases", pid=os.getpid())
            if time.monotonic() >= deadline:
                raise RuntimeError("Standard lease wait deadline exceeded.") from exc
            time.sleep(v1.POLL_SECONDS)
    touched, lease = False, None
    try:
        suite.write_json(root / "standard-lock-ownership.json", v1.lock_ownership())
        services, targets = v1.snapshot_originals(root, "admission.json")
        for spec in run.SPECS.values():
            resources.port_free(spec["port"])
        suite.write_json(root / "coordinator-owner.json", runtime.process(os.getpid()))
        lease = Lease(root, services, targets)
        v1.idle_originals()
        suite.require(all(resources.same_identity(old, resources.service(lane))
                          for lane, old in services.items()), "Service changed at admission.")
        admission.four_gpu_mapping(services, runtime.gpu_inventory())
        touched = True
        suite.write_json(root / "shutdown-intent.json", {
            "at_utc": runtime.now(), "services": services, "target_gpu_uuids": targets})
        suite.require(runtime.stop_tree([p for s in services.values() for p in s["process_tree"]]),
                      "Original shutdown incomplete.")
        lease.require_empty()
        runtime.status(root, status="leased_originals_stopped", pid=os.getpid())
        yield lease
    finally:
        try:
            if touched:
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                for sig in handlers:
                    signal.signal(sig, signal.SIG_IGN)
                cleanup_error = None
                try:
                    try:
                        lease.cleanup()
                    except BaseException as exc:
                        cleanup_error = exc
                    lease.restore()
                    if cleanup_error is not None:
                        raise cleanup_error
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
        finally:
            manager.__exit__(None, None, None)


def source_hashes():
    return {**run.source_hashes(), **{
        name: suite.digest(REPO / name) for name in (
            "scripts/stronger_local_coordinator_v2.py",
            "tests/test_stronger_local_coordinator_v2.py")}}


def register(root, materials):
    before = source_hashes()
    base = v1.register(root, materials)
    version = root / "coordinator-v2"
    version.mkdir()
    with (version / "safety-tests.log").open("xb") as log:
        result = subprocess.run(
            [runtime.PYTHON, "-B", "-m", "unittest", "tests.test_stronger_local_coordinator_v2", "-v"],
            cwd=REPO, env=v1.child_environment(root), stdout=log, stderr=subprocess.STDOUT,
            timeout=180, check=False)
    suite.require(result.returncode == 0, "V2 identity-handshake tests failed.")
    suite.require(before == source_hashes(), "Source drift during v2 registration.")
    run.base.invariance.freeze_sources(version, before)
    registration = {
        "registered_at_utc": runtime.now(), "coordinator_version": 2,
        "source_sha256": before,
        "base_registration_sha256": suite.digest(root / "registration.json"),
        "safety_tests_sha256": suite.digest(version / "safety-tests.log"),
        "maximum_post_exec_identity_seconds_per_original": IDENTITY_SECONDS,
        "preserves_v1_sources_and_registrations": True,
        "model_calls": 0, "state": "prepared_not_launched",
    }
    suite.write_json(version / "registration.json", registration)
    suite.write_json(version / "registration.sha256", suite.digest(version / "registration.json"))
    runtime.status(root, status="prepared_not_launched", coordinator_version=2, model_calls=0)
    return {**base, "coordinator_v2": registration}


def verify_registration(root):
    base = v1.verify_registration(root)
    version = root / "coordinator-v2"
    value = suite.read_json(version / "registration.json")
    suite.require(suite.digest(version / "registration.json") ==
                  suite.read_json(version / "registration.sha256"), "V2 registration drift.")
    suite.require(value["source_sha256"] == source_hashes()
                  and value["base_registration_sha256"] == suite.digest(root / "registration.json")
                  and value["safety_tests_sha256"] == suite.digest(version / "safety-tests.log")
                  and value["maximum_post_exec_identity_seconds_per_original"] == IDENTITY_SECONDS,
                  "V2 source/input/test/configuration drift.")
    for name, digest in value["source_sha256"].items():
        suite.require(suite.digest(version / "source-snapshot" / name) == digest,
                      "V2 frozen source drift.")
    return {**base, "coordinator_v2": value}


def terminal_audits(root, result, failure):
    for lane in run.SPECS:
        lane_root = root / lane
        if not (lane_root / "registration.json").exists():
            continue
        try:
            if not (lane_root / "run").exists():
                run.preserve_unattempted(lane_root, "Coordinator stopped before this lane started.")
            verification = run.verify(lane_root)
            if not (lane_root / "verification.json").exists():
                command = [
                    runtime.PYTHON, "-B", str(REPO / "scripts/stronger_local_replay_v1.py"),
                    "--root", str(lane_root), "--bundle", str(run.base.BUNDLE)]
                with (root / f"{lane}-terminal-replay.log").open("xb") as log:
                    audit = subprocess.run(
                        command, cwd=REPO,
                        env=v1.child_environment(root, CUDA_VISIBLE_DEVICES=""),
                        stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False)
                suite.require(audit.returncode == 0, "Terminal independent replay failed.")
                suite.write_json(lane_root / "verification.json", {
                    "runner": verification, "independent_replay_sha256":
                        suite.digest(lane_root / "independent-replay.json"),
                    "construction_gaps_sha256": suite.digest(lane_root / "construction-gaps.csv")})
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
                   "coordinator_version": 2}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    children, result, failure = v1.Children(root), {}, None
    try:
        v1.snapshot_originals(root, "resources-before-downloads.json")
        public = [admission.validate_metadata(
            spec, admission.get_json(
                f"https://huggingface.co/api/models/{spec['model']}/revision/{spec['revision']}?blobs=true"),
            admission.get_json(f"https://huggingface.co/{spec['model']}/resolve/{spec['revision']}/config.json"),
            admission.get_json(f"https://huggingface.co/{spec['model']}/resolve/{spec['revision']}/model.safetensors.index.json"))
            for spec in run.SPECS.values()]
        import shutil
        storage = admission.storage_admission(shutil.disk_usage(REPO).free, public)
        existing = 0
        for lane, model in zip(run.SPECS, public, strict=True):
            snapshot = materials / lane / "snapshot"
            for name, shard in model["shards"].items():
                paths = (snapshot / name, snapshot / (name + ".partial"))
                existing += min(shard["bytes"], sum(
                    path.stat().st_size for path in paths if path.is_file() and not path.is_symlink()))
        storage["existing_snapshot_bytes"] = existing
        suite.require(storage["available_bytes"] + existing >=
                      storage["repository_bytes_upper_bound"] + admission.HEADROOM_BYTES,
                      "Insufficient exact-byte download headroom.")
        suite.write_json(root / "storage-before-downloads.json", storage)
        for lane in run.SPECS:
            verify_registration(root)
            runtime.status(root, status="downloading", lane=lane, pid=os.getpid(), model_calls=0)
            command = [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_runtime_v1.py"),
                       "download", "--lane", lane, "--materials", str(materials)]
            child = children.start(command, v1.child_environment(root, CUDA_VISIBLE_DEVICES=""),
                                   f"{lane}-download")
            children.wait(child, registration["maximum_download_seconds_per_model"])
            run.verify_artifacts(lane, materials)
            run.register_lane(root / lane, lane, materials)
        with acquire(root / "resources",
                     wait_seconds=registration["maximum_lease_wait_seconds"]) as lease:
            for lane, spec in run.SPECS.items():
                verify_registration(root)
                run.load(root / lane)
                checkpoint = run.verify_artifacts(lane, materials)
                expected = lease.targets[:spec["tensor_parallel_size"]]
                lease.require_empty()
                server = lease.start(
                    v1.server_command(lane, checkpoint["snapshot"]),
                    v1.child_environment(root, CUDA_VISIBLE_DEVICES=",".join(expected)),
                    f"{lane}-server")
                v1.ready(lease, server, lane, materials, expected,
                         timeout=registration["maximum_startup_seconds_per_model"])
                runtime.status(root, status="inference", lane=lane, pid=os.getpid(),
                               server_pid=server["identity"]["pid"], coordinator_version=2)
                worker = lease.start(
                    [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_runtime_v1.py"),
                     "run", "--root", str(root / lane)],
                    v1.child_environment(root, CUDA_VISIBLE_DEVICES=""), f"{lane}-runner")
                lease.wait(worker, registration["maximum_run_seconds_per_lane"] + 120,
                           heartbeat=lambda: lease.check_server(server, expected), allow_failure=True)
                verification = run.verify(root / lane)
                audit = lease.start(
                    [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_replay_v1.py"),
                     "--root", str(root / lane), "--bundle", str(run.base.BUNDLE)],
                    v1.child_environment(root, CUDA_VISIBLE_DEVICES=""), f"{lane}-replay")
                lease.wait(audit, 300)
                independent = suite.read_json(root / lane / "independent-replay.json")
                suite.require(independent["verified"] and independent["episodes"] == 960,
                              "Independent replay failed.")
                result[lane] = verification
                suite.write_json(root / lane / "verification.json", {
                    "runner": verification, "independent_replay_sha256":
                        suite.digest(root / lane / "independent-replay.json"),
                    "construction_gaps_sha256": suite.digest(root / lane / "construction-gaps.csv")})
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
            failure = {**(failure or {}), "download_cleanup_error": str(exc)[:1000]}
        failure = terminal_audits(root, result, failure)
        path = root / "resources/restoration.json"
        restoration = suite.read_json(path) if path.exists() else None
        terminal = {
            "at_utc": runtime.now(), "status": "failed" if failure else "completed",
            "coordinator_version": 2, "lanes": result, "failure": failure,
            "original_services_restored": restoration["original_services_restored"]
            if restoration else None,
            "resources_touched": (root / "resources/shutdown-intent.json").exists(),
            "restoration": restoration}
        suite.write_json(root / "terminal.json", terminal)
        runtime.status(root, **terminal)
    return terminal


def recover(root, *, allow_standard_lock_access=False):
    v1.require_standard_lock_access(allow_standard_lock_access)
    verify_registration(root)
    owner = suite.read_json(root / "start.json")["identity"]
    suite.require(not runtime.alive(owner), "Coordinator still alive; recovery refused.")
    resource_root = root / "resources"
    captured = suite.read_json(resource_root / "admission.json")
    lease = Lease(resource_root, captured["services"], captured["target_gpu_uuids"])
    with suite.lane_locks(tuple(LANES)):
        for path in sorted(resource_root.glob("*-process.json")):
            if path.name.startswith("restore-"):
                continue
            value = suite.read_json(path)
            identity = value["identity"]
            lease.entries.append({
                "child": None, "identity": identity,
                "label": path.name.removesuffix("-process.json"),
                "known": {(p["pid"], p["start_ticks"]): p for p in value["known_tree"]}})
        lease.cleanup()
        lease.restore()
    return suite.read_json(resource_root / "restoration.json")


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
        result = register(root, args.materials)
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
