#!/usr/bin/env python3
"""Scoped GPU leases with identity-checked shutdown and original-service restoration."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from types import SimpleNamespace

from scripts import larger_local_pilot_runtime as runtime

suite = runtime.suite
LANES = {"qwen08": (0, 8000, "Qwen3.5-0.8B"),
         "qwen2": (1, 8001, "Qwen3.5-2B")}
PROTECTED = {"qwen4": (2, 8002, "Qwen3.5-4B"),
             "qwen9": (3, 8003, "Qwen3.5-9B")}


def service(lane):
    gpu, port, model = {**LANES, **PROTECTED}[lane]
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [v.decode() for v in (entry / "cmdline").read_bytes().split(b"\0") if v]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "--port" not in argv or "--served-model-name" not in argv:
            continue
        if argv[argv.index("--port") + 1] != str(port):
            continue
        if argv[argv.index("--served-model-name") + 1] != model:
            continue
        identity = runtime.process(int(entry.name))
        suite.require("serve" in argv
                      and argv[argv.index("serve") + 1] == f"/fastdata/xianya/models/{model}"
                      and argv[argv.index("--host") + 1] == "127.0.0.1"
                      and "--trust-remote-code" not in argv
                      and identity["cuda_environment"].get("CUDA_VISIBLE_DEVICES") == str(gpu),
                      f"Unexpected {lane} service configuration; no shutdown permitted.")
        identity["process_tree"] = runtime.descendants(identity["pid"])
        matches.append(identity)
    suite.require(len(matches) == 1, f"Expected exactly one original {lane} service.")
    return matches[0]


def same_identity(before, after):
    keys = ("pid", "start_ticks", "argv", "cuda_environment", "cwd")
    return all(before[k] == after[k] for k in keys)


def preflight_originals():
    configs = suite.shared.model_configs(models=tuple(LANES))
    asyncio.run(suite.shared.vllm_preflight(
        [SimpleNamespace(model=c) for c in configs.values()]))


def port_free(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def lock_owner_evidence(pid):
    """Expose only this process's standard FLOCK ownership, never other lock metadata."""
    result = {}
    for lane in LANES:
        import tempfile
        path = Path(tempfile.gettempdir()) / f"jevgraphbench-paired-{os.getuid()}-{lane}.lock"
        stat = path.stat()
        device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
        rows = [line.split() for line in Path("/proc/locks").read_text().splitlines()]
        suite.require(any(len(row) >= 6 and row[1:4] == ["FLOCK", "ADVISORY", "WRITE"]
                          and row[4] == str(pid) and row[5] == device for row in rows),
                      f"Coordinator does not own the unchanged {lane} lease.")
        result[lane] = {"path": str(path), "device_inode": device, "owner_pid": pid}
    return result


class Lease:
    def __init__(self, root, services, protected, targets):
        self.root, self.services, self.protected = root, services, protected
        self.targets, self.children = targets, []

    def start(self, command, environment, label):
        for item in self.protected.values():
            suite.require(runtime.alive(item), "Protected service changed before child launch.")
        identity = runtime.process(os.getpid())
        suite.write_json(self.root / f"{label}-launch-intent.json", {
            "argv": command, "parent": identity, "at_utc": runtime.now(),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "retry_forbidden": True})
        with (self.root / f"{label}.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=runtime.REPO, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        owned = runtime.process(child.pid)
        self.children.append((child, owned))
        suite.write_json(self.root / f"{label}-process.json", owned)
        return child

    def stop(self, child):
        identity = next(identity for process, identity in self.children if process is child)
        known = runtime.descendants(child.pid) if runtime.alive(identity) else [identity]
        suite.require(runtime.stop_tree(known), "Owned worker tree did not stop.")
        child.wait(timeout=30)

    def empty(self):
        inventory = runtime.gpu_inventory()
        return not any(uuid in self.targets for _, uuid, _ in inventory["processes"])

    def cleanup(self):
        for child, identity in reversed(self.children):
            if runtime.alive(identity):
                self.stop(child)
            else:
                child.wait(timeout=30)
        # Never kill an uncaptured occupant or restore a server onto its GPU.
        while not self.empty():
            runtime.status(self.root, status="cleanup_blocked_leases_retained",
                           message="Target GPU still occupied; no unowned process will be killed.")
            time.sleep(30)

    def restore(self):
        suite.require(self.empty(), "Target GPU occupied before restoration.")
        restored = {}
        for lane, (_, port, _) in LANES.items():
            old = self.services[lane]
            suite.require(not runtime.alive(old), "Original service unexpectedly alive.")
            port_free(port)
            env = {k: v for k, v in os.environ.items()
                   if k not in {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"}}
            env.update(old["cuda_environment"])
            suite.write_json(self.root / f"restore-{lane}-intent.json", {
                "argv": old["argv"], "cwd": old["cwd"],
                "cuda_environment": old["cuda_environment"], "retry_forbidden": True})
            with (self.root / f"restore-{lane}.log").open("xb") as log:
                child = subprocess.Popen(old["argv"], cwd=old["cwd"], env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            restored[lane] = runtime.process(child.pid)
            suite.write_json(self.root / f"restore-{lane}-process.json", restored[lane])
        deadline = time.monotonic() + 1200
        while True:
            suite.require(all(runtime.alive(r) for r in restored.values()),
                          "An original-service restoration exited before readiness.")
            try:
                preflight_originals()
                break
            except Exception as exc:
                runtime.status(self.root, status="restoring_services",
                               last_error_type=type(exc).__name__, last_error=str(exc)[:500])
                if time.monotonic() >= deadline:
                    raise RuntimeError("Original service readiness deadline exceeded.") from exc
                time.sleep(10)
        for lane in restored:
            restored[lane]["process_tree"] = runtime.descendants(restored[lane]["pid"])
            suite.require(same_identity(restored[lane], service(lane)),
                          "Restored listener has unexpected identity.")
        suite.require(runtime.validate_mapping(restored, runtime.gpu_inventory()) == self.targets,
                      "Restored GPU mapping changed.")
        suite.require(all(runtime.alive(p) for p in self.protected.values()),
                      "Protected services changed during run.")
        suite.write_json(self.root / "restoration.json", {
            "at_utc": runtime.now(), "services": restored, "models_verified_idle": True,
            "protected_services_unchanged": True})


@contextmanager
def acquire(root, *, wait_seconds=24 * 60 * 60):
    suite.require(not root.exists(), "Resource output exists; no resume or duplicate launch.")
    root.mkdir(parents=True)
    deadline = time.monotonic() + wait_seconds
    while True:
        locks = suite.lane_locks(tuple(LANES))
        try:
            locks.__enter__()
            break
        except suite.shared.Refusal as exc:
            runtime.status(root, status="waiting_for_standard_leases", message=str(exc),
                           pid=os.getpid())
            if time.monotonic() >= deadline:
                raise RuntimeError("Shared resource wait deadline exceeded.") from exc
            time.sleep(30)
    touched, lease = False, None
    try:
        services = {lane: service(lane) for lane in LANES}
        protected = {lane: service(lane) for lane in PROTECTED}
        targets = runtime.validate_mapping(services, runtime.gpu_inventory())
        preflight_originals()
        port_free(8010)
        suite.write_json(root / "admission.json", {
            "at_utc": runtime.now(), "parent": runtime.process(os.getpid()),
            "services": services, "protected": protected, "target_gpu_uuids": targets,
            "leases": lock_owner_evidence(os.getpid()), "models_verified_idle": True})
        lease = Lease(root, services, protected, targets)
        preflight_originals()
        for lane, original in services.items():
            suite.require(same_identity(original, service(lane)), "Service changed at admission.")
        runtime.validate_mapping(services, runtime.gpu_inventory())
        owned = [p for s in services.values() for p in s["process_tree"]]
        touched = True
        suite.require(runtime.stop_tree(owned), "Original service shutdown incomplete.")
        suite.require(lease.empty(), "Unexpected target-GPU occupant after shutdown.")
        runtime.status(root, status="gpus_released", target_gpu_uuids=targets)
        yield lease
    except BaseException as exc:
        error = {"status": "resource_or_job_failure", "error_type": type(exc).__name__,
                 "message": str(exc)[:1000], "resources_touched": touched}
        suite.write_json(root / "failure.json", error)
        runtime.status(root, **error)
        raise
    finally:
        try:
            if touched:
                old_handlers = {sig: signal.getsignal(sig) for sig in
                                (signal.SIGTERM, signal.SIGINT)}
                for sig in old_handlers:
                    signal.signal(sig, signal.SIG_IGN)
                try:
                    lease.cleanup()
                    lease.restore()
                    runtime.status(root, status="restored", original_services_restored=True)
                finally:
                    for sig, handler in old_handlers.items():
                        signal.signal(sig, handler)
        finally:
            locks.__exit__(None, None, None)
