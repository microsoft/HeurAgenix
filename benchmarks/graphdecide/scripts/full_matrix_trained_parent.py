#!/usr/bin/env python3
"""Resource-only entry amendment: trained workers under a verified parent lease."""
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scripts import full_matrix_trained as full

ENTRY = "scripts/full_matrix_trained_parent.py"
COORDINATOR = "scripts/full_matrix_qwen27_coordinator.py"
AMENDMENT = "parent-lease-registration.json"
LANES = ("qwen08", "qwen2")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def lock_paths():
    # These are identities only: this child never opens/stats a lock path.
    return {lane: f"/tmp/jevgraphbench-paired-{os.getuid()}-{lane}.lock" for lane in LANES}


def process_identity(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text()
    fields = stat.rsplit(")", 1)[1].split()
    require(fields[0] not in ("Z", "X"), "Coordinator is not a live process")
    return {"pid": pid, "start_ticks": int(fields[19]),
            "argv": [s.decode() for s in (root / "cmdline").read_bytes().split(b"\0") if s]}


def lock_evidence(pid, expected):
    """Validate the parent's actual fdinfo FLOCKs without touching lock files."""
    descriptors = Path("/proc") / str(pid) / "fd"
    found = {}
    for fd in descriptors.iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        for lane, path in lock_paths().items():
            if target != path:
                continue
            info = (descriptors.parent / "fdinfo" / fd.name).read_text()
            for line in info.splitlines():
                parts = line.split()
                if (len(parts) >= 9 and parts[0] == "lock:" and
                        parts[2:5] == ["FLOCK", "ADVISORY", "WRITE"] and
                        parts[5] == str(pid)):
                    found[lane] = {"path": path, "device_inode": parts[6], "owner_pid": pid}
    require(found == expected and set(found) == set(LANES),
            "Parent does not hold both unchanged standard FLOCK leases")
    return found


def verify_coordinator(path, expected_hash):
    path = path.resolve()
    require(path.is_relative_to(REPO) and path.name == "registration.json",
            "Coordinator registration must be a repository-local frozen registration")
    require(full.digest(path) == expected_hash and
            full.read(path.with_suffix(".sha256")) == expected_hash,
            "Coordinator registration hash mismatch")
    registration = full.read(path)
    require(COORDINATOR in registration["source_sha256"] and
            set(registration["resources"]) == set(LANES) and
            registration["restore_originals"] is True and
            registration.get("thinking") is False,
            "Coordinator does not register the required restoring resource ownership")
    for name, expected in registration["source_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Invalid coordinator source path")
        require(full.digest(REPO / relative) == expected and
                full.digest(path.parent / "source-snapshot" / relative) == expected,
                "Coordinator frozen source changed")


def verify_lease(root, handoff, *, inspect_coordinator=True):
    require(handoff.get("approved") is True and
            handoff.get("prior_gpu_owner_released") is True and
            handoff.get("parent_owns_standard_locks") is True,
            "Missing explicit parent lease authorization")
    require(handoff.get("registration_sha256") == full.digest(root / "registration.json") and
            handoff.get("parent_lease_registration_sha256") == full.digest(root / AMENDMENT),
            "Handoff does not authorize these frozen trained registrations")
    pid = handoff.get("parent_pid")
    require(type(pid) is int and pid > 1 and os.getppid() == pid,
            "Trained entry must be the coordinator's direct child")
    identity = process_identity(pid)
    require(type(handoff.get("parent_start_ticks")) is int and
            identity["start_ticks"] == handoff["parent_start_ticks"],
            "Parent PID/startticks mismatch")
    require(any(Path(arg).resolve() == REPO / COORDINATOR for arg in identity["argv"][1:3]),
            "Parent is not the registered matrix coordinator")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible.startswith("GPU-") and "," not in visible and
            handoff.get("cuda_visible_devices") == visible,
            "Parent must supply exactly one matching GPU UUID")
    require(handoff.get("resource_models") == list(LANES), "Wrong parent resource lanes")
    lock_evidence(pid, handoff.get("locks"))
    if inspect_coordinator:
        verify_coordinator(Path(handoff["coordination_registration"]),
                           handoff["coordination_registration_sha256"])
    return identity


def register(root):
    require(not any(root.glob("run-*")) and not (root / "launch.json").exists(),
            "Resource amendment must precede all trained inference")
    require(not (root / AMENDMENT).exists(), "Parent lease amendment is immutable")
    full.verify_registration(root)
    own_hash = full.digest(REPO / ENTRY)
    snapshot = root / "frozen-parent-lease" / ENTRY
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes((REPO / ENTRY).read_bytes())
    snapshot.chmod(0o444)
    registration = {
        "schema_version": 1, "registered_at": time.time(), "benchmark_calls_before_amendment": 0,
        "base_registration_sha256": full.digest(root / "registration.json"),
        "entry_sha256": own_hash,
        "resource_only_amendment": True, "model_protocol_panel_unchanged": True,
        "parent_owned_lease_required": True, "parent_direct_child_required": True,
        "parent_pid_and_start_ticks_required": True, "parent_entry": COORDINATOR,
        "parent_frozen_registration_required": True, "lock_paths": lock_paths(),
        "coordinator_thinking": False,
        "lock_evidence": "/proc/<parent>/fd symlink identities and fdinfo FLOCK owner/device/inode",
        "child_acquires_locks": False, "child_changes_services": False,
        "cuda_visible_devices": "one physical GPU0 UUID supplied by the coordinator",
        "gpu_admission": "physical GPUs 0 and 1 empty before and after the sequential trained workers",
        "parent_death_signal": "SIGTERM; worker bound to trained entry using existing SIGKILL prctl",
        "lease_checks": "admission, worker startup, loading response, every decision dispatch",
        "policy": {"retry": False, "repair": False, "resume": False},
    }
    full.write(root / AMENDMENT, registration)
    full.write(root / "parent-lease-registration.sha256", full.digest(root / AMENDMENT))
    for name in (AMENDMENT, "parent-lease-registration.sha256"):
        (root / name).chmod(0o444)
    full.status(root / "status.json", {
        "status": "registered_awaiting_parent_owned_lease", "benchmark_calls": 0,
        "gpu_workers_started": False, "active_entry": ENTRY,
        "active_registration": AMENDMENT,
        "active_registration_sha256": full.digest(root / AMENDMENT)})
    return registration


def verify_entry(root):
    registration = full.read(root / AMENDMENT)
    require(full.digest(root / AMENDMENT) == full.read(root / "parent-lease-registration.sha256") and
            registration["base_registration_sha256"] == full.digest(root / "registration.json") and
            registration["entry_sha256"] == full.digest(REPO / ENTRY) ==
            full.digest(root / "frozen-parent-lease" / ENTRY) and
            registration["lock_paths"] == lock_paths() and
            registration["parent_owned_lease_required"] is True,
            "Frozen parent-lease entry changed")
    return registration


def bind_parent(pid):
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGTERM) == 0 and os.getppid() == pid,
            "Cannot bind trained entry lifetime to coordinator")


def launch(root, handoff_path):
    require(not any(root.glob("run-*")) and not (root / "launch.json").exists(),
            "Prior execution exists; no retry/resume")
    verify_entry(root)
    handoff = full.read(handoff_path)
    parent_identity = verify_lease(root, handoff)
    bind_parent(parent_identity["pid"])
    registration, instances = full.verify_registration(root)
    verify_lease(root, handoff)
    before = full.parent.inspect_empty_devices()
    require(before["physical_gpus"][0]["uuid"] == handoff["cuda_visible_devices"],
            "Handed GPU UUID is not the preregistered physical GPU0")
    full.write(root / "gpu-admission.json", before)
    full.write(root / "launch.json", {
        "pid": os.getpid(), "at": time.time(), "handoff": handoff,
        "parent": parent_identity, "active_entry": ENTRY,
        "registration_sha256": full.digest(root / "registration.json"),
        "parent_lease_registration_sha256": full.digest(root / AMENDMENT)})

    class LeasedWorker(full.Worker):
        def __init__(self, *args):
            verify_entry(root)
            verify_lease(root, handoff)
            super().__init__(*args)

        def receive(self, *args, **kwargs):
            verify_lease(root, handoff, inspect_coordinator=False)
            return super().receive(*args, **kwargs)

        def predict(self, *args, **kwargs):
            verify_entry(root)
            verify_lease(root, handoff)
            return super().predict(*args, **kwargs)

    prior_handoff = os.environ.get("TRAINED_JEVLIKE_GPU_HANDOFF")
    os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"] = "approved"
    interrupted, blocked = False, None
    try:
        for candidate in full.CANDIDATES:
            interrupted |= full.evaluate(root, candidate, instances, registration, blocked=blocked,
                                         worker_factory=LeasedWorker)
            if interrupted:
                blocked = "Coordinator interrupted; remaining model not dispatched"
            if not full.read(root / f"run-{candidate}/status.json")["worker_released"]:
                blocked = "Prior worker release failed; remaining model not dispatched"
        full.pilot.verify_files(registration["code_sha256"])
        full.pilot.verify_artifacts(registration["models"])
        verify_entry(root)
        proof = {name: full.verify_terminal(root, name, instances) for name in full.CANDIDATES}
        full.write(root / "terminal-verification.json", proof)
    finally:
        if prior_handoff is None:
            os.environ.pop("TRAINED_JEVLIKE_GPU_HANDOFF", None)
        else:
            os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"] = prior_handoff
        after = full.parent.inspect_empty_devices()
        full.write(root / "gpu-release.json", after)
        full.status(root / "status.json", {
            "status": "terminal_verified" if (root / "terminal-verification.json").exists()
                      else "aborted_unverified",
            "active_entry": ENTRY, "gpu_workers_released": True,
            "parent_owns_service_restoration": True,
            "outcomes": {name: full.read(root / f"run-{name}/status.json")
                         for name in full.CANDIDATES if (root / f"run-{name}/status.json").exists()}})
    return full.read(root / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "dry-run", "run"))
    parser.add_argument("--root", type=Path, default=full.DEFAULT_ROOT)
    parser.add_argument("--handoff", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "register":
        result = register(root)
    elif args.command == "dry-run":
        result = verify_entry(root)
        full.verify_registration(root)
    else:
        require(args.handoff is not None, "run requires a parent coordination handoff")
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        result = launch(root, args.handoff)
    print(full.suite.canonical(result))


if __name__ == "__main__":
    main()
