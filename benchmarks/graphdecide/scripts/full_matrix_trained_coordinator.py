#!/usr/bin/env python3
"""Independent trained-only lease; frozen inference and parent-owned restoration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scripts import full_matrix_gpu_resources as resources
from scripts import full_matrix_trained as full
from scripts import full_matrix_trained_parent as leased

ENTRY = "scripts/full_matrix_trained_coordinator.py"
DIRECTORY = "standalone-coordinator"
WAIT_SECONDS = 86400
RUN_SECONDS = 2 * (600 + 1920 * 180 + 20) + 3600
require = leased.require


def source_hashes():
    paths = {REPO / ENTRY}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).suffix == ".py" and Path(name).resolve().is_relative_to(REPO):
            paths.add(Path(name).resolve())
    return {str(p.relative_to(REPO)): full.digest(p) for p in sorted(paths)}


def register(root):
    require(not any(root.glob("run-*")) and not (root / "launch.json").exists(),
            "Registration must precede all trained inference")
    out = root / DIRECTORY
    require(not out.exists(), "Standalone coordinator registration is immutable")
    base, _ = full.verify_registration(root)
    leased.verify_entry(root)
    out.mkdir()
    hashes = source_hashes()
    for name, expected in hashes.items():
        target = out / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
        target.chmod(0o444)
        require(full.digest(target) == expected, "Source changed during snapshot")
    registration = {
        "schema_version": 1, "registered_at": time.time(),
        "base_registration_sha256": full.digest(root / "registration.json"),
        "prior_lease_registration_sha256": full.digest(root / leased.AMENDMENT),
        "source_sha256": hashes, "entry": ENTRY, "thinking": False,
        "resource_only_amendment": True, "model_protocol_panel_unchanged": True,
        "resources": list(leased.LANES), "restore_originals": True,
        "resource_api": "scripts.full_matrix_gpu_resources.acquire",
        "fresh_resource_root": str(out / "resources"),
        "maximum_resource_wait_seconds": WAIT_SECONDS,
        "maximum_supervisor_seconds": RUN_SECONDS,
        "scheduled_episodes_per_model": base["scheduled_episodes_per_model"],
        "maximum_calls_per_model": base["maximum_calls_per_model"],
        "worker_parent_identity_and_standard_lease_required": True,
        "coordinator_approval_required_before_acquire": True,
        "child_acquires_locks": False, "worker_changes_services": False,
        "retry": False, "resume": False, "new_model_calls": 0,
    }
    full.write(out / "registration.json", registration)
    full.write(out / "registration.sha256", full.digest(out / "registration.json"))
    (out / "registration.json").chmod(0o444)
    (out / "registration.sha256").chmod(0o444)
    full.status(root / "status.json", {
        "status": "standalone_registered_awaiting_parent_approval", "benchmark_calls": 0,
        "gpu_workers_started": False, "active_entry": ENTRY,
        "active_registration": str((out / "registration.json").relative_to(root))})
    return registration


def verify(root, *, base=True):
    out = root / DIRECTORY
    registration = full.read(out / "registration.json")
    require(full.digest(out / "registration.json") == full.read(out / "registration.sha256") and
            registration["base_registration_sha256"] == full.digest(root / "registration.json") and
            registration["prior_lease_registration_sha256"] == full.digest(root / leased.AMENDMENT),
            "Standalone coordinator registration changed")
    for name, expected in registration["source_sha256"].items():
        require(full.digest(REPO / name) == expected and
                full.digest(out / "source-snapshot" / name) == expected, "Coordinator source drift")
    if base:
        full.verify_registration(root)
        leased.verify_entry(root)
    return registration


def check_handoff(root, handoff):
    verify(root, base=False)
    pid = handoff["parent_pid"]
    require(type(pid) is int and pid > 1 and os.getppid() == pid,
            "Supervisor must be the trained coordinator's direct child")
    owner = leased.process_identity(pid)
    require(owner["start_ticks"] == handoff["parent_start_ticks"] and
            type(handoff["parent_start_ticks"]) is int and
            any(Path(arg).resolve() == REPO / ENTRY for arg in owner["argv"][1:3]) and
            "run" in owner["argv"], "Parent coordinator identity changed")
    require(handoff["coordinator_registration_sha256"] ==
            full.digest(root / DIRECTORY / "registration.json") and
            handoff["base_registration_sha256"] == full.digest(root / "registration.json"),
            "Parent handoff registration mismatch")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible.startswith("GPU-") and "," not in visible and
            visible == handoff["cuda_visible_devices"], "Expected single parent-leased GPU UUID")
    leased.lock_evidence(pid, handoff["locks"])
    return owner


def worker_phase(root, handoff_path):
    require(not any(root.glob("run-*")) and not (root / "launch.json").exists(),
            "No inference retry/resume")
    handoff = full.read(handoff_path)
    owner = check_handoff(root, handoff)
    leased.bind_parent(owner["pid"])
    registration, instances = full.verify_registration(root)
    check_handoff(root, handoff)
    before = full.parent.inspect_empty_devices()
    require(before["physical_gpus"][0]["uuid"] == handoff["cuda_visible_devices"],
            "GPU UUID is not physical GPU0")
    full.write(root / "gpu-admission.json", before)
    full.write(root / "launch.json", {"pid": os.getpid(), "parent": owner,
                                     "handoff": handoff, "active_entry": ENTRY, "at": time.time()})

    class Worker(full.Worker):
        def __init__(self, *args):
            check_handoff(root, handoff)
            super().__init__(*args)

        def receive(self, *args, **kwargs):
            check_handoff(root, handoff)
            return super().receive(*args, **kwargs)

        def predict(self, *args, **kwargs):
            check_handoff(root, handoff)
            return super().predict(*args, **kwargs)

    previous = os.environ.get("TRAINED_JEVLIKE_GPU_HANDOFF")
    os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"] = "approved"
    blocked = None
    try:
        for candidate in full.CANDIDATES:
            interrupted = full.evaluate(root, candidate, instances, registration,
                                        blocked=blocked, worker_factory=Worker)
            if interrupted:
                blocked = "Coordinator interrupted; remaining model not dispatched"
            if not full.read(root / f"run-{candidate}/status.json")["worker_released"]:
                blocked = "Prior worker release failed"
        full.pilot.verify_files(registration["code_sha256"])
        full.pilot.verify_artifacts(registration["models"])
        verify(root, base=False)
        full.write(root / "terminal-verification.json", {
            name: full.verify_terminal(root, name, instances) for name in full.CANDIDATES})
    finally:
        if previous is None:
            os.environ.pop("TRAINED_JEVLIKE_GPU_HANDOFF", None)
        else:
            os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"] = previous
        full.write(root / "gpu-release.json", full.parent.inspect_empty_devices())
        full.status(root / "status.json", {
            "status": "terminal_verified" if (root / "terminal-verification.json").exists()
                      else "aborted_unverified",
            "active_entry": ENTRY, "gpu_workers_released": True,
            "outcomes": {name: full.read(root / f"run-{name}/status.json")
                         for name in full.CANDIDATES if (root / f"run-{name}/status.json").exists()}})


def launch(root, approval_path):
    registration = verify(root)
    out = root / DIRECTORY
    approval = full.read(approval_path)
    require(approval.get("approved") is True and
            approval.get("coordinator_registration_sha256") == full.digest(out / "registration.json") and
            approval.get("base_registration_sha256") == full.digest(root / "registration.json"),
            "Explicit parent approval must authorize both frozen registrations")
    require(not (out / "start.json").exists() and not (root / "launch.json").exists() and
            not any(root.glob("run-*")), "No coordinator or inference resume")
    full.write(out / "start.json", {"pid": os.getpid(), "at": time.time(), "approval": approval})
    try:
        with resources.acquire(out / "resources",
                               wait_seconds=registration["maximum_resource_wait_seconds"]) as lease:
            verify(root)
            owner = leased.process_identity(os.getpid())
            handoff_path = out / "worker-handoff.json"
            full.write(handoff_path, {
                "parent_pid": owner["pid"], "parent_start_ticks": owner["start_ticks"],
                "coordinator_registration_sha256": full.digest(out / "registration.json"),
                "base_registration_sha256": full.digest(root / "registration.json"),
                "cuda_visible_devices": lease.targets[0],
                "locks": resources.lock_owner_evidence(os.getpid())})
            child = lease.start(
                [full.pilot.PYTHON, str(REPO / ENTRY), "worker-phase", "--root", str(root),
                 "--handoff", str(handoff_path)],
                {**os.environ, "CUDA_VISIBLE_DEVICES": lease.targets[0],
                 "PYTHONDONTWRITEBYTECODE": "1"}, "trained-supervisor")
            try:
                code = child.wait(timeout=registration["maximum_supervisor_seconds"])
            except BaseException:
                lease.stop(child)
                raise
            require(code == 0, "Trained supervisor failed; restoration remains parent-owned")
            _, instances = full.verify_registration(root)
            proof = {name: full.verify_terminal(root, name, instances) for name in full.CANDIDATES}
            require(proof == full.read(root / "terminal-verification.json"), "Terminal proof mismatch")
            full.write(out / "run-verification.json", proof)
        full.status(out / "status.json", {"status": "terminal_verified_services_restored",
                                        "original_services_restored": True})
    except BaseException as error:
        full.status(out / "status.json", {"status": "failed", "error_type": type(error).__name__,
                                        "error": str(error)[:1200]})
        raise
    return full.read(out / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "dry-run", "run", "worker-phase"))
    parser.add_argument("--root", type=Path, default=full.DEFAULT_ROOT)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--handoff", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "register":
        result = register(root)
    elif args.command == "dry-run":
        result = verify(root)
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        if args.command == "run":
            require(args.approval is not None, "run requires --approval")
            result = launch(root, args.approval)
        else:
            require(args.handoff is not None, "worker-phase requires --handoff")
            result = worker_phase(root, args.handoff)
    print(full.suite.canonical(result))


if __name__ == "__main__":
    main()
