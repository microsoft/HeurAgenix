#!/usr/bin/env python3
"""Parent-only GPU entry: standard locks, empty-device admission, frozen pilot."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_trained_jevlike_pilot as pilot
from scripts import extended_graph_suite as suite
from scripts.prepare_trained_jevlike import digest, save

RESOURCES = ("qwen08", "qwen2")
PHYSICAL_GPUS = (0, 1)
MAX_IDLE_MEMORY_MIB = 256


def nvidia_query(fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15)
    return [[cell.strip() for cell in row]
            for row in csv.reader(io.StringIO(result.stdout)) if row]


def inspect_empty_devices():
    gpu_rows = nvidia_query(("index", "uuid", "memory.used"))
    mapping = {}
    for row in gpu_rows:
        if len(row) != 3:
            raise RuntimeError("Unrecognized GPU inventory")
        index, uuid, memory = row
        if int(index) in PHYSICAL_GPUS:
            mapping[int(index)] = {"uuid": uuid, "memory_used_mib": int(memory)}
    if set(mapping) != set(PHYSICAL_GPUS) or len({v["uuid"] for v in mapping.values()}) != 2:
        raise RuntimeError("Required physical GPUs 0 and 1 are not uniquely present")
    target_uuids = {value["uuid"] for value in mapping.values()}
    occupants = []
    for row in nvidia_query(("gpu_uuid", "pid", "process_name"), "compute-apps"):
        if len(row) != 3:
            raise RuntimeError("Unrecognized GPU process inventory")
        uuid, pid, name = row
        if uuid in target_uuids:
            occupants.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name})
    observation = {"at": time.time(), "physical_gpus": mapping, "compute_occupants": occupants,
                   "idle_memory_cap_mib": MAX_IDLE_MEMORY_MIB}
    if occupants or any(v["memory_used_mib"] > MAX_IDLE_MEMORY_MIB for v in mapping.values()):
        error = RuntimeError("GPUs 0/1 are not empty; no service may be stopped by this entry")
        error.observation = observation
        raise error
    return observation


def admit(root, handoff):
    entry = root / "parent-entry-registration.json"
    pinned = json.loads(entry.read_text())
    if digest(entry) != json.loads((root / "parent-entry-registration.sha256").read_text()):
        raise RuntimeError("Parent entry registration changed")
    if digest(__file__) != pinned["parent_entry_sha256"]:
        raise RuntimeError("Parent entry source changed after registration")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This parent entry is preregistered for physical GPU 0 only")
    path = root / "benchmark-registration-parent-locks.json"
    if digest(path) != pinned["benchmark_registration_sha256"]:
        raise RuntimeError("Frozen trained registration changed")
    registration = json.loads(path.read_text())
    pilot.verify_handoff(handoff, path)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if packages != registration["packages"]:
        raise RuntimeError("Runtime packages changed after registration")
    pilot.verify_files(registration["code_sha256"])
    pilot.verify_artifacts(registration["models"])
    instances, shared = pilot.panel(Path(registration["selection"]))
    if shared["selection_sha256"] != registration["selection_sha256"]:
        raise RuntimeError("Shared selected panel changed")
    if any(root.glob("run-*")):
        raise RuntimeError("Benchmark output already exists; no retry/resume")
    return instances, registration


def launch(root, handoff):
    root, handoff = root.resolve(), handoff.resolve()
    status_path = root / "parent-entry-status.json"
    status = {"status": "admission", "pid": os.getpid(), "started_at": time.time(),
              "resource_models": list(RESOURCES), "gpu_workers_released": False}
    save(status_path, status)
    started = False
    try:
        with suite.lane_locks(RESOURCES):
            before = inspect_empty_devices()
            save(root / "gpu-admission-observation.json", before)
            instances, registration = admit(root, handoff)
            status.update(status="running")
            save(status_path, status)
            started = True
            prior_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            selected_uuid = before["physical_gpus"][0]["uuid"]
            save(root / "gpu-execution-device.json", {
                "physical_gpu_index": 0, "cuda_visible_devices": selected_uuid,
                "admitted_parent_cuda_visible_devices": prior_visible})
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_uuid
            try:
                pilot.run_locked(root, instances, registration)
            finally:
                if prior_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prior_visible
                after = inspect_empty_devices()
                save(root / "gpu-release-observation.json", after)
                status["gpu_workers_released"] = True
            outcomes = json.loads((root / "status.json").read_text())["outcomes"]
            status["model_outcomes"] = outcomes
            status["status"] = ("completed" if all(
                row["status"] == "completed" for row in outcomes.values())
                else "completed_with_model_failures")
    except BaseException as error:
        status.update(status="aborted" if started else "refused_before_inference",
                      error_type=type(error).__name__, error=str(error)[:1200])
        if hasattr(error, "observation"):
            save(root / "gpu-refusal-observation.json", error.observation)
        raise
    finally:
        status["finished_at"] = time.time()
        save(status_path, status)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    launch(args.root, args.handoff)


if __name__ == "__main__":
    main()
