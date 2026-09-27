#!/usr/bin/env python3
"""Parent-launched durable queue: validated 27B release, trained pilot, restoration."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import launch_trained_jevlike_parent as trained
from scripts.prepare_trained_jevlike import digest, save

PYTHON = "/fastdata/xianya/vllm-venv/bin/python"
LANES = {"qwen08": (0, 8000, "Qwen3.5-0.8B"), "qwen2": (1, 8001, "Qwen3.5-2B")}
MAX_WAIT_SECONDS = 24 * 60 * 60
CANCELLED = False


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def identity(pid, *, with_argv=True):
    try:
        root = Path(f"/proc/{int(pid)}")
        stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "start_ticks": stat[19], "state": stat[0],
                "ppid": int(stat[1]), "session": int(stat[3]),
                "argv": [x.decode(errors="replace") for x in
                         (root / "cmdline").read_bytes().split(b"\0") if x] if with_argv else []}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    current = identity(record["pid"])
    return bool(current and current["start_ticks"] == record["start_ticks"]
                and current["state"] not in {"Z", "X"})


def processes():
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                row = identity(int(path.name), with_argv=False)
                if row:
                    rows[row["pid"]] = row
            except PermissionError:
                continue
    return rows


def tree(record):
    table = processes()
    current = table.get(record["pid"])
    if not current or current["start_ticks"] != record["start_ticks"]:
        return []
    selected = {current["pid"]}
    while True:
        expanded = selected | {pid for pid, row in table.items() if row["ppid"] in selected}
        if expanded == selected:
            break
        selected = expanded
    # The 27B server and restoration services create their own sessions. Include
    # reparented same-session workers as well as current descendants.
    if current["session"] == current["pid"]:
        selected |= {pid for pid, row in table.items() if row["session"] == current["session"]}
    result = []
    for pid in sorted(selected):
        row = identity(pid)
        if row and row["start_ticks"] == table[pid]["start_ticks"]:
            result.append(row)
    return result


def flag(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv),
            f"Saved service has missing/duplicate {name}")
    return argv[argv.index(name) + 1]


def validate_services(start):
    require(set(start["services"]) == set(LANES), "Unexpected saved restoration lanes")
    require(start["locks"] == list(LANES), "Prior supervisor used different resources")
    for lane, (gpu, port, model) in LANES.items():
        service = start["services"][lane]
        argv = service["argv"]
        require(isinstance(argv, list) and all(type(x) is str and x for x in argv)
                and argv[:3] == [PYTHON, "/fastdata/xianya/vllm-venv/bin/vllm", "serve"],
                "Saved original command is not the approved vLLM entry")
        require(flag(argv, "--port") == str(port)
                and flag(argv, "--host") == "127.0.0.1"
                and flag(argv, "--served-model-name") == model,
                "Saved original endpoint/model identity differs")
        require(service["cwd"] == str(REPO)
                and service["cuda_environment"].get("CUDA_VISIBLE_DEVICES") == str(gpu)
                and set(service["cuda_environment"]) <= {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"},
                "Saved original cwd/CUDA settings differ")
        require(service["pid"] > 0 and service["start_ticks"], "Missing original process identity")
    require(len(start["target_gpu_uuids"]) == 2
            and len(set(start["target_gpu_uuids"])) == 2, "Ambiguous target GPU mapping")


def verify_trained_registration(root):
    path = root / "benchmark-registration-parent-locks.json"
    entry = root / "parent-entry-registration.json"
    require(path.is_file() and entry.is_file(), "Missing trained registrations")
    require(digest(path) == read(root / "benchmark-registration-parent-locks.sha256"),
            "Trained registration digest mismatch")
    require(digest(entry) == read(root / "parent-entry-registration.sha256"),
            "Parent entry registration digest mismatch")
    metadata = read(entry)
    require(metadata["benchmark_registration_sha256"] == digest(path)
            and digest(REPO / metadata["parent_entry"]) == metadata["parent_entry_sha256"],
            "Frozen parent entry drift")
    trained.pilot.verify_files(read(path)["code_sha256"])
    require(not any(root.glob("run-*")), "Trained execution already attempted; no overwrite/resume")
    return digest(path), digest(entry)


def prepare(prior_root, trained_root, output):
    prior_root, trained_root, output = (p.resolve() for p in (prior_root, trained_root, output))
    require(all(p.is_relative_to(REPO) for p in (prior_root, trained_root, output)),
            "Queue artifacts must remain in the repository")
    require(not output.exists(), "Queue already exists; no overwrite")
    start_path, server_path = prior_root / "supervisor-start.json", prior_root / "server-process.json"
    start, server = read(start_path), read(server_path)
    validate_services(start)
    supervisor = identity(start["pid"])
    require(supervisor and alive(supervisor), "Capture prior supervisor identity while it is still alive")
    require("scripts/larger_local_pilot_runtime.py" in supervisor["argv"]
            and "launch" in supervisor["argv"], "Unexpected prior supervisor command")
    require(server["lock_owner_supervisor_pid"] == supervisor["pid"]
            and server["gpu_uuids"] == start["target_gpu_uuids"], "Prior server ownership mismatch")
    registration, entry = verify_trained_registration(trained_root)
    execution = prior_root / "execution.json"
    require(digest(execution) == read(prior_root / "execution.json.sha256"),
            "Prior execution registration drift")
    output.mkdir(parents=True)
    manifest = {
        "version": 1, "prepared_at": time.time(), "prior_root": str(prior_root),
        "trained_root": str(trained_root), "prior_supervisor": supervisor,
        "prior_server": server["identity"], "initial_server_tree": tree(server["identity"]),
        "start_sha256": digest(start_path), "server_sha256": digest(server_path),
        "prior_execution_sha256": digest(execution), "trained_registration_sha256": registration,
        "parent_entry_registration_sha256": entry, "source_sha256": digest(__file__),
        "max_wait_seconds": MAX_WAIT_SECONDS, "trained_timeout_seconds": 18 * 60 * 60,
        "restore_startup_timeout_seconds": 1200, "poll_seconds": 30,
        "resources": list(LANES), "retry_inference": False, "retry_service_start": False,
        "restore_only_recorded_original_services": True,
    }
    save(output / "registration.json", manifest)
    save(output / "registration.sha256", digest(output / "registration.json"))
    (output / "queue-source.py").write_bytes(Path(__file__).read_bytes())
    (output / "queue-source.py").chmod(0o444)
    save(output / "status.json", {"status": "prepared_not_started", "resource_actions": 0})
    return manifest


def load_registration(output):
    path = output / "registration.json"
    require(digest(path) == read(output / "registration.sha256"), "Queue registration changed")
    manifest = read(path)
    require(digest(__file__) == manifest["source_sha256"], "Queue source changed after preparation")
    prior = Path(manifest["prior_root"])
    for file, key in (("supervisor-start.json", "start_sha256"),
                      ("server-process.json", "server_sha256"),
                      ("execution.json", "prior_execution_sha256")):
        require(digest(prior / file) == manifest[key], f"Prior {file} changed")
    validate_services(read(prior / "supervisor-start.json"))
    return manifest


def port_free(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def validate_release(manifest, handoff, observed):
    for key in ("owned_processes_stopped", "target_gpus_free",
                "protected_services_alive", "ready_after_supervisor_exit"):
        require(handoff.get(key) is True, f"Invalid prior handoff: {key}")
    require(handoff.get("old_services_restored") is False, "Prior services already restored")
    require(not alive(manifest["prior_supervisor"]), "Prior supervisor is still alive")
    require(not alive(manifest["prior_server"])
            and not any(alive(row) for row in observed), "Prior server tree is still alive")
    start = read(Path(manifest["prior_root"]) / "supervisor-start.json")
    require(all(alive(row) for row in start["protected"]), "Protected service identity changed")
    for service in start["services"].values():
        require(not alive(service)
                and not any(alive(row) for row in service["process_tree"]),
                "An original service/worker is still present")
    for port in (8000, 8001, 8010):
        port_free(port)
    observation = trained.inspect_empty_devices()
    require([observation["physical_gpus"][i]["uuid"] for i in (0, 1)]
            == start["target_gpu_uuids"], "Physical GPU mapping changed")
    return observation


def wait_for_release(output, manifest):
    deadline = time.monotonic() + manifest["max_wait_seconds"]
    prior = Path(manifest["prior_root"])
    observed = {(r["pid"], r["start_ticks"]): r for r in manifest["initial_server_tree"]}
    while True:
        require(not CANCELLED, "Queue cancelled before resource handoff")
        for row in tree(manifest["prior_server"]):
            observed[row["pid"], row["start_ticks"]] = row
        save(output / "observed-server-tree.json", list(observed.values()))
        supervisor_alive = alive(manifest["prior_supervisor"])
        handoff_path = prior / "handoff.json"
        if not supervisor_alive and handoff_path.is_file():
            load_registration(output)
            observation = validate_release(manifest, read(handoff_path), list(observed.values()))
            save(output / "validated-prior-release.json", {
                "handoff_sha256": digest(handoff_path), "validated_at": time.time(),
                "prior_supervisor": manifest["prior_supervisor"], "gpu_observation": observation})
            return
        save(output / "status.json", {
            "status": "waiting_for_prior_release", "pid": os.getpid(),
            "prior_supervisor_alive": supervisor_alive, "handoff_present": handoff_path.exists(),
            "deadline_seconds_remaining": max(0, deadline - time.monotonic()),
            "resource_actions": 0, "updated_at": time.time()})
        require(time.monotonic() < deadline, "24-hour prior handoff deadline exceeded")
        time.sleep(min(manifest["poll_seconds"], max(0, deadline - time.monotonic())))


def wait_process(process, seconds, output, trained_root):
    deadline, terminate_at, next_report = time.monotonic() + seconds, None, 0
    while process.poll() is None:
        if time.monotonic() >= next_report:
            summaries = {}
            for candidate in ("decider", "kev"):
                path = trained_root / f"run-{candidate}/status.json"
                if path.exists():
                    row = read(path)
                    summaries[candidate] = {key: row[key] for key in (
                        "status", "calls", "failure_count", "unattempted_episodes") if key in row}
            save(output / "trained-progress.json", {
                "queue_pid": os.getpid(), "trained_pid": process.pid,
                "trained_process_alive": True, "updated_at": time.time(),
                "deadline_seconds_remaining": max(0, deadline - time.monotonic()),
                "models": summaries})
            next_report = time.monotonic() + 30
        if (CANCELLED or time.monotonic() >= deadline) and terminate_at is None:
            # This is our own trained supervisor only; its normal handler preserves
            # partial attempts and stops its owned worker before exiting.
            process.send_signal(signal.SIGTERM)
            terminate_at = time.monotonic()
            save(output / "trained-stop-request.json", {
                "pid": process.pid, "signal": "SIGTERM", "reason": "cancel_or_deadline"})
        if terminate_at is not None:
            require(time.monotonic() - terminate_at < 240,
                    "Trained supervisor did not stop; refuse restoration while ownership is uncertain")
        time.sleep(2)
    return process.returncode


def invoke_trained(output, manifest):
    root = Path(manifest["trained_root"])
    registration, entry = verify_trained_registration(root)
    require(registration == manifest["trained_registration_sha256"]
            and entry == manifest["parent_entry_registration_sha256"], "Trained pin changed")
    handoff = output / "trained-handoff.json"
    with handoff.open("x") as stream:
        json.dump({"approved": True, "registration_sha256": registration,
                   "prior_gpu_owner_released": True, "cuda_visible_devices": "0",
                   "authorized_by": "parent_launched_registered_queue_after_validated_27b_release",
                   "prior_release_sha256": digest(output / "validated-prior-release.json")}, stream)
    with (output / "trained-supervisor.log").open("x") as log:
        process = subprocess.Popen(
            [PYTHON, str(REPO / "scripts/launch_trained_jevlike_parent.py"),
             "--root", str(root), "--handoff", str(handoff)], cwd=REPO,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1"},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    record = identity(process.pid)
    require(record is not None, "Unable to record trained supervisor identity")
    save(output / "trained-process.json", record)
    save(output / "status.json", {"status": "trained_running", "pid": os.getpid(),
                                 "trained_supervisor": record, "updated_at": time.time()})
    code = wait_process(process, manifest["trained_timeout_seconds"], output, root)
    save(output / "trained-exit.json", {"returncode": code, "at": time.time(),
                                      "identity_exited": not alive(record)})
    return code


def listeners(port):
    import psutil
    return [row.pid for row in psutil.net_connections(kind="tcp")
            if row.status == "LISTEN" and row.laddr.port == port]


def http_get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        require(response.status == 200, "Restored endpoint returned non-200")
        return response.read().decode("utf-8")


def check_endpoint(port, model, owned_pids):
    owners = listeners(port)
    if not owners:
        return False
    require(all(pid is not None and pid in owned_pids for pid in owners),
            "Port listener belongs to a replacement/unknown service")
    body = json.loads(http_get(f"http://127.0.0.1:{port}/v1/models"))
    require(isinstance(body, dict) and isinstance(body.get("data"), list)
            and [item.get("id") for item in body["data"]] == [model],
            "Restored port serves the wrong model identity")
    metrics = http_get(f"http://127.0.0.1:{port}/metrics")
    pattern = re.compile(r"^(vllm:num_requests_(?:running|waiting))(?:\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$")
    seen = set()
    for line in metrics.splitlines():
        match = pattern.fullmatch(line)
        if match:
            value = float(match[2])
            require(math.isfinite(value) and value == 0, "Restored endpoint is not idle")
            seen.add(match[1])
        elif re.match(r"^vllm:num_requests_(?:running|waiting)(?:\{|\s|$)", line):
            raise RuntimeError("Unrecognized restored occupancy metric")
    require(seen == {"vllm:num_requests_running", "vllm:num_requests_waiting"},
            "Restored endpoint lacks required occupancy gauges")
    return True


def restore_inventory(start, restored):
    rows = trained.nvidia_query(("gpu_uuid", "pid", "process_name"), "compute-apps")
    allowed = {row["pid"] for record in restored.values() for row in tree(record)}
    for uuid, pid, _ in rows:
        if uuid in start["target_gpu_uuids"]:
            require(int(pid) in allowed, "Unknown target-GPU process; refuse restoration")
    return rows


def verify_restored_mapping(lane, start, restored):
    rows = restore_inventory(start, restored)
    own = {row["pid"] for row in tree(restored[lane])}
    locations = {uuid for uuid, pid, _ in rows if int(pid) in own}
    require(locations == {start["target_gpu_uuids"][LANES[lane][0]]},
            "Restored engine is missing or allocated on the wrong GPU")


def restore_originals(output, manifest):
    start_path = Path(manifest["prior_root"]) / "supervisor-start.json"
    require(digest(start_path) == manifest["start_sha256"], "Saved original restoration record changed")
    start = read(start_path)
    validate_services(start)
    trained_record = output / "trained-process.json"
    if trained_record.exists():
        prior = read(trained_record)
        require(not alive(prior) and not tree(prior), "Trained supervisor remains alive")
    restored = {}
    with trained.suite.lane_locks(tuple(LANES)):
        # Admission must be inside the same original leases, not merely before
        # waiting for them. Never stop an occupant or reuse a listening port.
        observation = trained.inspect_empty_devices()
        require([observation["physical_gpus"][i]["uuid"] for i in (0, 1)]
                == start["target_gpu_uuids"], "GPU identity changed before restoration")
        require(all(alive(row) for row in start["protected"]), "Protected service changed")
        for lane, (_, port, _) in LANES.items():
            port_free(port)
            require(not alive(start["services"][lane]), "Original service is already running")
        save(output / "restoration-admission.json", observation)
        for lane, (gpu, port, model) in LANES.items():
            service = start["services"][lane]
            marker = output / f"restore-{lane}-launch.json"
            require(not marker.exists(), "Service launch already attempted; no retry")
            port_free(port)
            rows = restore_inventory(start, restored)
            require(not any(uuid == start["target_gpu_uuids"][gpu] for uuid, _, _ in rows),
                    "Next restoration GPU is not empty")
            env = {k: v for k, v in os.environ.items()
                   if k not in {"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"}}
            env.update(service["cuda_environment"])
            # Persist intent before spawn. A crash in the spawn/identity gap must
            # never lead to a duplicate launch on an automatic rerun.
            with marker.open("x") as stream:
                json.dump({"status": "launch_intent", "argv": service["argv"],
                           "cwd": service["cwd"], "cuda_environment": service["cuda_environment"],
                           "at": time.time(), "retry_forbidden": True}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            with (output / f"restore-{lane}.log").open("x") as log:
                process = subprocess.Popen(
                    service["argv"], cwd=service["cwd"], env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = identity(process.pid)
            require(record is not None, "Restoration spawned but process identity is unavailable")
            restored[lane] = record
            save(output / f"restore-{lane}-process.json", {
                **record, "cwd": service["cwd"], "cuda_environment": service["cuda_environment"],
                "detached": True, "parent_death_signal": False, "verified_ready": False})
            save(output / "status.json", {"status": "restoring", "lane": lane,
                                         "pid": os.getpid(), "service_pid": process.pid})
            deadline = time.monotonic() + manifest["restore_startup_timeout_seconds"]
            while True:
                require(process.poll() is None and alive(record),
                        "Restored process exited; no relaunch")
                own = {row["pid"] for row in tree(record)}
                restore_inventory(start, restored)
                try:
                    ready = check_endpoint(port, model, own)
                except (urllib.error.URLError, ConnectionError, TimeoutError):
                    ready = False
                if ready:
                    verify_restored_mapping(lane, start, restored)
                    break
                save(output / f"restore-{lane}-readiness.json", {
                    "pid": process.pid, "alive": alive(record), "ready": False,
                    "last_checked_at": time.time(),
                    "deadline_seconds_remaining": max(0, deadline - time.monotonic())})
                require(time.monotonic() < deadline, "Restoration readiness timed out; no relaunch")
                time.sleep(5)
            verified = {**read(output / f"restore-{lane}-process.json"),
                        "verified_ready": True, "verified_at": time.time(),
                        "process_tree": tree(record), "model": model, "port": port,
                        "running_requests": 0, "waiting_requests": 0}
            save(output / f"restore-{lane}-process.json", verified)
        require(all(alive(row) for row in start["protected"]), "Protected service changed")
        for lane, (_, port, model) in LANES.items():
            require(alive(restored[lane]) and check_endpoint(
                port, model, {row["pid"] for row in tree(restored[lane])}),
                "Restored service lost verified liveness")
            verify_restored_mapping(lane, start, restored)
        save(output / "restoration.json", {"status": "verified", "services": {
            lane: read(output / f"restore-{lane}-process.json") for lane in LANES},
            "protected_services_preserved": True, "at": time.time()})


def follow_through(output, manifest):
    """Once handoff is validated, always attempt safe restoration after child exit."""
    trained_error, restoration_error, returncode = None, None, None
    try:
        returncode = invoke_trained(output, manifest)
    except BaseException as error:
        trained_error = {"type": type(error).__name__, "message": str(error)[:1600]}
        save(output / "trained-failure.json", trained_error)
    finally:
        try:
            restore_originals(output, manifest)
        except BaseException as error:
            restoration_error = {"type": type(error).__name__, "message": str(error)[:1600]}
            restoration_error["launched_service_liveness"] = {
                path.name: {"pid": read(path)["pid"], "alive": alive(read(path)),
                            "verified_ready": read(path)["verified_ready"]}
                for path in output.glob("restore-*-process.json")}
            save(output / "restoration-failure.json", restoration_error)
    result = {
        "status": "restoration_failed" if restoration_error else
                  ("restored_after_trained_failure" if trained_error or returncode != 0 else "completed"),
        "trained_returncode": returncode, "trained_error": trained_error,
        "restoration_error": restoration_error, "finished_at": time.time(),
        "partial_inference_artifacts_preserved": True,
    }
    root = Path(manifest["trained_root"])
    for name in ("parent-entry-status.json", "status.json"):
        if (root / name).exists():
            result[name] = read(root / name)
    if not restoration_error and result.get("parent-entry-status.json", {}).get("status") != "completed":
        result["status"] = "restored_with_incomplete_or_failed_trained_results"
    save(output / "status.json", result)
    return result


def run(output):
    manifest = load_registration(output)
    with (output / "queue-start.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "identity": identity(os.getpid()), "at": time.time(),
                   "authorization": "parent_detached_launch_for_registered_follow_through"}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        wait_for_release(output, manifest)
        load_registration(output)
        return follow_through(output, manifest)
    except BaseException as error:
        save(output / "status.json", {"status": "blocked_before_trained_launch",
             "error_type": type(error).__name__, "error": str(error)[:1600],
             "resource_actions": 0, "finished_at": time.time()})
        raise


def main():
    def cancel(signum, frame):
        global CANCELLED
        CANCELLED = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "inspect"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--trained-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.prior_root is not None and args.trained_root is not None,
                "prepare needs --prior-root and --trained-root")
        result = prepare(args.prior_root, args.trained_root, args.output)
    elif args.command == "inspect":
        result = {"registration": load_registration(args.output),
                  "status": read(args.output / "status.json")}
    else:
        result = run(args.output.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "run" and result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
