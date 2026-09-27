#!/usr/bin/env python3
"""Parent-operated pinned download, GPU supervisor, and bounded pilot execution."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import larger_local_pilot as panel
from src.clients.base import ClientTimeoutError, InvalidResponseError
from src.clients.vllm_client import VLLMClient

suite, canonical, release = panel.suite, panel.canonical, panel.release
MODEL = "Qwen/Qwen3.5-27B"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
PYTHON = "/fastdata/xianya/vllm-venv/bin/python"
ROOT = REPO / "output/larger-local-pilot-execution-20260924-v1"
ARTIFACTS = REPO / "output/larger-local-pilot-model-20260924-v1"
OLD = {"qwen08": (713065, 0), "qwen2": (713069, 1)}
PROTECTED = (713076, 713079)
RESOURCES = tuple(OLD)


def now():
    return panel.invariance.now()


def local(path):
    path = path.resolve()
    suite.require(path.is_relative_to(REPO), "Artifacts must remain inside the repository.")
    return path


def write(path, value):
    suite.write_json(path, value)


def status(root, **value):
    suite.set_status(root, {"at_utc": now(), **value})


def verify_manifest(root, filename):
    data = suite.read_json(root / filename)
    suite.require(suite.digest(root / filename) == suite.read_json(root / (filename + ".sha256")),
                  "Manifest digest mismatch.")
    return data


def download(root):
    """Hub local-dir partial files permit restart of this step, never inference."""
    from huggingface_hub import HfApi, snapshot_download
    root = local(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "artifacts.json").exists():
        return verify_artifacts(root)
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    suite.require(info.sha == REVISION, "Hub revision mismatch.")
    snapshot = root / "snapshot"
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=snapshot,
                      cache_dir=root / "hub-cache", max_workers=4,
                      ignore_patterns=["*.py", "*.bin", "*.pt"])
    files = {}
    for sibling in info.siblings:
        name = sibling.rfilename
        if name.endswith((".py", ".bin", ".pt")):
            continue
        path = snapshot / name
        suite.require(path.is_file() and not path.is_symlink(), "Missing regular checkpoint file.")
        actual = suite.digest(path)
        lfs = getattr(sibling, "lfs", None)
        remote = getattr(lfs, "sha256", None) if lfs else None
        suite.require(remote is None or remote == actual, "Hub LFS checksum mismatch.")
        size = path.stat().st_size
        suite.require(sibling.size is None or sibling.size == size, "Hub size mismatch.")
        files[name] = {"sha256": actual, "bytes": size, "hub_lfs_sha256": remote}
    config = suite.read_json(snapshot / "config.json")
    suite.require(not config.get("quantization_config"), "Quantized checkpoint is forbidden.")
    index = suite.read_json(snapshot / "model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    suite.require(len(shards) == 11 and shards <= set(files), "Unexpected shard inventory.")
    dtypes = Counter()
    for name in sorted(shards):
        with (snapshot / name).open("rb") as stream:
            count = int.from_bytes(stream.read(8), "little")
            suite.require(0 < count < 100_000_000, "Invalid safetensors header.")
            header = json.loads(stream.read(count))
        dtypes.update(v["dtype"] for key, v in header.items() if key != "__metadata__")
    suite.require(dtypes["BF16"] > 0 and not (set(dtypes) - {"BF16", "F32"}),
                  "Expected BF16 checkpoint, with optional FP32 tensors only.")
    manifest = {"model": MODEL, "revision": REVISION, "downloaded_at_utc": now(),
                "snapshot": str(snapshot), "files": files, "tensor_dtypes": dict(dtypes),
                "trust_remote_code": False, "resumable_download_only": True}
    write(root / "artifacts.json", manifest)
    write(root / "artifacts.json.sha256", suite.digest(root / "artifacts.json"))
    return manifest


def verify_artifacts(root):
    manifest = verify_manifest(root, "artifacts.json")
    suite.require(manifest["model"] == MODEL and manifest["revision"] == REVISION
                  and manifest["trust_remote_code"] is False, "Checkpoint identity differs.")
    snapshot = local(Path(manifest["snapshot"]))
    for name, entry in manifest["files"].items():
        path = snapshot / name
        suite.require(path.resolve().is_relative_to(snapshot) and path.is_file()
                      and not path.is_symlink() and path.stat().st_size == entry["bytes"]
                      and suite.digest(path) == entry["sha256"], "Checkpoint drift.")
    return manifest


def load_panel(root):
    registration = suite.read_json(root / "registration.json")
    suite.require(suite.digest(root / "registration.json") ==
                  suite.read_json(root / "registration.sha256"), "Panel registration drift.")
    suite.require(suite.digest(root / "pilot-cases.jsonl") == registration["selection_sha256"]
                  and suite.digest(root / "initial-requests.jsonl") ==
                  registration["initial_requests_sha256"], "Panel artifacts drift.")
    for name, digest in registration["source_sha256"].items():
        suite.require(suite.digest(REPO / name) == digest, "Frozen panel dependency drift.")
    bundle = Path(registration["bundle"])
    suite.require(suite.digest(bundle / "manifest.json") == registration["bundle_manifest_sha256"],
                  "Release identity drift.")
    manifest = release.manifest_at(bundle)
    labeled = {i["id"]: i for i in release.labeled_instances(bundle, registration["split"], manifest)}
    cases = suite.read_rows(root / "pilot-cases.jsonl")
    instances = [labeled[i["instance_id"]] for i in cases]
    suite.require(len(instances) == 64 and len({i["id"] for i in instances}) == 64
                  and sum(i["query_budget"] for i in instances) == 160, "Panel cardinality drift.")
    expected = suite.read_rows(root / "initial-requests.jsonl")
    actual = [{"instance_id": i["id"], "request": suite.json_value(
        asdict(canonical.public_request(i, [])))} for i in instances]
    suite.require(expected == actual, "Canonical initial request mismatch.")
    return registration, instances


def server_command(snapshot):
    return [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(snapshot), "--served-model-name", "Qwen3.5-27B",
            "--host", "127.0.0.1", "--port", "8010", "--tensor-parallel-size", "2",
            "--dtype", "bfloat16", "--language-model-only", "--max-model-len", "16384",
            "--gpu-memory-utilization", "0.90", "--max-num-seqs", "1",
            "--generation-config", "vllm", "--reasoning-parser", "qwen3",
            "--default-chat-template-kwargs", '{"enable_thinking": false}']


def register(root, panel_root, artifacts):
    root, panel_root, artifacts = map(local, (root, panel_root, artifacts))
    suite.require(not root.exists(), "Execution registration exists; no overwrite.")
    registration, instances = load_panel(panel_root)
    checkpoint = verify_artifacts(artifacts)
    canonical.tasks.validate_instances(instances)
    root.mkdir(parents=True)
    hashes = {**registration["source_sha256"],
              "scripts/larger_local_pilot_runtime.py": suite.digest(__file__)}
    panel.invariance.freeze_sources(root, hashes)
    value = {"registered_at_utc": now(), "model_calls": 0, "panel": str(panel_root),
             "panel_registration_sha256": suite.digest(panel_root / "registration.json"),
             "artifacts": str(artifacts), "artifacts_sha256": suite.digest(artifacts / "artifacts.json"),
             "model": MODEL, "revision": REVISION, "code_sha256": hashes,
             "configurations": {k: asdict(v) for k, v in panel.configurations().items()},
             "episodes_per_profile": 64, "maximum_calls_per_profile": 160,
             "maximum_calls_total": 480, "resources": list(RESOURCES),
             "server_command": server_command(checkpoint["snapshot"]),
             "server_cuda_visible_devices": "0,1",
             "configuration_effect": "Thinking, budget, and constraint change together; not "
                                     "isolated reasoning. Historical references are not latency controls.",
             "policy": {"retry": False, "repair": False, "resume_inference": False,
                        "auto_restore_old_services": False, "all_planned_denominators": True,
                        "response_storage": "Allowlisted structural diagnostics and final-content "
                                            "length/hash only; no reasoning text or full bodies.",
                        "lock_api": "extended_graph_suite.lane_locks unchanged standard namespace"}}
    write(root / "execution.json", value)
    write(root / "execution.json.sha256", suite.digest(root / "execution.json"))
    return value


def execution(root):
    value = verify_manifest(root, "execution.json")
    suite.require(suite.digest(Path(value["panel"]) / "registration.json") ==
                  value["panel_registration_sha256"], "Panel registration changed.")
    for name, digest in value["code_sha256"].items():
        suite.require(suite.digest(REPO / name) == digest, "Execution source drift.")
    suite.require(suite.digest(Path(value["artifacts"]) / "artifacts.json") ==
                  value["artifacts_sha256"], "Model manifest changed.")
    _, instances = load_panel(Path(value["panel"]))
    return value, instances


def process(pid):
    base = Path(f"/proc/{pid}")
    stat = (base / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [v.decode() for v in (base / "cmdline").read_bytes().split(b"\0") if v]
    env = dict(v.decode().split("=", 1) for v in (base / "environ").read_bytes().split(b"\0")
               if v.startswith((b"CUDA_VISIBLE_DEVICES=", b"CUDA_DEVICE_ORDER=")))
    return {"pid": pid, "ppid": int(stat[1]), "start_ticks": stat[19], "state": stat[0],
            "argv": argv, "cuda_environment": env, "cwd": os.readlink(base / "cwd")}


def alive(identity):
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[19] == identity["start_ticks"] and stat[0] != "Z"
    except FileNotFoundError:
        return False


def descendants(pid):
    parents = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                text = (path / "stat").read_text().rsplit(")", 1)[1].split()
                parents[int(path.name)] = int(text[1])
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    selected = {pid}
    while True:
        expanded = selected | {child for child, parent in parents.items() if parent in selected}
        if expanded == selected:
            return [process(child) for child in sorted(selected)]
        selected = expanded


def gpu_inventory():
    def query(kind, fields):
        return [line.split(", ") for line in subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader"],
            text=True).strip().splitlines() if line]
    return {"gpus": query("gpu", "index,uuid"),
            "processes": query("compute-apps", "pid,gpu_uuid,used_memory")}


def validate_mapping(services, inventory):
    by_gpu = {int(index): uuid for index, uuid in inventory["gpus"]}
    suite.require(set(by_gpu) >= {0, 1, 2, 3}, "Expected four physical GPUs.")
    allowed = set()
    for lane, (_, gpu) in OLD.items():
        service = services[lane]
        suite.require(service["cuda_environment"].get("CUDA_VISIBLE_DEVICES") == str(gpu),
                      "Original CUDA mapping differs.")
        children = {p["pid"] for p in service["process_tree"]}
        matches = [(int(pid), uuid) for pid, uuid, _ in inventory["processes"] if int(pid) in children]
        suite.require(matches and all(uuid == by_gpu[gpu] for _, uuid in matches),
                      "Original engines do not match physical GPU mapping.")
        allowed.update(children)
    suite.require(all(int(pid) in allowed for pid, uuid, _ in inventory["processes"]
                      if uuid in {by_gpu[0], by_gpu[1]}), "Unowned process on target GPU.")
    return [by_gpu[0], by_gpu[1]]


def port_free():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 8010))


def stop_tree(identities):
    """Signal only captured, still-matching process identities; never a shell/group."""
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        for identity in identities:
            if alive(identity):
                os.kill(identity["pid"], sig)
        deadline = time.monotonic() + wait
        while any(alive(identity) for identity in identities) and time.monotonic() < deadline:
            time.sleep(0.2)
        if not any(alive(identity) for identity in identities):
            return True
    return False


def response_diagnostics(body):
    """Persist only structural metadata and final-content length/hash, never text."""
    result = {"response_is_object": isinstance(body, dict)}
    if not isinstance(body, dict):
        return result
    choices = body.get("choices")
    if not isinstance(choices, list):
        return result
    result["choice_count"] = len(choices)
    if len(choices) != 1 or not isinstance(choices[0], dict):
        return result
    choice = choices[0]
    finish = choice.get("finish_reason")
    allowed = ("stop", "length", "tool_calls", "function_call", "content_filter", "error", "abort")
    result["finish_reason"] = finish if type(finish) is str and finish in allowed else "unrecognized"
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            result["final_content_characters"] = len(content)
            result["final_content_sha256"] = hashlib.sha256(
                content.encode("utf-8", errors="surrogatepass")).hexdigest()
        result["reasoning_present"] = bool(message.get("reasoning") or message.get("reasoning_content"))
    return result


class RecordingClient(VLLMClient):
    """Attach allowlisted failure diagnostics without changing the existing parser."""
    def _parse_response(self, request, body):
        try:
            return super()._parse_response(request, body)
        except InvalidResponseError as exc:
            exc.response_diagnostics = response_diagnostics(body)
            exc.raw_output = None
            raise


async def run_profile(root, profile, *, client_factory=RecordingClient,
                      preflight=suite.shared.vllm_preflight):
    _, instances = execution(root)
    config = panel.configurations()[profile]
    output = root / profile
    suite.require(not output.exists(), "Profile already attempted; no resume or retry.")
    output.mkdir()
    write(output / "manifest.json", {"profile": profile, "model": asdict(config),
          "execution_sha256": suite.digest(root / "execution.json"),
          "planned_episodes": 64, "maximum_calls": 160, "fresh": True})
    attempts, client = [], None
    state = {"status": "running", "profile": profile, "calls": 0}
    status(output, **state)
    with (output / "attempts.jsonl").open("x") as stream:
        try:
            await preflight([SimpleNamespace(model=config)])
            client = client_factory(**config.client_kwargs())
            await asyncio.wait_for(client.initialize(), config.timeout_seconds)
            for instance in instances:
                history = []
                while (request := canonical.public_request(instance, history)) is not None:
                    suite.require(len(attempts) < 160 and len(history) < instance["query_budget"],
                                  "Preregistered call budget exceeded.")
                    execution(root)
                    diagnostics, exception_type, started = None, None, time.perf_counter()
                    try:
                        response = await asyncio.wait_for(client.predict(request), config.timeout_seconds)
                        diagnostics = response_diagnostics(response.raw_output)
                        record = suite.response_record(response, request, config)
                    except BaseException as exc:
                        if isinstance(exc, TimeoutError):
                            exc = ClientTimeoutError("Pilot outer request timeout")
                        exception_type = type(exc).__name__[:80]
                        diagnostics = getattr(exc, "response_diagnostics", None) or diagnostics or (
                            response_diagnostics(getattr(exc, "raw_output", None)))
                        record = {"status": "failure", "selected_option_id": None,
                                  "resolved_model": None, "usage": suite.safe_usage(getattr(exc, "usage", None)),
                                  **suite.failure(exc)}
                        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                            record["fatal"] = True
                    row = {"instance_id": instance["id"], "step": len(history),
                           "request": asdict(request), "response_diagnostics": diagnostics,
                           "exception_type": exception_type,
                           "latency_seconds": time.perf_counter() - started, **record}
                    suite.append_row(stream, row)
                    stream.flush()
                    os.fsync(stream.fileno())
                    attempts.append(suite.json_value(row))
                    status(output, **{**state, "calls": len(attempts)})
                    if record["status"] == "failure":
                        if record["fatal"]:
                            state.update(status="aborted", diagnostic_code=record["diagnostic_code"])
                        break
                    history.append(record["selected_option_id"])
                if state["status"] != "running":
                    break
            if state["status"] == "running":
                state["status"] = ("completed_with_failures" if any(
                    a["status"] == "failure" for a in attempts) else "completed")
        except BaseException as exc:
            state.update(status="aborted", diagnostic_code=suite.failure(exc)["diagnostic_code"],
                         exception_type=type(exc).__name__[:80])
        finally:
            if client is not None:
                try:
                    await client.aclose()
                except BaseException as exc:
                    state.update(status="aborted", diagnostic_code="shutdown_failed",
                                 exception_type=type(exc).__name__[:80])
            rows = canonical.replay(instances, attempts, config, native_action=False)
            panel.invariance.write_rows(output / "scores.jsonl", rows)
            write(output / "summary.json", suite.summarize(rows))
            anomalies = []
            for task in sorted({i["task"] for i in instances}):
                subset = [r for r in rows if r["task"] == task]
                correct = sum(bool(r.get("correct", r.get("optimal", False))) for r in subset)
                if not correct:
                    identifiers = {r["instance_id"] for r in subset}
                    own_attempts = [a for a in attempts if a["instance_id"] in identifiers]
                    anomalies.append({"task": task, "planned": len(subset),
                                      "all_wrong_or_failed": True,
                                      "instance_ids": [r["instance_id"] for r in subset],
                                      "feasible_episodes": sum(r["feasible"] for r in subset),
                                      "failure_diagnostics": dict(Counter(
                                          a["diagnostic_code"] for a in own_attempts
                                          if a["status"] == "failure")),
                                      "selected_ids": dict(Counter(
                                          a["selected_option_id"] for a in own_attempts
                                          if a["status"] == "success")),
                                      "exact_answer_counts": dict(Counter(
                                          r["answer"] for r in subset if r["kind"] == "exact")),
                                      "inspection": "Canonical replay validates action mapping and own "
                                                    "history. Independent references were verified. "
                                                    "Allowlisted response diagnostics support review; "
                                                    "no retry or replacement permitted."})
            write(output / "anomalies.json", anomalies)
            state["calls"] = len(attempts)
            status(output, **state)
    return state


def unattempted_profile(root, profile, reason):
    _, instances = execution(root)
    output = root / profile
    if output.exists():
        return
    output.mkdir()
    config = panel.configurations()[profile]
    write(output / "manifest.json", {"profile": profile, "model": asdict(config),
          "execution_sha256": suite.digest(root / "execution.json"),
          "planned_episodes": 64, "maximum_calls": 160, "fresh": True, "unattempted": True})
    (output / "attempts.jsonl").touch(exist_ok=False)
    rows = canonical.replay(instances, [], config, native_action=False)
    panel.invariance.write_rows(output / "scores.jsonl", rows)
    write(output / "summary.json", suite.summarize(rows))
    status(output, status="unattempted", reason=reason, calls=0)


def wait_ready(server, root, *, timeout_seconds=1200,
               preflight=suite.shared.vllm_preflight, clock=time.monotonic, sleep=time.sleep):
    deadline, last_error = clock() + timeout_seconds, None
    while True:
        suite.require(server.poll() is None,
                      f"Qwen27B server exited during startup; last preflight error: {last_error}")
        try:
            asyncio.run(preflight([
                SimpleNamespace(model=panel.configurations()["qwen27_grammar"])]))
            return
        except Exception as exc:
            last_error = {"error_type": type(exc).__name__, "message": str(exc)[:500]}
            status(root, status="server_loading", last_preflight_error=last_error)
            if clock() >= deadline:
                raise RuntimeError(
                    f"Qwen27B startup readiness timeout; last preflight error: "
                    f"{last_error['error_type']}: {last_error['message']}") from None
            sleep(5)


def supervisor(root):
    value, _ = execution(root)
    suite.require(not (root / "supervisor-start.json").exists(), "Supervisor cannot resume.")
    checkpoint = verify_artifacts(Path(value["artifacts"]))
    port_free()
    # This unchanged shared API is invoked by the parent, never by offline preparation.
    with suite.lane_locks(RESOURCES):
        protected = [process(pid) for pid in PROTECTED]
        observation = suite.read_json(Path(value["panel"]) / "resource-recovery-observation.json")
        previous = {i["pid"]: i for i in observation["services"]}
        services = {}
        for lane, (pid, _) in OLD.items():
            info = process(pid)
            suite.require(info["argv"] == previous[pid]["argv"]
                          and info["cuda_environment"] == previous[pid]["cuda_environment"],
                          "Old service configuration changed.")
            info["process_tree"] = descendants(pid)
            info["restore_command"] = shlex.join(["env", *(
                f"{k}={v}" for k, v in info["cuda_environment"].items()), *info["argv"]])
            services[lane] = info
        inventory = gpu_inventory()
        targets = validate_mapping(services, inventory)
        configs = suite.shared.model_configs(models=RESOURCES)
        asyncio.run(suite.shared.vllm_preflight([SimpleNamespace(model=c) for c in configs.values()]))
        write(root / "supervisor-start.json", {
            "pid": os.getpid(), "started_at_utc": now(), "services": services,
            "protected": protected, "inventory": inventory, "target_gpu_uuids": targets,
            "zero_running_waiting_preflight": True, "locks": list(RESOURCES)})
        server, owned, touched = None, [], False
        try:
            # Recheck occupancy and process/GPU identities immediately before any stop.
            asyncio.run(suite.shared.vllm_preflight([SimpleNamespace(model=c) for c in configs.values()]))
            for info in services.values():
                suite.require(alive(info), "Old server process identity changed.")
            validate_mapping(services, gpu_inventory())
            owned = [p for info in services.values() for p in info["process_tree"]]
            touched = True
            suite.require(stop_tree(owned), "Original selected service processes did not exit.")
            inventory = gpu_inventory()
            suite.require(not any(uuid in targets for _, uuid, _ in inventory["processes"]),
                          "Target GPUs remain occupied; refusing new server.")
            status(root, status="gpus_freed", target_gpu_uuids=targets)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(targets)
            command = server_command(checkpoint["snapshot"])
            with (root / "server.log").open("xb") as log:
                server = subprocess.Popen(command, env=env, cwd=REPO, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            owned = [process(server.pid)]
            write(root / "server-process.json", {"identity": owned[0], "argv": command,
                  "gpu_uuids": targets, "locks_inherited": False,
                  "lock_owner_supervisor_pid": os.getpid()})
            wait_ready(server, root)
            owned = descendants(server.pid)
            inventory = gpu_inventory()
            children = {i["pid"] for i in owned}
            gpu_rows = [(int(pid), uuid) for pid, uuid, _ in inventory["processes"]
                        if int(pid) in children]
            suite.require({uuid for _, uuid in gpu_rows} == set(targets),
                          "New server has unexpected physical GPU mapping.")
            suite.require(all(alive(p) for p in protected), "Protected service identity changed.")
            status(root, status="model_ready", server_pid=server.pid, target_gpu_uuids=targets)
            for profile in ("qwen27_grammar", "qwen27_thinking"):
                state = asyncio.run(run_profile(root, profile))
                suite.require(state["status"].startswith("completed"),
                              "Pilot aborted; inspect preserved all-planned results.")
            status(root, status="completed", profiles=["qwen27_grammar", "qwen27_thinking"])
        except BaseException as exc:
            status(root, status="blocked_or_aborted", error_type=type(exc).__name__,
                   message=str(exc), resources_touched=touched)
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if server is not None and server.poll() is None:
                known = {p["pid"]: p for p in owned}
                known.update({p["pid"]: p for p in descendants(server.pid)})
                owned = list(known.values())
            if touched:
                clean = stop_tree(owned)
                while not clean:
                    status(root, status="cleanup_blocked_locks_retained",
                           surviving_pids=[i["pid"] for i in owned if alive(i)])
                    time.sleep(30)
                    clean = not any(alive(i) for i in owned)
                inventory = gpu_inventory()
                free = not any(uuid in targets for _, uuid, _ in inventory["processes"])
                while not free:
                    status(root, status="gpu_occupancy_cleanup_blocked_locks_retained",
                           note="Uncaptured or unrelated target-GPU process remains; no unsafe killing.")
                    time.sleep(30)
                    inventory = gpu_inventory()
                    free = not any(uuid in targets for _, uuid, _ in inventory["processes"])
                write(root / "handoff.json", {"at_utc": now(), "owned_processes_stopped": True,
                      "target_gpus_free": free, "protected_services_alive": all(alive(p) for p in protected),
                      "ready_after_supervisor_exit": free, "old_services_restored": False,
                      "note": "Shared locks release on supervisor exit; parent coordinates trained pilot."})
            if server is not None:
                server.wait()
            for profile in ("qwen27_grammar", "qwen27_thinking"):
                unattempted_profile(root, profile, "Supervisor terminated before this profile.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "register", "launch", "qwen9", "dry-run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--panel", type=Path, default=panel.DEFAULT_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    if args.command == "download":
        result = download(args.artifacts)
    elif args.command == "register":
        result = register(args.root, args.panel, args.artifacts)
    elif args.command == "dry-run":
        registration, instances = load_panel(args.panel)
        result = {"model_calls": 0, "cases": len(instances),
                  "maximum_calls": sum(i["query_budget"] for i in instances),
                  "independent_validation": canonical.tasks.validate_instances(instances),
                  "source_guard": True, "checkpoint_revision": REVISION,
                  "server_command": server_command(args.artifacts.resolve() / "snapshot")}
    else:
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Supervisor signal {signum}")
        signal.signal(signal.SIGTERM, interrupt)
        if args.command == "launch":
            supervisor(local(args.root))
            result = {"status": "supervisor_exited", "root": str(args.root)}
        else:
            with suite.lane_locks(("qwen9",)):
                result = asyncio.run(run_profile(local(args.root), "qwen9_thinking"))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
