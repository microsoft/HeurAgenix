#!/usr/bin/env python3
"""One-shot, bounded four-GPU coordinator with identity-checked restoration.

Registration/metadata preparation never acquire GPU leases. Launch/recover require
explicit access to the *unchanged* standard lock namespace; the flag does not
override execution-environment restrictions.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import stronger_local_runtime_v1 as run

admission, suite, runtime = run.admission, run.suite, run.runtime
resources = admission.resources
LANES = admission.LANES
DOWNLOAD_SECONDS = 4 * 3600
LEASE_WAIT_SECONDS = 3600
STARTUP_SECONDS = 1200
RESTORE_SECONDS = 1200
POLL_SECONDS = 5
ENV_NAMES = {
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TMPDIR", "TEMP", "TMP",
}
ENV_PREFIXES = ("CUDA_", "NCCL_", "VLLM_", "TORCH_")


def relevant_env(name):
    return name in ENV_NAMES or name.startswith(ENV_PREFIXES)


def restoration_environment(identity):
    entries = Path(f"/proc/{identity['pid']}/environ").read_bytes().split(b"\0")
    captured = {}
    for entry in entries:
        if not entry:
            continue
        key, value = entry.decode().split("=", 1)
        if relevant_env(key):
            suite.require(not value or not any(
                word in key for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")),
                "Secret-bearing restoration setting cannot be persisted.")
            captured[key] = value
    return captured


def originals():
    values = {lane: resources.service(lane) for lane in LANES}
    for original in values.values():
        suite.require(Path(f"/proc/{original['pid']}").stat().st_uid == os.getuid()
                      and original["cwd"] == str(REPO), "Original is not user/project-owned.")
        original["restore_environment"] = restoration_environment(original)
    return values


def idle_originals():
    configs = suite.shared.model_configs(models=tuple(LANES))
    asyncio.run(suite.shared.vllm_preflight(
        [SimpleNamespace(model=c) for c in configs.values()]))


def snapshot_originals(root, filename):
    values = originals()
    inventory = runtime.gpu_inventory()
    targets = admission.four_gpu_mapping(values, inventory)
    idle_originals()
    suite.write_json(root / filename, {
        "at_utc": runtime.now(), "services": values, "inventory": inventory,
        "target_gpu_uuids": targets, "models_verified_idle": True})
    return values, targets


def child_environment(root, **updates):
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Only child scratch/cache locations change; coordinator standard locks do not.
    env.update(TMPDIR=str(work), HF_HOME=str(work / "hf-home"),
               XDG_CACHE_HOME=str(work / "cache"), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_DISABLE_TELEMETRY="1")
    env.update(updates)
    return env


class Children:
    def __init__(self, root):
        self.root, self.entries = root, []

    def start(self, command, environment, label):
        suite.require(not (self.root / f"{label}-launch-intent.json").exists(),
                      "Child launch already attempted; retry forbidden.")
        suite.write_json(self.root / f"{label}-launch-intent.json", {
            "argv": command, "at_utc": runtime.now(), "parent": runtime.process(os.getpid()),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES")})
        with (self.root / f"{label}.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = runtime.process(child.pid)
        entry = {"child": child, "identity": identity, "label": label,
                 "known": {(identity["pid"], identity["start_ticks"]): identity}}
        self.entries.append(entry)
        self.capture(entry)
        return entry

    def capture(self, entry):
        if runtime.alive(entry["identity"]):
            for identity in runtime.descendants(entry["identity"]["pid"]):
                entry["known"][(identity["pid"], identity["start_ticks"])] = identity
        run.atomic_json(self.root / f"{entry['label']}-process.json", {
            "at_utc": runtime.now(), "identity": entry["identity"],
            "known_tree": list(entry["known"].values())})

    def stop(self, entry):
        self.capture(entry)
        suite.require(runtime.stop_tree(list(entry["known"].values())), "Owned child did not stop.")
        if entry.get("child") is not None:
            entry["child"].wait(timeout=30)

    def cleanup(self):
        errors = []
        for entry in reversed(self.entries):
            try:
                self.stop(entry)
            except BaseException as exc:
                errors.append({"label": entry["label"], "error": str(exc)[:500]})
        suite.require(not errors, f"Owned-child cleanup failed: {errors}")

    def wait(self, entry, timeout, *, heartbeat=None, allow_failure=False):
        deadline = time.monotonic() + timeout
        while True:
            self.capture(entry)
            if heartbeat is not None:
                heartbeat()
            result = entry["child"].poll()
            if result is not None:
                suite.require(allow_failure or result == 0,
                              f"{entry['label']} exited {result}; see its log.")
                return result
            suite.require(time.monotonic() < deadline, f"{entry['label']} deadline exceeded.")
            runtime.status(self.root, status="downloading" if entry["label"].endswith("-download")
                           else "child_running", label=entry["label"],
                           pid=os.getpid(), child_pid=entry["identity"]["pid"],
                           deadline_seconds_remaining=max(0, deadline - time.monotonic()))
            time.sleep(POLL_SECONDS)


class Lease(Children):
    def __init__(self, root, services, targets):
        super().__init__(root)
        self.services, self.targets = services, targets

    def require_empty(self, targets=None):
        targets = self.targets if targets is None else targets
        suite.require(not any(uuid in targets for _, uuid, _ in runtime.gpu_inventory()["processes"]),
                      "Restoration blocked by unexpected GPU occupant; no unowned process killed.")

    def check_server(self, entry, expected):
        self.capture(entry)
        suite.require(runtime.alive(entry["identity"]), "Owned inference server exited.")
        known = {p["pid"] for p in entry["known"].values() if runtime.alive(p)}
        occupants = [(int(pid), uuid) for pid, uuid, _ in runtime.gpu_inventory()["processes"]
                     if uuid in self.targets]
        suite.require(all(pid in known and uuid in expected for pid, uuid in occupants),
                      "Unexpected GPU occupant during execution; no unowned process killed.")
        suite.require({uuid for _, uuid in occupants} == set(expected),
                      "Inference server GPU mapping differs.")

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
                if previous.exists():
                    captured = suite.read_json(previous)
                    suite.require(runtime.alive(captured)
                                  and resources.same_identity(captured, resources.service(lane)),
                                  "Previous restoration attempt is not a matching live service.")
                    restoration[lane] = captured
                    continue
                self.require_empty([self.targets[gpu]])
                resources.port_free(port)
                env = {key: value for key, value in os.environ.items() if not relevant_env(key)}
                env.update(old["restore_environment"])
                intent = self.root / f"restore-{lane}-intent.json"
                suite.require(not intent.exists(), "Unresolved prior restoration launch intent.")
                suite.write_json(intent, {
                    "at_utc": runtime.now(), "argv": old["argv"], "cwd": old["cwd"],
                    "environment": old["restore_environment"]})
                with (self.root / f"restore-{lane}.log").open("xb") as log:
                    child = subprocess.Popen(old["argv"], cwd=old["cwd"], env=env,
                                             stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                identity = runtime.process(child.pid)
                restoration[lane] = identity
                suite.write_json(previous, identity)
            except BaseException as exc:
                blocked[lane] = {"error_type": type(exc).__name__, "message": str(exc)[:1000]}
        deadline = time.monotonic() + RESTORE_SECONDS
        ready = False
        while not blocked:
            try:
                suite.require(all(runtime.alive(identity) for identity in restoration.values()),
                              "Restored service exited before readiness.")
                idle_originals()
                current = originals()
                suite.require(all(resources.same_identity(restoration[lane], current[lane])
                                  for lane in LANES), "Restored original identity differs.")
                admission.four_gpu_mapping(current, runtime.gpu_inventory())
                restoration = current
                ready = True
                break
            except Exception as exc:
                runtime.status(self.root, status="restoring", last_error=str(exc)[:500],
                               pid=os.getpid())
                if time.monotonic() >= deadline:
                    blocked["readiness"] = {"message": str(exc)[:1000]}
                    break
                time.sleep(POLL_SECONDS)
        run.atomic_json(self.root / "restoration.json", {
            "at_utc": runtime.now(), "original_services_restored": ready,
            "services": restoration, "blocked": blocked,
            "unknown_occupants_never_stopped": True})
        runtime.status(self.root, status="restored" if ready else "restoration_blocked",
                       original_services_restored=ready, blocked=blocked, pid=os.getpid())
        suite.require(ready, f"Restoration incomplete; explicit blockers: {blocked}")


def require_standard_lock_access(allowed):
    suite.require(allowed, "Launch requires permitted access to unchanged standard lock paths; "
                  "metadata/register do not require that access.")
    # This code path is parent-executable only where external lock access is permitted.
    import tempfile
    prior = suite.read_json(admission.PRIOR_ADMISSION)
    roots = {Path(v["path"]).parent for v in prior["leases"].values()}
    suite.require(len(roots) == 1 and Path(tempfile.gettempdir()) == next(iter(roots)),
                  "Standard lock namespace changed; refusing substituted locks.")


def lock_ownership():
    import tempfile
    result = {}
    rows = [line.split() for line in Path("/proc/locks").read_text().splitlines()]
    for lane in LANES:
        path = Path(tempfile.gettempdir()) / f"jevgraphbench-paired-{os.getuid()}-{lane}.lock"
        stat = path.stat()
        inode = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
        suite.require(any(len(row) >= 6 and row[1:4] == ["FLOCK", "ADVISORY", "WRITE"]
                          and row[4] == str(os.getpid()) and row[5] == inode for row in rows),
                      "Coordinator lacks an unchanged standard lane lock.")
        result[lane] = {"path": str(path), "device_inode": inode, "owner_pid": os.getpid()}
    return result


@contextmanager
def acquire(root, *, wait_seconds=LEASE_WAIT_SECONDS):
    suite.require(not root.exists(), "Resource acquisition exists; no inference resume.")
    root.mkdir()
    deadline, manager = time.monotonic() + wait_seconds, None
    while True:
        manager = suite.lane_locks(tuple(LANES))
        try:
            manager.__enter__()
            break
        except suite.shared.Refusal as exc:
            runtime.status(root, status="waiting_for_standard_leases", pid=os.getpid())
            if time.monotonic() >= deadline:
                raise RuntimeError("Standard lease wait deadline exceeded.") from exc
            time.sleep(POLL_SECONDS)
    touched, lease = False, None
    try:
        suite.write_json(root / "standard-lock-ownership.json", lock_ownership())
        services, targets = snapshot_originals(root, "admission.json")
        for spec in run.SPECS.values():
            resources.port_free(spec["port"])
        suite.write_json(root / "coordinator-owner.json", runtime.process(os.getpid()))
        lease = Lease(root, services, targets)
        idle_originals()
        suite.require(all(resources.same_identity(old, resources.service(lane))
                          for lane, old in services.items()), "Service changed at admission.")
        admission.four_gpu_mapping(services, runtime.gpu_inventory())
        touched = True
        suite.write_json(root / "shutdown-intent.json", {"at_utc": runtime.now(),
                         "services": services, "target_gpu_uuids": targets})
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


def server_command(lane, snapshot):
    spec = run.SPECS[lane]
    command = [runtime.PYTHON, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(snapshot), "--served-model-name", spec["served_name"],
               "--host", "127.0.0.1", "--port", str(spec["port"]),
               "--tensor-parallel-size", str(spec["tensor_parallel_size"]),
               "--dtype", "bfloat16", "--max-model-len", "16384",
               "--gpu-memory-utilization", "0.90", "--max-num-seqs", "1",
               "--generation-config", "vllm"]
    if lane == "qwen38_27b_bf16":
        command += ["--language-model-only", "--reasoning-parser", "qwen3",
                    "--default-chat-template-kwargs", '{"enable_thinking": false}']
    return command


async def served_tokenization(lane, materials):
    import httpx
    config = run.configuration(lane)
    await suite.shared.vllm_preflight([SimpleNamespace(model=config)])
    local = suite.read_json(materials / lane / "tokenizer-preflight.json")
    url = config.base_url.rsplit("/v1", 1)[0] + "/tokenize"
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.post(url, json={
            "model": config.model, "messages": local["messages"], "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}})
        suite.require(response.status_code == 200, "Served tokenizer endpoint failed.")
        body = response.json()
    suite.require(body.get("tokens") == local["token_ids"], "Actual served tokenization differs.")
    return {"at_utc": runtime.now(), "model": config.model, "token_count": len(local["token_ids"]),
            "tokenizer_preflight_sha256": suite.digest(materials / lane / "tokenizer-preflight.json"),
            "actual_token_ids_match": True, "nonthinking_verified": True, "model_calls": 0}


def ready(lease, server, lane, materials, expected, timeout=STARTUP_SECONDS):
    deadline = time.monotonic() + timeout
    while True:
        lease.capture(server)
        suite.require(runtime.alive(server["identity"]), "Server exited during startup.")
        try:
            result = asyncio.run(served_tokenization(lane, materials))
        except Exception as exc:
            runtime.status(lease.root, status="server_starting", lane=lane,
                           server_pid=server["identity"]["pid"], last_error=str(exc)[:500])
            if time.monotonic() >= deadline:
                raise RuntimeError("Server startup deadline exceeded.") from exc
            time.sleep(POLL_SECONDS)
            continue
        lease.check_server(server, expected)
        suite.write_json(lease.root / f"{lane}-ready.json", result)
        return


def register(root, materials):
    root, materials = runtime.local(root), runtime.local(materials)
    suite.require(not root.exists(), "Coordinator registration exists; no overwrite.")
    root.mkdir(parents=True)
    before = run.source_hashes()
    with (root / "safety-tests.log").open("xb") as log:
        test = subprocess.run(
            [runtime.PYTHON, "-B", "-m", "unittest", "tests.test_stronger_local_execution_v1",
             "tests.test_stronger_local_preflight_v1", "-v"],
            cwd=REPO, env=child_environment(root), stdout=log, stderr=subprocess.STDOUT,
            timeout=180, check=False)
    suite.require(test.returncode == 0, "Safety tests failed; see safety-tests.log.")
    suite.require(run.source_hashes() == before, "Sources changed during safety tests.")
    run.base.invariance.freeze_sources(root, before)
    value = {
        "registered_at_utc": runtime.now(), "state": "prepared_not_launched",
        "materials": str(materials), "models": run.SPECS, "source_sha256": before,
        "maximum_download_seconds_per_model": DOWNLOAD_SECONDS,
        "maximum_lease_wait_seconds": LEASE_WAIT_SECONDS,
        "maximum_startup_seconds_per_model": STARTUP_SECONDS,
        "maximum_run_seconds_per_lane": run.MAX_SECONDS,
        "maximum_restoration_seconds": RESTORE_SECONDS,
        "safety_tests_sha256": suite.digest(root / "safety-tests.log"),
        "all_four_gpus_authorized": True, "originals_must_be_idle": True,
        "standard_locks_unchanged": True, "model_calls": 0,
        "real_vs_planned": "Registration date is actual; download/inference dates recorded on execution",
    }
    suite.write_json(root / "registration.json", value)
    suite.write_json(root / "registration.sha256", suite.digest(root / "registration.json"))
    runtime.status(root, status="prepared_not_launched", model_calls=0)
    return value


def verify_registration(root):
    value = suite.read_json(root / "registration.json")
    suite.require(suite.digest(root / "registration.json") ==
                  suite.read_json(root / "registration.sha256"), "Coordinator registration drift.")
    run.check_sources(root, value)
    suite.require(value["models"] == run.SPECS
                  and value["maximum_run_seconds_per_lane"] == run.MAX_SECONDS
                  and suite.digest(root / "safety-tests.log") == value["safety_tests_sha256"],
                  "Coordinator configuration/test evidence drift.")
    return value


def launch(root, *, allow_standard_lock_access=False):
    require_standard_lock_access(allow_standard_lock_access)
    registration = verify_registration(root)
    materials = Path(registration["materials"])
    with (root / "start.json").open("x") as stream:
        json.dump({"at_utc": runtime.now(), "identity": runtime.process(os.getpid())}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    children = Children(root)
    result, failure = {}, None
    try:
        snapshot_originals(root, "resources-before-downloads.json")
        public = [admission.validate_metadata(
            spec, admission.get_json(
                f"https://huggingface.co/api/models/{spec['model']}/revision/{spec['revision']}?blobs=true"),
            admission.get_json(f"https://huggingface.co/{spec['model']}/resolve/{spec['revision']}/config.json"),
            admission.get_json(f"https://huggingface.co/{spec['model']}/resolve/{spec['revision']}/model.safetensors.index.json"))
            for spec in run.SPECS.values()]
        import shutil
        storage = admission.storage_admission(shutil.disk_usage(REPO).free, public)
        # Count only selected existing files/partials inside the single snapshot layout.
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
            child = children.start(command, child_environment(root, CUDA_VISIBLE_DEVICES=""),
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
                server = lease.start(server_command(lane, checkpoint["snapshot"]),
                                     child_environment(root, CUDA_VISIBLE_DEVICES=",".join(expected)),
                                     f"{lane}-server")
                ready(lease, server, lane, materials, expected,
                      timeout=registration["maximum_startup_seconds_per_model"])
                runtime.status(root, status="inference", lane=lane, pid=os.getpid(),
                               server_pid=server["identity"]["pid"])
                worker = lease.start(
                    [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_runtime_v1.py"),
                     "run", "--root", str(root / lane)],
                    child_environment(root, CUDA_VISIBLE_DEVICES=""), f"{lane}-runner")
                lease.wait(worker, registration["maximum_run_seconds_per_lane"] + 120,
                           heartbeat=lambda: lease.check_server(server, expected),
                           allow_failure=True)
                verification = run.verify(root / lane)
                audit = lease.start(
                    [runtime.PYTHON, "-B", "-u", str(REPO / "scripts/stronger_local_replay_v1.py"),
                     "--root", str(root / lane), "--bundle", str(run.base.BUNDLE)],
                    child_environment(root, CUDA_VISIBLE_DEVICES=""), f"{lane}-replay")
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
        runtime.status(root, status="completed", pid=os.getpid(), lanes=result,
                       original_services_restored=True)
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        try:
            children.cleanup()
        except BaseException as exc:
            failure = {**(failure or {}), "download_cleanup_error": str(exc)[:1000]}
        # Restoration has already run on the exception path. Recheck preserved
        # failed/unattempted lanes independently too; they are not completed results.
        for lane in run.SPECS:
            lane_root = root / lane
            if not (lane_root / "registration.json").exists():
                continue
            try:
                if not (lane_root / "run").exists():
                    run.preserve_unattempted(lane_root, "Coordinator stopped before this lane started.")
                verification = run.verify(lane_root)
                if not (lane_root / "verification.json").exists():
                    command = [runtime.PYTHON, "-B", str(REPO / "scripts/stronger_local_replay_v1.py"),
                               "--root", str(lane_root), "--bundle", str(run.base.BUNDLE)]
                    with (root / f"{lane}-terminal-replay.log").open("xb") as log:
                        audit = subprocess.run(command, cwd=REPO,
                                               env=child_environment(root, CUDA_VISIBLE_DEVICES=""),
                                               stdout=log, stderr=subprocess.STDOUT,
                                               timeout=300, check=False)
                    suite.require(audit.returncode == 0, "Terminal independent replay failed.")
                    suite.write_json(lane_root / "verification.json", {
                        "runner": verification, "independent_replay_sha256":
                            suite.digest(lane_root / "independent-replay.json"),
                        "construction_gaps_sha256": suite.digest(lane_root / "construction-gaps.csv")})
                result[lane] = verification
            except BaseException as exc:
                failure = {**(failure or {}), f"{lane}_terminal_audit_error": str(exc)[:1000]}
        restoration_path = root / "resources/restoration.json"
        restoration = suite.read_json(restoration_path) if restoration_path.exists() else None
        terminal = {
            "at_utc": runtime.now(), "status": "failed" if failure else "completed",
            "lanes": result, "failure": failure,
            "original_services_restored": restoration["original_services_restored"]
            if restoration else None,
            "resources_touched": (root / "resources/shutdown-intent.json").exists(),
            "restoration": restoration,
        }
        suite.write_json(root / "terminal.json", terminal)
        runtime.status(root, **terminal)
    return terminal


def recover(root, *, allow_standard_lock_access=False):
    require_standard_lock_access(allow_standard_lock_access)
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
                "child": None, "identity": identity, "label": path.name.removesuffix("-process.json"),
                "known": {(p["pid"], p["start_ticks"]): p for p in value["known_tree"]}})
        lease.cleanup()
        lease.restore()
    return suite.read_json(resource_root / "restoration.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "verify", "launch", "recover"))
    parser.add_argument("--root", type=Path, default=run.ROOT)
    parser.add_argument("--materials", type=Path, default=run.MATERIALS)
    parser.add_argument("--allow-standard-lock-access", action="store_true",
                        help="Only use where the existing external lock namespace is permitted.")
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
