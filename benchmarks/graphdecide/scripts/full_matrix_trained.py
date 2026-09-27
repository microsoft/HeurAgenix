#!/usr/bin/env python3
"""Fresh full-canonical Decider/Kev registration and parent-gated execution.

The historical pilot is a read-only source of pinned implementations and weights,
never a source of selected cases, answers, or scores.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import evaluate_graph_release as evaluation
from scripts import launch_trained_jevlike_parent as parent
from scripts import run_trained_jevlike_pilot as pilot
from scripts import trained_jevlike_adapter as adapter
from scripts.prepare_trained_jevlike import PINS, SOURCES, digest

suite, release = evaluation.suite, evaluation.release
DEFAULT_ASSETS = REPO / "output/trained-jevlike-pilot-20260924-v1"
DEFAULT_BUNDLE = REPO / "output/graphbench-portable-release-20260924-v2"
DEFAULT_ROOT = REPO / "output/full-matrix-trained-20260925-v1"
CANDIDATES = ("decider", "kev")
SPLIT = "structural_challenge"
RESOURCES = ("qwen08", "qwen2")
COUNTS = (960, 1920)


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(suite.canonical(value).encode()).hexdigest()


def write(path, value):
    suite.write_json(Path(path), value)


def status(path, value):
    """Atomic, durable status replacement; ledgers and registrations are exclusive."""
    path = Path(path)
    pending = path.with_suffix(".next")
    with pending.open("w") as stream:
        stream.write(suite.canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def packages():
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


def canonical_panel(bundle):
    if bundle.resolve() != DEFAULT_BUNDLE.resolve():
        raise RuntimeError("This runner requires the existing canonical v2 release")
    release.verify(bundle)
    manifest = release.manifest_at(bundle)
    if "explicit_actions" not in manifest["protocols"]:
        raise RuntimeError("Canonical explicit_actions protocol is absent")
    instances = release.labeled_instances(bundle, SPLIT, manifest)
    if (len(instances), sum(i["query_budget"] for i in instances)) != COUNTS:
        raise RuntimeError("Expected all 960 canonical episodes / maximum 1920 calls")
    if len({i["id"] for i in instances}) != COUNTS[0]:
        raise RuntimeError("Duplicate canonical instance")
    return instances


def check_assets(assets, prior):
    if assets.resolve() != DEFAULT_ASSETS.resolve():
        raise RuntimeError("This runner reuses only the existing pinned trained assets")
    if read(assets / "artifacts.json") != prior["models"]:
        raise RuntimeError("Historical checkpoint inventory changed")
    pilot.verify_artifacts(prior["models"])
    for name, revision in SOURCES.items():
        checkout = assets / "upstream" / name
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip()
        if actual != revision or dirty:
            raise RuntimeError("Upstream checkout is not the clean pinned revision")
        if read(assets / f"cpu-smoke-{name}/status.json")["status"] != "complete":
            raise RuntimeError("Historical authored CPU preflight is missing")
    expected_deps = {k for k in prior["code_sha256"]
                     if Path(k).is_relative_to(assets.relative_to(REPO) / "deps")}
    actual_deps = {str(p.relative_to(REPO)) for p in (assets / "deps").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts}
    if actual_deps != expected_deps:
        raise RuntimeError("Isolated dependency file inventory changed")


def register(root, assets, bundle):
    root, assets, bundle = root.resolve(), assets.resolve(), bundle.resolve()
    if root.exists():
        raise RuntimeError("Output exists; registration cannot overwrite or resume")
    if not all(p.is_relative_to(REPO) for p in (root, assets, bundle)):
        raise RuntimeError("All registration paths must remain in this repository")
    prior_path = assets / "benchmark-registration-parent-locks.json"
    if digest(prior_path) != read(assets / "benchmark-registration-parent-locks.sha256"):
        raise RuntimeError("Historical registration hash changed")
    prior = read(prior_path)
    entry = assets / "parent-entry-registration.json"
    if (digest(entry) != read(assets / "parent-entry-registration.sha256") or
            read(entry)["parent_entry_sha256"] != digest(parent.__file__)):
        raise RuntimeError("Historical standard-lock parent helper changed")
    pilot.verify_files(prior["code_sha256"])
    check_assets(assets, prior)
    if packages() != prior["packages"] or Path(sys.executable).resolve() != Path(pilot.PYTHON).resolve():
        raise RuntimeError("Use the unchanged pinned shared Python environment")
    instances = canonical_panel(bundle)
    hashes = {**prior["code_sha256"], **evaluation.code_hashes(),
              "scripts/launch_trained_jevlike_parent.py": digest(parent.__file__),
              "scripts/full_matrix_trained.py": digest(__file__)}
    if any(hashes[k] != v for k, v in prior["code_sha256"].items()):
        raise RuntimeError("Cannot replace a historical source gate")
    root.mkdir(parents=True)
    write(root / "panel.json", [i["id"] for i in instances])
    write(root / "initial-requests.json", [
        asdict(evaluation.public_request(i, [])) for i in instances])
    for name, expected in hashes.items():
        source = REPO / name
        snapshot = root / "frozen-source" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        snapshot.chmod(0o444)
        if digest(snapshot) != expected:
            raise RuntimeError("Source changed during snapshot")
    registration = {
        "schema_version": 1, "registered_at": time.time(),
        "scope": "full_canonical_structural_challenge", "protocol": "explicit_actions",
        "assets": str(assets), "bundle": str(bundle), "split": SPLIT,
        "historical_registration_sha256": digest(prior_path),
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "panel_sha256": digest(root / "panel.json"),
        "initial_requests_sha256": digest(root / "initial-requests.json"),
        "code_sha256": hashes, "models": prior["models"], "packages": packages(),
        "python": pilot.PYTHON, "python_sha256": digest(Path(pilot.PYTHON).resolve()),
        "candidates": list(CANDIDATES), "scheduled_episodes_per_model": COUNTS[0],
        "maximum_calls_per_model": COUNTS[1], "context_cap": adapter.CONTEXT_LIMIT,
        "loading_timeout_seconds": 600, "inference_timeout_seconds": 180,
        "model_settings": {
            "decider": {"dtype": "bfloat16", "temperature": 1.164},
            "kev": {"dtype": "float32", "temperature": 2.406050072164233}},
        "policy": {"retry": False, "repair": False, "resume": False, "truncate": False,
                   "own_history": True, "all_scheduled_in_denominator": True,
                   "private_labels_sent": False, "score_based_selection": False,
                   "unsupported_is_failure": True, "models_sequential": True},
        "resources": {"standard_lock_models": list(RESOURCES),
                      "physical_gpu": 0, "require_empty_physical_gpus": [0, 1],
                      "explicit_parent_handoff": True},
        "dependency_deviations": prior["dependency_deviations"],
        "request_identity": "initial requests frozen; subsequent public_request(instance, own_history) "
                            "generated by frozen sources and durably logged before dispatch"}
    write(root / "registration.json", registration)
    write(root / "registration.sha256", digest(root / "registration.json"))
    for name in ("registration.json", "registration.sha256", "panel.json", "initial-requests.json"):
        (root / name).chmod(0o444)
    pilot.verify_files(hashes)
    status(root / "status.json", {"status": "registered_awaiting_parent_handoff",
                                 "benchmark_calls": 0, "gpu_workers_started": False})
    return registration


def verify_registration(root, *, assets=True):
    registration = read(root / "registration.json")
    if digest(root / "registration.json") != read(root / "registration.sha256"):
        raise RuntimeError("Frozen registration changed")
    pilot.verify_files(registration["code_sha256"])
    for name, expected in registration["code_sha256"].items():
        snapshot = root / "frozen-source" / name
        if snapshot.is_symlink() or digest(snapshot) != expected:
            raise RuntimeError("Frozen source snapshot changed")
    for name, key in (("panel.json", "panel_sha256"),
                      ("initial-requests.json", "initial_requests_sha256")):
        if digest(root / name) != registration[key]:
            raise RuntimeError("Frozen request identity changed")
    if (packages() != registration["packages"] or
            digest(Path(registration["python"]).resolve()) != registration["python_sha256"] or
            Path(sys.executable).resolve() != Path(registration["python"]).resolve()):
        raise RuntimeError("Frozen runtime changed")
    bundle = Path(registration["bundle"])
    if digest(bundle / "manifest.json") != registration["bundle_manifest_sha256"]:
        raise RuntimeError("Canonical bundle manifest changed")
    instances = canonical_panel(bundle)
    if ([i["id"] for i in instances] != read(root / "panel.json") or
            suite.json_value([asdict(evaluation.public_request(i, [])) for i in instances])
            != read(root / "initial-requests.json")):
        raise RuntimeError("Full canonical request replay mismatch")
    prior_path = Path(registration["assets"]) / "benchmark-registration-parent-locks.json"
    if digest(prior_path) != registration["historical_registration_sha256"]:
        raise RuntimeError("Historical source registration changed")
    if assets:
        check_assets(Path(registration["assets"]), read(prior_path))
    return registration, instances


class Worker:
    """Existing public-only worker, with deadlines covering both pipe directions."""

    def __init__(self, assets, candidate, output):
        self.log = (output / "worker.log").open("x")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "TRAINED_JEVLIKE_SUPERVISOR_PID": str(os.getpid())}
        try:
            self.process = subprocess.Popen(
                [pilot.PYTHON, str(REPO / "scripts/trained_jevlike_worker.py"),
                 "--root", str(assets), "--candidate", candidate, "--device", "cuda"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                env=env, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        try:
            self.buffer = b""
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            write(output / "worker-process.json", {"pid": self.process.pid, "device": "cuda"})
        except BaseException:
            self.close()
            raise

    @staticmethod
    def remaining(deadline):
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Finite worker deadline exceeded")
        return value

    def receive(self, seconds=None, *, deadline=None):
        deadline = deadline if deadline is not None else time.monotonic() + seconds
        while b"\n" not in self.buffer:
            ready, _, _ = select.select([self.process.stdout], [], [], self.remaining(deadline))
            if not ready:
                raise TimeoutError("Worker response deadline exceeded")
            part = os.read(self.process.stdout.fileno(), 65536)
            if not part:
                raise RuntimeError("Worker exited before a complete response")
            self.buffer += part
            if len(self.buffer) > 16 * 1024 * 1024:
                raise RuntimeError("Worker response exceeded bounded wire size")
        line, self.buffer = self.buffer.split(b"\n", 1)
        result = json.loads(line)
        if not isinstance(result, dict):
            raise RuntimeError("Invalid worker response")
        return result

    def predict(self, request, seconds):
        deadline = time.monotonic() + seconds
        payload = memoryview((json.dumps(asdict(request), allow_nan=False) + "\n").encode())
        while payload:
            _, ready, _ = select.select([], [self.process.stdin], [], self.remaining(deadline))
            if not ready:
                raise TimeoutError("Worker input deadline exceeded")
            payload = payload[os.write(self.process.stdin.fileno(), payload):]
        return self.receive(deadline=deadline)

    def close(self):
        try:
            self.process.stdin.close()
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=5)
            self.process.stdout.close()
        finally:
            self.log.close()


def model_id(candidate):
    return "@".join(PINS[candidate])


def failure(error):
    return {"status": "failure", "selected_option_id": None, "resolved_model": None,
            "fatal": True, "strict_status": "failure",
            "diagnostic_code": getattr(error, "diagnostic_code", type(error).__name__),
            "error": str(error)[:1200],
            "action_observation": getattr(error, "action_observation", {
                "eligible": False, "selected_option_id": None}),
            "usage": {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}}


def score_rows(instances, attempts, candidate):
    scores = evaluation.replay(instances, attempts, SimpleNamespace(model=model_id(candidate)),
                               native_action=False)
    for row in scores:
        row.update(scheduled=True, unattempted=row["query_count"] == 0)
    return scores


def runtime_gate(ready, candidate, registration):
    expected = registration["model_settings"][candidate]
    runtime = ready.get("runtime", {})
    if (ready.get("status") != "ready" or runtime.get("candidate") != candidate or
            runtime.get("device") != "cuda" or runtime.get("context_limit") != adapter.CONTEXT_LIMIT or
            runtime.get("dtype") != expected["dtype"] or
            not math.isclose(runtime.get("temperature", -1), expected["temperature"], abs_tol=1e-12)):
        raise RuntimeError("Worker loading or frozen native configuration gate failed")


def evaluate(root, candidate, instances, registration, *, blocked=None, worker_factory=Worker):
    output = root / f"run-{candidate}"
    output.mkdir(exist_ok=False)
    write(output / "scheduled.json", [i["id"] for i in instances])
    attempts, worker, fatal, interrupted = [], None, blocked, False
    started = time.monotonic()
    status(output / "status.json", {"status": "loading", "calls": 0,
                                  "scheduled_episodes": len(instances)})
    with (output / "attempts.jsonl").open("x") as stream, (
            output / "dispatch.jsonl").open("x") as dispatch:
        try:
            if blocked:
                raise RuntimeError(blocked)
            pilot.verify_files(registration["code_sha256"])
            worker = worker_factory(Path(registration["assets"]), candidate, output)
            ready = worker.receive(registration["loading_timeout_seconds"])
            write(output / "runtime.json", ready)
            runtime_gate(ready, candidate, registration)
            for instance in instances:
                history = []
                while (request := evaluation.public_request(instance, history)) is not None:
                    pilot.verify_files(registration["code_sha256"])
                    row = {"instance_id": instance["id"], "step": len(history),
                           "request": asdict(request), "call_index": len(attempts),
                           "request_sha256": identity(asdict(request))}
                    suite.append_row(dispatch, row)
                    tick = time.monotonic()
                    try:
                        backend = worker.predict(request, registration["inference_timeout_seconds"])
                        if backend.get("status") != "success":
                            row.update(failure(RuntimeError(backend.get("error", "Worker failure"))),
                                       fatal=backend.get("fatal", True),
                                       diagnostic_code=backend.get("error_type", "worker_failure"),
                                       backend=backend)
                        else:
                            response, diagnostic = adapter.validate_result(
                                request, backend["result"], model_id(candidate))
                            row.update(status="success", selected_option_id=response.selected_option_id,
                                       resolved_model=model_id(candidate), fatal=False,
                                       probabilities=response.probabilities,
                                       probability_kind=response.probability_kind,
                                       usage=asdict(response.usage), backend=backend, **diagnostic)
                    except BaseException as error:
                        row.update(failure(error))
                        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                    row["latency_seconds"] = time.monotonic() - tick
                    suite.append_row(stream, row)
                    attempts.append(suite.json_value(row))
                    status(output / "status.json", {"status": "running", "calls": len(attempts),
                                                  "instance_id": instance["id"]})
                    if row["status"] != "success":
                        if row["fatal"]:
                            fatal = row["diagnostic_code"]
                        break
                    history.append(row["selected_option_id"])
                if fatal:
                    break
        except BaseException as error:
            fatal = f"{type(error).__name__}: {str(error)[:1200]}"
            interrupted |= isinstance(error, (KeyboardInterrupt, SystemExit))
        finally:
            if worker is not None:
                try:
                    worker.close()
                except BaseException as error:
                    fatal = f"worker_cleanup_failed: {error}"
                    interrupted |= isinstance(error, (KeyboardInterrupt, SystemExit))
            scores = score_rows(instances, attempts, candidate)
            with (output / "scores.jsonl").open("x") as scores_stream:
                for row in scores:
                    suite.append_row(scores_stream, row)
            write(output / "summary.json", suite.summarize(scores))
            status(output / "status.json", {
                "status": "aborted" if fatal else (
                    "completed_with_failures" if any(a["status"] == "failure" for a in attempts)
                    else "completed"),
                "fatal": fatal, "calls": len(attempts), "scheduled_episodes": len(instances),
                "unattempted_episodes": sum(s["unattempted"] for s in scores),
                "failure_count": sum(a["status"] == "failure" for a in attempts),
                "worker_released": worker is None or worker.process.poll() is not None,
                "wall_seconds": time.monotonic() - started})
    verify_terminal(root, candidate, instances)
    return interrupted


def verify_terminal(root, candidate, instances):
    output = root / f"run-{candidate}"
    attempts = suite.read_rows(output / "attempts.jsonl")
    dispatches = suite.read_rows(output / "dispatch.jsonl")
    if len(attempts) != len(dispatches):
        raise RuntimeError("Dispatch ledger has an unresolved call; never retry it")
    for index, (row, dispatch) in enumerate(zip(attempts, dispatches, strict=True)):
        if (dispatch["call_index"] != index or
                any(row.get(k) != v for k, v in dispatch.items()) or
                row["request_sha256"] != identity(row["request"])):
            raise RuntimeError("Durable dispatch/request identity mismatch")
        if row["status"] == "success":
            response, _ = adapter.validate_result(
                adapter.request_from_dict(row["request"]), row["backend"]["result"], model_id(candidate))
            if (response.selected_option_id != row["selected_option_id"] or
                    response.probabilities != row["probabilities"] or
                    asdict(response.usage) != row["usage"]):
                raise RuntimeError("Backend result replay mismatch")
    scores = score_rows(instances, attempts, candidate)
    outcome = read(output / "status.json")
    if (read(output / "scheduled.json") != [i["id"] for i in instances] or
            scores != suite.read_rows(output / "scores.jsonl") or
            suite.summarize(scores) != read(output / "summary.json") or
            outcome["calls"] != len(attempts) or
            outcome["scheduled_episodes"] != len(instances) or
            outcome["failure_count"] != sum(a["status"] == "failure" for a in attempts) or
            outcome["unattempted_episodes"] != sum(s["unattempted"] for s in scores) or
            len(attempts) > sum(i["query_budget"] for i in instances) or
            outcome["worker_released"] is not True or
            outcome["status"] not in ("completed", "completed_with_failures", "aborted")):
        raise RuntimeError("Terminal full-denominator replay failed")
    if outcome["status"] == "completed" and (
            len(attempts) != sum(i["query_budget"] for i in instances) or
            outcome["failure_count"] or outcome["unattempted_episodes"] or outcome["fatal"]):
        raise RuntimeError("Invalid clean completion claim")
    return {"status": "verified", "candidate": candidate, "episodes": len(scores),
            "calls": len(attempts), "outcome": outcome["status"],
            "sha256": {name: digest(output / name) for name in (
                "scheduled.json", "dispatch.jsonl", "attempts.jsonl", "scores.jsonl",
                "summary.json", "status.json")}}


def launch(root, handoff):
    if any(root.glob("run-*")) or (root / "launch.json").exists():
        raise RuntimeError("Prior execution exists; no retries or resume")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Parent launch requires CUDA_VISIBLE_DEVICES=0")
    registration, instances = verify_registration(root)
    pilot.verify_handoff(handoff, root / "registration.json")
    with suite.lane_locks(RESOURCES):
        before = parent.inspect_empty_devices()
        # Recheck mutable source gates after acquiring the unchanged shared locks.
        pilot.verify_files(registration["code_sha256"])
        write(root / "gpu-admission.json", before)
        write(root / "launch.json", {"pid": os.getpid(), "at": time.time(),
                                     "handoff": read(handoff),
                                     "registration_sha256": digest(root / "registration.json")})
        prior_visible = os.environ["CUDA_VISIBLE_DEVICES"]
        prior_handoff = os.environ.get("TRAINED_JEVLIKE_GPU_HANDOFF")
        os.environ["CUDA_VISIBLE_DEVICES"] = before["physical_gpus"][0]["uuid"]
        os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"] = "approved"
        interrupted, blocked = False, None
        try:
            for candidate in CANDIDATES:
                interrupted |= evaluate(root, candidate, instances, registration, blocked=blocked)
                if interrupted:
                    blocked = "Parent interrupted; remaining model not dispatched"
                if not read(root / f"run-{candidate}/status.json")["worker_released"]:
                    blocked = "Prior worker release failed; remaining model not dispatched"
            pilot.verify_files(registration["code_sha256"])
            pilot.verify_artifacts(registration["models"])
            proof = {name: verify_terminal(root, name, instances) for name in CANDIDATES}
            write(root / "terminal-verification.json", proof)
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior_visible
            if prior_handoff is None:
                os.environ.pop("TRAINED_JEVLIKE_GPU_HANDOFF", None)
            else:
                os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"] = prior_handoff
            after = parent.inspect_empty_devices()
            write(root / "gpu-release.json", after)
            status(root / "status.json", {
                "status": "terminal_verified" if (root / "terminal-verification.json").exists()
                          else "aborted_unverified",
                "gpu_workers_released": True,
                "outcomes": {name: read(root / f"run-{name}/status.json")
                             for name in CANDIDATES if (root / f"run-{name}/status.json").exists()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "dry-run", "run", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--handoff", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "register":
        register(root, args.assets, args.bundle)
    elif args.command == "run":
        if args.handoff is None:
            parser.error("run requires --handoff from the resource-owning parent")
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        launch(root, args.handoff)
    else:
        registration, instances = verify_registration(root)
        if args.command == "verify":
            proof = {name: verify_terminal(root, name, instances) for name in CANDIDATES}
            if proof != read(root / "terminal-verification.json"):
                raise RuntimeError("Terminal proof changed")
            print(json.dumps(proof, indent=2))
        else:
            print(json.dumps({"status": "offline_verified", "benchmark_calls": 0,
                              "episodes_per_model": len(instances),
                              "maximum_calls_per_model": sum(i["query_budget"] for i in instances),
                              "initial_requests": len(read(root / "initial-requests.json")),
                              "source_hash_gates": len(registration["code_sha256"]),
                              "registration_sha256": digest(root / "registration.json")},
                             indent=2))


if __name__ == "__main__":
    main()
