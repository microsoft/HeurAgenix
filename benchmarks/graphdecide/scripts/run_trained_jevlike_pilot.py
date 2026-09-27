#!/usr/bin/env python3
"""Bounded trained-model supervisor, with public-only workers and immutable inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
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
from scripts.prepare_trained_jevlike import digest, save, SOURCES, PINS
from scripts import trained_jevlike_adapter as adapter
from src.clients.base import DecisionOption, DecisionRequest, InvalidResponseError

PYTHON = "/fastdata/xianya/vllm-venv/bin/python"


class Worker:
    def __init__(self, root, candidate, device, output):
        self.log = (output / "worker.log").open("x")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        env["TRAINED_JEVLIKE_SUPERVISOR_PID"] = str(os.getpid())
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        self.process = subprocess.Popen(
            [PYTHON, str(REPO / "scripts/trained_jevlike_worker.py"),
             "--root", str(root), "--candidate", candidate, "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env, start_new_session=True)
        save(output / "worker-process.json", {"pid": self.process.pid, "device": device})

    def receive(self, seconds):
        readable, _, _ = select.select([self.process.stdout], [], [], seconds)
        if not readable:
            raise TimeoutError(f"Worker exceeded {seconds}s deadline")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Worker exited without a response")
        return json.loads(line)

    def predict(self, request, seconds=180):
        self.process.stdin.write(json.dumps(asdict(request), allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self.receive(seconds)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        self.log.close()


def source_manifest(root):
    paths = [REPO / "scripts" / name for name in (
        "prepare_trained_jevlike.py", "trained_jevlike_adapter.py",
        "trained_jevlike_worker.py", "run_trained_jevlike_pilot.py",
        "trained_jevlike_requirements.txt")]
    paths += [p for candidate in SOURCES for p in
              (root / "upstream" / candidate).rglob("*.py") if ".git" not in p.parts]
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(paths)}


def verify_files(hashes):
    for name, expected in hashes.items():
        if digest(REPO / name) != expected:
            raise RuntimeError(f"Frozen file changed: {name}")


def verify_artifacts(artifacts):
    for name, data in artifacts["models"].items():
        if data["revision"] != PINS[name][1]:
            raise RuntimeError("Checkpoint revision mismatch")
        for file, expected in data["sha256"].items():
            if digest(Path(data["path"]) / file) != expected:
                raise RuntimeError("Checkpoint artifact hash mismatch")


def panel(selection):
    from scripts import graph_release_api as release
    registration = json.loads((selection / "registration.json").read_text())
    cases = [json.loads(line) for line in (selection / "pilot-cases.jsonl").read_text().splitlines()]
    if digest(selection / "pilot-cases.jsonl") != registration["selection_sha256"]:
        raise RuntimeError("Shared selection hash mismatch")
    bundle = Path(registration["bundle"])
    if digest(bundle / "manifest.json") != registration["bundle_manifest_sha256"]:
        raise RuntimeError("Canonical bundle manifest mismatch")
    release.verify(bundle)
    instances = release.labeled_instances(bundle, registration["split"], release.manifest_at(bundle))
    by_id = {item["id"]: item for item in instances}
    selected = [by_id[case["instance_id"]] for case in cases]
    if len(selected) != 64 or len({item["id"] for item in selected}) != 64:
        raise RuntimeError("Expected exact shared 64-episode selection")
    return selected, registration


def register(root, selection, baseline=None):
    from scripts import evaluate_graph_release as evaluation
    instances, shared = panel(selection)
    target = root / "benchmark-registration.json"
    if target.exists():
        raise RuntimeError("Registration is immutable; no overwrite")
    if any(json.loads((root / f"cpu-smoke-{name}/status.json").read_text())["status"] != "complete"
           for name in SOURCES):
        raise RuntimeError("Both CPU implementation preflights must pass")
    artifacts = json.loads((root / "artifacts.json").read_text())
    for name, revision in SOURCES.items():
        checkout = root / "upstream" / name
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"],
                                        text=True).strip()
        if actual != revision or dirty:
            raise RuntimeError("Upstream source is not the exact clean pinned revision")
    verify_artifacts(artifacts)
    hashes = {**evaluation.code_hashes(), **source_manifest(root)}
    hashes.update({str(p.relative_to(REPO)): digest(p)
                   for p in (root / "deps").rglob("*") if p.is_file() and
                   "__pycache__" not in p.parts})
    for file, expected in hashes.items():
        source = REPO / file
        copy = root / "frozen-source" / file
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(source.read_bytes())
        copy.chmod(0o444)
        if digest(copy) != expected:
            raise RuntimeError("Source changed during snapshot")
    verify_files(hashes)
    save(root / "panel.json", [i["id"] for i in instances])
    save(root / "initial-requests.json", [
        asdict(evaluation.public_request(i, [])) for i in instances])
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    overlay = {d.metadata["Name"]: d.version for d in
               importlib.metadata.distributions(path=[str(root / "deps")])}
    registration = {
        "schema_version": 1, "registered_at": time.time(), "selection": str(selection),
        "selection_sha256": shared["selection_sha256"], "protocol": "explicit_actions",
        "bundle_manifest_sha256": shared["bundle_manifest_sha256"],
        "models": artifacts, "code_sha256": hashes, "packages": packages, "overlay": overlay,
        "scheduled_episodes_per_model": len(instances),
        "maximum_calls_per_model": sum(i["query_budget"] for i in instances),
        "context_cap": adapter.CONTEXT_LIMIT, "inference_timeout_seconds": 180,
        "loading_timeout_seconds": 600, "gpu_models_sequential": True,
        "policy": {"retry": False, "repair": False, "truncate": False,
                   "own_history": True, "all_scheduled_in_denominator": True,
                   "probability_calibration_claim": False, "private_labels_sent": False},
        "dependency_deviations": {
            "torch": "2.13.0 shared runtime; Kev upstream declares >=2.6,<2.9",
            "numpy": "2.3.5 shared runtime; Decider declares <2, Kev >=2.5.3; no numpy inference",
            "fused_kernels": "No flash-linear-attention/causal-conv1d; native reference kernels",
            "omitted": "Training/server dependencies unused; only accelerate/peft isolated overlay"},
    }
    if baseline is not None:
        rows = [json.loads(line) for line in baseline.read_text().splitlines()]
        by_id = {row["instance_id"]: row for row in rows}
        if any(i["id"] not in by_id for i in instances):
            raise RuntimeError("Inspected baseline lacks selected instances")
        save(root / "reused-jev-baseline.json", {
            "classification": "reused_inspected_baseline_not_new_inference",
            "source": str(baseline), "source_sha256": digest(baseline),
            "rows": [by_id[i["id"]] for i in instances]})
        registration["reused_baseline_sha256"] = digest(root / "reused-jev-baseline.json")
        registration["reused_baseline_source_sha256"] = digest(baseline)
    save(target, registration)
    save(root / "benchmark-registration.sha256", digest(target))
    save(root / "status.json", {"status": "ready_awaiting_explicit_gpu_handoff",
                               "benchmark_calls": 0, "scheduled_episodes": 128})


def verify_handoff(handoff_path, registration):
    """Explicit lifecycle approval, independently of standard resource locking."""
    handoff = json.loads(handoff_path.read_text())
    if (handoff.get("approved") is not True or
            handoff.get("registration_sha256") != digest(registration) or
            handoff.get("prior_gpu_owner_released") is not True or
            handoff.get("cuda_visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("Explicit handoff does not authorize this frozen registration/device")


def amend_parent_locks(root):
    """Resource-only pre-inference amendment; original registration stays immutable."""
    original = root / "benchmark-registration.json"
    target = root / "benchmark-registration-parent-locks.json"
    if target.exists() or any(root.glob("run-*")):
        raise RuntimeError("Amendment requires no existing amendment or benchmark runs")
    if digest(original) != json.loads((root / "benchmark-registration.sha256").read_text()):
        raise RuntimeError("Original registration changed")
    registration = json.loads(original.read_text())
    name = "scripts/run_trained_jevlike_pilot.py"
    prior = registration["code_sha256"][name]
    verify_files({p: h for p, h in registration["code_sha256"].items() if p != name})
    registration["code_sha256"][name] = digest(REPO / name)
    registration["resource_amendment"] = {
        "prior_registration_sha256": digest(original),
        "registered_at": time.time(), "benchmark_calls_before_amendment": 0,
        "changed_source": name, "prior_source_sha256": prior,
        "new_source_sha256": registration["code_sha256"][name],
        "reason": "Parent ordinary environment admits exact standard locks; remove descriptor-only gate.",
        "standard_lock_function": "scripts.extended_graph_suite.lane_locks",
        "lock_models": ["qwen08", "qwen2"], "relocated_namespace": False,
        "selection_models_inference_scoring_unchanged": True}
    snapshot = root / "frozen-source-parent-locks" / name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes((REPO / name).read_bytes())
    snapshot.chmod(0o444)
    save(target, registration)
    save(root / "benchmark-registration-parent-locks.sha256", digest(target))
    save(root / "status.json", {
        "status": "ready_for_parent_standard_lock_launch", "benchmark_calls": 0,
        "active_registration": target.name, "active_registration_sha256": digest(target),
        "scheduled_episodes": 128})


def evaluate(root, candidate, instances, registration):
    from scripts import evaluate_graph_release as evaluation
    suite = evaluation.suite
    output = root / f"run-{candidate}"
    output.mkdir(exist_ok=False)
    model_id = f"{PINS[candidate][0]}@{PINS[candidate][1]}"
    attempts, worker, fatal = [], None, None
    interrupted = False
    start = time.perf_counter()
    save(output / "status.json", {"status": "loading", "benchmark_calls": 0})
    try:
        worker = Worker(root, candidate, "cuda", output)
        ready = worker.receive(600)
        save(output / "runtime.json", ready)
        if ready["status"] != "ready":
            raise RuntimeError(ready)
        with (output / "attempts.jsonl").open("x") as stream:
            for instance in instances:
                history = []
                while (request := evaluation.public_request(instance, history)) is not None:
                    verify_files(registration["code_sha256"])
                    save(output / "status.json", {"status": "running",
                                                 "instance_id": instance["id"],
                                                 "step": len(history), "calls": len(attempts)})
                    tick = time.perf_counter()
                    row = {"instance_id": instance["id"], "step": len(history),
                           "request": asdict(request)}
                    try:
                        backend = worker.predict(request)
                        if backend["status"] != "success":
                            row.update(status="failure", selected_option_id=None,
                                       resolved_model=None, fatal=backend.get("fatal", True),
                                       diagnostic_code=backend.get("error_type"), backend=backend,
                                       strict_status="failure",
                                       action_observation={"eligible": False, "selected_option_id": None},
                                       usage={"input_tokens": None, "output_tokens": None,
                                              "reasoning_tokens": None})
                        else:
                            response, diagnostic = adapter.validate_result(request, backend["result"], model_id)
                            row.update(status="success", selected_option_id=response.selected_option_id,
                                       resolved_model=model_id, fatal=False,
                                       probabilities=response.probabilities,
                                       probability_kind=response.probability_kind,
                                       usage=asdict(response.usage), backend=backend, **diagnostic)
                    except Exception as exc:
                        row.update(status="failure", selected_option_id=None, resolved_model=None,
                                   fatal=True, diagnostic_code=getattr(exc, "diagnostic_code", type(exc).__name__),
                                   strict_status="failure",
                                   action_observation=getattr(exc, "action_observation", {
                                       "eligible": False, "selected_option_id": None}),
                                   usage={"input_tokens": None, "output_tokens": None, "reasoning_tokens": None})
                    row["latency_seconds"] = time.perf_counter() - tick
                    suite.append_row(stream, row)
                    attempts.append(suite.json_value(row))
                    if row["status"] != "success":
                        if row["fatal"]:
                            fatal = row["diagnostic_code"]
                        break
                    history.append(row["selected_option_id"])
                if fatal:
                    break
    except BaseException as exc:
        fatal = f"{type(exc).__name__}: {str(exc)[:1200]}"
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
    finally:
        if worker is not None:
            worker.close()
        scores = evaluation.replay(instances, attempts, SimpleNamespace(model=model_id), native_action=False)
        for score in scores:
            score["scheduled"] = True
            score["unattempted"] = score["query_count"] == 0
        with (output / "scores.jsonl").open("x") as stream:
            for score in scores:
                suite.append_row(stream, score)
        save(output / "summary.json", suite.summarize(scores))
        save(output / "status.json", {
            "status": "aborted" if fatal else (
                "completed_with_failures" if any(a["status"] == "failure" for a in attempts) else "completed"),
            "fatal": fatal, "calls": len(attempts), "scheduled_episodes": len(instances),
            "unattempted_episodes": sum(s["unattempted"] for s in scores),
            "failure_count": sum(a["status"] == "failure" for a in attempts),
            "wall_seconds": time.perf_counter() - start})
    if interrupted:
        raise KeyboardInterrupt


def run_gpu(root, handoff):
    from scripts import extended_graph_suite as suite
    path = root / "benchmark-registration-parent-locks.json"
    expected = json.loads((root / "benchmark-registration-parent-locks.sha256").read_text())
    if digest(path) != expected:
        raise RuntimeError("Registration changed")
    registration = json.loads(path.read_text())
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if packages != registration["packages"]:
        raise RuntimeError("Shared runtime package versions changed after registration")
    verify_handoff(handoff, path)
    verify_files(registration["code_sha256"])
    verify_artifacts(registration["models"])
    instances, shared = panel(Path(registration["selection"]))
    if shared["selection_sha256"] != registration["selection_sha256"]:
        raise RuntimeError("Shared selected panel changed")
    with suite.lane_locks(("qwen08", "qwen2")):
        run_locked(root, instances, registration)


def run_locked(root, instances, registration):
    """Called only inside the ordinary parent's unchanged standard lock context."""
    os.environ["TRAINED_JEVLIKE_GPU_HANDOFF"] = "approved"
    completed = False
    try:
        for candidate in ("decider", "kev"):
            save(root / "status.json", {"status": "running", "candidate": candidate,
                                       "pid": os.getpid()})
            evaluate(root, candidate, instances, registration)
        verify_files(registration["code_sha256"])
        completed = True
    finally:
        save(root / "status.json", {
            "status": "finished" if completed else "interrupted", "gpu_workers_released": True,
            "outcomes": {name: json.loads((root / f"run-{name}/status.json").read_text())
                         if (root / f"run-{name}/status.json").exists()
                         else {"status": "unattempted", "scheduled_episodes": len(instances), "calls": 0}
                         for name in ("decider", "kev")}})
        from scripts import extended_graph_suite as suite
        comparison = {"scope": "descriptive_64_episode_pilot",
                      "scheduled_episodes_per_model": len(instances),
                      "models": {name: json.loads((root / f"run-{name}/summary.json").read_text())
                                 for name in ("decider", "kev")
                                 if (root / f"run-{name}/summary.json").exists()}}
        baseline = root / "reused-jev-baseline.json"
        if baseline.exists():
            if digest(baseline) != registration["reused_baseline_sha256"]:
                raise RuntimeError("Reused baseline changed")
            comparison["jev"] = {
                "classification": "reused_inspected_baseline_not_new_inference",
                "summary": suite.summarize(json.loads(baseline.read_text())["rows"])}
        save(root / "comparison.json", comparison)


def authored_requests():
    return [
        DecisionRequest("authored-binary", "There is exactly one edge: A--B.",
                        "Are A and B adjacent?", (
                            DecisionOption("no", "No, no edge joins A and B."),
                            DecisionOption("yes", "Yes, an edge joins A and B."))),
        DecisionRequest("authored-degree12", "A has exactly two neighbors: B and C.",
                        "What is the degree of A?", tuple(
                            DecisionOption(str(i), f"Exactly {i} neighbors.")
                            for i in [11, 0, 2, 1, 10, 3, 4, 5, 6, 7, 8, 9])),
    ]


def smoke(root, candidate):
    output = root / f"cpu-smoke-{candidate}"
    output.mkdir(exist_ok=False)
    save(output / "status.json", {"status": "loading_cpu", "pid": os.getpid()})
    worker = None
    rows = []
    try:
        worker = Worker(root, candidate, "cpu", output)
        ready = worker.receive(600)
        save(output / "runtime.json", ready)
        if ready["status"] != "ready":
            raise RuntimeError(ready)
        for request in authored_requests():
            save(output / "status.json", {"status": "cpu_forward", "request_id": request.request_id,
                                         "pid": os.getpid()})
            start = time.perf_counter()
            record = worker.predict(request)
            if record["status"] != "success":
                raise RuntimeError(record)
            response, diagnostics = adapter.validate_result(request, record["result"], candidate)
            rows.append({"request": asdict(request), "response": asdict(response),
                         "latency_seconds": time.perf_counter() - start,
                         "backend": record, **diagnostics})
            save(output / "results.json", rows)
        save(output / "status.json", {"status": "complete", "calls": len(rows),
                                     "cuda_initialized": ready["runtime"]["cuda_initialized"]})
    except Exception as exc:
        save(output / "status.json", {"status": "failed", "calls": len(rows),
                                     "error_type": type(exc).__name__, "error": str(exc)[:1400]})
        raise
    finally:
        if worker is not None:
            worker.close()


def main():
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", choices=("decider", "kev"))
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--register-selection", type=Path)
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--amend-parent-locks", action="store_true")
    args = parser.parse_args()
    if args.smoke_cpu:
        if args.candidate is None:
            parser.error("--candidate is required for smoke")
        smoke(args.root.resolve(), args.candidate)
    elif args.register_selection:
        register(args.root.resolve(), args.register_selection.resolve(), args.baseline_scores)
    elif args.amend_parent_locks:
        amend_parent_locks(args.root.resolve())
    elif args.run_gpu:
        if args.handoff is None:
            parser.error("GPU execution requires an explicit --handoff")
        run_gpu(args.root.resolve(), args.handoff)
    else:
        parser.error("GPU registration/run requires explicit handoff; CPU preparation only")


if __name__ == "__main__":
    main()
