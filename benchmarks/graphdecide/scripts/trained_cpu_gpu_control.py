#!/usr/bin/env python3
"""Preregistered four-forward, CPU-only device compatibility diagnostic."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import trained_jevlike_adapter as adapter
from scripts import run_trained_jevlike_pilot as frozen
from scripts.prepare_trained_jevlike import digest, save

MODELS = ("decider", "kev")
GATE = 0.01
LOAD_SECONDS, REQUEST_SECONDS, JOB_SECONDS = 300, 180, 1200


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def choose_cases(cases):
    chosen = [next(row for row in cases if row["task"] == task and row["node_count"] == size)
              for task, size in (("adjacency", 8), ("degree_exact", 12))]
    return sorted(chosen, key=lambda row: cases.index(row))


def wire_request(persisted, canonical):
    """Recover pre-log object insertion order, without changing any public values."""
    require(persisted == json.loads(json.dumps(canonical)), "Canonical public request differs")
    result = dict(persisted)
    # GPU logging sorts JSON object keys, whereas the frozen adapter renders state
    # in insertion order. Frozen request_for restores the original order using only
    # the public release. In degree requests, remaining_queries follows vertex.
    result["state"] = {key: persisted["state"][key] for key in canonical["state"]}
    return result


def register(source, output):
    from scripts import graph_release_api as release
    require(not output.exists(), "Control output exists; no overwrite")
    original = source / "benchmark-registration-parent-locks.json"
    registration = read(original)
    require(digest(original) == read(source / "benchmark-registration-parent-locks.sha256"),
            "Frozen benchmark registration changed")
    frozen.verify_files(registration["code_sha256"])
    frozen.verify_artifacts(registration["models"])
    proof_path = source / "parent-checkpoint-payload-verification.json"
    proof = read(proof_path)
    require(proof["verified"] is True and proof["files"] == 34
            and proof["safe_weights_only"] is True
            and proof["artifacts_sha256"] == digest(source / "artifacts.json"),
            "Missing/mismatched independent 34-file payload verification")
    selection = Path(registration["selection"])
    panel_path = selection / "pilot-cases.jsonl"
    require(digest(panel_path) == registration["selection_sha256"], "Shared panel changed")
    panel = [json.loads(line) for line in panel_path.read_text().splitlines()]
    chosen = choose_cases(panel)  # Selection precedes any GPU result inspection.
    shared = read(selection / "registration.json")
    rows, inputs = [], {}
    for candidate in MODELS:
        attempts_path = source / f"run-{candidate}/attempts.jsonl"
        runtime_path = source / f"run-{candidate}/runtime.json"
        inputs[str(attempts_path)] = digest(attempts_path)
        inputs[str(runtime_path)] = digest(runtime_path)
        attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
        runtime = read(runtime_path)
        require(runtime["status"] == "ready" and runtime["runtime"]["device"] == "cuda",
                "Expected saved GPU runtime")
        for case in chosen:
            matches = [row for row in attempts if row["instance_id"] == case["instance_id"]]
            require(len(matches) == 1 and matches[0]["step"] == 0
                    and matches[0]["status"] == "success", "Missing unique saved exact GPU attempt")
            gpu = matches[0]
            canonical = release.request_for(Path(shared["bundle"]), shared["split"],
                                            case["instance_id"], [], protocol="explicit_actions")
            request = wire_request(gpu["request"], canonical)
            req = adapter.request_from_dict(request)
            adapter.validate_result(req, gpu["backend"]["result"], candidate)
            require(gpu["probabilities"] == gpu["backend"]["result"]["probabilities"]
                    and gpu["selected_option_id"] == gpu["backend"]["result"]["selected_option_id"],
                    "Saved native GPU trace is inconsistent")
            if case["task"] == "degree_exact":
                require(len(req.options) == 12 and {o.id for o in req.options}
                        == {str(i) for i in range(12)}, "Twelve-option numeric identity mismatch")
            rows.append({
                "candidate": candidate, "panel_index": case["index"],
                "instance_id": case["instance_id"], "task": case["task"],
                "node_count": case["node_count"], "request": request,
                "persisted_gpu_request": gpu["request"],
                "original_state_key_order": list(canonical["state"]),
                "payload_sha256": json_digest(adapter.public_payload(req)),
                "gpu_result": gpu["backend"]["result"], "gpu_runtime": runtime,
            })
    require(len(rows) == 4, "Exactly four scheduled forwards required")
    for offset in (0, 1):
        require(rows[offset]["payload_sha256"] == rows[offset + 2]["payload_sha256"],
                "Models were not given identical public requests")
    output.mkdir(parents=True)
    save(output / "scheduled.json", rows)
    own = {str(path.relative_to(REPO)): digest(path) for path in (
        Path(__file__), REPO / "tests/test_trained_cpu_gpu_control.py")}
    for name in own:
        copy = output / "source-snapshot" / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((REPO / name).read_bytes())
        copy.chmod(0o444)
    manifest = {
        "version": 1, "registered_at": time.time(), "source_root": str(source),
        "classification": "posthoc_device_compatibility_diagnostic",
        "selection": "First n8 adjacency and first n12 degree in immutable panel order; no outcome filter",
        "panel_selection_sha256": registration["selection_sha256"],
        "frozen_registration_sha256": digest(original),
        "checkpoint_verification_sha256": digest(proof_path),
        "checkpoint_payload_files": 34, "models": registration["models"],
        "frozen_code_sha256": registration["code_sha256"], "control_code_sha256": own,
        "saved_gpu_inputs_sha256": inputs, "scheduled_sha256": digest(output / "scheduled.json"),
        "packages": registration["packages"], "overlay": registration["overlay"],
        "scheduled_forwards_per_model": 2, "maximum_forwards_total": 4,
        "load_timeout_seconds": LOAD_SECONDS, "request_timeout_seconds": REQUEST_SECONDS,
        "job_timeout_seconds": JOB_SECONDS, "torch_threads": 4, "torch_interop_threads": 1,
        "cuda_visible_devices": "", "native_dtype": {"decider": "bfloat16", "kev": "float32"},
        "native_temperature": {m: rows[MODELS.index(m) * 2]["gpu_runtime"]["runtime"]["temperature"]
                               for m in MODELS},
        "gate": {"max_abs_probability_difference_lte": GATE, "choice_agreement_required": True,
                 "scope": "descriptive_not_exact_floating_equality"},
        "policy": {"retry": False, "repair": False, "temperature_fit": False,
                   "kernel_override": False, "gold_labels_read": False,
                   "new_performance_cases": False, "calibration_claim": False,
                   "speed_ranking": False, "full_upstream_environment_reproduction": False},
        "state_serialization": "Restore pre-log key order from frozen public-only request_for; values are persisted GPU request values",
    }
    save(output / "registration.json", manifest)
    save(output / "registration.sha256", digest(output / "registration.json"))
    save(output / "status.json", {"status": "registered", "fresh_forwards": 0})
    return manifest


def verify(output):
    path = output / "registration.json"
    require(digest(path) == read(output / "registration.sha256"), "Control registration changed")
    manifest = read(path)
    require(digest(output / "scheduled.json") == manifest["scheduled_sha256"], "Scheduled requests changed")
    frozen.verify_files(manifest["control_code_sha256"])
    frozen.verify_files(manifest["frozen_code_sha256"])
    source = Path(manifest["source_root"])
    require(digest(source / "benchmark-registration-parent-locks.json")
            == manifest["frozen_registration_sha256"], "Frozen registration changed")
    for path, expected in manifest["saved_gpu_inputs_sha256"].items():
        require(digest(path) == expected, "Original GPU trace changed")
    return manifest


def assert_cpu(model, candidate, torch):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CPU-only CUDA invariant failed")
    native = model.m if candidate == "decider" else model[1]
    require(all(p.device.type == "cpu" for p in native.parameters()), "Non-CPU parameter found")
    require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1,
            "Unexpected CPU thread counts")


def worker(source, candidate):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden before imports")
    supervisor = int(os.environ["CONTROL_SUPERVISOR_PID"])
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGKILL) == 0 and os.getppid() == supervisor,
            "Cannot bind CPU worker to supervisor")
    sys.path.insert(0, str(source / "deps"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", TOKENIZERS_PARALLELISM="false")
    import torch
    require(not torch.cuda.is_initialized(), "CUDA initialized before CPU loading")
    wire = sys.stdout

    def emit(value):
        wire.write(json.dumps(value, allow_nan=False) + "\n")
        wire.flush()

    with contextlib.redirect_stdout(sys.stderr):
        try:
            start = time.monotonic()
            model, metadata = adapter.initialize(source, candidate, "cpu")
            assert_cpu(model, candidate, torch)
            emit({"status": "ready", "runtime": metadata,
                  "torch_interop_threads": torch.get_num_interop_threads(),
                  "loading_seconds": time.monotonic() - start})
        except Exception as error:
            emit({"status": "failure", "phase": "loading", "error_type": type(error).__name__,
                  "error": str(error)[:1600], "cuda_initialized": torch.cuda.is_initialized()})
            return 1
        for count, line in enumerate(sys.stdin):
            if count >= 2:
                raise RuntimeError("CPU worker refuses a third forward")
            request = adapter.request_from_dict(json.loads(line))
            try:
                assert_cpu(model, candidate, torch)
                start = time.monotonic()
                native = adapter.infer(model, candidate, request)
                assert_cpu(model, candidate, torch)
                emit({"status": "success", "result": native,
                      "worker_seconds": time.monotonic() - start,
                      "cuda_initialized": torch.cuda.is_initialized(),
                      "payload_sha256": json_digest(adapter.public_payload(request))})
            except Exception as error:
                emit({"status": "failure", "phase": "forward", "error_type": type(error).__name__,
                      "error": str(error)[:1600], "cuda_initialized": torch.cuda.is_initialized()})
                return 1
    return 0


class CPUWorker:
    def __init__(self, source, candidate, output):
        self.log = (output / f"{candidate}-worker.log").open("x")
        self.process = subprocess.Popen(
            [frozen.PYTHON, str(Path(__file__).resolve()), "worker", "--source", str(source),
             "--candidate", candidate],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            bufsize=1, start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
                 "MKL_NUM_THREADS": "4", "CONTROL_SUPERVISOR_PID": str(os.getpid()),
                 "PYTHONDONTWRITEBYTECODE": "1"})
        save(output / f"{candidate}-process.json", {"pid": self.process.pid, "device": "cpu"})

    def receive(self, seconds):
        require(seconds > 0, "Job time budget exhausted")
        if not select.select([self.process.stdout], [], [], seconds)[0]:
            raise TimeoutError("Bounded CPU worker deadline exceeded")
        line = self.process.stdout.readline()
        require(bool(line), "CPU worker exited without response")
        return json.loads(line)

    def predict(self, request, seconds):
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        return self.receive(seconds)

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
        self.log.close()


def compare_vector(request, gpu, cpu, candidate):
    request = adapter.request_from_dict(request)
    adapter.validate_result(request, gpu, candidate)
    adapter.validate_result(request, cpu, candidate)
    keys = [o.id for o in request.options]
    deltas = {key: abs(cpu["probabilities"][key] - gpu["probabilities"][key]) for key in keys}
    maximum, mean = max(deltas.values()), sum(deltas.values()) / len(keys)
    agree = cpu["selected_option_id"] == gpu["selected_option_id"]
    return {"option_ids": keys, "absolute_differences": deltas,
            "max_abs_difference": maximum, "mean_abs_difference": mean,
            "native_choice_agreement": agree,
            "cpu_argmax_id": max(keys, key=cpu["probabilities"].__getitem__),
            "gpu_argmax_id": max(keys, key=gpu["probabilities"].__getitem__),
            "argmax_agreement": max(keys, key=cpu["probabilities"].__getitem__)
                                == max(keys, key=gpu["probabilities"].__getitem__),
            "gate_pass": maximum <= GATE and agree,
            "input_token_count_match": cpu.get("input_tokens") == gpu.get("input_tokens")}


def offline_compare(output):
    manifest = verify(output)
    scheduled = read(output / "scheduled.json")
    results = read(output / "results.json")
    require(len(scheduled) == len(results) == 4, "Expected four scheduled result rows")
    comparisons = []
    for planned, actual in zip(scheduled, results, strict=True):
        require((actual["candidate"], actual["instance_id"])
                == (planned["candidate"], planned["instance_id"]), "Result order differs")
        row = {"candidate": planned["candidate"], "instance_id": planned["instance_id"],
               "task": planned["task"], "node_count": planned["node_count"],
               "panel_index": planned["panel_index"], "status": actual["status"],
               "native_temperature": manifest["native_temperature"][planned["candidate"]],
               "dtype": manifest["native_dtype"][planned["candidate"]],
               "gpu_result": planned["gpu_result"], "cpu_result": actual.get("result"),
               "gate_pass": None}
        if actual["status"] == "success":
            require(actual["payload_sha256"] == planned["payload_sha256"]
                    and actual["cuda_initialized"] is False, "CPU fidelity invariant failed")
            row.update(compare_vector(planned["request"], planned["gpu_result"],
                                      actual["result"], planned["candidate"]))
        else:
            row.update(error=actual.get("error"), error_type=actual.get("error_type"))
        comparisons.append(row)
    report = {"classification": manifest["classification"], "scheduled": 4,
              "successful": sum(r["status"] == "success" for r in comparisons),
              "failed": sum(r["status"] == "failure" for r in comparisons),
              "unattempted": sum(r["status"] == "unattempted" for r in comparisons),
              "gate_pass_count": sum(r["gate_pass"] is True for r in comparisons),
              "gate": manifest["gate"], "comparisons": comparisons,
              "registration_sha256": digest(output / "registration.json"),
              "results_sha256": digest(output / "results.json"),
              "offline_only": True, "model_calls_in_comparison": 0,
              "confidence_semantics": "native selected-option probability; no graph-calibration guarantee",
              "runtime_claim": "compatible-runtime upstream-inference pilot, not full-environment reproduction"}
    save(output / "comparison.json", report)
    return report


def run(output, worker_factory=CPUWorker):
    manifest = verify(output)
    with (output / "run-start.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_at": time.time(), "fresh_forwards": 0}, stream)
    started = time.monotonic()
    deadline = started + JOB_SECONDS
    scheduled = read(output / "scheduled.json")
    results = [{"candidate": row["candidate"], "instance_id": row["instance_id"],
                "status": "unattempted", "error": "not yet attempted"} for row in scheduled]
    save(output / "results.json", results)
    forwards, runtimes = 0, {}
    try:
        frozen.verify_artifacts(manifest["models"])
        import importlib.metadata
        require({d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
                == manifest["packages"], "Runtime packages changed")
        for candidate in MODELS:
            indices = [i for i, row in enumerate(scheduled) if row["candidate"] == candidate]
            child, current = None, indices[0]
            try:
                require(time.monotonic() < deadline, "Job deadline exhausted before loading")
                save(output / "status.json", {"status": "loading_cpu", "candidate": candidate,
                                              "fresh_forwards": forwards})
                child = worker_factory(Path(manifest["source_root"]), candidate, output)
                runtime = child.receive(min(LOAD_SECONDS, deadline - time.monotonic()))
                runtimes[candidate] = runtime
                save(output / "runtime.json", runtimes)
                require(runtime["status"] == "ready", f"CPU load failed: {runtime}")
                require(runtime["runtime"]["cuda_initialized"] is False
                        and runtime["runtime"]["dtype"] == manifest["native_dtype"][candidate]
                        and runtime["runtime"]["temperature"] == manifest["native_temperature"][candidate],
                        "CPU runtime differs from preregistered native setup")
                for current in indices:
                    planned = scheduled[current]
                    require(time.monotonic() < deadline and forwards < 4, "CPU job/call bound exhausted")
                    frozen.verify_files(manifest["frozen_code_sha256"])
                    save(output / "status.json", {"status": "cpu_forward", "candidate": candidate,
                          "instance_id": planned["instance_id"], "fresh_forwards": forwards + 1})
                    forwards += 1
                    result = child.predict(planned["request"],
                                           min(REQUEST_SECONDS, deadline - time.monotonic()))
                    results[current] = {"candidate": candidate,
                                        "instance_id": planned["instance_id"], **result}
                    save(output / "results.json", results)
                    require(result["status"] == "success", f"CPU forward failed: {result}")
                    compare_vector(planned["request"], planned["gpu_result"], result["result"], candidate)
            except Exception as error:
                results[current].update(status="failure", error_type=type(error).__name__,
                                        error=str(error)[:1800])
                save(output / "results.json", results)
            finally:
                if child is not None:
                    child.close()
    finally:
        save(output / "results.json", results)
        report = offline_compare(output)
        save(output / "status.json", {
            "status": "completed" if report["successful"] == 4 else "completed_with_failures",
            "fresh_forward_requests": forwards, "scheduled": 4,
            "successful": report["successful"], "failed": report["failed"],
            "unattempted": report["unattempted"], "gate_pass_count": report["gate_pass_count"],
            "wall_seconds": time.monotonic() - started, "no_retry": True,
            "cuda_visible_devices": "", "comparison_sha256": digest(output / "comparison.json")})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "run", "compare", "worker"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate", choices=MODELS)
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args.source.resolve(), args.candidate)
    require(args.output is not None, "--output is required")
    if args.command == "register":
        result = register(args.source.resolve(), args.output.resolve())
    elif args.command == "compare":
        result = offline_compare(args.output.resolve())
    else:
        def timeout(signum, frame):
            raise TimeoutError("1200-second CPU control job limit")
        signal.signal(signal.SIGALRM, timeout)
        signal.signal(signal.SIGTERM, timeout)
        signal.alarm(JOB_SECONDS)
        try:
            result = run(args.output.resolve())
        finally:
            signal.alarm(0)
    print(json.dumps({k: v for k, v in result.items()
                      if k in {"scheduled", "successful", "failed", "unattempted", "gate_pass_count",
                               "classification", "registered_at"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
