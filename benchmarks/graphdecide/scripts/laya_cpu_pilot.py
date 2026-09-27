#!/usr/bin/env python3
"""Pinned Laya typed-decision pilot: CPU only, lossless prompts, bounded workers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
SOURCE = "970dc8c5f63d7b886a68409493f37d569424f933"
REVISION = "1a793eb568e6718f15941d08f85432581df534e3"
WEIGHTS_SHA = "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e"
SELECTION_SHA = "175f80f4fa87e9147ffeb26204dcf7fa53acfad348d3a9d49caa37d3cef06f8e"
MODEL = f"convaiinnovations/laya-typed-decisions@{REVISION}:cpu-float32"
ROOT = REPO / "output/laya-cpu-pilot-20260924-v1"
SELECTION = REPO / "output/larger-local-pilot-20260924-v1"
PYTHON = "/fastdata/xianya/vllm-venv/bin/python"
LIMITS = {"context_tokens": 1024, "head_tokens": 256, "option_tokens": 48,
          "max_options": 12, "torch_threads": 4, "interop_threads": 1,
          "load_seconds": 300, "forward_seconds": 30, "job_seconds": 1800,
          "max_forwards": 160}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    pending = path.with_suffix(path.suffix + ".next")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def environment():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LAYA_CPU_AMP": "",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}


def imports(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES='' required")
    sys.path.insert(0, str(root / "upstream"))
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    return torch


def question(request):
    options = request["options"]
    ids = [o["id"] for o in options]
    if not 1 <= len(ids) <= 12 or len(ids) != len(set(ids)):
        raise ValueError("Expected 1..12 unique option IDs")
    if not all(isinstance(i, str) and i for i in ids):
        raise ValueError("Canonical IDs must remain nonempty strings")
    return {"type": "choice", "instructions": request["question"],
            "criteria": {o["id"]: o["description"] for o in options}}


def coverage(tok, request):
    """Compare native IDs with an unsliced construction of the exact native format.

    Native build_sequence also clips question/options inside head_max_len, so a
    total-token check alone is insufficient even for requests below 1024 tokens.
    """
    from laya.common import build_sequence, render_options, serialize_state
    definition = question(request)
    q = {"t": "choice", "ins": definition["instructions"], "crit": definition["criteria"]}
    state = serialize_state(request["state"])
    texts = ["choice question: " + q["ins"], *[" " + o for o in render_options(q)], state]
    parts = [tok(t, add_special_tokens=False, truncation=False)["input_ids"] for t in texts]
    head, opts, state_ids = parts[0], parts[1:-1], parts[-1]
    full = [tok.cls_token_id] + head + [tok.sep_token_id]
    markers = []
    for option in opts:
        markers.append(len(full))
        full += [tok.mask_token_id] + option
    full += [tok.sep_token_id] + state_ids + [tok.sep_token_id]
    native, native_markers = build_sequence(tok, request["state"], q, 1024, 256)
    reasons = []
    if any(tok.mask_token in text for text in texts):
        reasons.append("mask_literal_replacement")
    if len(full) > 1024:
        reasons.append("context_overlength")
    if any(len(p) > 48 for p in opts):
        reasons.append("option_truncation")
    option_sum = sum(len(p) + 1 for p in opts)
    if 256 - option_sum < 16 or len(head) > max(8, 256 - option_sum):
        reasons.append("head_truncation")
    if (native != full or native_markers != markers) and not reasons:
        reasons.append("native_sequence_mismatch")
    return {"compatible": not reasons and native == full and native_markers == markers,
            "reasons": reasons, "full_tokens": len(full), "native_tokens": len(native),
            "question_tokens": len(head), "option_tokens": [len(p) for p in opts],
            "state_tokens": len(state_ids), "option_count": len(opts),
            "native_sequence_sha256": hashlib.sha256(json.dumps(native).encode()).hexdigest()}


def panel():
    from scripts.run_trained_jevlike_pilot import panel as shared_panel
    instances, shared = shared_panel(SELECTION)
    assert shared["selection_sha256"] == SELECTION_SHA
    assert sum(i["query_budget"] for i in instances) == 160
    return instances, shared


def prepare_coverage(root):
    imports(root)
    from transformers import AutoTokenizer
    from scripts import evaluate_graph_release as evaluation
    cfg = json.loads((root / "model/rl_agent_config.json").read_text())
    assert cfg["max_len"] == 1024 and cfg["head_max_len"] == 256
    tok = AutoTokenizer.from_pretrained(root / "model/tokenizer",
                                       local_files_only=True, trust_remote_code=False)
    instances, shared = panel()
    records = []
    for instance in instances:
        request = asdict(evaluation.public_request(instance, []))
        records.append({"instance_id": instance["id"], "task": instance["task"],
                        "node_count": len(instance["state"]["nodes"]),
                        "request_sha256": hashlib.sha256(
                            json.dumps(request, sort_keys=True).encode()).hexdigest(),
                        **coverage(tok, request)})
    groups = defaultdict(list)
    for row in records:
        groups[(row["task"], row["node_count"])].append(row)
    result = {"selection_sha256": shared["selection_sha256"], "model_forwards": 0,
              "contract": LIMITS, "scheduled": 64,
              "compatible": sum(r["compatible"] for r in records), "rows": records,
              "by_task_size": [
                  {"task": task, "node_count": size, "scheduled": len(rows),
                   "compatible": sum(r["compatible"] for r in rows),
                   "min_full_tokens": min(r["full_tokens"] for r in rows),
                   "max_full_tokens": max(r["full_tokens"] for r in rows),
                   "reasons": dict(Counter(reason for r in rows for reason in r["reasons"]))}
                  for (task, size), rows in sorted(groups.items())]}
    if (root / "coverage.json").exists():
        raise RuntimeError("Coverage immutable")
    save(root / "coverage.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


def validate_answer(request, result, logits):
    ids = [o["id"] for o in request["options"]]
    probabilities = result["probabilities"]
    if result["choice"] not in ids or list(probabilities) != ids or len(logits) != len(ids):
        raise ValueError("Upstream returned illegal or reordered IDs")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in [*probabilities.values(), *logits]):
        raise ValueError("Nonfinite or nonnumeric native scores")
    if any(not 0 <= v <= 1 for v in probabilities.values()):
        raise ValueError("Invalid probability")
    if abs(sum(probabilities.values()) - 1) > len(ids) * 0.000051:
        raise ValueError("Rounded upstream probabilities outside rounding tolerance")
    if ids[max(range(len(ids)), key=lambda i: logits[i])] != result["choice"]:
        raise ValueError("Choice disagrees with raw-logit argmax")


def worker(root):
    # Bind only this worker to its parent, without touching service processes.
    import ctypes
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
        raise RuntimeError("Cannot set worker parent-death signal")
    if os.getppid() != int(os.environ["LAYA_SUPERVISOR_PID"]):
        raise RuntimeError("Supervisor disappeared")
    torch = imports(root)
    import contextlib
    wire = sys.stdout

    def emit(row):
        wire.write(json.dumps(row, allow_nan=False) + "\n")
        wire.flush()

    with contextlib.redirect_stdout(sys.stderr):
        from laya.agent import Agent
        start = time.monotonic()
        agent = Agent(str(root / "model"), device="cpu", fast=False, compile=False)
        assert not torch.cuda.is_initialized()
        assert all(p.device.type == "cpu" and p.dtype == torch.float32
                   for p in agent.model.parameters())
        assert not agent.amp_enabled
        captured = []

        def capture(_module, _args, outputs):
            captured.append(outputs[0].detach().float().cpu().tolist()[0])

        agent.model.register_forward_hook(capture)
        emit({"status": "ready", "loading_seconds": time.monotonic() - start,
              "torch": torch.__version__, "threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(),
              "parameter_count": sum(p.numel() for p in agent.model.parameters()),
              "dtype": "float32", "checkpoint_storage_dtype": "F16",
              "cuda_initialized": torch.cuda.is_initialized(),
              "temperature_raw": agent.temperature_raw, "temperature": agent.temperature,
              "temperature_by_options_raw": agent.temperature_by_options_raw,
              "temperature_by_options_applied": agent.temperature_by_options})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                check = coverage(agent.tok, request)
                if not check["compatible"]:
                    emit({"status": "unsupported", "coverage": check, "forward_count": 0})
                    continue
                captured.clear()
                start = time.monotonic()
                result = agent.system_one(request["state"], {"decision": question(request)})
                elapsed = time.monotonic() - start
                assert len(captured) == 1 and not torch.cuda.is_initialized()
                answer = result["answers"]["decision"]
                validate_answer(request, answer, captured[0])
                emit({"status": "success", "answer": answer, "raw_logits": captured[0],
                      "coverage": check, "usage": {**result["usage"], "reasoning_tokens": None},
                      "forward_count": 1, "worker_seconds": elapsed, "cuda_initialized": False})
            except Exception as exc:
                emit({"status": "fatal", "error": f"{type(exc).__name__}: {str(exc)[:1000]}"})
                return 1
    return 0


class Worker:
    def __init__(self, root, output):
        self.log = (output / "worker.log").open("x")
        self.process = subprocess.Popen(
            [PYTHON, str(Path(__file__).resolve()), "worker", "--root", str(root)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env={**environment(), "LAYA_SUPERVISOR_PID": str(os.getpid())},
            start_new_session=True, bufsize=0)
        self.pending = b""
        save(output / "worker-process.json", {"pid": self.process.pid,
                                              "supervisor_pid": os.getpid(), "device": "cpu"})

    def receive(self, seconds):
        deadline = time.monotonic() + seconds
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"Own worker exceeded {seconds:.3f}s deadline")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError("Own worker exited without response")
            self.pending += data
        line, self.pending = self.pending.split(b"\n", 1)
        return json.loads(line)

    def predict(self, request, seconds):
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        return self.receive(seconds)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.log.close()


def smoke_requests():
    ids = ["11", "0", "2", "alpha", "10", "3", "four", "5", "6", "7", "8", "9"]
    return [
        {"request_id": "smoke-binary", "state": {"edges": [["A", "B"]]},
         "question": "Are A and B adjacent?",
         "options": [{"id": "no", "description": "No direct edge."},
                     {"id": "yes", "description": "There is a direct edge."}]},
        {"request_id": "smoke-mixed12", "state": "A has exactly two neighbors: B and C.",
         "question": "Which option indicates the degree of A?",
         "options": [{"id": i, "description": f"Degree {n}."}
                     for i, n in zip(ids, [11, 0, 2, 1, 10, 3, 4, 5, 6, 7, 8, 9])]}]


def smoke(root):
    out = root / "cpu-smoke"
    out.mkdir()
    process, results, error = None, [], None
    start = time.monotonic()
    try:
        process = Worker(root, out)
        ready = process.receive(300)
        save(out / "runtime.json", ready)
        assert ready["status"] == "ready"
        for request in smoke_requests():
            result = process.predict(request, 30)
            results.append({"request": request, "result": result})
            assert result["status"] == "success"
            validate_answer(request, result["answer"], result["raw_logits"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process:
            process.close()
        save(out / "results.json", results)
        save(out / "status.json", {"status": "failed" if error else "complete",
                                  "error": error, "benchmark": False,
                                  "requests": len(results), "wall_seconds": time.monotonic() - start})
    if error:
        raise RuntimeError(error)


def register(root, baseline):
    from scripts import evaluate_graph_release as evaluation
    if (root / "registration.json").exists():
        raise RuntimeError("Registration immutable")
    assert json.loads((root / "cpu-smoke/status.json").read_text())["status"] == "complete"
    instances, shared = panel()
    hashes = evaluation.code_hashes()
    paths = [Path(__file__), REPO / "tests/test_laya_cpu_pilot.py",
             root / "coverage.json", root / "cpu-smoke/runtime.json",
             root / "hub-metadata.json", root / "hub-tree.json", root / "upstream.tar.gz"]
    paths += list((root / "upstream/laya").rglob("*.py"))
    paths += [root / "upstream/LICENSE", root / "upstream/pyproject.toml"]
    paths += [p for p in (root / "model").rglob("*") if p.is_file()]
    for path in paths:
        hashes[str(path.relative_to(REPO))] = digest(path)
    assert digest(root / "model/model.safetensors") == WEIGHTS_SHA
    for name, sha in hashes.items():
        source = REPO / name
        if source.suffix == ".py":
            target = root / "frozen-source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            target.chmod(0o444)
            assert digest(target) == sha
    registration = {
        "registered_at_unix": time.time(), "registered_before_official_inference": True,
        "source_revision": SOURCE, "checkpoint_revision": REVISION,
        "checkpoint_sha256": WEIGHTS_SHA, "model_id": MODEL, "limits": LIMITS,
        "scheduled_episodes": len(instances), "maximum_official_forwards": 160,
        "selection_sha256": shared["selection_sha256"],
        "bundle_manifest_sha256": shared["bundle_manifest_sha256"],
        "protocol": "explicit_actions", "files_sha256": hashes,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "policy": {"truncate": False, "resample": False, "repair": False, "retry": False,
                   "own_history": True, "calibration_fit": False, "calibration_claim": False,
                   "all_planned_in_denominator": True, "latency_ranking": False,
                   "separate_cpu_budget_not_gpu_equivalent": True},
        "native_probability_semantics": "4-decimal rounded softmax over typed-choice logits, "
            "using upstream clamped checkpoint temperature; confidence is normalized entropy, "
            "answer_confidence is max probability; neither asserted calibrated here",
        "precision_deviation": "F16 stored weights loaded by native CPU path into float32, no AMP",
        "source_pairing": "Agent local checkpoint, safetensors strict=True, local tokenizer and "
            "encoder config; no remote code, alternate next-token model, or base-model download"}
    if baseline:
        manifest_path = baseline.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["protocol"] == "explicit_actions"
        assert manifest["bundle_manifest_sha256"] == shared["bundle_manifest_sha256"]
        assert manifest["model"]["model"] == "jev-1.13.0"
        rows = read_rows(baseline)
        chosen = {r["instance_id"]: r for r in rows}
        assert len(chosen) == len(rows)
        assert all(i["id"] in chosen for i in instances)
        assert all(chosen[i["id"]]["task"] == i["task"] and
                   chosen[i["id"]]["node_count"] == len(i["state"]["nodes"]) and
                   chosen[i["id"]]["query_budget"] == i["query_budget"] for i in instances)
        selected_ids = {i["id"] for i in instances}
        baseline_attempts = read_rows(baseline.parent / "attempts.jsonl")
        grouped = defaultdict(list)
        for row in baseline_attempts:
            if row["instance_id"] in selected_ids:
                grouped[row["instance_id"]].append(row)
        selected_attempts = [r for i in instances for r in grouped[i["id"]]]
        replayed = evaluation.replay(
            instances, selected_attempts, SimpleNamespace(model="jev-1.13.0"), native_action=True)
        assert all(all(chosen[r["instance_id"]][k] == v for k, v in r.items()) for r in replayed)
        save(root / "reused-jev.json", {"source": str(baseline),
                                      "source_sha256": digest(baseline),
                                      "manifest_sha256": digest(manifest_path),
                                      "attempts_sha256": digest(baseline.parent / "attempts.jsonl"),
                                      "matching_public_requests_and_scores_replayed": True,
                                      "classification": "reused_inspected_not_new_inference",
                                      "rows": [chosen[i["id"]] for i in instances]})
        registration["reused_jev_sha256"] = digest(root / "reused-jev.json")
    save(root / "registration.json", registration)
    save(root / "registration.sha256", digest(root / "registration.json"))
    (root / "registration.json").chmod(0o444)


def verify_registration(root):
    assert digest(root / "registration.json") == json.loads(
        (root / "registration.sha256").read_text())
    registration = json.loads((root / "registration.json").read_text())
    for name, sha in registration["files_sha256"].items():
        if digest(REPO / name) != sha:
            raise RuntimeError(f"Frozen input changed: {name}")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    assert packages == registration["packages"]
    return registration


def summarize(root, instances, attempts, fatal, forwards, elapsed):
    from scripts import evaluate_graph_release as evaluation
    from scripts import extended_graph_suite as suite
    scores = evaluation.replay(instances, attempts, SimpleNamespace(model=MODEL), native_action=False)
    initial = {r["instance_id"]: r for r in json.loads((root / "coverage.json").read_text())["rows"]}
    by_id = defaultdict(list)
    for row in attempts:
        by_id[row["instance_id"]].append(row)
    for row in scores:
        own = by_id[row["instance_id"]]
        row["initial_context_compatible"] = initial[row["instance_id"]]["compatible"]
        row["scheduled"] = True
        row["forward_count"] = sum(a.get("forward_count", 0) for a in own)
        row["execution_category"] = (
            "unsupported_initial" if not row["initial_context_compatible"] else
            "infeasible_growing_context" if any(a.get("diagnostic_code") == "unsupported" for a in own) else
            "completed" if row["feasible"] else "failed_or_unattempted")
    save(root / "run/scores.json", scores)
    summary = {"status": "aborted" if fatal else "completed", "fatal": fatal,
               "scheduled_episodes": 64, "official_forwards": forwards,
               "attempt_records_including_unsupported": len(attempts),
               "wall_seconds": elapsed,
               "execution_categories": dict(Counter(r["execution_category"] for r in scores)),
               "whole_panel": suite.summarize(scores),
               "context_compatible_initial": suite.summarize(
                   [r for r in scores if r["initial_context_compatible"]])}
    reused = root / "reused-jev.json"
    if reused.exists():
        registration = json.loads((root / "registration.json").read_text())
        assert digest(reused) == registration["reused_jev_sha256"]
        reference = json.loads(reused.read_text())
        compatible = {r["instance_id"] for r in scores if r["initial_context_compatible"]}
        completed = {r["instance_id"] for r in scores if r["execution_category"] == "completed"}
        summary["reused_jev"] = {
            "classification": reference["classification"],
            "whole_panel": suite.summarize(reference["rows"]),
            "matching_initial_context": suite.summarize(
                [r for r in reference["rows"] if r["instance_id"] in compatible]),
            "matching_completed_episodes": suite.summarize(
                [r for r in reference["rows"] if r["instance_id"] in completed]),
            "completed_comparison_is_postselection": True}
    save(root / "run/summary.json", summary)
    return summary


def run(root):
    from scripts import evaluate_graph_release as evaluation
    start = time.monotonic()
    deadline = start + 1800
    registration = verify_registration(root)
    instances, shared = panel()
    assert shared["selection_sha256"] == registration["selection_sha256"]
    output = root / "run"
    output.mkdir()
    process, attempts, fatal, forwards = None, [], None, 0
    initial = {r["instance_id"]: r for r in json.loads((root / "coverage.json").read_text())["rows"]}
    try:
        process = Worker(root, output)
        ready = process.receive(min(300, max(0, deadline - time.monotonic())))
        save(output / "runtime.json", ready)
        assert ready["status"] == "ready"
        with (output / "attempts.jsonl").open("x") as stream:
            for instance in instances:
                history = []
                while (request := evaluation.public_request(instance, history)) is not None:
                    tick = time.monotonic()
                    if tick >= deadline or forwards >= 160:
                        raise TimeoutError("Total CPU job or forward budget exceeded")
                    public = asdict(request)
                    save(output / "status.json", {"status": "running", "instance_id": instance["id"],
                                                  "step": len(history), "forwards": forwards,
                                                  "supervisor_pid": os.getpid()})
                    row = {"instance_id": instance["id"], "step": len(history), "request": public,
                           "status": "failure", "selected_option_id": None, "resolved_model": None,
                           "fatal": False, "forward_count": 0,
                           "usage": {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}}
                    try:
                        if not history and not initial[instance["id"]]["compatible"]:
                            result = {"status": "unsupported", "coverage": initial[instance["id"]],
                                      "forward_count": 0}
                        else:
                            # Reserve one forward before sending: timed-out inference counts too.
                            forwards += 1
                            row["forward_count"] = 1
                            result = process.predict(public, min(30, deadline - tick))
                            if result.get("forward_count") == 0:
                                forwards -= 1
                                row["forward_count"] = 0
                        row["backend"] = result
                        row["diagnostic_code"] = result["status"]
                        if result["status"] == "success":
                            answer = result["answer"]
                            validate_answer(public, answer, result["raw_logits"])
                            row.update(status="success", selected_option_id=answer["choice"],
                                       resolved_model=MODEL, probabilities=answer["probabilities"],
                                       probability_kind="native_typed_choice_softmax_rounded_4dp",
                                       usage=result["usage"])
                        elif result["status"] != "unsupported":
                            row["fatal"] = True
                    except Exception as exc:
                        row.update(fatal=True, diagnostic_code=type(exc).__name__, error=str(exc)[:1000])
                    row["latency_seconds"] = time.monotonic() - tick
                    append(stream, row)
                    attempts.append(row)
                    if row["fatal"]:
                        fatal = row["diagnostic_code"]
                    if row["status"] != "success":
                        break
                    history.append(row["selected_option_id"])
                if fatal:
                    break
    except BaseException as exc:
        fatal = f"{type(exc).__name__}: {str(exc)[:1000]}"
    finally:
        if process:
            process.close()
        summary = summarize(root, instances, attempts, fatal, forwards, time.monotonic() - start)
        save(output / "status.json", {k: summary[k] for k in (
            "status", "fatal", "scheduled_episodes", "official_forwards", "wall_seconds",
            "execution_categories")})
    verify_registration(root)
    print(json.dumps({k: summary[k] for k in (
        "status", "fatal", "official_forwards", "execution_categories", "wall_seconds")}))


def replay(root):
    verify_registration(root)
    from scripts import evaluate_graph_release as evaluation
    instances, _ = panel()
    attempts = read_rows(root / "run/attempts.jsonl") if (root / "run/attempts.jsonl").exists() else []
    actual = evaluation.replay(instances, attempts, SimpleNamespace(model=MODEL), native_action=False)
    saved = json.loads((root / "run/scores.json").read_text())
    for new, old in zip(actual, saved):
        assert all(old[k] == v for k, v in new.items())
    assert len(saved) == len(actual) == 64
    for row in attempts:
        if row["status"] == "success":
            validate_answer(row["request"], row["backend"]["answer"], row["backend"]["raw_logits"])
    result = {"verified": True, "scheduled_episodes": 64, "attempts": len(attempts),
              "scores_sha256": digest(root / "run/scores.json"),
              "attempts_sha256": digest(root / "run/attempts.jsonl")}
    save(root / "run/offline-replay.json", result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("coverage", "worker", "smoke", "register", "run", "replay"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.mode == "register":
        register(args.root, args.baseline)
    else:
        {"coverage": prepare_coverage, "worker": worker, "smoke": smoke,
         "run": run, "replay": replay}[args.mode](args.root)


if __name__ == "__main__":
    main()
