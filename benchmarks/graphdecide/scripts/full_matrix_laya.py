#!/usr/bin/env python3
"""Full canonical Laya matrix: immutable registration, lossless CPU inference."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scripts import laya_cpu_pilot as pilot
from scripts import evaluate_graph_release as evaluation
from scripts import graph_release_api as release
from scripts import extended_graph_suite as suite
from src.benchmark import extended_tasks as tasks

ROOT = REPO / "output/full-matrix-laya-20260925-v1"
BUNDLE = REPO / "output/graphbench-portable-release-20260924-v2"
LIMITS = {**pilot.LIMITS, "job_seconds": 86400, "max_forwards": 1920,
          "maximum_attempts": 1920, "scheduled_episodes": 960}


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def public_request(instance, history):
    request = evaluation.public_request(instance, history)
    return None if request is None else canonical(asdict(request))


def panel():
    manifest = release.manifest_at(BUNDLE)
    instances = release.labeled_instances(BUNDLE, "structural_challenge", manifest)
    assert len(instances) == len({i["id"] for i in instances}) == 960
    assert sum(i["query_budget"] for i in instances) == 1920
    return instances


def process_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "startticks": int(fields[19])}


def tokenizer():
    pilot.imports(pilot.ROOT)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(pilot.ROOT / "model/tokenizer",
                                        local_files_only=True, trust_remote_code=False)


def pinned_artifacts():
    old = json.loads((pilot.ROOT / "registration.json").read_text())
    assert pilot.digest(pilot.ROOT / "registration.json") == json.loads(
        (pilot.ROOT / "registration.sha256").read_text())
    paths = {}
    for name, sha in old["files_sha256"].items():
        if name.startswith(str(pilot.ROOT.relative_to(REPO))) or name == "scripts/laya_cpu_pilot.py":
            assert pilot.digest(REPO / name) == sha, name
            paths[name] = sha
    assert pilot.digest(pilot.ROOT / "model/model.safetensors") == pilot.WEIGHTS_SHA
    return paths


def prepare(root):
    if root.exists():
        raise RuntimeError("New output required; no overwrite or resume")
    root.mkdir(parents=True)
    pinned = pinned_artifacts()
    validation = release.verify(BUNDLE)
    instances = panel()
    tok = tokenizer()
    coverage, request_count = [], 0
    with (root / "offline-request-validation.jsonl").open("x") as stream:
        for instance in instances:
            history = []
            while (request := public_request(instance, history)) is not None:
                # Compare independently against the portable bundle's public API.
                expected = release.request_for(BUNDLE, "structural_challenge",
                                               instance["id"], history,
                                               protocol="explicit_actions")
                assert request == canonical(expected)
                assert "private" not in request["state"]
                check = pilot.coverage(tok, request)
                if not history:
                    coverage.append({"instance_id": instance["id"], "task": instance["task"],
                                     "node_count": len(instance["state"]["nodes"]), **check})
                pilot.append(stream, {"instance_id": instance["id"], "history": history[:],
                                      "request": request, "coverage": check,
                                      "validation_only_not_model_history": True})
                history.append(request["options"][0]["id"])
                request_count += 1
            assert len(history) == instance["query_budget"]
    assert request_count == 1920
    pilot.save(root / "coverage.json", {"scheduled": 960, "model_calls": 0,
                                       "compatible": sum(r["compatible"] for r in coverage),
                                       "rows": coverage})
    pilot.save(root / "offline-validation.json", {
        "bundle": validation, "canonical_requests_checked": request_count,
        "reference_objectives_independently_verified": True,
        "scheduled_episodes": 960, "model_calls": 0,
        "cpu_available": len(os.sched_getaffinity(0)), "cpu_threads_required": 4,
        "load_average": list(os.getloadavg()), "pinned_artifacts_verified": len(pinned)})
    assert len(os.sched_getaffinity(0)) >= 4
    hashes = {**pinned, **evaluation.code_hashes()}
    for path in [Path(__file__), REPO / "tests/test_full_matrix_laya.py",
                 root / "coverage.json", root / "offline-validation.json",
                 root / "offline-request-validation.jsonl", BUNDLE / "manifest.json",
                 BUNDLE / "manifest.sha256"]:
        hashes[str(path.relative_to(REPO))] = pilot.digest(path)
    for name, sha in release.manifest_at(BUNDLE)["sha256"].items():
        hashes[str((BUNDLE / name).relative_to(REPO))] = sha
    for name, sha in hashes.items():
        if name.endswith(".py"):
            target = root / "frozen-source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
            target.chmod(0o444)
            assert pilot.digest(target) == sha
    registration = {
        "registered_at_unix": time.time(), "registered_before_any_new_model_calls": True,
        "model": pilot.MODEL, "source_revision": pilot.SOURCE,
        "checkpoint_revision": pilot.REVISION, "checkpoint_sha256": pilot.WEIGHTS_SHA,
        "bundle": str(BUNDLE), "split": "structural_challenge",
        "protocol": "explicit_actions", "history": "model-own-history-only",
        "limits": LIMITS, "retries": 0, "repair": False, "sample_selection": False,
        "unsupported_policy": "explicit failure retained in full 960 denominator",
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "files_sha256": hashes}
    pilot.save(root / "registration.json", registration)
    pilot.save(root / "registration.sha256", pilot.digest(root / "registration.json"))
    (root / "registration.json").chmod(0o444)
    verify_registration(root)
    print(json.dumps({"registered": True, "scheduled": 960,
                      "initial_compatible": sum(r["compatible"] for r in coverage)}))


def verify_registration(root):
    registration = json.loads((root / "registration.json").read_text())
    assert pilot.digest(root / "registration.json") == json.loads(
        (root / "registration.sha256").read_text())
    for name, sha in registration["files_sha256"].items():
        assert pilot.digest(REPO / name) == sha, f"Frozen input changed: {name}"
    assert registration["packages"] == {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    return registration


def derive(root, attempts, fatal, elapsed):
    instances = panel()
    scores = evaluation.replay(instances, canonical(attempts),
                               SimpleNamespace(model=pilot.MODEL), native_action=False)
    initial = {r["instance_id"]: r for r in json.loads(
        (root / "coverage.json").read_text())["rows"]}
    by_id = {}
    for attempt in attempts:
        by_id.setdefault(attempt["instance_id"], []).append(attempt)
    for row in scores:
        own = by_id.get(row["instance_id"], [])
        compatible = initial[row["instance_id"]]["compatible"]
        row.update(initial_context_compatible=compatible, scheduled=True,
                   forward_count=sum(a["forward_count"] for a in own),
                   execution_category=(
                       "unsupported_initial" if not compatible else
                       "unsupported_growing_context" if any(
                           a["diagnostic_code"] == "unsupported" for a in own) else
                       "completed" if row["feasible"] else "failed_or_unattempted"))
    summary = {"status": "aborted" if fatal else "completed", "fatal": fatal,
               "scheduled_episodes": 960, "official_forwards": sum(
                   a["forward_count"] for a in attempts), "attempts": len(attempts),
               "wall_seconds": elapsed,
               "execution_categories": dict(Counter(r["execution_category"] for r in scores)),
               "whole_panel": suite.summarize(scores),
               "initial_context_compatible": suite.summarize(
                   [r for r in scores if r["initial_context_compatible"]]),
               "option_count_histogram": dict(Counter(
                   len(a["request"]["options"]) for a in attempts))}
    return canonical(scores), canonical(summary)


def verify_results(root):
    verify_registration(root)
    out = root / "run"
    attempts = pilot.read_rows(out / "attempts.jsonl")
    saved = json.loads((out / "summary.json").read_text())
    scores, summary = derive(root, attempts, saved["fatal"], saved["wall_seconds"])
    assert scores == json.loads((out / "scores.json").read_text())
    assert summary == saved
    assert len(attempts) <= 1920 and summary["official_forwards"] <= 1920
    if not saved["fatal"]:
        assert all(r["feasible"] or r["failure_count"] for r in scores)
    runtime = json.loads((out / "runtime.json").read_text()) if (out / "runtime.json").exists() else {}
    checked_native = 0
    for row in attempts:
        if row["status"] != "success":
            continue
        result = row["backend"]
        pilot.validate_answer(row["request"], result["answer"], result["raw_logits"])
        ids = [o["id"] for o in row["request"]["options"]]
        k = len(ids)
        bucket = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        temperature = runtime["temperature_by_options_applied"].get(
            f"choice:{bucket}", runtime["temperature"][0])
        logits = [v / temperature for v in result["raw_logits"]]
        weights = [math.exp(v - max(logits)) for v in logits]
        assert max(abs(w / sum(weights) - result["answer"]["probabilities"][i])
                   for w, i in zip(weights, ids)) <= .000051
        assert result["coverage"]["compatible"] and result["cuda_initialized"] is False
        assert result["usage"]["input_tokens"] == result["coverage"]["full_tokens"]
        checked_native += 1
    objective_checks = 0
    for instance, row in zip(panel(), scores):
        if instance["kind"] == "optimization" and row["feasible"]:
            evaluate, _ = tasks._independent_objectives(instance["task"], instance["state"])
            assert math.isclose(evaluate(row["solution"]), row["objective"],
                                rel_tol=0, abs_tol=1e-12)
            objective_checks += 1
    report = {"verified": True, "scheduled_episodes": 960, "model_calls": 0,
              "canonical_own_history_replay": True, "native_score_checks": checked_native,
              "independent_completed_objective_checks": objective_checks,
              "attempts_sha256": pilot.digest(out / "attempts.jsonl"),
              "scores_sha256": pilot.digest(out / "scores.json"),
              "summary_sha256": pilot.digest(out / "summary.json")}
    if (out / "offline-replay.json").exists():
        assert report == json.loads((out / "offline-replay.json").read_text())
    else:
        pilot.save(out / "offline-replay.json", report)
    return report


def run(root):
    start = time.monotonic()
    deadline = start + LIMITS["job_seconds"]
    verify_registration(root)
    out = root / "run"
    out.mkdir()
    pilot.save(out / "supervisor-process.json", {
        **process_identity(os.getpid()), "started_at_unix": time.time(),
        "limits": LIMITS, "device": "cpu", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    initial = {r["instance_id"]: r for r in json.loads(
        (root / "coverage.json").read_text())["rows"]}
    process, fatal, forwards = None, None, 0
    try:
        process = pilot.Worker(pilot.ROOT, out)
        pilot.save(out / "worker-identity.json", process_identity(process.process.pid))
        ready = process.receive(min(300, deadline - time.monotonic()))
        pilot.save(out / "runtime.json", ready)
        assert ready["status"] == "ready" and ready["dtype"] == "float32"
        assert ready["threads"] == 4 and ready["interop_threads"] == 1
        assert ready["cuda_initialized"] is False
        with (out / "attempts.jsonl").open("x") as stream:
            for instance in panel():
                history = []
                while (public := public_request(instance, history)) is not None:
                    tick = time.monotonic()
                    row = {"instance_id": instance["id"], "step": len(history), "request": public,
                           "status": "failure", "selected_option_id": None, "resolved_model": None,
                           "fatal": False, "forward_count": 0,
                           "usage": {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None}}
                    pilot.save(out / "status.json", {
                        "status": "running", "instance_id": instance["id"], "step": len(history),
                        "forwards": forwards, "updated_at_unix": time.time(),
                        **process_identity(os.getpid())})
                    try:
                        if tick >= deadline:
                            raise TimeoutError("24-hour job deadline")
                        if not history and not initial[instance["id"]]["compatible"]:
                            result = {"status": "unsupported", "coverage": initial[instance["id"]],
                                      "forward_count": 0}
                        else:
                            if forwards >= LIMITS["max_forwards"]:
                                raise RuntimeError("Forward budget exhausted")
                            forwards += 1
                            row["forward_count"] = 1
                            result = process.predict(public, min(30, deadline - tick))
                            if result.get("forward_count") == 0:
                                forwards -= 1
                                row["forward_count"] = 0
                        row.update(backend=result, diagnostic_code=result["status"])
                        if result["status"] == "success":
                            pilot.validate_answer(public, result["answer"], result["raw_logits"])
                            row.update(status="success", selected_option_id=result["answer"]["choice"],
                                       resolved_model=pilot.MODEL, usage=result["usage"],
                                       probabilities=result["answer"]["probabilities"],
                                       probability_kind="native_typed_choice_softmax_rounded_4dp")
                        elif result["status"] != "unsupported":
                            row["fatal"] = True
                    except Exception as exc:
                        row.update(fatal=True, diagnostic_code=type(exc).__name__, error=str(exc)[:1000])
                    row["latency_seconds"] = time.monotonic() - tick
                    pilot.append(stream, row)
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
        if not (out / "attempts.jsonl").exists():
            (out / "attempts.jsonl").touch()
        attempts = pilot.read_rows(out / "attempts.jsonl")
        scores, summary = derive(root, attempts, fatal, time.monotonic() - start)
        pilot.save(out / "scores.json", scores)
        pilot.save(out / "summary.json", summary)
        pilot.save(out / "status.json", summary)
    verification = verify_results(root)
    pilot.save(out / "completion.json", {
        "status": summary["status"], "fatal": fatal, "finished_at_unix": time.time(),
        "scheduled_episodes": 960, "official_forwards": forwards,
        "sealed_replay": verification, "registration_sha256": pilot.digest(root / "registration.json")})
    print(json.dumps({k: summary[k] for k in (
        "status", "fatal", "official_forwards", "execution_categories")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.mode in ("prepare", "run") and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES='' required")
    result = {"prepare": prepare, "run": run, "verify": verify_results}[args.mode](args.root)
    if result:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
