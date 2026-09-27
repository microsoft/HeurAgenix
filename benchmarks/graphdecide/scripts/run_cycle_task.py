#!/usr/bin/env python3
"""Independent cycle lanes with explicit completed-predecessor handoff."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import cycle_task as task
from scripts import large_graph_matrix as predecessor
from scripts import paired_graph_ablation as shared
from src.clients.base import DecisionClientError, DecisionRequest, DecisionOption, InvalidResponseError

KIND = "cycle-seven-real-derived-v1"
ARTIFACTS = ("requests.jsonl", "labels.jsonl", "source-manifest.json", "prior-graphs.jsonl")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sources():
    pins = shared.code_hashes()
    for name in ("cycle_task.py", "run_cycle_task.py", "large_graph_matrix.py"):
        pins[f"scripts/{name}"] = shared.digest(REPO / "scripts" / name)
    return pins


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def validate_inputs(root):
    origin = shared.read_json(root / "source-manifest.json")
    shared.require(origin["kind"] == "cycle-matched-edge-subgraphs-v1" and origin["seed"] == task.SEED,
                   "Unexpected offline plan.")
    shared.require(shared.digest(root / "prior-graphs.jsonl") == origin["prior"]["sha256"], "Source graph pin mismatch.")
    records, skipped = [], []
    for source in shared.read_rows(root / "prior-graphs.jsonl"):
        pair = task.build_pair(source)
        if pair is None:
            skipped.append(source["graph_id"])
        else:
            records.extend(pair)
    random.Random(task.SEED).shuffle(records)
    # JSON-normalize tuples in request dataclasses before exact comparison.
    records = json.loads(json.dumps(records))
    requests, labels = shared.read_rows(root / "requests.jsonl"), shared.read_rows(root / "labels.jsonl")
    shared.require(requests == [r["request"] for r in records] and labels == [r["label"] for r in records],
                   "Regenerated requests/labels differ.")
    shared.require(len(requests) == origin["questions_per_model"] == 2 * origin["generated_pairs"]
                   and origin["seven_model_call_budget"] == 7 * len(requests)
                   and skipped == [s["graph_id"] for s in origin["skipped"]], "Coverage/budget mismatch.")
    return requests, labels


def plan(root, input_root, prior_run):
    root, input_root, prior_run = (Path(p).resolve() for p in (root, input_root, prior_run))
    root.mkdir(parents=True, exist_ok=False)
    original = shared.read_json(input_root / "manifest.json")
    for filename in ("requests.jsonl", "labels.jsonl"):
        shared.require(shared.digest(input_root / filename) == original["artifact_sha256"][filename], "Offline input changed.")
    paths = {"requests.jsonl": input_root / "requests.jsonl", "labels.jsonl": input_root / "labels.jsonl",
             "source-manifest.json": input_root / "manifest.json", "prior-graphs.jsonl": Path(original["prior"]["path"])}
    for name, path in paths.items():
        with (root / name).open("xb") as stream:
            stream.write(path.read_bytes())
    requests, _ = validate_inputs(root)
    _, prior_manifest = predecessor.load(prior_run, check_code=False)
    shared.require(original["prior"]["sha256"] == shared.digest(prior_run / "inputs/adjacency/graphs.jsonl"),
                   "Cycle source does not match declared predecessor.")
    settings = shared.model_configs("answer_only")
    m = {"kind": KIND, "root": str(root), "created_at": utc(), "models": list(settings),
         "calls_per_model": len(requests), "max_total_calls": len(settings) * len(requests),
         "settings": {n: {"provider": c.provider, "kwargs": c.client_kwargs()} for n, c in settings.items()},
         "input_sha256": {name: shared.digest(root / name) for name in ARTIFACTS}, "code_sha256": sources(),
         "predecessor": {"root": str(prior_run), "manifest_sha256": shared.digest(prior_run / "manifest.json")},
         "notice": "Real-derived matched n/m cycle pairs; not held-out induced graphs. No retries/resume or answer repair. Select only completed predecessor model lanes; GPT can be launched separately later."}
    shared.write_json(root / "manifest.json", m)
    with (root / "manifest.sha256").open("x") as stream:
        stream.write(shared.digest(root / "manifest.json") + "\n")
    return m


def load(root, check_code=True):
    root = Path(root).resolve()
    shared.require(shared.digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip(), "Manifest changed.")
    m = shared.read_json(root / "manifest.json")
    shared.require(m["kind"] == KIND and m["root"] == str(root) and m["models"] == list(shared.MODELS), "Plan identity mismatch.")
    if check_code:
        shared.require(sources() == m["code_sha256"], "Source drift.")
    shared.require(m["input_sha256"] == {name: shared.digest(root / name) for name in ARTIFACTS}, "Input pin mismatch.")
    settings = shared.model_configs("answer_only")
    shared.require(m["settings"] == {n: {"provider": c.provider, "kwargs": c.client_kwargs()} for n,c in settings.items()},
                   "Settings mismatch.")
    requests, labels = shared.read_rows(root / "requests.jsonl"), shared.read_rows(root / "labels.jsonl")
    shared.require(len(requests) == len(labels) == m["calls_per_model"] and m["max_total_calls"] == 7 * len(requests), "Budget mismatch.")
    return root, m, requests, labels


def check_predecessor(m, models):
    root = Path(m["predecessor"]["root"])
    shared.require(shared.digest(root / "manifest.json") == m["predecessor"]["manifest_sha256"], "Predecessor plan changed.")
    _, prior = predecessor.load(root, check_code=False)
    for model in models:
        entries = [e for e in prior["entries"] if e["model"] == model]
        shared.require(len(entries) == 2, "Missing predecessor tasks.")
        for e in entries:
            shared.completed_rows(root, e)


@contextmanager
def lane_locks(models):
    """Cycle-specific exclusion plus ordinary locks when predecessor releases them.

    The legacy large runner holds ALL model locks until the slowest GPT exits,
    even after Qwen is finished and closed. Admission therefore additionally
    requires complete, independently validated predecessor results per model.
    Never release or change another process's locks; never admit a running lane.
    """
    descriptors = []
    try:
        for model in sorted(models):
            filename = Path(tempfile.gettempdir()) / f"jevgraphbench-cycle-{os.getuid()}-{model}.lock"
            fd = os.open(filename, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            descriptors.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = Path(tempfile.gettempdir()) / f"jevgraphbench-paired-{os.getuid()}-{model}.lock"
            fd = os.open(old, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            descriptors.append(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass  # The verified completed-predecessor handoff is required below.
        yield
    finally:
        for fd in descriptors:
            os.close(fd)


def metrics(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None,
            "failures": sum(r["status"] != "success" for r in rows),
            "answers": dict(Counter(r["selected_option_id"] or "<missing>" for r in rows)),
            "errors": dict(Counter(r["error_type"] for r in rows if r["error_type"])),
            "mean_seconds": sum(r["elapsed_seconds"] for r in rows) / n if n else None}


def summarize(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["answer"]] = row
    complete = [p for p in pairs.values() if set(p) == {"yes", "no"}]
    return {"overall": metrics(rows),
            "by_node_count": {str(n): metrics([r for r in rows if r["node_count"] == n]) for n in sorted({r["node_count"] for r in rows})},
            "by_dataset": {d: metrics([r for r in rows if r["dataset"] == d]) for d in sorted({r["dataset"] for r in rows})},
            "by_answer": {a: metrics([r for r in rows if r["answer"] == a]) for a in ("yes", "no")},
            "pairs": {"n": len(complete), "both_correct": sum(p["yes"]["correct"] and p["no"]["correct"] for p in complete)}}


def report(root, models):
    root, m, requests, labels = load(root, check_code=False)
    models = shared.selected_models(models)
    results = {}
    for model in models:
        directory = root / "runs" / model
        status = shared.read_json(directory / "status.json")
        rows = shared.read_rows(directory / "attempts.jsonl")
        shared.require(status["status"] == "completed" and status["completed_calls"] == len(rows) == len(requests)
                       and status["settings"] == m["settings"][model]
                       and status["input_sha256"] == m["input_sha256"], "Incomplete or changed lane.")
        for row, label in zip(rows, labels):
            shared.require(all(row.get(k) == v for k,v in label.items()), "Label/order mismatch.")
            good = row["status"] == "success"
            shared.require(row["status"] in ("success", "failure") and type(row["correct"]) is bool
                           and (row["selected_option_id"] in ("yes", "no") if good else row["selected_option_id"] is None)
                           and row["correct"] == (good and row["selected_option_id"] == label["answer"]), "Score mismatch.")
        results[model] = summarize(rows)
        shared.require(results[model] == shared.read_json(directory / "summary.json"), "Summary mismatch.")
    return {"kind": KIND, "models": results, "notice": m["notice"], "status": "completed-selected-lanes"}


async def run(root, models, *, factory=None, preflight=True):
    root, m, requests, labels = load(root)
    models = shared.selected_models(models)
    validate_inputs(root)
    settings = shared.model_configs("answer_only", models=models)
    clients = {}
    with lane_locks(models):
        if preflight:
            check_predecessor(m, models)  # Before reservation or any network operation.
        directories = [root / "runs" / model for model in models]
        shared.require(not any(p.exists() or p.is_symlink() for p in directories), "No automatic retry/resume.")
        for directory in directories:
            directory.mkdir(parents=True)
        try:
            if preflight:
                shared.credential_preflight(models)
                await shared.vllm_preflight([SimpleNamespace(model=c) for c in settings.values()])
            for model, config in settings.items():
                kwargs = config.client_kwargs()
                if config.provider == "vllm":
                    kwargs["api_key"] = ""
                clients[model] = (shared.create_client if factory is None else factory)(config.provider, **kwargs)
            for model, client in clients.items():
                await asyncio.wait_for(client.initialize(), settings[model].timeout_seconds)

            async def lane(model):
                directory, client = root / "runs" / model, clients[model]
                status = {"status": "running", "started_at": utc(), "completed_calls": 0,
                          "expected_calls": len(requests), "settings": m["settings"][model],
                          "input_sha256": m["input_sha256"], "injected_client": factory is not None,
                          "response_protocol": "native-choice" if model == "jev" else "answer-only-v1",
                          "adapter_retries": 0, "sdk_retries": "not_controlled" if settings[model].provider == "github_copilot" else "not_applicable"}
                shared.write_json(directory / "status.json", status)
                rows = []
                try:
                    with (directory / "attempts.jsonl").open("x") as stream:
                        for payload, label in zip(requests, labels):
                            shared.require(sources() == m["code_sha256"], "Source drift during evaluation.")
                            request = DecisionRequest(**{**payload, "options": tuple(DecisionOption(**o) for o in payload["options"])})
                            row = {**label, "started_at": utc(), "status": "failure", "correct": False,
                                   "selected_option_id": None, "error_type": None, "http_status": None,
                                   "diagnostic_code": None, "finish_reason": None, "usage": None, "resolved_model": None}
                            started = time.perf_counter()
                            try:
                                response = await asyncio.wait_for(client.predict(request), settings[model].timeout_seconds)
                                if response.request_id != request.request_id or response.selected_option_id not in ("yes", "no"):
                                    raise InvalidResponseError("Response ID/choice mismatch")
                                row.update(status="success", selected_option_id=response.selected_option_id,
                                           correct=response.selected_option_id == label["answer"],
                                           usage=asdict(response.usage), resolved_model=response.resolved_model)
                            except (DecisionClientError, TimeoutError) as exc:
                                row.update(error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                                if isinstance(exc, InvalidResponseError):
                                    safe = InvalidResponseError("", diagnostic_code=exc.diagnostic_code, finish_reason=exc.finish_reason, usage=exc.usage)
                                    row.update(diagnostic_code=safe.diagnostic_code, finish_reason=safe.finish_reason,
                                               usage=asdict(safe.usage) if safe.usage else None)
                                if row["http_status"] in (400, 401, 403, 413, 422, 429):
                                    raise shared.Refusal("Systemic provider failure; stopping selected lanes.") from None
                            except BaseException as exc:
                                row.update(status="aborted", error_type=type(exc).__name__)
                                raise
                            finally:
                                row["elapsed_seconds"] = time.perf_counter() - started
                                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                                stream.flush()
                                rows.append(row)
                                status["completed_calls"] = len(rows)
                                save(directory / "status.json", status)
                    status["status"] = "completed"
                except BaseException as exc:
                    status.update(status="aborted", error_type=type(exc).__name__)
                    raise
                finally:
                    status["finished_at"] = utc()
                    save(directory / "status.json", status)
                    shared.write_json(directory / "summary.json", summarize(rows))

            tasks = [asyncio.create_task(lane(model)) for model in models]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        finally:
            await asyncio.gather(*(c.aclose() for c in clients.values()), return_exceptions=True)
    result = report(root, models)
    (root / "reports").mkdir(exist_ok=True)
    shared.write_json(root / "reports" / ("-".join(models) + ".json"), result)
    if all((root / "runs" / model / "summary.json").exists() for model in m["models"]):
        shared.write_json(root / "report.json", report(root, m["models"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--models", nargs="+", choices=shared.MODELS, default=list(shared.LEGACY_MODELS))
    args = parser.parse_args()
    try:
        if args.command == "plan":
            shared.require(args.input_root is not None and args.prior_run is not None, "Plan requires both source roots.")
            m = plan(args.root, args.input_root, args.prior_run)
            print(json.dumps({k: m[k] for k in ("root", "calls_per_model", "max_total_calls")}))
        elif args.command == "run":
            result = asyncio.run(run(args.root, args.models))
            print(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps(report(args.root, args.models), sort_keys=True))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Cycle run stopped ({type(exc).__name__}); no retry/resume; details suppressed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())