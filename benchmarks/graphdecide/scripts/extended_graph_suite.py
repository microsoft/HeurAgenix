#!/usr/bin/env python3
"""Frozen synthetic graph plans and fail-closed, iterative model lanes."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import paired_graph_ablation as shared
from src.benchmark import extended_tasks
from src.clients.base import (
    ClientTimeoutError, DecisionClientError, DecisionResponse, InvalidResponseError,
    ProviderHTTPError, TokenUsage,
)
from src.clients.registry import create_client

Refusal = shared.Refusal
require = shared.require
digest = shared.digest
MODELS = (*shared.MODELS, "qwen9think", "qwen9recommended", "qwen9thinkrecommended",
          "qwen9constrained")
DEFAULT_MODELS = shared.MODELS
QWEN9_LANES = {"qwen9", "qwen9think", "qwen9recommended", "qwen9thinkrecommended",
               "qwen9constrained"}
SEED = 20260926
SOURCE_SNAPSHOT = "frozen-source"
TERMINAL = {"completed", "completed_with_failures", "aborted", "interrupted"}
USAGE_KEYS = ("input_tokens", "output_tokens", "reasoning_tokens")
DIAGNOSTICS = InvalidResponseError._DIAGNOSTIC_CODES | {
    "timeout", "http_error", "client_error", "internal_error", "interrupted",
    "preflight_failed", "initialization_failed", "shutdown_failed",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def environment_versions():
    packages = {name: version(name) for name in ("networkx", "httpx", "PyYAML")}
    try:
        packages["github-copilot-sdk"] = version("github-copilot-sdk")
    except PackageNotFoundError:
        packages["github-copilot-sdk"] = None
    return {"python": platform.python_version(), "packages": packages}


def json_value(value):
    return json.loads(canonical(value))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def append_row(stream, row):
    stream.write(canonical(row) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def selected_models(models):
    names = tuple(models)
    require(bool(names) and len(set(names)) == len(names) and set(names) <= set(MODELS),
            "Select distinct supported model names.")
    return names


def model_configs(models=DEFAULT_MODELS, *, seed=SEED, no_think_max_tokens=4096):
    require(type(no_think_max_tokens) is int and no_think_max_tokens > 0,
            "No-think token limit must be a positive integer.")
    configs = shared.model_configs("answer_only")
    configs["qwen9think"] = replace(configs["qwen9"], think=True, max_tokens=8192)
    configs["qwen9recommended"] = replace(
        configs["qwen9"], think=False, temperature=1.0, top_p=1.0,
        top_k=40, presence_penalty=2.0)
    configs["qwen9thinkrecommended"] = replace(
        configs["qwen9"], think=True, temperature=1.0, top_p=0.95,
        top_k=20, presence_penalty=1.5, max_tokens=8192, timeout_seconds=360.0)
    configs["qwen9constrained"] = replace(configs["qwen9"], constrain_choices=True)
    return {name: replace(configs[name], seed=seed, max_tokens=(
                no_think_max_tokens if configs[name].think is False else configs[name].max_tokens))
            if configs[name].provider == "vllm" else configs[name]
            for name in selected_models(models)}


def code_hashes():
    paths = {Path(__file__).resolve(), Path(shared.__file__).resolve(), REPO / "src" / "__init__.py"}
    if (REPO / "scripts" / "__init__.py").is_file():
        paths.add(REPO / "scripts" / "__init__.py")
    for directory in ("benchmark", "clients", "datasets"):
        paths.update((REPO / "src" / directory).rglob("*.py"))
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(paths)}


def verify_source_snapshot(root, hashes):
    snapshot = root / SOURCE_SNAPSHOT
    require(snapshot.is_dir() and not snapshot.is_symlink(), "Frozen source snapshot missing.")
    require({p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*.py")} == set(hashes),
            "Frozen source snapshot file set mismatch.")
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".py",
                "Invalid frozen source path.")
        path = snapshot / relative
        require(not any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents),
                "Symlinked frozen source refused.")
        require(path.is_file() and digest(path) == expected, "Frozen source snapshot hash mismatch.")
        require(path.stat().st_mode & 0o222 == 0, "Frozen source files must remain read-only.")


def freeze_source(root, hashes):
    snapshot = root / SOURCE_SNAPSHOT
    snapshot.mkdir(exist_ok=False)
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".py",
                "Invalid source path.")
        source = REPO / relative
        require(not source.is_symlink() and source.resolve().is_relative_to(REPO),
                "Symlinked source refused.")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as original, target.open("xb") as frozen:
            frozen.write(original.read())
            frozen.flush()
            os.fsync(frozen.fileno())
        require(digest(target) == expected, "Source changed while freezing.")
        target.chmod(0o444)
    verify_source_snapshot(root, hashes)
    require(code_hashes() == hashes, "Source changed while freezing.")


def budgets(instances, model_count):
    per_model = sum(i["query_budget"] for i in instances)
    return {"instances_per_model": len(instances), "max_calls_per_model": per_model,
            "max_calls_all_models": per_model * model_count,
            "by_task": {task: {"instances": sum(i["task"] == task for i in instances),
                              "max_calls_per_model": sum(i["query_budget"] for i in instances
                                                         if i["task"] == task)}
                        for task in sorted({i["task"] for i in instances})}}


def plan(root, *, design="pilot", seed=SEED, samples_per_size=2, models=DEFAULT_MODELS,
         node_counts=None, no_think_max_tokens=4096, prior_pool=None):
    names = selected_models(models)
    require(design in ("pilot", "curriculum", "structural_dev", "structural_holdout"), "Unknown design.")
    require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer.")
    require(type(samples_per_size) is int and samples_per_size > 0,
            "Samples per size must be positive.")
    require(design != "pilot" or node_counts is None,
            "Node-count selection does not apply to the legacy pilot.")
    require((prior_pool is not None) == (design == "structural_holdout"),
            "Only structural_holdout requires a pinned prior graph pool.")
    default_sizes = range(8, 13) if design == "structural_holdout" else range(5, 13)
    sizes = tuple(default_sizes) if node_counts is None else tuple(node_counts)
    require(bool(sizes) and all(type(n) is int and 5 <= n <= 12 for n in sizes)
            and len(set(sizes)) == len(sizes), "Select distinct node counts from 5 through 12.")
    require(design != "structural_holdout" or all(n >= 8 for n in sizes),
            "The structural holdout is defined only for sizes 8 through 12.")
    configs = model_configs(names, seed=seed, no_think_max_tokens=no_think_max_tokens)
    root = Path(root).resolve()
    require(not root.exists(), "Plan root already exists; use a fresh root.")
    before = code_hashes()
    metadata, pool_data, pool_sha = None, None, None
    if design == "curriculum":
        from src.benchmark.extended_tasks import build_curriculum
        instances = build_curriculum(seed=seed, node_counts=sizes,
                                     samples_per_size=samples_per_size)
    elif design in ("structural_dev", "structural_holdout"):
        from src.benchmark.structural_tasks import build_structural_bank
        prior = []
        if prior_pool is not None:
            prior_pool = Path(prior_pool)
            require(prior_pool.is_file() and not prior_pool.is_symlink(),
                    "Prior pool must be a regular nonsymlinked file.")
            prior_pool = prior_pool.resolve()
            pool_data = prior_pool.read_bytes()
            pool_sha = hashlib.sha256(pool_data).hexdigest()
            prior = [json.loads(line) for line in pool_data.decode("utf-8").splitlines()]
            require(bool(prior), "Structural holdout requires a nonempty prior pool.")
        instances, metadata = build_structural_bank(
            seed=seed, node_counts=sizes, repetitions=samples_per_size, prior_instances=prior)
        require(all(len(i["state"]["nodes"]) in sizes for i in instances),
                "Generated graph size differs from the requested structural sizes.")
        require(set(metadata["cases"]) == {i["id"] for i in instances},
                "Structural metadata must cover every generated instance.")
        require(all(isinstance(metadata["cases"][i["id"]].get("cluster_id"), str)
                    and metadata["cases"][i["id"]]["cluster_id"] for i in instances),
                "Structural instances need nonempty generation cluster IDs.")
        if prior_pool is not None:
            require(digest(prior_pool) == pool_sha, "Prior pool changed during generation.")
        require({i["task"] for i in instances} == set(
            (*extended_tasks._ALL_EXACT, *extended_tasks._OPTIMIZATION)),
            "Structural design must cover all ten tasks; inspect generation coverage.")
    else:
        instances = extended_tasks.build_instances(seed=seed)
    validation = extended_tasks.validate_instances(instances)
    require(bool(instances), "Empty instance plan.")
    require(code_hashes() == before, "Code changed during planning.")
    root.mkdir(parents=True, exist_ok=False)
    with (root / "instances.jsonl").open("x", encoding="utf-8") as stream:
        for instance in instances:
            append_row(stream, instance)
    input_hashes = {"instances.jsonl": digest(root / "instances.jsonl")}
    if metadata is not None:
        write_json(root / "design_metadata.json", metadata)
        input_hashes["design_metadata.json"] = digest(root / "design_metadata.json")
    if pool_data is not None:
        with (root / "prior_graph_pool.jsonl").open("xb") as stream:
            stream.write(pool_data)
        require(digest(root / "prior_graph_pool.jsonl") == pool_sha, "Copied prior pool mismatch.")
        input_hashes["prior_graph_pool.jsonl"] = pool_sha
    freeze_source(root, before)
    manifest = {
        "schema_version": 1, "design": design, "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_versions(),
        "no_think_max_tokens": no_think_max_tokens,
        "node_counts": list(sizes) if design != "pilot" else None,
        "samples_per_size": samples_per_size if design != "pilot" else None,
        "models": {name: asdict(config) for name, config in configs.items()},
        "input_sha256": input_hashes,
        "code_sha256": before, "source_snapshot": SOURCE_SNAPSHOT, "validation": validation,
        "budgets": budgets(instances, len(names)),
        "policy": {"retries": 0, "repair": False, "resume": False,
                   "failure": "stop instance; fatal failures abort model lane",
                   "private_labels_sent": False},
    }
    if prior_pool is not None:
        manifest["prior_pool_source"] = {"path": str(prior_pool), "sha256": pool_sha}
    write_json(root / "manifest.json", manifest)
    write_json(root / "manifest.sha256", digest(root / "manifest.json"))
    return {"status": "planned", "design": design, "models": list(names),
            "no_think_max_tokens": no_think_max_tokens, "budgets": manifest["budgets"]}


def load_plan(root):
    root = Path(root).resolve()
    require(not (root / "manifest.json").is_symlink()
            and not (root / "instances.jsonl").is_symlink(), "Symlinked inputs refused.")
    require(digest(root / "manifest.json") == read_json(root / "manifest.sha256"),
            "Manifest hash mismatch.")
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == 1, "Unsupported manifest.")
    require(manifest["code_sha256"] == code_hashes(), "Frozen source code changed.")
    if "source_snapshot" in manifest:
        require(manifest["source_snapshot"] == SOURCE_SNAPSHOT, "Unknown source snapshot layout.")
        verify_source_snapshot(root, manifest["code_sha256"])
    input_names = {"instances.jsonl"}
    if manifest["design"] in ("structural_dev", "structural_holdout"):
        input_names.add("design_metadata.json")
    if manifest["design"] == "structural_holdout":
        input_names.add("prior_graph_pool.jsonl")
    require(set(manifest["input_sha256"]) == input_names, "Unexpected frozen input files.")
    require(all(not (root / name).is_symlink() and digest(root / name) ==
                manifest["input_sha256"][name] for name in input_names),
            "Frozen input hash mismatch.")
    configs = model_configs(tuple(manifest["models"]), seed=manifest["seed"],
                            no_think_max_tokens=manifest.get("no_think_max_tokens", 4096))
    require(manifest["models"] == {n: asdict(c) for n, c in configs.items()},
            "Frozen model configuration mismatch.")
    instances = read_rows(root / "instances.jsonl")
    require(manifest["budgets"] == budgets(instances, len(configs)), "Budget mismatch.")
    require(len({i["id"] for i in instances}) == len(instances), "Duplicate instance ID.")
    return root, manifest, instances, configs


@contextmanager
def lane_locks(models):
    with shared.lane_locks(sorted({"qwen9" if n in QWEN9_LANES else n for n in models})):
        yield


def safe_usage(usage):
    result = {key: None for key in USAGE_KEYS}
    if type(usage) is TokenUsage:
        for key in USAGE_KEYS:
            value = getattr(usage, key)
            if value is None or type(value) is int and value >= 0:
                result[key] = value
    return result


def failure(exc):
    code, http_status, fatal = "internal_error", None, True
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        code = "interrupted"
    elif isinstance(exc, ProviderHTTPError):
        code = "http_error"
        if type(exc.status_code) is int and 100 <= exc.status_code <= 599:
            http_status = exc.status_code
        fatal = not (http_status == 408 or http_status is not None and 500 <= http_status <= 599)
    elif isinstance(exc, ClientTimeoutError):
        code, fatal = "timeout", False
    elif isinstance(exc, InvalidResponseError):
        code = exc.diagnostic_code if exc.diagnostic_code in DIAGNOSTICS else "invalid_response"
        fatal = code in {"finish_error", "finish_abort"}
    elif isinstance(exc, DecisionClientError):
        # SDK transport errors may conceal auth/quota/context failures; fail closed.
        code = "client_error"
    return {"diagnostic_code": code, "http_status": http_status, "fatal": fatal}


def response_record(response, request, config):
    if (not isinstance(response, DecisionResponse)
            or response.request_id != request.request_id
            or type(response.selected_option_id) is not str
            or response.selected_option_id not in {o.id for o in request.options}):
        raise InvalidResponseError("Invalid decision response", diagnostic_code="invalid_choice")
    # Do not persist arbitrary provider text as a purported model identifier.
    if response.resolved_model != config.model:
        raise InvalidResponseError("Unrecognized resolved model", diagnostic_code="invalid_model")
    if type(response.usage) is not TokenUsage or any(
            v is not None and (type(v) is not int or v < 0)
            for v in (getattr(response.usage, k) for k in USAGE_KEYS)):
        raise InvalidResponseError("Invalid token usage", diagnostic_code="invalid_usage")
    return {"status": "success", "selected_option_id": response.selected_option_id,
            "resolved_model": response.resolved_model, "usage": safe_usage(response.usage),
            "diagnostic_code": None, "http_status": None, "fatal": False}


def validate_attempt(row, request, config):
    require(set(row) == {"instance_id", "step", "request", "status", "selected_option_id",
                         "resolved_model", "usage", "diagnostic_code", "http_status",
                         "fatal", "latency_seconds"}, "Unexpected attempt fields.")
    require(row["request"] == json_value(asdict(request)), "Public request replay mismatch.")
    require(type(row["step"]) is int and row["step"] >= 0, "Invalid step.")
    latency = row["latency_seconds"]
    require(type(latency) in (int, float) and math.isfinite(latency) and latency >= 0,
            "Invalid latency.")
    require(type(row["fatal"]) is bool and type(row["usage"]) is dict
            and set(row["usage"]) == set(USAGE_KEYS)
            and all(v is None or type(v) is int and v >= 0 for v in row["usage"].values()),
            "Invalid response metadata.")
    if row["status"] == "success":
        require(type(row["selected_option_id"]) is str
                and row["selected_option_id"] in {o.id for o in request.options}
                and row["resolved_model"] == config.model and row["diagnostic_code"] is None
                and row["http_status"] is None and not row["fatal"], "Invalid success.")
    else:
        require(row["status"] == "failure" and row["selected_option_id"] is None
                and row["resolved_model"] is None and row["diagnostic_code"] in DIAGNOSTICS
                and (row["http_status"] is None or type(row["http_status"]) is int
                     and 100 <= row["http_status"] <= 599), "Invalid failure.")


def replay(instances, attempts, config):
    """Check exact request/decision history, then score every planned instance."""
    cursor, aborted, scored = 0, False, []
    for instance in instances:
        decisions, own = [], []
        while cursor < len(attempts) and attempts[cursor]["instance_id"] == instance["id"]:
            require(not aborted and not (own and own[-1]["status"] == "failure"),
                    "Calls continued after a failure.")
            row = attempts[cursor]
            request = extended_tasks.next_request(instance, decisions)
            require(request is not None and row["step"] == len(decisions),
                    "Calls exceeded budget or history order.")
            validate_attempt(row, request, config)
            own.append(row)
            cursor += 1
            if row["status"] == "success":
                decisions.append(row["selected_option_id"])
            else:
                aborted = row["fatal"]
        result = extended_tasks.score(instance, decisions)
        if cursor < len(attempts):
            require(result["feasible"] or bool(own) and own[-1]["status"] == "failure",
                    "History skipped an unfinished instance.")
        row = {"instance_id": instance["id"], "task": instance["task"], "kind": instance["kind"],
               "node_count": len(instance["state"]["nodes"]),
               "edge_count": len(instance["state"]["edges"]),
               "query_budget": instance["query_budget"], "query_count": len(own),
               "successful_queries": len(decisions),
               "failure_count": sum(r["status"] == "failure" for r in own),
               "decisions": decisions, **result}
        if instance["kind"] == "optimization":
            baseline = instance["private"]["baseline"]
            gap = abs(baseline["objective"] - instance["private"]["objective"])
            row.update(baseline_method=baseline["method"], baseline_objective=baseline["objective"],
                       baseline_absolute_gap=gap,
                       baseline_optimal=math.isclose(gap, 0, rel_tol=0, abs_tol=1e-12))
        else:
            row.update(answer=str(instance["private"]["answer"]),
                       prediction=decisions[0] if decisions else None)
        scored.append(row)
    require(cursor == len(attempts), "Unknown/out-of-order attempt instance.")
    return scored


def mean(values):
    return sum(values) / len(values) if values else None


def metrics(rows):
    feasible = [r for r in rows if r["feasible"]]
    result = {"instances": len(rows), "feasible": len(feasible),
              "incomplete_or_invalid": len(rows) - len(feasible),
              "unattempted": sum(r["query_count"] == 0 for r in rows),
              "attempts": sum(r["query_count"] for r in rows),
              "failure_count": sum(r["failure_count"] for r in rows),
              "instances_with_failure": sum(r["failure_count"] > 0 for r in rows),
              "feasibility_rate": len(feasible) / len(rows),
              "rate_denominator": "all planned instances",
              "feasible_mean_denominator": len(feasible)}
    if rows[0]["kind"] == "optimization":
        result.update(
            mean_objective_feasible=mean([r["objective"] for r in feasible]),
            mean_absolute_gap_feasible=mean([r["absolute_gap"] for r in feasible]),
            optimal_count=sum(r["optimal"] for r in rows),
            optimality_rate=sum(r["optimal"] for r in rows) / len(rows),
            optimality_rate_feasible=mean([int(r["optimal"]) for r in feasible]),
            baseline_mean_objective=mean([r["baseline_objective"] for r in rows]),
            baseline_mean_absolute_gap=mean([r["baseline_absolute_gap"] for r in rows]),
            baseline_optimality_rate=mean([int(r["baseline_optimal"]) for r in rows]),
            baseline_denominator=len(rows),
            baseline_mean_objective_on_model_feasible=mean(
                [r["baseline_objective"] for r in feasible]),
            baseline_mean_absolute_gap_on_model_feasible=mean(
                [r["baseline_absolute_gap"] for r in feasible]))
    else:
        result.update(correct=sum(r["correct"] for r in rows),
                      accuracy=sum(r["correct"] for r in rows) / len(rows),
                      accuracy_feasible=mean([int(r["correct"]) for r in feasible]),
                      label_counts=dict(Counter(r["answer"] for r in rows)),
                      prediction_counts=dict(Counter(r["prediction"] for r in feasible)),
                      missing_predictions=len(rows) - len(feasible))
    return result


def summarize(rows):
    tasks = sorted({r["task"] for r in rows})
    by_task = {task: metrics([r for r in rows if r["task"] == task]) for task in tasks}
    by_size = {
        task: {str(n): metrics([r for r in rows if r["task"] == task and r["node_count"] == n])
               for n in sorted({r["node_count"] for r in rows if r["task"] == task})}
        for task in tasks}
    for task in tasks:
        if "accuracy" in by_task[task]:
            by_task[task]["macro_size_accuracy"] = mean(
                [cell["accuracy"] for cell in by_size[task].values()])
            by_task[task]["macro_size_denominator"] = len(by_size[task])
    return {"denominators": "Rates include every planned instance, including unattempted/failed; "
                            "objective/gap means use feasible instances only. "
                            "Classical baseline all-instance and matched-feasible means are separate. "
                            "Exact macro_size_accuracy equally weights each planned graph size.",
            "by_task": by_task, "by_task_and_size": by_size}


def set_status(output, status):
    next_path = output / "status.next.json"
    write_json(next_path, status)
    os.replace(next_path, output / "status.json")


async def run_lane(root, manifest, instances, name, config, *, client_factory, preflight,
                   endpoint_lock):
    output = root / "runs" / name
    lane_started = time.perf_counter()
    status = {"schema_version": 1, "model": name, "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest(root / "manifest.json"), "diagnostic_code": None,
              "injected_client": client_factory is not create_client}
    set_status(output, status)
    attempts, client, interrupted = [], None, None
    with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
        try:
            async with endpoint_lock:
                try:
                    await preflight([SimpleNamespace(model=config)])
                except Exception:
                    status.update(status="aborted", diagnostic_code="preflight_failed")
                    return
                try:
                    client = client_factory(config.provider, **config.client_kwargs())
                    await client.initialize()
                except Exception:
                    status.update(status="aborted", diagnostic_code="initialization_failed")
                    return
                require(code_hashes() == manifest["code_sha256"]
                        and all(digest(root / name) == value
                                for name, value in manifest["input_sha256"].items()),
                        "Frozen artifacts changed before prediction.")
                for instance in instances:
                    decisions = []
                    while (request := extended_tasks.next_request(instance, decisions)) is not None:
                        public = json_value(asdict(request))
                        started = time.perf_counter()
                        try:
                            client.validate_request(request)
                            response = await client.predict(request)
                            record = response_record(response, request, config)
                        except BaseException as exc:
                            record = {"status": "failure", "selected_option_id": None,
                                      "resolved_model": None,
                                      "usage": safe_usage(getattr(exc, "usage", None)), **failure(exc)}
                            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                                interrupted = exc
                        record.update(instance_id=instance["id"], step=len(decisions), request=public,
                                      latency_seconds=time.perf_counter() - started)
                        append_row(stream, record)
                        attempts.append(record)
                        if record["status"] == "failure":
                            if record["fatal"]:
                                status.update(status="interrupted" if interrupted else "aborted",
                                              diagnostic_code=record["diagnostic_code"])
                            break
                        decisions.append(record["selected_option_id"])
                    if status["status"] != "running":
                        break
                if status["status"] == "running":
                    status["status"] = ("completed_with_failures" if any(
                        a["status"] == "failure" for a in attempts) else "completed")
        except BaseException as exc:
            status.update(status="interrupted" if isinstance(
                exc, (asyncio.CancelledError, KeyboardInterrupt)) else "aborted",
                          diagnostic_code=failure(exc)["diagnostic_code"])
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                interrupted = exc
        finally:
            if client is not None:
                try:
                    await client.aclose()
                except BaseException:
                    if status["status"] in ("completed", "completed_with_failures"):
                        status.update(status="aborted", diagnostic_code="shutdown_failed")
            rows = replay(instances, attempts, config)
            with (output / "scores.jsonl").open("x", encoding="utf-8") as scores:
                for row in rows:
                    append_row(scores, row)
            write_json(output / "summary.json", summarize(rows))
            status["artifact_sha256"] = {file: digest(output / file)
                                        for file in ("attempts.jsonl", "scores.jsonl", "summary.json")}
            status["attempts"] = len(attempts)
            status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            status["wall_seconds"] = time.perf_counter() - lane_started
            set_status(output, status)
    if interrupted is not None:
        raise interrupted


async def run_live(root, models=None, *, client_factory=create_client,
                   preflight=shared.vllm_preflight):
    root, manifest, instances, configs = load_plan(root)
    names = selected_models(tuple(configs) if models is None else models)
    require(set(names) <= set(configs), "Selected model was not frozen in this plan.")
    with lane_locks(names):
        for name in names:
            output = root / "runs" / name
            require(not output.exists() and not output.is_symlink(),
                    "Selected lane output already exists; no resume or overwrite.")
        (root / "runs").mkdir(exist_ok=True)
        require(not (root / "runs").is_symlink(), "Symlinked run directory refused.")
        for name in names:
            (root / "runs" / name).mkdir(exist_ok=False)
        locks = {}
        await asyncio.gather(*(run_lane(
            root, manifest, instances, name, configs[name], client_factory=client_factory,
            preflight=preflight, endpoint_lock=locks.setdefault(
                configs[name].base_url or name, asyncio.Lock())) for name in names))
    return report(root, names)


def report(root, models=None):
    root, manifest, instances, configs = load_plan(root)
    names = selected_models(tuple(configs) if models is None else models)
    require(set(names) <= set(configs), "Report model was not frozen in plan.")
    result = {"design": manifest["design"], "budgets": manifest["budgets"],
              "no_think_max_tokens": manifest.get("no_think_max_tokens", 4096), "models": {}}
    for name in names:
        output = root / "runs" / name
        if not output.exists():
            result["models"][name] = {"status": "not_started"}
            continue
        require(not output.is_symlink(), "Symlinked output refused.")
        status = read_json(output / "status.json")
        require(status["model"] == name and status["status"] in TERMINAL,
                "Lane is unsealed/active; report refuses incomplete artifact verification.")
        require(status["diagnostic_code"] is None or status["diagnostic_code"] in DIAGNOSTICS,
                "Invalid lane diagnostic.")
        require(status["manifest_sha256"] == digest(root / "manifest.json"),
                "Lane manifest mismatch.")
        require(set(status["artifact_sha256"]) == {"attempts.jsonl", "scores.jsonl", "summary.json"},
                "Missing lane artifact hashes.")
        for file, expected in status["artifact_sha256"].items():
            require(not (output / file).is_symlink() and digest(output / file) == expected,
                    "Lane artifact hash mismatch.")
        attempts = read_rows(output / "attempts.jsonl")
        require(status["attempts"] == len(attempts), "Attempt count mismatch.")
        rows = replay(instances, attempts, configs[name])
        summary = summarize(rows)
        require(rows == read_rows(output / "scores.jsonl")
                and summary == read_json(output / "summary.json"), "Recomputed scores differ.")
        if status["status"] in ("completed", "completed_with_failures"):
            require(all(r["feasible"] or r["failure_count"] == 1 for r in rows),
                    "Completed lane contains unattempted/incomplete instances.")
            require(not any(a["fatal"] for a in attempts), "Completed lane contains fatal failure.")
            require((status["status"] == "completed") == all(
                a["status"] == "success" for a in attempts), "Completion status mismatch.")
        result["models"][name] = {"status": status["status"], "attempts": len(attempts),
                                  "diagnostic_code": status["diagnostic_code"],
                                  "injected_client": status["injected_client"], **summary}
    return result


async def run_cli(root, models):
    """CLI-only SIGTERM cancellation lets active lanes seal their known prefixes."""
    loop, task = asyncio.get_running_loop(), asyncio.current_task()

    def stop():
        if not task.cancelling():
            task.cancel()

    loop.add_signal_handler(signal.SIGTERM, stop)
    try:
        return await run_live(root, models)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run", "report"):
        child = commands.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--models", nargs="+", choices=MODELS)
        if command == "plan":
            child.add_argument("--design", choices=(
                "pilot", "curriculum", "structural_dev", "structural_holdout"), default="pilot")
            child.add_argument("--seed", type=int, default=SEED)
            child.add_argument("--samples-per-size", type=int, default=2)
            child.add_argument("--node-counts", nargs="+", type=int, choices=range(5, 13),
                               help="Sizes 5..12; structural holdout defaults to 8..12")
            child.add_argument("--prior-pool", type=Path,
                               help="Required graph-pool.jsonl for structural_holdout only")
            child.add_argument("--no-think-max-tokens", type=int, default=4096,
                               help="Positive completion-token cap for no-think vLLM lanes only")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.root, design=args.design, seed=args.seed,
                          samples_per_size=args.samples_per_size,
                          models=args.models or DEFAULT_MODELS, node_counts=args.node_counts,
                          no_think_max_tokens=args.no_think_max_tokens,
                          prior_pool=args.prior_pool)
        elif args.command == "run":
            result = asyncio.run(run_cli(args.root, args.models))
        else:
            result = report(args.root, args.models)
        print(canonical(result))
        return int(args.command == "run" and any(
            v["status"] in ("aborted", "interrupted") for v in result["models"].values()))
    except KeyboardInterrupt:
        print('{"status":"interrupted","diagnostic_code":"interrupted"}', file=sys.stderr)
        return 130
    except asyncio.CancelledError:
        print('{"status":"interrupted","diagnostic_code":"interrupted"}', file=sys.stderr)
        return 143
    except Exception:
        # Never echo client exceptions, paths, response text, or credentials.
        print('{"status":"refused","diagnostic_code":"verification_or_execution_failed"}',
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
