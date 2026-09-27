"""Sequential single-decision runs, append-only attempt logs and explicit failures."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from contextlib import AsyncExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Callable

from src.clients.base import BaseDecisionClient, DecisionClientError, InvalidResponseError
from src.clients.registry import create_client
from src.datasets import GraphDataset, load_graph
from .config import BenchmarkConfig
from .prepare import PreparedBenchmark, prepare
from .progress import EvaluationProgress


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_json(row) + "\n")


def _percentile(values: list[float], fraction: float) -> float | None:
    """Linear interpolation between sorted observations (including singleton)."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _latency(rows: list[dict]) -> dict:
    values = [r["elapsed_seconds"] for r in rows]
    return {"count": len(values), "mean_seconds": statistics.mean(values) if values else None,
            "p50_seconds": _percentile(values, 0.5), "p95_seconds": _percentile(values, 0.95),
            "total_seconds": sum(values)}


def _metrics(rows: list[dict]) -> dict:
    count = len(rows)
    success = [r for r in rows if r["status"] == "success"]
    correct = sum(r["correct"] for r in rows)
    usage = {}
    for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
        known = [r["usage"][field] for r in success if r["usage"][field] is not None]
        usage[field] = {"known_sum": sum(known) if known else None, "reported_calls": len(known)}
    brier = [(r["probabilities"]["yes"] - (r["answer"] == "yes")) ** 2
             for r in success if r["probabilities"] is not None]
    return {
        "attempts": count, "unique_questions": len({r["request_id"] for r in rows}),
        "unique_graphs": len({r["graph_id"] for r in rows}),
        "successes": len(success), "failures": count - len(success), "correct": correct,
        "accuracy": correct / count if count else None,
        "accuracy_on_success": correct / len(success) if success else None,
        "failure_rate": (count - len(success)) / count if count else None,
        "label_counts": dict(Counter(r["answer"] for r in rows)),
        "error_counts": dict(Counter(r["error_type"] for r in rows if r["status"] != "success")),
        "diagnostic_counts": dict(Counter(r["diagnostic_code"] for r in rows
                         if r["status"] != "success" and r.get("diagnostic_code"))),
        "finish_reason_counts": dict(Counter(r["finish_reason"] for r in rows
                            if r["status"] != "success" and r.get("finish_reason"))),
        "resolved_models": dict(Counter(r["resolved_model"] for r in success)),
        "latency_all": _latency(rows), "latency_success": _latency(success),
        "token_usage": usage, "brier_yes": statistics.mean(brier) if brier else None,
        "probability_calls": len(brier),
        "cost": None,  # No invented pricing; token use is not a billing receipt.
    }


def summarize(rows: list[dict], *, expected_calls: int, preparation: dict) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"], row["node_count"], row["threshold"])].append(row)
    coverage = {}
    for cell in preparation.get("strata", []):
        key = (cell["dataset"], cell["task"], cell["node_count"], cell["threshold"])
        groups.setdefault(key, [])  # Retain unavailable strata with null accuracy.
        coverage[key] = cell
    return {
        "schema_version": 1, "expected_calls": expected_calls,
        "completed_calls": len(rows),
        "execution_coverage": len(rows) / expected_calls if expected_calls else None,
        "preparation": preparation, "overall": _metrics(rows),
        "by_task": {task: _metrics([r for r in rows if r["task"] == task])
                    for task in sorted({r["task"] for r in rows})},
        "by_dataset": {name: _metrics([r for r in rows if r["dataset"] == name])
                   for name in sorted({r["dataset"] for r in rows})},
        "by_node_count": {str(n): _metrics([r for r in rows if r["node_count"] == n])
                  for n in sorted({r["node_count"] for r in rows})},
        "by_threshold": {str(k): _metrics([r for r in rows if r["threshold"] == k])
                 for k in sorted({r["threshold"] for r in rows if r["threshold"] is not None})},
        "by_repetition": {str(rep): _metrics([r for r in rows if r["repetition"] == rep])
                          for rep in sorted({r["repetition"] for r in rows})},
        "strata": [{"dataset": key[0], "task": key[1], "node_count": key[2],
                    "threshold": key[3], "generation": coverage.get(key),
                    **_metrics(value)} for key, value in sorted(groups.items())],
    }


def export_prepared(output: Path, prepared: PreparedBenchmark) -> dict[str, str]:
    _write_jsonl(output / "requests.jsonl", (asdict(e.request) for e in prepared.examples))
    _write_jsonl(output / "labels.jsonl", (e.label_record() for e in prepared.examples))
    _write_jsonl(output / "graphs.jsonl", prepared.graphs)
    _write_json(output / "sources.json", prepared.sources)
    _write_json(output / "preparation.json", prepared.summary())
    hashes = {}
    for name in ("requests.jsonl", "labels.jsonl", "graphs.jsonl", "sources.json", "preparation.json"):
        with (output / name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


async def run_experiment(
    config: BenchmarkConfig, *,
    client: BaseDecisionClient | None = None,
    loader: Callable[..., GraphDataset] = load_graph,
    progress: bool = False,
    prepared_input: PreparedBenchmark | None = None,
) -> dict:
    """Prepare samples and run evaluation. Owns/closes the client once entered.

    An injected client is only for tests/custom callers; the CLI never offers a
    mock model whose numbers could be mistaken for actual Jev performance.
    A trusted orchestrator may provide prevalidated, pinned prepared_input to
    avoid CPU-bound regeneration blocking other asynchronous model lanes.
    The orchestrator must verify the exported artifacts before any prediction.
    Progress is opt-in for library callers and goes to stderr, not artifacts.
    """
    if config.planned_questions * config.run.repetitions > config.run.max_calls:
        raise ValueError("Planned calls exceed max_calls")
    output = config.run.output_dir
    output.mkdir(parents=True, exist_ok=False)  # Never append to or overwrite a prior run.
    _write_json(output / "config.json", config.snapshot())
    started = time.perf_counter()
    run = {
        "schema_version": 1, "status": "preparing",
        "config_sha256": config.sha256, "started_at_utc": _utc(),
        "requested_model": config.model.model, "provider": config.model.provider,
        "model_settings": config.model.client_kwargs(),
        "response_protocol": ("native-choice" if config.model.provider == "typesafe" else
                      "answer-only-v1" if config.model.output_format == "answer_only" else
                      "json-choice-v1"),
        "versions": {"python": platform.python_version(), "networkx": version("networkx"),
                     "httpx": version("httpx"), "pyyaml": version("PyYAML")},
        "protocol": "topology-choice-v1", "concurrency": 1,
        "retries": None if config.model.provider == "github_copilot" else 0,
        "adapter_retries": 0,
        "sdk_retries": "not_controlled" if config.model.provider == "github_copilot" else "not_applicable",
        "injected_client": client is not None,
    }
    if config.model.provider == "github_copilot":
        try:
            run["versions"]["github-copilot-sdk"] = version("github-copilot-sdk")
        except PackageNotFoundError:
            run["versions"]["github-copilot-sdk"] = None
    if prepared_input is not None:
        run["injected_prepared"] = True
    _write_json(output / "run.json", run)
    rows: list[dict] = []
    prepared = None
    run_started = None
    display = EvaluationProgress(progress)
    try:
        display.preparing(config.planned_graphs)
        prepared = prepare(config, loader) if prepared_input is None else prepared_input
        if (prepared.planned_graphs != config.planned_graphs
                or prepared.planned_questions != config.planned_questions
                or len(prepared.examples) * config.run.repetitions > config.run.max_calls):
            raise ValueError("Prepared input does not match the configured budget")
        run["preparation_seconds"] = time.perf_counter() - started
        if prepared_input is None:
            run["artifact_sha256"] = export_prepared(output, prepared)
        else:
            # Each lane exports to its own directory. Keep large JSON encoding
            # and disk writes off the event loop while sibling requests run.
            run["artifact_sha256"] = await asyncio.to_thread(export_prepared, output, prepared)
        expected = len(prepared.examples) * config.run.repetitions
        run["expected_calls"] = expected
        if not expected:
            raise ValueError("No questions generated; inspect preparation.json before executing")
        display.start(len(prepared.graphs), len(prepared.examples), config.run.repetitions)
        run["status"] = "running"
        _write_json(output / "run.json", run)
        if client is None:
            # Adapters read credentials internally, never from configuration.
            client = create_client(config.model.provider, **config.model.client_kwargs())
        run_started = time.perf_counter()
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(client)
            initialization_started = time.perf_counter()
            await asyncio.wait_for(client.initialize(), timeout=config.model.timeout_seconds)
            run["initialization_seconds"] = time.perf_counter() - initialization_started
            stream = stack.enter_context((output / "attempts.jsonl").open("x", encoding="utf-8"))
            for repetition in range(config.run.repetitions):
                for example in prepared.examples:
                    row = {**example.label_record(), "repetition": repetition,
                           "attempt_id": f"{example.request.request_id}-r{repetition:03d}",
                           "started_at_utc": _utc(), "status": "failure", "correct": False,
                           "selected_option_id": None, "resolved_model": None,
                           "probabilities": None, "probability_kind": None, "confidence": None,
                           "usage": {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None},
                           "error_type": None, "http_status": None,
                           "diagnostic_code": None, "finish_reason": None}
                    call_started = time.perf_counter()
                    try:
                        response = await asyncio.wait_for(client.predict(example.request),
                                                          timeout=config.model.timeout_seconds)
                        if response.request_id != example.request.request_id or response.selected_option_id not in ("yes", "no"):
                            raise InvalidResponseError("Response does not match request")
                        row.update(status="success", correct=response.selected_option_id == example.answer,
                                   selected_option_id=response.selected_option_id,
                                   resolved_model=response.resolved_model, probabilities=response.probabilities,
                                   probability_kind=response.probability_kind, confidence=response.confidence,
                                   usage=asdict(response.usage))
                    except (DecisionClientError, TimeoutError) as exc:
                        # Never save exception text, raw provider bodies or credentials.
                        row.update(error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                        if isinstance(exc, InvalidResponseError):
                            # Revalidate at the persistence boundary: custom clients
                            # can mutate exception attributes after construction.
                            safe = InvalidResponseError(
                                "", diagnostic_code=exc.diagnostic_code,
                                finish_reason=exc.finish_reason, usage=exc.usage,
                            )
                            row.update(diagnostic_code=safe.diagnostic_code, finish_reason=safe.finish_reason)
                            if safe.usage is not None:
                                row["usage"] = asdict(safe.usage)
                    except BaseException as exc:
                        # A cancelled/interrupted request may already be billable.
                        # Retain this attempt, but do not swallow cancellation or bugs.
                        row.update(status="aborted", error_type=type(exc).__name__)
                        raise
                    finally:
                        row["elapsed_seconds"] = time.perf_counter() - call_started
                        stream.write(_json(row) + "\n")
                        stream.flush()
                        rows.append(row)
                        display.advance(correct=row["correct"], status=row["status"])
        run["status"] = "completed"
        return summarize(rows, expected_calls=expected, preparation=prepared.summary())
    except BaseException as exc:
        run.update(status="aborted", error_type=type(exc).__name__)
        raise
    finally:
        run["finished_at_utc"] = _utc()
        run["total_seconds"] = time.perf_counter() - started
        run["execution_seconds"] = time.perf_counter() - run_started if run_started is not None else None
        run["completed_calls"] = len(rows)
        _write_json(output / "run.json", run)
        if prepared is not None:
            summary = summarize(rows, expected_calls=len(prepared.examples) * config.run.repetitions,
                                preparation=prepared.summary())
            summary["run_status"] = run["status"]
            summary["execution_seconds"] = run["execution_seconds"]
            _write_json(output / "summary.json", summary)
        display.finish(run["status"])