#!/usr/bin/env python3
"""Replay failed Jev public requests; save allowlisted probability diagnostics only."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import paired_graph_ablation as shared
from src.benchmark import extended_tasks
from src.clients.base import DecisionClientError, InvalidResponseError
from src.clients.registry import create_client


def diagnostics(request, body):
    answers = body.get("answers") if isinstance(body, dict) else None
    answer = answers.get("decision") if isinstance(answers, dict) else None
    answer = answer if isinstance(answer, dict) else {}
    probabilities = answer.get("probabilities")
    ids = {o.id for o in request.options}
    selected = answer.get("choice")
    numeric = (isinstance(probabilities, dict) and all(
        type(v) in (int, float) and math.isfinite(v) for v in probabilities.values()))
    return {
        "option_count": len(ids),
        "choice_is_string": isinstance(selected, str),
        "choice_is_supplied": isinstance(selected, str) and selected in ids,
        "probability_keys_match": isinstance(probabilities, dict) and set(probabilities) == ids,
        "probability_count": len(probabilities) if isinstance(probabilities, dict) else None,
        "probability_sum": sum(probabilities.values()) if numeric else None,
        "probability_range_valid": numeric and all(0 <= p <= 1 for p in probabilities.values()),
        "selected_probability": probabilities.get(selected)
        if numeric and isinstance(selected, str) and selected in ids else None,
        "max_probability": max(probabilities.values()) if numeric and probabilities else None,
    }


async def run(root, output, *, limit=16, repetitions=1, lock_wait_seconds=0):
    manifest = shared.read_json(root / "manifest.json")
    shared.require(shared.digest(extended_tasks.__file__) ==
                   manifest["code_sha256"]["src/benchmark/extended_tasks.py"],
                   "Request-generation source changed; use the matching source snapshot.")
    instances = {i["id"]: i for i in shared.read_rows(root / "instances.jsonl")}
    histories, prior = {}, []
    for row in shared.read_rows(root / "runs/jev/attempts.jsonl"):
        history = histories.setdefault(row["instance_id"], [])
        request = extended_tasks.next_request(instances[row["instance_id"]], history)
        shared.require(json.loads(json.dumps(asdict(request))) == row["request"],
                       "Original request/history mismatch.")
        if row["status"] == "failure":
            prior.append(request)
        else:
            history.append(row["selected_option_id"])
    prior = prior[:limit]
    shared.require(bool(prior), "No failed Jev attempts to diagnose.")
    shared.require(not output.exists(), "Diagnostic output exists.")
    deadline = time.monotonic() + lock_wait_seconds
    while True:
        lock = shared.lane_locks(("jev",))
        try:
            lock.__enter__()
            break
        except shared.Refusal:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(5)
    try:
        output.mkdir(parents=True, exist_ok=False)
        shared.write_json(output / "manifest.json", {
            "source_manifest_sha256": shared.digest(root / "manifest.json"),
            "diagnostic_code_sha256": shared.digest(Path(__file__)),
            "client_code_sha256": shared.digest(REPO / "src/clients/typesafe.py"),
            "planned_replays": len(prior) * repetitions,
            "repetitions_per_failed_request": repetitions,
            "request_reconstruction": "next_request from frozen instance and original history; preserve state key order",
            "notice": "New diagnostic attempts, not replacements for original failures; no raw provider text.",
        })
        config = shared.model_configs("answer_only", models=("jev",))["jev"]
        async with create_client(config.provider, **config.client_kwargs()) as client:
            with (output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
                for trial, request in ((trial, row) for trial in range(repetitions) for row in prior):
                    started = time.perf_counter()
                    row = {"request": asdict(request), "trial": trial,
                           "status": "success", "error": None}
                    try:
                        response = await client.predict(request)
                        body = response.raw_output
                    except InvalidResponseError as exc:
                        body = exc.raw_output
                        allowed = {"probabilities must sum to 1",
                                   "choice does not have the highest probability",
                                   "probabilities must cover exactly the supplied options",
                                   "choice is not a supplied option"}
                        row.update(status="failure", error=str(exc) if str(exc) in allowed
                                   else "other_invalid_response")
                    except DecisionClientError as exc:
                        row.update(status="failure", error=type(exc).__name__)
                        body = None
                    row.update(diagnostics=diagnostics(request, body),
                               latency_seconds=time.perf_counter() - started)
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
        return {"diagnostic_attempts": len(prior) * repetitions, "output": str(output)}
    finally:
        lock.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--lock-wait-seconds", type=int, default=0)
    args = parser.parse_args()
    shared.require(args.limit > 0 and args.repetitions > 0 and args.lock_wait_seconds >= 0,
                   "Invalid diagnostic budget.")
    print(json.dumps(asyncio.run(run(args.root, args.output, limit=args.limit,
                                    repetitions=args.repetitions,
                                    lock_wait_seconds=args.lock_wait_seconds))))


if __name__ == "__main__":
    main()
