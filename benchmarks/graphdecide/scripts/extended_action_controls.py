#!/usr/bin/env python3
"""Authored forced-action controls for suspicious all-one-side construction."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts.extended_response_diagnostic import diagnostics
from src.benchmark import extended_tasks as tasks
from src.clients.base import DecisionClientError, DecisionOption
from src.clients.registry import create_client


def cases():
    rows = []
    for task in ("maxcut_construct", "community_bipartition_construct"):
        for index, (edges, history) in enumerate((([(0, 1)], []), ([(1, 2)], ["B"]))):
            instance = tasks._record(task, index, tasks._graph(5, edges))
            request = tasks.next_request(instance, history)
            values = {}
            for action in ("A", "B"):
                possible = [tasks.score(instance, history + [action] + list(tail))["objective"]
                            for tail in product(("A", "B"),
                                                repeat=instance["query_budget"] - len(history) - 1)]
                values[action] = max(possible)
            expected = max(values, key=values.get)
            if values["A"] == values["B"]:
                raise ValueError("Authored control does not force an action")
            for arm in ("baseline", "described_options", "reversed_options"):
                options = request.options
                if arm == "described_options":
                    options = tuple(DecisionOption(o.id, f"Assign the current vertex to side {o.id}.")
                                    for o in options)
                elif arm == "reversed_options":
                    options = options[::-1]
                rows.append((replace(request, request_id=f"{request.request_id}:{arm}", options=options),
                             {"task": task, "case": index, "arm": arm, "expected": expected}))
    return rows


async def run(root, models, *, controls=None, purpose=None, provenance=None,
              constrain_local_choices=False):
    configs = suite.model_configs(models, seed=20260928, no_think_max_tokens=64)
    suite.require(type(constrain_local_choices) is bool, "Select a boolean local choice constraint.")
    if constrain_local_choices:
        for name, config in configs.items():
            if config.provider == "vllm":
                suite.require(config.think is False, "Choice constraints require no-thinking models.")
                configs[name] = replace(config, constrain_choices=True)
    controls = cases() if controls is None else controls
    suite.require(bool(controls), "Empty control plan.")
    suite.require(not root.exists(), "Control directory already exists.")
    with suite.lane_locks(models):
        await suite.shared.vllm_preflight([SimpleNamespace(model=c) for c in configs.values()])
        root.mkdir(parents=True, exist_ok=False)
        suite.write_json(root / "manifest.json", {
            "purpose": purpose or "Development diagnosis only; forced optimal next actions on single-edge graphs.",
            "provenance": provenance,
            "constrain_local_choices": constrain_local_choices,
            "models": {name: asdict(c) for name, c in configs.items()},
            "cases": [{"request": asdict(request), "label": label} for request, label in controls],
            "code_sha256": {**suite.code_hashes(), str(Path(__file__).relative_to(REPO)):
                            suite.digest(__file__)},
            "planned_calls": len(controls) * len(configs),
        })
        results = {}
        for name, config in configs.items():
            rows = []
            async with create_client(config.provider, **config.client_kwargs()) as client:
                await client.initialize()
                with (root / f"{name}.jsonl").open("x") as stream:
                    for request, label in controls:
                        started, body = time.perf_counter(), None
                        try:
                            response = await client.predict(request)
                            body = response.raw_output
                            record = suite.response_record(response, request, config)
                        except DecisionClientError as exc:
                            body = getattr(exc, "raw_output", None)
                            record = {"status": "failure", "selected_option_id": None,
                                      **suite.failure(exc)}
                        expected = label.get("optimal_option_ids")
                        if expected is None:
                            expected = [label["expected"]]
                        row = {**label, **record, "request_id": request.request_id,
                               "latency_seconds": time.perf_counter() - started,
                               "correct": record["selected_option_id"] in expected}
                        if config.provider == "typesafe":
                            row["native_diagnostics"] = diagnostics(request, body)
                        suite.append_row(stream, row)
                        rows.append(row)
                        if record["status"] == "failure" and record["fatal"]:
                            raise suite.Refusal("Fatal control-lane failure; partial rows preserved.")
            results[name] = {"correct": sum(r["correct"] for r in rows), "attempts": len(rows),
                             "failures": sum(r["status"] == "failure" for r in rows)}
        suite.write_json(root / "summary.json", results)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=suite.MODELS, default=["jev", "qwen4"])
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.root, args.models)), sort_keys=True))


if __name__ == "__main__":
    main()
