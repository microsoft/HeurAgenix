#!/usr/bin/env python3
"""Small, fixed graph sanity and representation-invariance diagnostic, not a leaderboard."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import paired_graph_ablation as shared
from src.benchmark.config import PresentationConfig
from src.benchmark.presentation import question_for
from src.clients.base import DecisionRequest, DecisionOption, DecisionClientError, InvalidResponseError

VARIANTS = ("baseline", "node_permutation", "edge_order", "option_order")
SEED = 20260925


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value, *, exclusive=False):
    """Atomic status updates; only new experiment directories may be written."""
    if exclusive:
        shared.write_json(path, value)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)


def sources():
    hashes = shared.code_hashes()
    hashes[Path(__file__).relative_to(REPO).as_posix()] = shared.digest(Path(__file__))
    return hashes


def visible_truth(request):
    """Independent BFS, parsing query arguments from actual model-visible text."""
    state = request["state"]
    assert set(state) == {"nodes", "edges"} and state["nodes"] == list(range(6))
    adjacent = re.fullmatch(
        r"Use only the given undirected, unweighted graph\. Are nodes (\d+) and (\d+) directly connected by an edge\?",
        request["question"],
    )
    distance = re.fullmatch(
        r"Use only the given undirected, unweighted graph\. Is there a path from node (\d+) to node (\d+) using at most (\d+) edges\?",
        request["question"],
    )
    assert adjacent or distance
    if adjacent:
        u, v = map(int, adjacent.groups())
        threshold = 1
    else:
        u, v, threshold = map(int, distance.groups())
    neighbors = {n: set() for n in state["nodes"]}
    seen = set()
    for a, b in state["edges"]:
        assert a != b and a in neighbors and b in neighbors
        edge = frozenset((a, b))
        assert edge not in seen
        seen.add(edge)
        neighbors[a].add(b)
        neighbors[b].add(a)
    queue, dist = deque([u]), {u: 0}
    while queue:
        x = queue.popleft()
        for y in neighbors[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    return "yes" if dist.get(v, float("inf")) <= threshold else "no"


def examples():
    """Twelve predeclared cases, balanced in each task, each asked four ways.

    Expected answers are authored independently, then checked with visible_truth.
    All IDs/labels/variant names stay outside the model-visible prompt.
    """
    graphs = [
        ("path", [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
         [(0, 1, None, "yes"), (0, 2, None, "no"), (0, 2, 2, "yes"), (0, 3, 2, "no")]),
        ("cycle", [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5]],
         [(5, 0, None, "yes"), (0, 2, None, "no"), (0, 2, 2, "yes"), (0, 3, 2, "no")]),
        ("disconnected_triangles", [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]],
         [(2, 0, None, "yes"), (2, 3, None, "no"), (3, 5, 2, "yes"), (0, 3, 2, "no")]),
    ]
    permutation = [4, 1, 5, 0, 2, 3]
    rows = []
    for graph, original, questions in graphs:
        for index, (u, v, threshold, answer) in enumerate(questions):
            case = f"{graph}-{index}"
            for variant in VARIANTS:
                edges = [e[:] for e in original]
                a, b = u, v
                if variant == "node_permutation":
                    edges = [[permutation[x], permutation[y]] for x, y in edges]
                    a, b = permutation[u], permutation[v]
                if variant == "edge_order":
                    random.Random(SEED).shuffle(edges)
                options = ("no", "yes") if variant == "option_order" else ("yes", "no")
                request = DecisionRequest(
                    request_id=f"{case}.{variant}", state={"nodes": list(range(6)), "edges": edges},
                    question=question_for(a, b, threshold, PresentationConfig()),
                    options=tuple(DecisionOption(o) for o in options),
                )
                record = asdict(request)
                assert visible_truth(record) == answer
                rows.append({"request": record, "label": {
                    "request_id": request.request_id, "case_id": case, "graph": graph,
                    "task": "adjacency" if threshold is None else "distance_threshold",
                    "variant": variant, "answer": answer,
                }})
    # Avoid placing equivalent questions next to one another; identical order per model.
    random.Random(SEED).shuffle(rows)
    return rows


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def plan(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = examples()
    for key, name in (("request", "requests.jsonl"), ("label", "labels.jsonl")):
        with (root / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row[key], sort_keys=True, allow_nan=False) + "\n")
    settings = shared.model_configs("answer_only")
    manifest = {
        "kind": "tiny-graph-sanity-v1", "root": str(root), "seed": SEED,
        "created_at": utc(), "models": list(shared.MODELS), "variants": list(VARIANTS),
        "unique_cases": 12, "unique_graphs": 3, "calls_per_model": len(rows),
        "max_total_calls": len(rows) * len(settings), "max_jev_calls": len(rows),
        "settings": {m: {"provider": c.provider, "kwargs": c.client_kwargs()} for m, c in settings.items()},
        "code_sha256": sources(),
        "artifact_sha256": {n: shared.digest(root / n) for n in ("requests.jsonl", "labels.jsonl")},
        "notice": "Diagnostic only. No parameter/prompt tuning; repeated isomorphic questions are not independent samples.",
    }
    save(root / "manifest.json", manifest, exclusive=True)
    (root / "manifest.sha256").write_text(shared.digest(root / "manifest.json") + "\n")
    return manifest


def load(root, *, check_code=True):
    root = Path(root).resolve()
    shared.require(shared.digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip(),
                   "Manifest changed.")
    m = shared.read_json(root / "manifest.json")
    shared.require(m["kind"] == "tiny-graph-sanity-v1" and m["root"] == str(root), "Wrong or relocated plan.")
    if check_code:
        shared.require(m["code_sha256"] == sources(), "Source drift; refusing live execution.")
    for name, digest in m["artifact_sha256"].items():
        shared.require(shared.digest(root / name) == digest, "Input artifact changed.")
    requests, labels = json_rows(root / "requests.jsonl"), json_rows(root / "labels.jsonl")
    canonical = json.loads(json.dumps(examples()))
    shared.require(requests == [e["request"] for e in canonical]
                   and labels == [e["label"] for e in canonical], "Unexpected diagnostic inputs/order.")
    settings = shared.model_configs("answer_only")
    shared.require(m["models"] == list(settings) and m["calls_per_model"] == 48
                   and m["max_total_calls"] == 336 and m["max_jev_calls"] == 48, "Budget/roster mismatch.")
    shared.require(m["settings"] == {k: {"provider": c.provider, "kwargs": c.client_kwargs()}
                                     for k, c in settings.items()}, "Model settings mismatch.")
    return root, m, requests, labels


def metrics(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None,
            "failures": sum(r["status"] != "success" for r in rows),
            "answers": dict(Counter(r["selected_option_id"] or "<no_answer>" for r in rows)),
            "error_types": dict(Counter(r["error_type"] for r in rows if r["error_type"]))}


def summarize(rows):
    result = {"overall": metrics(rows), "by_task_variant": {}, "invariance": {}}
    indexed = {(r["case_id"], r["variant"]): r for r in rows}
    for task in ("adjacency", "distance_threshold"):
        result["by_task_variant"][task] = {
            variant: metrics([r for r in rows if r["task"] == task and r["variant"] == variant])
            for variant in VARIANTS
        }
    for variant in VARIANTS[1:]:
        paired = [(r, indexed[r["case_id"], variant]) for r in rows
                  if r["variant"] == "baseline" and (r["case_id"], variant) in indexed]
        valid = [(a, b) for a, b in paired if a["status"] == b["status"] == "success"]
        result["invariance"][variant] = {
            "paired_cases": len(paired), "valid_pairs": len(valid),
            "answer_changed": sum(a["selected_option_id"] != b["selected_option_id"] for a, b in valid),
            "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in valid),
            "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in valid),
        }
    return result


def report(root, models=None):
    root, m, requests, labels = load(root, check_code=False)
    chosen = m["models"] if models is None else models
    shared.require(bool(chosen) and set(chosen) <= set(m["models"]), "Unknown models.")
    results = {}
    for model in chosen:
        directory = root / "runs" / model
        status = shared.read_json(directory / "status.json")
        rows = json_rows(directory / "attempts.jsonl")
        shared.require(status["status"] == "completed" and len(rows) == len(requests), "Incomplete lane.")
        for row, label in zip(rows, labels):
            shared.require(all(row[k] == v for k, v in label.items()), "Attempt/label mismatch.")
            valid = row["status"] == "success"
            shared.require(row["status"] in ("success", "failure")
                           and (row["selected_option_id"] in ("yes", "no") if valid
                                else row["selected_option_id"] is None)
                           and row["correct"] == (valid and row["selected_option_id"] == label["answer"]),
                           "Invalid score.")
        results[model] = summarize(rows)
        shared.require(results[model] == shared.read_json(directory / "summary.json"), "Summary mismatch.")
    return {"kind": m["kind"], "status": "completed-selected-lanes", "models": results,
            "notice": m["notice"]}


async def run(root, *, factory=None, preflight=True):
    root, m, requests, labels = load(root)
    injected = factory is not None
    factory = shared.create_client if factory is None else factory
    # Never resume partial outputs; reserve the entire execution before any network use.
    runs = root / "runs"
    runs.mkdir(exist_ok=False)
    settings = shared.model_configs("answer_only")
    clients = {}
    with shared.lane_locks(m["models"]):
        try:
            if preflight:
                shared.credential_preflight(m["models"])
                await shared.vllm_preflight([SimpleNamespace(model=c) for c in settings.values()])
            for model, config in settings.items():
                kwargs = config.client_kwargs()
                if config.provider == "vllm":
                    kwargs["api_key"] = ""
                clients[model] = factory(config.provider, **kwargs)
            # All identities/authentication checked before any scored generation.
            for model, client in clients.items():
                await asyncio.wait_for(client.initialize(), settings[model].timeout_seconds)
            shared.require(sources() == m["code_sha256"], "Code drift during preflight.")

            async def lane(model):
                directory = runs / model
                directory.mkdir(exist_ok=False)
                config, client = settings[model], clients[model]
                status = {"status": "running", "model": config.model, "provider": config.provider,
                          "response_protocol": "native-choice" if model == "jev" else "answer-only-v1",
                          "settings": m["settings"][model], "injected_client": injected,
                          "adapter_retries": 0, "sdk_retries": "not_controlled" if config.provider == "github_copilot" else "not_applicable",
                          "started_at": utc(), "expected_calls": len(requests), "completed_calls": 0}
                save(directory / "status.json", status, exclusive=True)
                rows = []
                try:
                    with (directory / "attempts.jsonl").open("x", encoding="utf-8") as stream:
                        for payload, label in zip(requests, labels):
                            shared.require(sources() == m["code_sha256"], "Code drift during execution.")
                            request = DecisionRequest(**{**payload, "options": tuple(DecisionOption(**o) for o in payload["options"])})
                            row = {**label, "status": "failure", "selected_option_id": None,
                                   "correct": False, "error_type": None, "http_status": None,
                                   "diagnostic_code": None, "finish_reason": None, "usage": None,
                                   "resolved_model": None, "started_at": utc()}
                            started = time.perf_counter()
                            try:
                                response = await asyncio.wait_for(client.predict(request), config.timeout_seconds)
                                if response.request_id != request.request_id or response.selected_option_id not in ("yes", "no"):
                                    raise InvalidResponseError("Invalid choice/ID")
                                row.update(status="success", selected_option_id=response.selected_option_id,
                                           correct=response.selected_option_id == label["answer"],
                                           resolved_model=response.resolved_model, usage=asdict(response.usage))
                            except (DecisionClientError, TimeoutError) as exc:
                                row.update(error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                                if isinstance(exc, InvalidResponseError):
                                    safe = InvalidResponseError("", diagnostic_code=exc.diagnostic_code,
                                                                finish_reason=exc.finish_reason, usage=exc.usage)
                                    row.update(diagnostic_code=safe.diagnostic_code, finish_reason=safe.finish_reason,
                                               usage=asdict(safe.usage) if safe.usage else None)
                                if row["http_status"] in (401, 403):
                                    raise shared.AuthenticationAbort("Authentication rejected; stop all lanes.") from None
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
                    save(directory / "summary.json", summarize(rows), exclusive=True)

            tasks = [asyncio.create_task(lane(model)) for model in m["models"]]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        finally:
            await asyncio.gather(*(client.aclose() for client in clients.values()), return_exceptions=True)
    result = report(root)
    save(root / "report.json", result, exclusive=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            result = plan(args.root)
            print(json.dumps({k: result[k] for k in ("kind", "root", "calls_per_model", "max_total_calls")}))
        elif args.command == "run":
            asyncio.run(run(args.root))
            print("Completed all seven diagnostic lanes; report.json written.")
        else:
            print(json.dumps(report(args.root), indent=2, sort_keys=True))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Diagnostic stopped ({type(exc).__name__}); no retry/resume. Details suppressed to protect credentials.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())