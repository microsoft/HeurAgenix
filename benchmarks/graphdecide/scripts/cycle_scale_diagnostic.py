"""Mixed-size synthetic Jev cycle diagnostic, separate from benchmark scores."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import networkx as nx
from scripts import run_cycle_task as cycle
from scripts import large_graph_matrix as large
from src.clients.base import DecisionRequest, DecisionOption, DecisionClientError

shared = cycle.shared
SIZES = (5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
SEED = 20260930
SOURCE = REPO / "output/cycle-five-20260924T071448Z-1325232"
PRIOR = REPO / "output/large-seven-20260924T053848Z-1273903"


def pins():
    return {**cycle.sources(), "scripts/cycle_scale_diagnostic.py": shared.digest(__file__)}


def build_cases():
    result = []
    for family in range(20):
        rng = random.Random(SEED + family)
        # One fixed core per family, expanded without changing the cycle label.
        growth = [(rng.randrange(v) if family < 10 else v - 1, v) for v in range(5, 128)]
        priorities = list(range(128))
        rng.shuffle(priorities)
        for mode in ("branches", "isolates"):
            if mode == "isolates" and family >= 5:
                continue
            for n in SIZES:
                # Rank a fixed permutation restricted to this size. Same mapping
                # for both labels/modes; vertex names may change between sizes.
                order = sorted(range(n), key=lambda v: priorities[v])
                mapping = {old: new for new, old in enumerate(order)}
                additions = [e for e in growth if e[1] < n] if mode == "branches" else []
                for answer, core in (("no", [(0, 1), (1, 2), (2, 3)]),
                                     ("yes", [(0, 1), (1, 2), (2, 0)])):
                    edges = [[mapping[a], mapping[b]] for a, b in core + additions]
                    random.Random(SEED + family * 1000 + n).shuffle(edges)
                    state = {"nodes": list(range(n)), "edges": edges}
                    g = nx.Graph()
                    g.add_nodes_from(state["nodes"])
                    g.add_edges_from(edges)
                    # NetworkX cycle_basis, own DFS, and cycle rank agree.
                    rank = len(edges) - n + nx.number_connected_components(g)
                    assert cycle.task.has_cycle(state) == bool(nx.cycle_basis(g)) == (answer == "yes")
                    assert rank == (1 if answer == "yes" else 0)
                    assert len(edges) == (n - 2 if mode == "branches" else 3)
                    request = DecisionRequest(f"scale-{mode}-{family}-{n}-{answer}", state,
                                              cycle.task.QUESTION, (DecisionOption("yes"), DecisionOption("no")))
                    result.append({"request": asdict(request), "label": {
                        "request_id": request.request_id, "answer": answer, "mode": mode,
                        "family": family, "growth": "random_tree" if family < 10 else "path",
                        "node_count": n, "edge_count": len(edges), "cycle_rank": rank,
                        "pair_id": f"{mode}-{family}-{n}"}})
    random.Random(SEED).shuffle(result)
    return json.loads(json.dumps(result))


def plan(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = build_cases()
    assert len(cases) == 550
    for key in ("request", "label"):
        with (root / f"{key}s.jsonl").open("x") as f:
            for c in cases:
                f.write(json.dumps(c[key], sort_keys=True) + "\n")
    config = shared.model_configs("answer_only", models=("jev",))["jev"]
    m = {"kind": "mixed-cycle-size-diagnostic-v1", "created_at": datetime.now(timezone.utc).isoformat(),
         "root": str(root), "seed": SEED, "sizes": list(SIZES), "expected_calls": 550,
         "mode_calls": {"branches": 440, "isolates": 110}, "code_sha256": pins(),
         "input_sha256": {n: shared.digest(root / n) for n in ("requests.jsonl", "labels.jsonl")},
         "model": {"provider": config.provider, "kwargs": config.client_kwargs()},
         "notice": "Synthetic nested triangle-vs-forest cores, not real graph benchmark. Twenty primary families (10 random-tree,10 path growth), five isolate-only controls. Counts and labels matched per size/pair; globally shuffled mixed-size schedule. Labels/IDs excluded from API body. Per-size families are related, NOT independent datasets. Fixed permutation ranks/edge shuffle may change serialized appearance across sizes. All positive cycles are triangles; no general cycle-length threshold claim. Control n=5 prompts deliberately repeat five primary pairs. No retries/resume, no output tuning, native Jev selected choice only."}
    shared.write_json(root / "manifest.json", m)
    (root / "manifest.sha256").write_text(shared.digest(root / "manifest.json") + "\n")
    return m


def load(root, check_code=True):
    root = Path(root).resolve()
    assert shared.digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip()
    m = shared.read_json(root / "manifest.json")
    assert m["root"] == str(root) and m["expected_calls"] == 550
    if check_code:
        assert pins() == m["code_sha256"]
    assert m["input_sha256"] == {n: shared.digest(root / n) for n in ("requests.jsonl", "labels.jsonl")}
    requests, labels = shared.read_rows(root / "requests.jsonl"), shared.read_rows(root / "labels.jsonl")
    assert [{"request": r, "label": l} for r, l in zip(requests, labels)] == build_cases()
    assert len(requests) == len(labels) == 550
    return root, m, requests, labels


def stats(rows):
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["pair_id"]].append(r)
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "failures": sum(r["status"] != "success" for r in rows),
            "answers": dict(Counter(r["selected"] or "<missing>" for r in rows)),
            "yes_correct": sum(r["correct"] for r in rows if r["answer"] == "yes"),
            "no_correct": sum(r["correct"] for r in rows if r["answer"] == "no"),
            "both_correct_pairs": sum(len(p) == 2 and all(r["correct"] for r in p) for p in pairs.values()),
            "pairs": len(pairs)}


def report(root):
    root, m, requests, labels = load(root, check_code=False)
    assert shared.read_json(root / "status.json")["status"] == "completed"
    rows = shared.read_rows(root / "attempts.jsonl")
    assert len(rows) == len(labels)
    for r, label in zip(rows, labels):
        assert all(r[k] == v for k, v in label.items())
        assert r["status"] in ("success", "failure")
        good = r["status"] == "success"
        assert r["correct"] == (good and r["selected"] == label["answer"])
        if good:
            assert r["raw_choice"] == r["selected"] and r["selected"] in ("yes", "no")
            assert r["wire_verified"] and r["resolved_model"] == "jev-1.13.0"
        else:
            assert r["selected"] is None
    return {"overall": stats(rows), "by_mode_size": {
        mode: {str(n): stats([r for r in rows if r["mode"] == mode and r["node_count"] == n]) for n in SIZES}
        for mode in ("branches", "isolates")}, "primary_by_growth_size": {
        growth: {str(n): stats([r for r in rows if r["mode"] == "branches" and r["growth"] == growth and r["node_count"] == n]) for n in SIZES}
        for growth in ("random_tree", "path")}, "notice": m["notice"]}


async def run(root, factory=None, preflight=True):
    root, m, requests, labels = load(root)
    config = shared.model_configs("answer_only", models=("jev",))["jev"]
    assert m["model"] == {"provider": config.provider, "kwargs": config.client_kwargs()}
    with cycle.lane_locks(("jev",)):
        if preflight:
            large.report(PRIOR, models=("jev",))
            cycle.report(SOURCE, ("jev",))
            large.load(PRIOR, check_code=True)
            shared.credential_preflight(("jev",))
        status = {"status": "initializing", "completed_calls": 0, "expected_calls": 550,
                  "started_at": datetime.now(timezone.utc).isoformat(), "injected_client": factory is not None}
        shared.write_json(root / "status.json", status)
        client = (shared.create_client if factory is None else factory)(config.provider, **config.client_kwargs())
        capture = {}
        async def on_request(request):
            capture["payload"] = json.loads(request.content)
        async def on_response(response):
            await response.aread()
            if response.status_code == 200:
                capture["raw"] = response.json().get("answers", {}).get("decision", {}).get("choice")
        client._http.event_hooks = {"request": [on_request], "response": [on_response]}
        try:
            await client.initialize()
            status["status"] = "running"
            with (root / "attempts.jsonl").open("x") as f:
                for payload, label in zip(requests, labels):
                    assert pins() == m["code_sha256"]
                    request = DecisionRequest(**{**payload, "options": tuple(DecisionOption(**o) for o in payload["options"])})
                    capture.clear()
                    row = {**label, "status": "failure", "selected": None, "correct": False, "error_type": None}
                    start = time.perf_counter()
                    try:
                        response = await asyncio.wait_for(client.predict(request), config.timeout_seconds)
                        expected = {"model": config.model, "state": request.state, "questions": {"decision": {
                            "type": "choice", "instructions": request.question, "criteria": {"yes": None, "no": None}}}}
                        assert capture["payload"] == expected
                        raw = capture["raw"]
                        assert raw in ("yes", "no") and raw == response.selected_option_id
                        assert response.request_id == request.request_id and response.resolved_model == config.model
                        row.update(status="success", selected=raw, raw_choice=raw, correct=raw == label["answer"],
                                   wire_verified=True, wire_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                                   resolved_model=response.resolved_model, usage=asdict(response.usage))
                    except (DecisionClientError, TimeoutError) as exc:
                        row.update(error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                        if row["http_status"] in (400, 401, 403, 413, 422, 429):
                            raise
                    except BaseException as exc:
                        row.update(status="aborted", error_type=type(exc).__name__)
                        raise
                    finally:
                        row["elapsed_seconds"] = time.perf_counter() - start
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                        f.flush()
                        status["completed_calls"] += 1
                        cycle.save(root / "status.json", status)
                        if status["completed_calls"] % 50 == 0:
                            print("Completed", status["completed_calls"], "/ 550", flush=True)
            status["status"] = "completed"
        except BaseException as exc:
            status.update(status="aborted", error_type=type(exc).__name__)
            raise
        finally:
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            cycle.save(root / "status.json", status)
            await client.aclose()
    result = report(root)
    shared.write_json(root / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = plan(args.root) if args.command == "plan" else asyncio.run(run(args.root)) if args.command == "run" else report(args.root)
        print(json.dumps(result, sort_keys=True))
    except BaseException as exc:
        print("Scale diagnostic stopped:", type(exc).__name__, "; no resume/retry; details suppressed.", file=sys.stderr)
        sys.exit(2)