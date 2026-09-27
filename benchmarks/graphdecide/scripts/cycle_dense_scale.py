"""Every integer size 5..12, paired nested graphs and mixed request order."""
import asyncio
from collections import Counter
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
from scripts import cycle_scale_diagnostic as previous
from src.clients.base import DecisionRequest, DecisionOption, DecisionClientError

shared, cycle = previous.shared, previous.cycle
SIZES = tuple(range(5, 13))
SEED = 20261001


def pins():
    return {**previous.pins(), "scripts/cycle_dense_scale.py": shared.digest(__file__)}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cases():
    records = []
    for family in range(20):
        rng = random.Random(SEED + family)
        mapping = list(range(5))
        rng.shuffle(mapping)
        order = list(range(3))
        rng.shuffle(order)
        cores = {"no": [(0, 1), (1, 2), (2, 3)], "yes": [(0, 1), (1, 2), (2, 0)]}
        edges = {a: [[mapping[c[i][0]], mapping[c[i][1]]] for i in order] for a, c in cores.items()}
        for n in SIZES:
            if n > 5:
                v = n - 1
                parent = rng.randrange(v) if family < 10 else (mapping[0] if v == 5 else v - 1)
                position = rng.randrange(len(edges["no"]) + 1)
                for a in edges:
                    edges[a].insert(position, [parent, v])
            for answer in ("no", "yes"):
                state = {"nodes": list(range(n)), "edges": [e[:] for e in edges[answer]]}
                g = nx.Graph()
                g.add_nodes_from(state["nodes"])
                g.add_edges_from(state["edges"])
                assert len(g.edges) == n - 2
                assert cycle.task.has_cycle(state) == bool(nx.cycle_basis(g)) == (answer == "yes")
                assert len(g.edges) - n + nx.number_connected_components(g) == int(answer == "yes")
                request = DecisionRequest(f"dense-{family}-{n}-{answer}", state, cycle.task.QUESTION,
                                          (DecisionOption("yes"), DecisionOption("no")))
                records.append({"request": asdict(request), "label": {
                    "request_id": request.request_id, "node_count": n, "edge_count": n - 2,
                    "family": family, "growth": "random_tree" if family < 10 else "path",
                    "answer": answer, "pair_id": f"{family}-{n}"}})
    random.Random(SEED).shuffle(records)
    return json.loads(json.dumps(records))


def plan(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = cases()
    assert len(records) == 320
    for key in ("request", "label"):
        with (root / f"{key}s.jsonl").open("x") as f:
            for record in records:
                f.write(json.dumps(record[key], sort_keys=True) + "\n")
    m = {"kind": "cycle-every-integer-5-12-v1", "root": str(root), "seed": SEED,
         "sizes": list(SIZES), "expected_calls": 320, "created_at": datetime.now(timezone.utc).isoformat(),
         "model": shared.model_configs("answer_only", models=("jev",))["jev"].client_kwargs(),
         "code_sha256": pins(), "input_sha256": {n: shared.digest(root / n) for n in ("requests.jsonl", "labels.jsonl")},
         "notice": "Posthoc synthetic follow-up, separate from prior curve. Every integer size,20 paired families (10 random-tree/10 path),new seed, globally shuffled schedule. One leaf edge added each step,labels stable. Unlike prior rank-permutation protocol,existing node names and relative edge order remain fixed across sizes. Positive cycles are triangles; smallest graphs are isomorphic relabelings,not independent topologies. Correlated family trajectories,not a universal difficulty/size threshold. IDs/labels never sent. No retries/resume, raw native choice only."}
    shared.write_json(root / "manifest.json", m)
    with (root / "manifest.sha256").open("x") as f:
        f.write(shared.digest(root / "manifest.json") + "\n")
    return m


def load(root):
    root = Path(root).resolve()
    assert shared.digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip()
    m = shared.read_json(root / "manifest.json")
    assert m["root"] == str(root) and m["expected_calls"] == 320
    assert m["input_sha256"] == {n: shared.digest(root / n) for n in ("requests.jsonl", "labels.jsonl")}
    requests, labels = shared.read_rows(root / "requests.jsonl"), shared.read_rows(root / "labels.jsonl")
    assert len(requests) == len(labels) == 320
    assert [{"request": r, "label": l} for r, l in zip(requests, labels)] == cases()
    return root, m, requests, labels


def report(root):
    root, m, requests, labels = load(root)
    assert shared.read_json(root / "status.json")["status"] == "completed"
    rows = shared.read_rows(root / "attempts.jsonl")
    assert len(rows) == 320
    for row, label, payload in zip(rows, labels, requests):
        assert all(row[k] == v for k, v in label.items())
        good = row["status"] == "success"
        assert row["status"] in ("success", "failure")
        assert type(row["correct"]) is bool and row["correct"] == (good and row["selected"] == label["answer"])
        if good:
            assert row["raw_choice"] == row["selected"] and row["selected"] in ("yes", "no")
            assert row["wire_sha256"] == sha(wire(payload, m["model"]["model"]))
            assert row["resolved_model"] == m["model"]["model"]
        else:
            assert row["selected"] is None
    lookup = {(r["family"], r["answer"], r["node_count"]): r for r in rows}
    trajectories = [{"family": f, "answer": a, "selected": [lookup[f,a,n]["selected"] for n in SIZES],
                     "correct": [lookup[f,a,n]["correct"] for n in SIZES]} for f in range(20) for a in ("no", "yes")]
    transitions = {}
    for n in SIZES[1:]:
        transitions[str(n)] = dict(Counter(
            ("correct" if lookup[f,a,n-1]["correct"] else "wrong") + "_to_" +
            ("correct" if lookup[f,a,n]["correct"] else "wrong")
            for f in range(20) for a in ("no", "yes")))
    return {"overall": previous.stats(rows),
            "by_size": {str(n): previous.stats([r for r in rows if r["node_count"] == n]) for n in SIZES},
            "by_growth_size": {g: {str(n): previous.stats([r for r in rows if r["growth"] == g and r["node_count"] == n]) for n in SIZES} for g in ("random_tree", "path")},
            "trajectories": trajectories, "adjacent_transitions": transitions, "notice": m["notice"]}


def wire(payload, model):
    return {"model": model, "state": payload["state"], "questions": {"decision": {
        "type": "choice", "instructions": payload["question"], "criteria": {o["id"]: o["description"] for o in payload["options"]}}}}


async def run(root, factory=None, preflight=True):
    root, m, requests, labels = load(root)
    assert pins() == m["code_sha256"]
    config = shared.model_configs("answer_only", models=("jev",))["jev"]
    assert config.client_kwargs() == m["model"]
    with cycle.lane_locks(("jev",)):
        if preflight:
            previous.large.load(previous.PRIOR, check_code=True)
            previous.large.report(previous.PRIOR, models=("jev",))
            cycle.report(previous.SOURCE, ("jev",))
            previous.report(REPO / "output/cycle-scale-jev-v1")
            shared.credential_preflight(("jev",))
        status = {"status": "initializing", "completed_calls": 0, "expected_calls": 320,
                  "started_at": datetime.now(timezone.utc).isoformat(), "injected_client": factory is not None}
        shared.write_json(root / "status.json", status)
        client = None
        try:
            client = (shared.create_client if factory is None else factory)(config.provider, **config.client_kwargs())
            capture = {}
            async def on_request(r):
                capture["payload"] = json.loads(r.content)
            async def on_response(r):
                await r.aread()
                if r.status_code == 200:
                    capture["raw"] = r.json().get("answers", {}).get("decision", {}).get("choice")
            client._http.event_hooks = {"request": [on_request], "response": [on_response]}
            await client.initialize()
            status["status"] = "running"
            with (root / "attempts.jsonl").open("x") as stream:
                for payload, label in zip(requests, labels):
                    assert pins() == m["code_sha256"]
                    req = DecisionRequest(**{**payload, "options": tuple(DecisionOption(**o) for o in payload["options"])})
                    row = {**label, "status": "failure", "selected": None, "correct": False, "error_type": None}
                    capture.clear()
                    start = time.perf_counter()
                    try:
                        response = await asyncio.wait_for(client.predict(req), config.timeout_seconds)
                        expected = wire(payload, config.model)
                        assert capture["payload"] == expected
                        raw = capture["raw"]
                        assert raw in ("yes", "no") and raw == response.selected_option_id
                        assert response.request_id == req.request_id and response.resolved_model == config.model
                        row.update(status="success", selected=raw, raw_choice=raw, correct=raw == label["answer"],
                                   wire_sha256=sha(expected), resolved_model=response.resolved_model, usage=asdict(response.usage))
                    except (DecisionClientError, TimeoutError) as exc:
                        row.update(error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                        if row["http_status"] in (400,401,403,413,422,429):
                            raise
                    except BaseException as exc:
                        row.update(status="aborted", error_type=type(exc).__name__)
                        raise
                    finally:
                        row["elapsed_seconds"] = time.perf_counter() - start
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                        status["completed_calls"] += 1
                        cycle.save(root / "status.json", status)
                        if status["completed_calls"] % 40 == 0:
                            print("Completed", status["completed_calls"], "/320", flush=True)
            status["status"] = "completed"
        except BaseException as exc:
            status.update(status="aborted", error_type=type(exc).__name__)
            raise
        finally:
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            cycle.save(root / "status.json", status)
            if client is not None:
                await client.aclose()
    result = report(root)
    shared.write_json(root / "report.json", result)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "report"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    try:
        result = plan(args.root) if args.command == "plan" else asyncio.run(run(args.root)) if args.command == "run" else report(args.root)
        print(json.dumps(result, sort_keys=True))
    except BaseException as exc:
        print("Dense diagnostic stopped:", type(exc).__name__, "; no retry/resume; details suppressed.", file=sys.stderr)
        sys.exit(2)