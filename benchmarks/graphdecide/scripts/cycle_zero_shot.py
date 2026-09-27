"""Paired Jev zero-shot screen: six arms, fixed historical graphs, no examples."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import networkx as nx
from scripts import cycle_dense_scale as dense
from src.clients.base import DecisionRequest, DecisionOption, DecisionClientError

shared, cycle = dense.shared, dense.cycle
SEED = 20261002
ARMS = ("baseline", "criteria", "semantics", "adjacency", "natural_edges", "algorithm")
SIZES = (5, 8, 12, 32)
DENSE = REPO / "output/cycle-dense-jev-v1"
REAL = dense.previous.SOURCE
FILES = ("bank.json", "requests.jsonl", "labels.jsonl")
SEMANTICS = (
    "Does the simple undirected graph specified by `nodes` and `edges` contain a cycle? "
    "`nodes` lists every vertex, including isolated vertices. Each pair [u, v] in `edges` "
    "is one undirected edge between u and v, not two directed edges. The edge list is "
    "complete: unlisted edges do not exist. A cycle passes through at least three distinct "
    "vertices and returns to its starting vertex along graph edges. Traversing one edge "
    "back and forth is not a cycle. Check all connected components. Return only yes or no."
)
CRITERIA = {
    "yes": "The graph contains a cycle through at least three distinct vertices in at least one connected component.",
    "no": "The graph contains no cycle through at least three distinct vertices in any connected component.",
}
ALGORITHM = (
    " Use depth-first search over every connected component, tracking visited vertices "
    "and each vertex's parent. At a vertex, visit each unvisited neighbor recursively. "
    "An edge to an already visited neighbor other than the parent indicates a cycle. "
    "The edge back to the parent alone is not a cycle. Answer no only if every component "
    "has been traversed without finding such an edge."
)
NOTICE = (
    "Development screen, not held-out confirmation or capability ceiling. Eighty fixed "
    "historical questions: 5/8/12-node synthetic nested triangle/forest families and "
    "32-node real-derived pairs. Size is confounded with bank at 32. Small isomorphic "
    "copies and nested sizes are correlated. Six independent baseline-relative arms; "
    "no examples, combined prompt, retries, resume, voting, tools, or probability scoring."
)


def pins():
    return {**dense.pins(), "scripts/cycle_zero_shot.py": shared.digest(__file__)}


def graph_of(state):
    """Independently reconstruct every rendered topology before any network call."""
    nodes = state["nodes"]
    assert nodes == list(range(len(nodes)))
    if "adjacency" in state:
        adjacency = state["adjacency"]
        assert set(adjacency) == {str(n) for n in nodes}
        for a in nodes:
            neighbors = adjacency[str(a)]
            assert len(neighbors) == len(set(neighbors))
            assert all(b in nodes and a != b and a in adjacency[str(b)] for b in neighbors)
        edges = [[a, b] for a in nodes for b in adjacency[str(a)] if a < b]
    elif isinstance(state["edges"], str):
        edges = []
        for line in state["edges"].splitlines():
            match = re.fullmatch(r"Node (\d+) and node (\d+) are joined by an undirected edge\.", line)
            assert match
            edges.append([int(v) for v in match.groups()])
    else:
        edges = state["edges"]
    assert all(a in nodes and b in nodes and a != b for a, b in edges)
    assert len(edges) == len({tuple(sorted(e)) for e in edges})
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def source_bank():
    """Selection depends on source IDs/strata only, never model outcomes."""
    dense.load(DENSE)
    dense.report(DENSE)
    cycle.load(REAL)
    cycle.validate_inputs(REAL)
    cycle.report(REAL, ("jev",))
    selected = []
    for name, root in (("dense", DENSE), ("real", REAL)):
        requests = shared.read_rows(root / "requests.jsonl")
        labels = shared.read_rows(root / "labels.jsonl")
        allowed = set()
        if name == "real":
            for dataset, count in (("facebook", 3), ("ca-grqc", 3), ("power", 2), ("human-ppi", 2)):
                pairs = sorted({l["pair_id"] for l in labels if l["dataset"] == dataset and l["node_count"] == 32})
                allowed.update(pairs[:count])
        # Read outcomes only after the answer-independent membership rule is fixed.
        attempts = shared.read_rows(root / ("attempts.jsonl" if name == "dense" else "runs/jev/attempts.jsonl"))
        for i, (request, label, old) in enumerate(zip(requests, labels, attempts)):
            take = (label["node_count"] in (5, 8, 12) and label["family"] in (*range(5), *range(10, 15))) if name == "dense" else label["pair_id"] in allowed
            if not take:
                continue
            cluster = f"dense-family-{label['family']}" if name == "dense" else f"real-{label['pair_id']}"
            selected.append({"request": request, "source_index": i, "source_bank": name,
                "source_root": str(root), "source_label": label,
                "pair_id": f"{name}-{label['pair_id']}", "cluster": cluster,
                "answer": label["answer"], "node_count": label["node_count"],
                "historical_selected": old["selected" if name == "dense" else "selected_option_id"]})
    selected.sort(key=lambda c: c["request"]["request_id"])
    assert len(selected) == 80
    assert Counter((c["node_count"], c["answer"]) for c in selected) == {(n, a): 10 for n in SIZES for a in ("yes", "no")}
    return selected


def render(case, arm):
    request = deepcopy(case["request"])
    request["request_id"] += f"--{arm}"
    state = request["state"]
    if arm == "criteria":
        request["options"] = [{"id": a, "description": CRITERIA[a]} for a in ("yes", "no")]
    elif arm == "semantics":
        request["question"] = SEMANTICS
    elif arm == "adjacency":
        adjacency = {str(n): [] for n in state["nodes"]}
        # Neighbor order follows the original edge stream, not an oracle traversal.
        for a, b in state["edges"]:
            adjacency[str(a)].append(b)
            adjacency[str(b)].append(a)
        request["state"] = {"nodes": state["nodes"], "adjacency": adjacency}
    elif arm == "natural_edges":
        state["edges"] = "\n".join(f"Node {a} and node {b} are joined by an undirected edge." for a, b in state["edges"])
    elif arm == "algorithm":
        request["question"] += ALGORITHM
    else:
        assert arm == "baseline"
    original, rendered = graph_of(case["request"]["state"]), graph_of(request["state"])
    assert set(original.nodes) == set(rendered.nodes)
    assert {frozenset(e) for e in original.edges} == {frozenset(e) for e in rendered.edges}
    assert (not nx.is_forest(rendered)) == (case["answer"] == "yes")
    assert cycle.task.has_cycle(case["request"]["state"]) == (case["answer"] == "yes")
    return request


def schedule(bank):
    # Six rounds with rotating arm assignment: each question sees each arm once.
    # Each round contains all 80 questions; same-question calls are separated.
    order = list(range(len(bank)))
    random.Random(SEED).shuffle(order)
    offsets = {i: j % len(ARMS) for j, i in enumerate(order)}
    records = []
    for round_id in range(len(ARMS)):
        order = list(range(len(bank)))
        random.Random(SEED + round_id + 1).shuffle(order)
        for i in order:
            c = bank[i]
            arm = ARMS[(offsets[i] + round_id) % len(ARMS)]
            request = render(c, arm)
            label = {"request_id": request["request_id"], "base_id": c["request"]["request_id"],
                     "arm": arm, "round": round_id, "answer": c["answer"], "pair_id": c["pair_id"],
                     "cluster": c["cluster"], "source_bank": c["source_bank"], "node_count": c["node_count"]}
            records.append({"request": request, "label": label})
    assert len(records) == 480
    return records


def plan(root):
    bank = source_bank()
    records = schedule(bank)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    shared.write_json(root / "bank.json", bank)
    for key in ("request", "label"):
        with (root / f"{key}s.jsonl").open("x") as f:
            for record in records:
                f.write(json.dumps(record[key], sort_keys=True) + "\n")
    config = shared.model_configs("answer_only", models=("jev",))["jev"]
    source_pins = {str(p): shared.digest(p) for r in (DENSE, REAL) for p in (r / "manifest.json", r / "requests.jsonl", r / "labels.jsonl")}
    m = {"kind": "cycle-zero-shot-six-arm-v1", "root": str(root), "created_at": cycle.utc(),
         "seed": SEED, "expected_calls": 480, "questions": 80, "arms": list(ARMS), "sizes": list(SIZES),
         "model": config.client_kwargs(), "code_sha256": pins(), "source_sha256": source_pins,
         "input_sha256": {n: shared.digest(root / n) for n in FILES}, "notice": NOTICE}
    shared.write_json(root / "manifest.json", m)
    with (root / "manifest.sha256").open("x") as f:
        f.write(shared.digest(root / "manifest.json") + "\n")
    return m


def load(root):
    root = Path(root).resolve()
    assert shared.digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip()
    m = shared.read_json(root / "manifest.json")
    assert m["kind"] == "cycle-zero-shot-six-arm-v1" and m["root"] == str(root)
    assert m["arms"] == list(ARMS) and m["expected_calls"] == 480
    assert m["input_sha256"] == {n: shared.digest(root / n) for n in FILES}
    assert all(shared.digest(p) == h for p, h in m["source_sha256"].items())
    requests, labels = (shared.read_rows(root / n) for n in FILES[1:])
    records = [{"request": r, "label": l} for r, l in zip(requests, labels)]
    assert len(requests) == len(labels) == 480
    assert records == schedule(shared.read_json(root / "bank.json"))
    return root, m, requests, labels


def paired_stats(rows, baseline):
    deltas = defaultdict(list)
    flips = []
    gains = losses = 0
    for row in rows:
        base = baseline[row["base_id"]]
        delta = int(row["correct"]) - int(base["correct"])
        deltas[row["cluster"]].append(delta)
        gains += delta == 1
        losses += delta == -1
        if row["selected"] != base["selected"]:
            flips.append({"base_id": row["base_id"], "answer": row["answer"], "node_count": row["node_count"],
                          "baseline": base["selected"], "selected": row["selected"], "correctness_delta": delta})
    # Stratify by bank; resample whole nested families or whole real-source pairs.
    strata = defaultdict(list)
    for cluster in deltas:
        strata["dense" if cluster.startswith("dense-") else "real"].append(cluster)
    rng = random.Random(SEED)
    boot = []
    for _ in range(2000):
        values = []
        for clusters in strata.values():
            for _ in clusters:
                values.extend(deltas[rng.choice(clusters)])
        boot.append(sum(values) / len(values))
    boot.sort()
    return {"gains": gains, "losses": losses, "accuracy_delta": (gains - losses) / len(rows),
            "cluster_bootstrap_95_interval": [boot[49], boot[1949]],
            "clusters": len(deltas), "flips": flips}


def report(root):
    root, m, requests, labels = load(root)
    status = shared.read_json(root / "status.json")
    rows = shared.read_rows(root / "attempts.jsonl")
    assert status["status"] == "completed" and status["completed_calls"] == len(rows) == 480
    for row, label, payload in zip(rows, labels, requests):
        assert all(row[k] == v for k, v in label.items())
        good = row["status"] == "success"
        assert row["status"] in ("success", "failure")
        assert type(row["correct"]) is bool and row["correct"] == (good and row["selected"] == label["answer"])
        if good:
            assert row["raw_choice"] == row["selected"] and row["selected"] in ("yes", "no")
            assert row["wire_sha256"] == dense.sha(dense.wire(payload, m["model"]["model"]))
            assert row["resolved_model"] == m["model"]["model"]
        else:
            assert row["selected"] is None
    baseline = {r["base_id"]: r for r in rows if r["arm"] == "baseline"}
    arms = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        arms[arm] = {"overall": dense.previous.stats(arm_rows),
                     "by_size": {str(n): dense.previous.stats([r for r in arm_rows if r["node_count"] == n]) for n in SIZES},
                     "paired_vs_baseline": paired_stats(arm_rows, baseline),
                     "input_tokens": sum(r.get("usage", {}).get("input_tokens") or 0 for r in arm_rows),
                     "usage_missing": sum(r.get("usage", {}).get("input_tokens") is None for r in arm_rows)}
    bank = shared.read_json(root / "bank.json")
    return {"arms": arms, "baseline_historical_matches": sum(baseline[c["request"]["request_id"]]["selected"] == c["historical_selected"] for c in bank),
            "expected_calls": 480, "notice": NOTICE,
            "uncertainty_note": "Exploratory 2000-resample stratified family/pair bootstrap, not multiple-comparison adjusted; isomorphism and source overlap may leave further dependence."}


async def run(root, factory=None, preflight=True):
    root, m, requests, labels = load(root)
    assert pins() == m["code_sha256"]
    config = shared.model_configs("answer_only", models=("jev",))["jev"]
    assert config.client_kwargs() == m["model"]
    with cycle.lane_locks(("jev",)):
        if preflight:
            dense.previous.large.load(dense.previous.PRIOR, check_code=True)
            dense.previous.large.report(dense.previous.PRIOR, models=("jev",))
            assert shared.read_json(root / "bank.json") == source_bank()
            shared.credential_preflight(("jev",))
        # Exclusive reservation: refuse reruns before creating a client.
        status = {"status": "initializing", "completed_calls": 0, "expected_calls": 480,
                  "started_at": cycle.utc(), "injected_client": factory is not None}
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
                    body = r.json()
                    if isinstance(body, dict) and isinstance(body.get("answers"), dict):
                        answer = body["answers"].get("decision")
                        if isinstance(answer, dict):
                            capture["raw"] = answer.get("choice")
            client._http.event_hooks = {"request": [on_request], "response": [on_response]}
            await client.initialize()
            status["status"] = "running"
            with (root / "attempts.jsonl").open("x") as stream:
                for payload, label in zip(requests, labels):
                    assert pins() == m["code_sha256"]
                    assert m["input_sha256"] == {n: shared.digest(root / n) for n in FILES}
                    req = DecisionRequest(**{**payload, "options": tuple(DecisionOption(**o) for o in payload["options"])})
                    row = {**label, "status": "failure", "selected": None, "correct": False, "error_type": None}
                    capture.clear()
                    start = time.perf_counter()
                    try:
                        response = await asyncio.wait_for(client.predict(req), config.timeout_seconds)
                        expected = dense.wire(payload, config.model)
                        assert capture["payload"] == expected
                        raw = capture["raw"]
                        assert raw in ("yes", "no") and raw == response.selected_option_id
                        assert response.request_id == req.request_id and response.resolved_model == config.model
                        row.update(status="success", selected=raw, raw_choice=raw, correct=raw == label["answer"],
                                   wire_sha256=dense.sha(expected), resolved_model=response.resolved_model, usage=asdict(response.usage))
                    except (DecisionClientError, TimeoutError) as exc:
                        row.update(error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                        if row["http_status"] in (400, 401, 403, 413, 422, 429):
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
                            print("Completed", status["completed_calls"], "/480", flush=True)
            status["status"] = "completed"
        except BaseException as exc:
            status.update(status="aborted", error_type=type(exc).__name__)
            raise
        finally:
            status["finished_at"] = cycle.utc()
            cycle.save(root / "status.json", status)
            if client is not None:
                await client.aclose()
    result = report(root)
    shared.write_json(root / "report.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run", "report"))
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    try:
        result = plan(args.root) if args.command == "plan" else asyncio.run(run(args.root)) if args.command == "run" else report(args.root)
        if args.command == "plan":
            print(json.dumps({k: result[k] for k in ("root", "questions", "expected_calls", "arms", "notice")}))
        else:
            print(json.dumps({a: v["overall"] for a, v in result["arms"].items()}, sort_keys=True))
    except BaseException as exc:
        print("Zero-shot screen stopped:", type(exc).__name__, "; no retry/resume; details suppressed.", file=sys.stderr)
        sys.exit(2)