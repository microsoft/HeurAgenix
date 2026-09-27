#!/usr/bin/env python3
"""Pinned seven-model large real-graph replication; new samples, not node-held-out."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml
from scripts import paired_graph_ablation as shared
from src.benchmark.config import BenchmarkConfig, DataConfig, RunConfig, SamplingConfig, TaskConfig, load_config
from src.benchmark.prepare import prepare
from src.benchmark.runner import export_prepared, run_experiment
from src.clients.base import DecisionClientError

SEED = 20260926
DATASETS = shared.DATASETS
NODE_COUNTS = (32, 64, 128)
SAMPLES = 20
TASKS = {"adjacency": TaskConfig(4, 0, ()),
         "distance_threshold": TaskConfig(0, 4, (2, 3, 4, 6))}
KIND = "large-seven-new-samples-not-node-heldout-v1"
DEFAULT_PRIORS = (
    REPO / "output/adjacency.jev/graphs.jsonl",
    REPO / "output/single-prompt-seven-20260924T024842Z-1187579/inputs/baseline_nodes_edges/adjacency/graphs.jsonl",
)


def sources():
    result = shared.code_hashes()
    result[Path(__file__).relative_to(REPO).as_posix()] = shared.digest(Path(__file__))
    return result


def configs(root, data_dir):
    for model, settings in shared.model_configs("answer_only").items():
        for task, questions in TASKS.items():
            count = len(DATASETS) * len(NODE_COUNTS) * SAMPLES
            per_graph = questions.adjacency_questions + questions.distance_questions_per_threshold * len(questions.distance_thresholds)
            yield model, task, BenchmarkConfig(
                1, RunConfig(SEED, 1, count * per_graph, root / "runs" / f"{model}.{task}"),
                DataConfig(DATASETS, data_dir), SamplingConfig(NODE_COUNTS, SAMPLES, 1000),
                questions, settings,
            )


def validate_prepared(config, prepared):
    shared.require(len(prepared.graphs) == config.planned_graphs, "Missing graph stratum.")
    counts = Counter((g["dataset"], g["node_count"]) for g in prepared.graphs)
    shared.require(counts == Counter({(d, n): SAMPLES for d in DATASETS for n in NODE_COUNTS}),
                   "Graph coverage mismatch.")
    shared.require(0 < len(prepared.examples) <= config.run.max_calls, "Invalid question count.")
    balances = Counter((e.dataset, e.node_count, e.task, e.threshold, e.answer) for e in prepared.examples)
    for dataset, n, task, threshold, _ in balances:
        shared.require(balances[dataset, n, task, threshold, "yes"] == balances[dataset, n, task, threshold, "no"],
                       "Unbalanced labels.")
    # Parse visible text, not label coordinates, and independently run deque BFS.
    neighbors = {}
    distances = {}
    for example in prepared.examples:
        request = example.request
        if example.graph_id not in neighbors:
            state = request.state
            shared.require(set(state) == {"nodes", "edges"} and state["nodes"] == list(range(example.node_count)),
                           "Unexpected visible state.")
            adjacent = {n: set() for n in state["nodes"]}
            for a, b in state["edges"]:
                adjacent[a].add(b)
                adjacent[b].add(a)
            neighbors[example.graph_id] = adjacent
        pattern = (r"Use only the given undirected, unweighted graph\. Are nodes (\d+) and (\d+) directly connected by an edge\?"
                   if example.task == "adjacency" else
                   r"Use only the given undirected, unweighted graph\. Is there a path from node (\d+) to node (\d+) using at most (\d+) edges\?")
        match = re.fullmatch(pattern, request.question)
        shared.require(match is not None, "Unexpected question wording.")
        args = list(map(int, match.groups()))
        u, v = args[:2]
        k = args[2] if len(args) == 3 else 1
        key = (example.graph_id, u)
        if key not in distances:
            queue, distance = deque([u]), {u: 0}
            while queue:
                a = queue.popleft()
                for b in neighbors[example.graph_id][a]:
                    if b not in distance:
                        distance[b] = distance[a] + 1
                        queue.append(b)
            distances[key] = distance
        distance = distances[key].get(v, float("inf"))
        shared.require((u, v) == (example.u, example.v) and distance == example.distance
                       and (args[2] if len(args) == 3 else None) == example.threshold
                       and example.answer == ("yes" if distance <= k else "no"), "Visible ground truth mismatch.")


def overlap_audit(graphs, prior_graphs):
    prior_sets = {(g["dataset"], frozenset(g["original_ids_by_anonymous_id"])) for g in prior_graphs}
    nodes = defaultdict(set)
    for g in prior_graphs:
        nodes[g["dataset"]].update(g["original_ids_by_anonymous_id"])
    result = {}
    for dataset in DATASETS:
        current = [g for g in graphs if g["dataset"] == dataset]
        selected = set().union(*(set(g["original_ids_by_anonymous_id"]) for g in current))
        duplicates = sum((dataset, frozenset(g["original_ids_by_anonymous_id"])) in prior_sets for g in current)
        shared.require(duplicates == 0, "Exact old source-node-set sample repeated; stop before inference.")
        result[dataset] = {"graphs": len(current), "unique_source_nodes": len(selected),
                           "source_nodes_shared_with_prior": len(selected & nodes[dataset]),
                           "graphs_sharing_prior_nodes": sum(bool(set(g["original_ids_by_anonymous_id"]) & nodes[dataset]) for g in current),
                           "exact_prior_node_set_duplicates": duplicates}
    return result


def plan(root, *, data_dir=None, priors=DEFAULT_PRIORS, loader=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data_dir = (REPO / "data/real" if data_dir is None else Path(data_dir)).resolve()
    frozen = sources()
    prior_graphs, prior_pins = [], []
    (root / "priors").mkdir()
    for index, path in enumerate(priors):
        path = Path(path).resolve()
        digest = shared.digest(path)
        target = root / "priors" / f"{index}.graphs.jsonl"
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        shared.require(shared.digest(target) == digest, "Prior changed during copy.")
        prior_pins.append({"source": str(path), "copy": target.relative_to(root).as_posix(), "sha256": digest})
        prior_graphs.extend(shared.read_rows(target))
    cache = shared.CachedLoader(loader)
    prepared_tasks, input_hashes, entries = {}, {}, []
    (root / "configs").mkdir()
    for model, task, config in configs(root, data_dir):
        if task not in prepared_tasks:
            prepared = prepare(config, cache)
            validate_prepared(config, prepared)
            prepared_tasks[task] = prepared
            target = root / "inputs" / task
            target.mkdir(parents=True)
            input_hashes[task] = export_prepared(target, prepared)
        filename = root / "configs" / f"{model}.{task}.yaml"
        with filename.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config.snapshot(), stream, sort_keys=True)
        shared.require(load_config(filename) == config, "Config did not round trip.")
        entries.append({"id": f"{model}.{task}", "model": model, "task": task,
                        "config": filename.relative_to(root).as_posix(), "yaml_sha256": shared.digest(filename),
                        "config_sha256": config.sha256, "effective_config": config.snapshot(),
                        "inputs": f"inputs/{task}", "input_sha256": input_hashes[task],
                        "expected_calls": len(prepared_tasks[task].examples), "max_calls": config.run.max_calls})
    shared.require(len({h["graphs.jsonl"] for h in input_hashes.values()}) == 1, "Tasks have different graphs.")
    shared.require(len({h["sources.json"] for h in input_hashes.values()}) == 1, "Tasks have different sources.")
    audit = overlap_audit(prepared_tasks["adjacency"].graphs, prior_graphs)
    shared.require(sources() == frozen, "Code changed during planning.")
    expected = {model: sum(e["expected_calls"] for e in entries if e["model"] == model) for model in shared.MODELS}
    manifest = {"kind": KIND, "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                "root": str(root), "data_dir": str(data_dir), "model_order": list(shared.MODELS),
                "task_order": list(TASKS), "code_sha256": frozen, "priors": prior_pins,
                "overlap_audit": audit, "entries": entries,
                "budget": {"expected_by_model": expected, "expected_total": sum(expected.values()),
                           "max_total": sum(e["max_calls"] for e in entries), "expected_jev": expected["jev"]},
                "notice": "Fresh sampled subgraphs; source nodes may overlap prior and sibling samples. Not a node-held-out final test. No protocol tuning or retries."}
    shared.write_json(root / "manifest.json", manifest)
    with (root / "manifest.sha256").open("x") as stream:
        stream.write(shared.digest(root / "manifest.json") + "\n")
    return manifest


def load(root, *, check_code=True):
    root = Path(root).resolve()
    shared.require(shared.digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip(), "Manifest changed.")
    m = shared.read_json(root / "manifest.json")
    shared.require(m["kind"] == KIND and m["root"] == str(root) and m["seed"] == SEED
                   and m["model_order"] == list(shared.MODELS) and m["task_order"] == list(TASKS), "Wrong plan identity.")
    if check_code:
        shared.require(m["code_sha256"] == sources(), "Source drift.")
    expected = list(configs(root, Path(m["data_dir"])))
    shared.require(len(m["entries"]) == len(expected), "Wrong entry count.")
    for (model, task, config), e in zip(expected, m["entries"]):
        shared.require(e["id"] == f"{model}.{task}" and e["model"] == model and e["task"] == task
                       and e["config"] == f"configs/{model}.{task}.yaml" and e["inputs"] == f"inputs/{task}"
                       and e["effective_config"] == config.snapshot() and e["max_calls"] == config.run.max_calls
                       and e["config_sha256"] == config.sha256, "Config/entry mismatch.")
        shared.require(shared.digest(root / e["config"]) == e["yaml_sha256"]
                       and load_config(root / e["config"]) == config, "Config file changed.")
        shared.require(e["input_sha256"] == {n: shared.digest(root / e["inputs"] / n) for n in shared.ARTIFACTS}, "Inputs changed.")
        shared.require(0 < e["expected_calls"] == shared.read_json(root / e["inputs"] / "preparation.json")["generated_questions"] <= e["max_calls"], "Invalid budget.")
    for pin in m["priors"]:
        shared.require(shared.digest(root / pin["copy"]) == pin["sha256"], "Prior copy changed.")
    by_model = {model: sum(e["expected_calls"] for e in m["entries"] if e["model"] == model) for model in shared.MODELS}
    shared.require(m["budget"] == {"expected_by_model": by_model, "expected_total": sum(by_model.values()),
                                  "max_total": sum(e["max_calls"] for e in m["entries"]), "expected_jev": by_model["jev"]}, "Budget changed.")
    return root, m


def regenerate(root, manifest, loader=None):
    cache = shared.CachedLoader(loader)
    prepared_tasks = {}
    for e in manifest["entries"]:
        if e["task"] in prepared_tasks:
            continue
        config = load_config(root / e["config"])
        prepared = prepare(config, cache)
        validate_prepared(config, prepared)
        with tempfile.TemporaryDirectory() as tmp:
            hashes = export_prepared(Path(tmp), prepared)
        shared.require(hashes == e["input_sha256"], "Regenerated input/source mismatch.")
        prepared_tasks[e["task"]] = prepared
    prior_graphs = [g for pin in manifest["priors"] for g in shared.read_rows(root / pin["copy"])]
    shared.require(overlap_audit(prepared_tasks["adjacency"].graphs, prior_graphs) == manifest["overlap_audit"], "Overlap audit mismatch.")
    return prepared_tasks


class GuardedClient(shared.GuardedClient):
    async def initialize(self):
        shared.require(not self.stop.is_set() and sources() == self.flight.manifest["code_sha256"], "Stopped or source drift.")
        output = self.flight.root / "runs" / self.entry["id"]
        shared.require({n: shared.digest(output / n) for n in shared.ARTIFACTS} == self.entry["input_sha256"], "Runner input mismatch.")
        run = shared.read_json(output / "run.json")
        shared.require(shared.read_json(output / "config.json") == self.entry["effective_config"]
                       and run["config_sha256"] == self.entry["config_sha256"]
                       and run["expected_calls"] == self.entry["expected_calls"], "Runner config/budget mismatch.")
        await self.delegate.initialize()

    async def predict(self, request):
        shared.require(sources() == self.flight.manifest["code_sha256"], "Source drift during generation.")
        try:
            return await super().predict(request)
        except DecisionClientError as exc:
            # Do not hammer a quota-limited provider or accumulate context-limit errors.
            if getattr(exc, "status_code", None) in (400, 413, 422, 429):
                self.stop.set()
                raise shared.Refusal("Systemic HTTP failure; no retry/resume.") from None
            raise


def report(root, models=None):
    root, m = load(root, check_code=False)
    selected = shared.selected_models(shared.MODELS if models is None else models)
    rows, runs = {}, []
    for e in m["entries"]:
        if e["model"] not in selected:
            continue
        rows[e["model"], e["task"]] = shared.completed_rows(root, e)
        runs.append({"id": e["id"], "summary": shared.read_json(root / "runs" / e["id"] / "summary.json")})
    pairs = []
    if "jev" in selected:
        for model in selected:
            if model != "jev":
                for task in TASKS:
                    pairs.append({"left": "jev", "right": model, "task": task,
                                  "counts": shared.matched_counts(rows["jev", task], rows[model, task])})
    return {"kind": KIND, "status": "completed-selected-lanes", "models": list(selected),
            "budget": m["budget"], "notice": m["notice"], "runs": runs, "cross_model": pairs}


async def run(root, *, factory=None, loader=None, preflight=True):
    root, m = load(root)
    # A partial execution is never silently resumed or replaced.
    (root / "runs").mkdir(exist_ok=False)
    clients = {}
    with shared.lane_locks(shared.MODELS):
        prepared = regenerate(root, m, loader)
        configurations = {e["id"]: load_config(root / e["config"]) for e in m["entries"]}
        stop = asyncio.Event()
        flight = shared.Preflight(root, m, m["entries"], [], shared.CachedLoader(loader))
        try:
            if preflight:
                shared.credential_preflight(shared.MODELS)
                await shared.vllm_preflight(list(configurations.values()))
            factory = shared.create_client if factory is None else factory
            # Initialize both task clients before any model generates a scored answer.
            for e in m["entries"]:
                config = configurations[e["id"]]
                kwargs = config.model.client_kwargs()
                if config.model.provider == "vllm":
                    kwargs["api_key"] = ""
                clients[e["id"]] = factory(config.model.provider, **kwargs)
            for e in m["entries"]:
                await asyncio.wait_for(clients[e["id"]].initialize(), configurations[e["id"]].model.timeout_seconds)

            async def lane(model):
                for e in m["entries"]:
                    if e["model"] != model:
                        continue
                    guarded = GuardedClient(clients[e["id"]], flight, e, stop)
                    await run_experiment(configurations[e["id"]], client=guarded,
                                         prepared_input=prepared[e["task"]], progress=False)
                    shared.completed_rows(root, e)

            tasks = [asyncio.create_task(lane(model)) for model in shared.MODELS]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                stop.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        finally:
            await asyncio.gather(*(c.aclose() for c in clients.values()), return_exceptions=True)
    result = report(root)
    shared.write_json(root / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            m = plan(args.root)
            print(json.dumps({"root": m["root"], "budget": m["budget"], "overlap_audit": m["overlap_audit"]}, sort_keys=True))
        elif args.command == "run":
            asyncio.run(run(args.root))
            print("All seven lanes complete; final report saved.")
        else:
            print(json.dumps(report(args.root), sort_keys=True, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Large matrix stopped ({type(exc).__name__}); no retry/resume; exception text suppressed.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())