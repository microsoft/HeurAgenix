#!/usr/bin/env python3
"""Offline paired development plan and explicitly invoked, fail-closed live lanes.

This is deliberately separate from the benchmark CLI. No command downloads data,
starts services, resumes jobs, or reads credential files. See docs/paired-ablations.md.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import warnings

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import httpx
import yaml

from src.benchmark.config import (
    BenchmarkConfig, DataConfig, GraphExclusionConfig, ModelConfig,
    PresentationConfig, RunConfig, SamplingConfig, TaskConfig, load_config,
)
from src.benchmark.prepare import excluded_source_nodes, prepare
from src.benchmark.runner import export_prepared, run_experiment, summarize
from src.clients.base import BaseDecisionClient, DecisionClientError
from src.clients.registry import create_client
from src.datasets import load_graph

SEED = 20260925
DEFAULT_ROOT = REPO / "output/paired-graph-dev-20260923-v4-seven-models"
DEFAULT_PRIOR = REPO / "output/adjacency.jev/graphs.jsonl"
DATASETS = ("facebook", "ca-grqc", "power", "human-ppi")
LEGACY_MODELS = ("jev", "qwen08", "qwen2", "qwen4", "qwen9")
MODELS = (*LEGACY_MODELS, "gpt54", "gpt6astra")
ARMS = {
    "baseline_nodes_edges": PresentationConfig("baseline", "nodes_edges"),
    "algorithmic_v1_nodes_edges": PresentationConfig("algorithmic_v1", "nodes_edges"),
    "baseline_adjacency_list_v1": PresentationConfig("baseline", "adjacency_list_v1"),
    "algorithmic_v1_adjacency_list_v1": PresentationConfig("algorithmic_v1", "adjacency_list_v1"),
}
BASELINE = next(iter(ARMS))
TASKS = {"adjacency": TaskConfig(4, 0, ()),
         "distance_threshold": TaskConfig(0, 2, (2, 3, 4))}
ARTIFACTS = ("requests.jsonl", "labels.jsonl", "graphs.jsonl", "sources.json", "preparation.json")


class Refusal(RuntimeError):
    """Only controlled, credential-free messages are displayed by the CLI."""


class AuthenticationAbort(Refusal):
    """Not a DecisionClientError: the runner must abort, not continue the batch."""


def require(condition, message):
    if not condition:
        raise Refusal(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def code_hashes():
    # Include the orchestration and dataset code too: additions/deletions also drift.
    paths = [Path(__file__).resolve()]
    for directory in ("benchmark", "clients", "datasets"):
        paths.extend((REPO / "src" / directory).rglob("*.py"))
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(paths)}


def model_configs(output_format=None, models=MODELS):
    settings = {
        "jev": ModelConfig(provider="typesafe", model="jev-1.13.0", timeout_seconds=30.0),
        "qwen08": ModelConfig(provider="vllm", model="Qwen3.5-0.8B", timeout_seconds=180.0,
                              think=False, temperature=0.0, top_p=1.0, seed=SEED,
                              max_tokens=4096, base_url="http://127.0.0.1:8000/v1"),
        "qwen2": ModelConfig(provider="vllm", model="Qwen3.5-2B", timeout_seconds=180.0,
                             think=False, temperature=0.0, top_p=1.0, seed=SEED,
                             max_tokens=4096, base_url="http://127.0.0.1:8001/v1"),
        "qwen4": ModelConfig(provider="vllm", model="Qwen3.5-4B", timeout_seconds=180.0,
                             think=False, temperature=0.0, top_p=1.0, seed=SEED,
                             max_tokens=4096, base_url="http://127.0.0.1:8002/v1"),
        "qwen9": ModelConfig(provider="vllm", model="Qwen3.5-9B", timeout_seconds=180.0,
                             think=False, temperature=0.0, top_p=1.0, seed=SEED,
                             max_tokens=4096, base_url="http://127.0.0.1:8003/v1"),
        "gpt54": ModelConfig(provider="github_copilot", model="gpt-5.4",
                             timeout_seconds=180.0, max_tokens=4096),
        "gpt6astra": ModelConfig(provider="github_copilot", model="gpt-6-astra",
                                 timeout_seconds=180.0, max_tokens=4096),
    }
    return {name: replace(settings[name], output_format=output_format)
            if settings[name].provider != "typesafe" else settings[name] for name in models}


def arm_order():
    order = list(ARMS)
    random.Random(SEED).shuffle(order)
    return order


def configurations(root, data_dir, prior, prior_sha, output_format=None, models=MODELS, arms=None):
    for model, settings in model_configs(output_format, models).items():
        for arm in (arm_order() if arms is None else arms):
            for task, questions in TASKS.items():
                run_id = f"{model}.{arm}.{task}"
                config = BenchmarkConfig(
                    schema_version=1,
                    run=RunConfig(seed=SEED, repetitions=1,
                                  max_calls=32 if task == "adjacency" else 48,
                                  output_dir=root / "runs" / run_id),
                    data=DataConfig(datasets=DATASETS, data_dir=data_dir),
                    sampling=SamplingConfig(
                        node_counts=(32,), samples_per_size=2, max_attempts=1000,
                        exclude_graphs=GraphExclusionConfig(path=prior, sha256=prior_sha,
                                                           overlap="source_nodes")),
                    tasks=questions, model=settings, presentation=ARMS[arm],
                )
                yield model, arm, task, run_id, config


class CachedLoader:
    """One verified local load per source and resolved data directory per command."""

    def __init__(self, loader=None):
        self.loader = load_graph if loader is None else loader
        self.cache = {}

    def __call__(self, name, *, data_dir):
        key = (name, Path(data_dir).resolve())
        if key not in self.cache:
            self.cache[key] = self.loader(name, data_dir=key[1])
        return self.cache[key]


def validate_prepared(config, prepared):
    require(len(prepared.graphs) == config.planned_graphs == 8,
            "Plan requires all eight node-disjoint graphs; do not change the seed to rescue a plan.")
    require(0 < len(prepared.examples) <= config.run.max_calls,
            "Invalid generated question budget.")
    blocked = {name: set(nodes) for name, nodes in excluded_source_nodes(config).items()}
    counts = {name: 0 for name in DATASETS}
    for graph in prepared.graphs:
        name = graph["dataset"]
        nodes = set(graph["original_ids_by_anonymous_id"])
        require(graph["node_count"] == len(nodes) == 32 and not nodes & blocked[name],
                "Prior/sibling source-node overlap or invalid graph size.")
        blocked[name].update(nodes)
        counts[name] += 1
    require(all(n == 2 for n in counts.values()), "Expected two graphs per source.")


def validate_pairing(entries):
    models = tuple(dict.fromkeys(e["model"] for e in entries))
    require(len({e["input_sha256"]["graphs.jsonl"] for e in entries}) == 1,
            "Graphs differ across tasks, arms, or models.")
    require(len({e["input_sha256"]["sources.json"] for e in entries}) == 1,
            "Raw provenance differs across runs.")
    for task in TASKS:
        cells = [e for e in entries if e["task"] == task]
        for artifact in ("labels.jsonl", "preparation.json"):
            require(len({e["input_sha256"][artifact] for e in cells}) == 1,
                    "Labels/coverage differ across arms or models.")
        for arm in dict.fromkeys(e["arm"] for e in entries):
            cells_arm = [e for e in cells if e["arm"] == arm]
            require(len(cells_arm) == len(models) and len({
                json.dumps(e["input_sha256"], sort_keys=True) for e in cells_arm}) == 1,
                "Models must receive byte-identical inputs within each arm/task.")


def budgets(entries):
    models = tuple(dict.fromkeys(e["model"] for e in entries))
    maximum = 80 * len({e["arm"] for e in entries})
    actual = {m: sum(e["expected_calls"] for e in entries if e["model"] == m) for m in models}
    require(all(0 < n <= maximum for n in actual.values()), "Per-model budget exceeded.")
    return {"max_per_arm_model": 80, "max_per_model": maximum, "max_jev": maximum,
            "max_total": maximum * len(models), "expected_by_model": actual, "expected_total": sum(actual.values())}


def plan(root=DEFAULT_ROOT, *, data_dir=None, prior=DEFAULT_PRIOR, loader=None,
    output_format="answer_only", models=MODELS, single_prompt=False):
    """Reserve a NEW root first. A failed/incomplete reservation is never reused."""
    require(output_format in (None, "json", "answer_only"), "Unsupported local output format.")
    models = tuple(models)
    require(models in (LEGACY_MODELS, MODELS), "Unsupported model roster.")
    arms = [BASELINE] if single_prompt else arm_order()
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    data_dir = (REPO / "data/real" if data_dir is None else Path(data_dir)).expanduser().resolve()
    prior = Path(prior).expanduser().resolve()
    frozen_code = code_hashes()
    prior_sha = digest(prior)
    with (root / "prior-graphs.jsonl").open("xb") as stream:
        stream.write(prior.read_bytes())
    require(digest(root / "prior-graphs.jsonl") == prior_sha, "Prior changed while planning.")
    cache = CachedLoader(loader)
    entries = []
    for model, arm, task, run_id, config in configurations(root, data_dir, prior, prior_sha, output_format, models, arms):
        config_path = root / "configs" / f"{run_id}.yaml"
        config_path.parent.mkdir(exist_ok=True)
        with config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config.snapshot(), stream, sort_keys=True)
        require(load_config(config_path) == config, "YAML snapshot did not round-trip.")
        prepared = prepare(config, cache)
        validate_prepared(config, prepared)
        shared = root / "inputs" / arm / task
        if not shared.exists():
            shared.mkdir(parents=True, exist_ok=False)
            hashes = export_prepared(shared, prepared)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                hashes = export_prepared(Path(tmp), prepared)
            require(hashes == {name: digest(shared / name) for name in ARTIFACTS},
                    "Model-specific input drift during planning.")
        entries.append({
            "id": run_id, "model": model, "arm": arm, "task": task,
            "config": config_path.relative_to(root).as_posix(),
            "config_sha256": config.sha256, "yaml_sha256": digest(config_path),
            "effective_config": config.snapshot(),
            "inputs": shared.relative_to(root).as_posix(), "input_sha256": hashes,
            "expected_calls": len(prepared.examples), "max_calls": config.run.max_calls,
        })
    validate_pairing(entries)
    require(code_hashes() == frozen_code and digest(prior) == prior_sha,
            "Code/prior changed during planning; use a new root after resolving drift.")
    manifest = {
        "schema_version": 1, "kind": "paired-development-not-final", "seed": SEED,
        "root": str(root), "data_dir": str(data_dir),
        "model_order": list(models),
        "local_output_format": output_format,
        "prior": {"path": str(prior), "sha256": prior_sha, "copy": "prior-graphs.jsonl",
                  "overlap": "source_nodes"},
        "code_sha256": frozen_code, "arm_order": arms,
        "task_order": list(TASKS),
        "lane_order": {m: [e["id"] for e in entries if e["model"] == m] for m in models},
        "budget": budgets(entries), "entries": entries,
    }
    write_json(root / "manifest.json", manifest)
    with (root / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(digest(root / "manifest.json") + "\n")
    return manifest


def load_plan(root, *, check_code=True):
    root = Path(root).expanduser().resolve()
    require(digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip(),
            "Manifest checksum mismatch or unfinished plan.")
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == 1 and manifest["kind"] == "paired-development-not-final"
            and manifest["seed"] == SEED and manifest["root"] == str(root),
            "Wrong protocol, seed, or relocated plan.")
    arms = manifest["arm_order"]
    require(arms in (arm_order(), [BASELINE]) and manifest["task_order"] == list(TASKS),
            "Execution order changed.")
    if check_code:
        require(manifest["code_sha256"] == code_hashes(), "Source code drift since plan; live run refused.")
    pin = manifest["prior"]
    require(pin["copy"] == "prior-graphs.jsonl" and pin["overlap"] == "source_nodes",
            "Invalid prior pin.")
    require(digest(root / pin["copy"]) == pin["sha256"], "Pinned prior copy changed.")
    if check_code:
        require(digest(Path(pin["path"])) == pin["sha256"], "Original prior artifact changed.")
    # Missing key denotes the historical JSON protocol; never reinterpret old results.
    output_format = manifest.get("local_output_format")
    require(output_format in (None, "json", "answer_only"), "Unsupported local output format.")
    # Older five-model manifests did not pin a separate roster field.
    models = tuple(manifest.get("model_order", LEGACY_MODELS))
    require(models in (LEGACY_MODELS, MODELS), "Unsupported model roster.")
    expected = list(configurations(root, Path(manifest["data_dir"]), Path(pin["path"]), pin["sha256"],
                                   output_format, models, arms))
    require(len(manifest["entries"]) == len(expected) == len(models) * len(arms) * len(TASKS),
            "Expected the complete pinned model matrix; create a new plan for this protocol.")
    for entry, (model, arm, task, run_id, config) in zip(manifest["entries"], expected):
        require((entry["id"], entry["model"], entry["arm"], entry["task"])
                == (run_id, model, arm, task), "Configuration matrix/order changed.")
        require(entry["config"] == f"configs/{run_id}.yaml"
                and entry["inputs"] == f"inputs/{arm}/{task}", "Unexpected artifact location.")
        path = root / entry["config"]
        require(digest(path) == entry["yaml_sha256"] and load_config(path) == config
                and entry["effective_config"] == config.snapshot()
                and entry["config_sha256"] == config.sha256, "Effective configuration changed.")
        require(entry["max_calls"] == config.run.max_calls
                and type(entry["expected_calls"]) is int
                and 0 < entry["expected_calls"] <= config.run.max_calls, "Invalid call budget.")
        require(set(entry["input_sha256"]) == set(ARTIFACTS), "Missing artifact pins.")
        require({n: digest(root / entry["inputs"] / n) for n in ARTIFACTS} == entry["input_sha256"],
                "Shared input artifact changed.")
    require(manifest["lane_order"] == {m: [e["id"] for e in manifest["entries"] if e["model"] == m]
                                         for m in models}, "Lane order changed.")
    require(manifest["budget"] == budgets(manifest["entries"]), "Budget manifest changed.")
    validate_pairing(manifest["entries"])
    return root, manifest


def completed_rows(root, entry):
    """Verify completed status, exact inputs/config/counts, labels and derived summary."""
    output = root / "runs" / entry["id"]
    require(output.is_dir() and not output.is_symlink(), "Run missing or unsafe; no partial reports/resume.")
    run, summary = read_json(output / "run.json"), read_json(output / "summary.json")
    count = entry["expected_calls"]
    require(run["status"] == summary["run_status"] == "completed", "Run is incomplete/failed; never resume it.")
    require(run["config_sha256"] == entry["config_sha256"]
            and read_json(output / "config.json") == entry["effective_config"], "Completed run config mismatch.")
    settings = entry["effective_config"]["model"]
    protocol = ("native-choice" if settings["provider"] == "typesafe" else
                "answer-only-v1" if settings.get("output_format") == "answer_only" else "json-choice-v1")
    require(run["response_protocol"] == protocol, "Completed run response protocol mismatch.")
    require(run["artifact_sha256"] == entry["input_sha256"]
            and {n: digest(output / n) for n in ARTIFACTS} == entry["input_sha256"],
            "Completed run inputs mismatch.")
    require(run["expected_calls"] == run["completed_calls"] == count, "Completed call count mismatch.")
    labels = read_rows(output / "labels.jsonl")
    rows = read_rows(output / "attempts.jsonl")
    require(len(rows) == len(labels) == count
            and len({r["request_id"] for r in rows}) == count, "Missing/duplicate attempts.")
    for row, label in zip(rows, labels):
        require(all(row.get(k) == v for k, v in label.items())
                and row["repetition"] == 0 and row["attempt_id"] == f"{label['request_id']}-r000",
                "Attempt request IDs/order/answers do not match pinned labels.")
        require(row["status"] in ("success", "failure") and type(row["correct"]) is bool
                and row["http_status"] not in (401, 403), "Invalid or systemic failure in completed run.")
        selected = row["selected_option_id"]
        require((row["status"] == "success" and selected in ("yes", "no")
                 and row["correct"] == (selected == label["answer"]))
                or (row["status"] == "failure" and selected is None and not row["correct"]),
                "Attempt correctness is inconsistent.")
    derived = summarize(rows, expected_calls=count, preparation=read_json(output / "preparation.json"))
    require(all(summary.get(k) == v for k, v in derived.items()), "Summary does not match attempts.")
    return rows


def selected_models(models):
    selected = tuple(models)
    require(bool(selected) and len(set(selected)) == len(selected) and set(selected) <= set(MODELS),
            "Select distinct models from " + ", ".join(MODELS) + ".")
    return selected


@dataclass
class Preflight:
    root: Path
    manifest: dict
    pending: list
    skipped: list
    loader: CachedLoader


def offline_preflight(root, models=MODELS, *, skip_completed=False, loader=None):
    selected = selected_models(models)
    root, manifest = load_plan(root)
    pending, skipped = [], []
    # Check ALL selected outputs before any regeneration, GET, or client creation.
    for entry in manifest["entries"]:
        if entry["model"] not in selected:
            continue
        output = root / "runs" / entry["id"]
        if output.exists() or output.is_symlink():
            require(skip_completed, "Run output already exists; only verified --skip-completed is permitted.")
            completed_rows(root, entry)
            skipped.append(entry)
        else:
            pending.append(entry)
    cache = CachedLoader(loader)
    # Regenerate every configuration, not only the next run, using four raw loads total.
    for entry in manifest["entries"]:
        config = load_config(root / entry["config"])
        prepared = prepare(config, cache)
        validate_prepared(config, prepared)
        with tempfile.TemporaryDirectory() as tmp:
            hashes = export_prepared(Path(tmp), prepared)
        require(hashes == entry["input_sha256"] and len(prepared.examples) == entry["expected_calls"],
                "Regenerated inputs/provenance/coverage differ from plan.")
    require(code_hashes() == manifest["code_sha256"], "Code changed during preflight.")
    return Preflight(root, manifest, pending, skipped, cache)


def credential_preflight(models, prompt_api_key=False):
    """Return a prompted key only in memory; normal credentials stay in the environment."""
    if "jev" not in models:
        return None
    key = os.environ.get("TYPESAFE_API_KEY")
    prompted = False
    if not key:
        require(prompt_api_key, "TYPESAFE_API_KEY is missing; no live calls were made.")
        require(sys.stdin.isatty() and sys.stderr.isatty(), "Secret prompt requires stdin/stderr TTYs.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("TYPESAFE_API_KEY (not saved): ")
            prompted = True
        except (getpass.GetPassWarning, EOFError, OSError):
            raise Refusal("Secure controlling-terminal prompt unavailable; refusing fallback.") from None
    require(isinstance(key, str) and bool(key.strip()) and key.strip().isascii()
            and not any(c.isspace() for c in key.strip()), "Invalid TYPESAFE_API_KEY format.")
    return key.strip() if prompted else None


async def vllm_preflight(configs, *, transport=None):
    """GET only; both gauges must exist and every reported engine must be idle."""
    metric = re.compile(r"^(vllm:num_requests_(?:running|waiting))(?:\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$")
    endpoints = {c.model.base_url: c.model.model for c in configs if c.model.provider == "vllm"}
    async with httpx.AsyncClient(timeout=10, follow_redirects=False, transport=transport,
                                 trust_env=False) as client:
        for base, identity in endpoints.items():
            models_response = await client.get(base.rstrip("/") + "/models")
            require(models_response.status_code == 200, "vLLM /models GET failed.")
            body = models_response.json()
            require(isinstance(body, dict) and isinstance(body.get("data"), list)
                    and all(isinstance(row, dict) for row in body["data"])
                    and [row.get("id") for row in body["data"]] == [identity],
                    "vLLM served model identity must match exactly.")
            metrics_response = await client.get(base.rsplit("/v1", 1)[0] + "/metrics")
            require(metrics_response.status_code == 200, "vLLM /metrics GET failed.")
            seen = set()
            for line in metrics_response.text.splitlines():
                if not line or line.startswith("#"):
                    continue
                match = metric.fullmatch(line)
                if match:
                    name, text = match.groups()
                    value = float(text)
                    require(math.isfinite(value) and value == 0, "vLLM is busy; do not resume/interrupt old jobs.")
                    seen.add(name)
                elif re.match(r"^vllm:num_requests_(?:running|waiting)(?:\{|\s|$)", line):
                    raise Refusal("Unrecognized vLLM occupancy metric; refusing live calls.")
            require(seen == {"vllm:num_requests_running", "vllm:num_requests_waiting"},
                    "vLLM occupancy gauges missing; refusing live calls.")


async def copilot_preflight(configs):
    """Authenticate and check both exact model IDs/modes, with no generation."""
    checked = set()
    for config in configs:
        if config.model.provider != "github_copilot" or config.model.model in checked:
            continue
        client = create_client("github_copilot", **config.model.client_kwargs())
        try:
            await client.initialize()
        except (DecisionClientError, ImportError):
            raise Refusal("Copilot preflight failed: check SDK installation, login and exact model availability; no generation started.") from None
        finally:
            await client.aclose()
        checked.add(config.model.model)


@contextmanager
def lane_locks(models):
    """Advisory per-user locks also exclude this tool's lanes from other plan roots."""
    descriptors = []
    try:
        for model in sorted(models):
            path = Path(tempfile.gettempdir()) / f"jevgraphbench-paired-{os.getuid()}-{model}.lock"
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            descriptors.append(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Refusal("A selected paired lane is already active; no concurrent same-model runs.") from None
        yield
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


class GuardedClient(BaseDecisionClient):
    """Transparent adapter except for pinned-input and fatal-auth guards.

    The runner truthfully records injected_client=true. The delegate is the real
    registry adapter, not a mock; no metadata is rewritten to disguise injection.
    """

    def __init__(self, delegate, flight, entry, stop):
        self.delegate, self.flight, self.entry, self.stop = delegate, flight, entry, stop

    @property
    def capabilities(self):
        return self.delegate.capabilities

    async def initialize(self):
        require(not self.stop.is_set(), "Another lane has aborted.")
        require(code_hashes() == self.flight.manifest["code_sha256"], "Source code drift before run.")
        output = self.flight.root / "runs" / self.entry["id"]
        require({n: digest(output / n) for n in ARTIFACTS} == self.entry["input_sha256"],
                "Runner inputs differ from preflight; refusing first model call.")
        run = read_json(output / "run.json")
        require(read_json(output / "config.json") == self.entry["effective_config"]
            and run["config_sha256"] == self.entry["config_sha256"]
            and run["expected_calls"] == self.entry["expected_calls"],
            "Runner configuration/budget differs from preflight.")
        await self.delegate.initialize()

    async def predict(self, request):
        require(not self.stop.is_set(), "Another lane has aborted.")
        try:
            return await self.delegate.predict(request)
        except DecisionClientError as exc:
            if getattr(exc, "status_code", None) in (401, 403):
                self.stop.set()
                raise AuthenticationAbort("HTTP 401/403: all selected lanes aborted; no retry/resume.") from None
            raise

    async def aclose(self):
        await self.delegate.aclose()


async def run_live(root, models=MODELS, *, skip_completed=False, prompt_api_key=False,
                   parallel_models=False, loader=None):
    models = selected_models(models)
    # Must precede even offline regeneration and all HTTP GETs.
    prompted_key = credential_preflight(models, prompt_api_key)
    with lane_locks(models):
        flight = offline_preflight(root, models, skip_completed=skip_completed, loader=loader)
        configs = [load_config(flight.root / e["config"]) for e in flight.pending]
        await vllm_preflight(configs)  # ALL pending endpoints before the first billable call.
        await copilot_preflight(configs)  # Auth/model discovery only, before Jev or any other generation.
        stop = asyncio.Event()

        async def lane(model):
            try:
                for entry in flight.pending:
                    if entry["model"] != model:
                        continue
                    require(not stop.is_set(), "Another lane has aborted.")
                    config = load_config(flight.root / entry["config"])
                    require(config.sha256 == entry["config_sha256"], "Config changed after preflight.")
                    if config.model.provider == "vllm":
                        await vllm_preflight([config])  # Detect intervening foreign work between runs.
                    kwargs = config.model.client_kwargs()
                    if model == "jev" and prompted_key is not None:
                        kwargs["api_key"] = prompted_key
                    elif config.model.provider == "vllm":
                        kwargs["api_key"] = ""  # Pinned local unauthenticated endpoints, no token-cache lookup.
                    delegate = create_client(config.model.provider, **kwargs)
                    guarded = GuardedClient(delegate, flight, entry, stop)
                    try:
                        await run_experiment(config, client=guarded, loader=flight.loader, progress=False)
                    finally:
                        await guarded.aclose()  # Also close if output reservation failed before runner context.
                    completed_rows(flight.root, entry)
            except BaseException:
                stop.set()
                raise

        if parallel_models:
            tasks = [asyncio.create_task(lane(model)) for model in models]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                stop.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        else:
            for model in models:
                await lane(model)
    return report(root, models)


def matched_counts(left, right):
    a = {r["request_id"]: r for r in left}
    b = {r["request_id"]: r for r in right}
    require(len(a) == len(left) == len(b) == len(right) and a.keys() == b.keys(),
            "Pairwise request IDs differ.")
    result = {}
    for answer in ("all", "yes", "no"):
        counts = dict(n=0, both_correct=0, left_only_correct=0, right_only_correct=0,
                      neither_correct=0, correctness_changed=0, selection_changed=0)
        for request_id, lrow in a.items():
            rrow = b[request_id]
            require(lrow["answer"] == rrow["answer"], "Pairwise ground-truth answers differ.")
            if answer != "all" and lrow["answer"] != answer:
                continue
            lc, rc = lrow["correct"], rrow["correct"]
            key = ("both_correct" if lc and rc else "left_only_correct" if lc else
                   "right_only_correct" if rc else "neither_correct")
            counts["n"] += 1
            counts[key] += 1
            counts["correctness_changed"] += lc != rc
            counts["selection_changed"] += lrow["selected_option_id"] != rrow["selected_option_id"]
        result[answer] = counts
    return result


def report(root, models=None):
    """Read-only, complete selected lanes only; historical code drift is allowed."""
    root, manifest = load_plan(root, check_code=False)
    roster = tuple(manifest.get("model_order", LEGACY_MODELS))
    models = selected_models(roster if models is None else models)
    require(set(models) <= set(roster), "Requested model is not in this plan.")
    rows, summaries = {}, []
    for entry in manifest["entries"]:
        if entry["model"] not in models:
            continue
        cell = completed_rows(root, entry)
        rows[entry["model"], entry["arm"], entry["task"]] = cell
        summary = read_json(root / "runs" / entry["id"] / "summary.json")
        summaries.append({"id": entry["id"], "overall": summary["overall"],
                          "preparation": summary["preparation"]})
    pairs = []
    for model in models:
        for task in TASKS:
            for arm in manifest["arm_order"]:
                if arm != BASELINE:
                    pairs.append({"model": model, "task": task, "left": BASELINE, "right": arm,
                                  "counts": matched_counts(rows[model, BASELINE, task], rows[model, arm, task])})
    cross = []
    if "jev" in models:
        for model in (m for m in models if m != "jev"):
            for arm in manifest["arm_order"]:
                for task in TASKS:
                    cross.append({"arm": arm, "task": task, "left": "jev", "right": model,
                                  "counts": matched_counts(rows["jev", arm, task], rows[model, arm, task])})
    return {"status": "completed-selected-development-lanes-not-final", "models": list(models),
            "local_output_format": manifest.get("local_output_format") or "json",
            "notice": "Development ablation only: report all arms; no winner or held-out claim.",
            "budget": manifest["budget"], "runs": summaries,
            "versus_baseline": pairs, "cross_model": cross}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "preflight", "run", "report"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if command == "plan":
            p.add_argument("--data-dir", type=Path, default=REPO / "data/real")
            p.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
            p.add_argument("--output-format", choices=("json", "answer_only"), default="answer_only")
            p.add_argument("--single-prompt", action="store_true",
                           help="Use only baseline nodes/edges for both tasks, 80 calls per model.")
        else:
            p.add_argument("--models", nargs="+", choices=MODELS,
                           default=None if command == "report" else list(MODELS))
        if command in ("preflight", "run"):
            p.add_argument("--skip-completed", action="store_true")
            p.add_argument("--prompt-api-key", action="store_true")
        if command == "run":
            p.add_argument("--parallel-models", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.root, data_dir=args.data_dir, prior=args.prior,
                          output_format=args.output_format, single_prompt=args.single_prompt)
            result = {"status": "offline-plan-created", "root": result["root"],
                      "local_output_format": result["local_output_format"],
                      "budget": result["budget"], "arm_order": result["arm_order"]}
        elif args.command == "preflight":
            credential_preflight(args.models, args.prompt_api_key)
            with lane_locks(args.models):
                flight = offline_preflight(args.root, args.models, skip_completed=args.skip_completed)
                configs = [load_config(flight.root / e["config"]) for e in flight.pending]
                asyncio.run(vllm_preflight(configs))
                asyncio.run(copilot_preflight(configs))
            result = {"status": "preflight-passed-no-generation", "pending_runs": len(flight.pending),
                      "verified_skips": len(flight.skipped)}
        elif args.command == "run":
            result = asyncio.run(run_live(args.root, args.models, skip_completed=args.skip_completed,
                                         prompt_api_key=args.prompt_api_key, parallel_models=args.parallel_models))
        else:
            result = report(args.root, args.models)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Refusal as exc:
        print(f"Refused: {exc}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
        # Never echo arbitrary server bodies, filesystem contents, or secrets.
        print("Refused: invalid/missing artifacts or failed transport; no automatic retry.", file=sys.stderr)
    except Exception:
        print("Aborted: unexpected failure; exception details suppressed to protect credentials.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Interrupted: in-flight calls may be billable; partial runs cannot be resumed.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())