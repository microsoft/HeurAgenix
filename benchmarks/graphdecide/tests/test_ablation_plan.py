"""Paired-plan tests with tiny local path graphs and mock adapters; no live API."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import getpass
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import warnings

import httpx
import networkx as nx
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import paired_graph_ablation as ablation
from src.benchmark.config import load_config
from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, DecisionResponse, ProviderHTTPError,
    UnsupportedRequestError,
)
from src.datasets import GraphDataset, LoadStats, get_source


class Fixture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.root = self.directory / "plan"
        self.prior = self.directory / "prior.jsonl"
        self.data = self.directory / "data"
        self.calls = Counter()
        self.graph = nx.freeze(nx.path_graph(256))
        self.prior.write_text("".join(json.dumps({
            "dataset": name, "graph_id": f"{name}-old", "node_count": 1,
            "original_ids_by_anonymous_id": [0],
        }) + "\n" for name in ablation.DATASETS))

    def loader(self, name, **kwargs):
        self.calls[name] += 1
        graph = self.graph
        stats = LoadStats(255, 0, 0, 256, 255, 0, 1)
        return GraphDataset(get_source(name), graph, stats, "f" * 64)

    def plan(self, **kwargs):
        # An accidental client construction or HTTP operation is a test failure.
        with patch.object(ablation, "create_client", side_effect=AssertionError("offline")), \
                patch("httpx.AsyncClient", side_effect=AssertionError("offline")), \
                patch("httpx.Client", side_effect=AssertionError("offline")):
            return ablation.plan(self.root, data_dir=self.data, prior=self.prior, loader=self.loader, **kwargs)


class MockDecisionClient(BaseDecisionClient):
    def __init__(self, model="mock", *, auth_status=None):
        self.model = model
        self.auth_status = auth_status
        self.requests = []
        self.closed = False

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        self.requests.append(request.request_id)
        if self.auth_status is not None:
            raise ProviderHTTPError(self.auth_status)
        return DecisionResponse(request.request_id, "yes", self.model)

    async def aclose(self):
        self.closed = True


class PlanTests(unittest.TestCase):
    def test_single_prompt_seven_models_pins_and_budget(self):
        f = self.fixture
        manifest = f.plan(single_prompt=True)
        self.assertEqual(manifest["arm_order"], [ablation.BASELINE])
        self.assertEqual(len(manifest["entries"]), 14)
        self.assertEqual(manifest["budget"]["expected_total"], 560)
        self.assertEqual(manifest["budget"]["max_jev"], 80)
        self.assertEqual(manifest["budget"]["expected_by_model"], dict.fromkeys(ablation.MODELS, 80))
        self.assertEqual(ablation.load_plan(f.root)[1], manifest)
        flight = ablation.offline_preflight(f.root, loader=f.loader)
        self.assertEqual(len(flight.pending), 14)
        f.root = f.directory / "four-arms"
        full = f.plan()
        baseline = [e for e in full["entries"] if e["arm"] == ablation.BASELINE]
        self.assertEqual([e["input_sha256"] for e in manifest["entries"]],
                         [e["input_sha256"] for e in baseline])

    def test_output_protocol_changes_only_local_model_settings(self):
        f = self.fixture
        old = f.plan(output_format=None)
        old_root = f.root
        # Historical manifests omitted this key. Preserve their read-only interpretation.
        old.pop("local_output_format")
        (old_root / "manifest.json").write_text(json.dumps(old))
        (old_root / "manifest.sha256").write_text(ablation.digest(old_root / "manifest.json"))
        ablation.load_plan(old_root, check_code=False)
        f.root = f.directory / "answer-only"
        new = f.plan()
        for before, after in zip(old["entries"], new["entries"]):
            self.assertEqual(before["id"], after["id"])
            self.assertEqual(before["input_sha256"], after["input_sha256"])
            b, a = before["effective_config"]["model"], after["effective_config"]["model"]
            self.assertEqual(a, b if before["model"] == "jev" else {**b, "output_format": "answer_only"})
        self.assertEqual(old["budget"], new["budget"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)

    def test_offline_matrix_absolute_snapshots_hashes_pairing_and_budget(self):
        f = self.fixture
        manifest = f.plan()
        self.assertEqual(manifest["local_output_format"], "answer_only")
        self.assertEqual(f.calls, Counter({name: 1 for name in ablation.DATASETS}))
        self.assertEqual(len(list((f.root / "configs").glob("*.yaml"))), 56)
        self.assertEqual(manifest["model_order"], list(ablation.MODELS))
        self.assertEqual(manifest["budget"]["expected_total"], 2240)
        self.assertEqual(manifest["budget"]["expected_by_model"],
             dict(jev=320, qwen08=320, qwen2=320, qwen4=320, qwen9=320, gpt54=320, gpt6astra=320))
        self.assertEqual(manifest["arm_order"], ablation.arm_order())
        graph_hashes, label_hashes, request_hashes = set(), {}, {}
        for entry in manifest["entries"]:
            config = load_config(f.root / entry["config"])
            self.assertEqual(config.snapshot(), entry["effective_config"])
            self.assertEqual(config.sha256, entry["config_sha256"])
            self.assertEqual(config.run.seed, 20260925)
            self.assertEqual(config.run.repetitions, 1)
            self.assertEqual(config.sampling.node_counts, (32,))
            self.assertEqual(config.sampling.samples_per_size, 2)
            self.assertEqual(config.data.datasets, ablation.DATASETS)
            self.assertEqual(config.sampling.exclude_graphs.path, f.prior.resolve())
            self.assertEqual(config.sampling.exclude_graphs.sha256, ablation.digest(f.prior))
            self.assertEqual(config.sampling.exclude_graphs.overlap, "source_nodes")
            self.assertTrue(config.data.data_dir.is_absolute())
            self.assertTrue(config.run.output_dir.is_absolute())
            self.assertEqual(config.run.max_calls, 32 if entry["task"] == "adjacency" else 48)
            if entry["model"] == "jev":
                self.assertEqual(config.model.client_kwargs(), {"model": "jev-1.13.0", "timeout_seconds": 30.0})
            elif entry["model"] in ("gpt54", "gpt6astra"):
                self.assertEqual(config.model.provider, "github_copilot")
                self.assertEqual(config.model.client_kwargs(), {
                    "model": "gpt-5.4" if entry["model"] == "gpt54" else "gpt-6-astra",
                    "timeout_seconds": 180.0, "think": None, "reasoning_effort": None,
                    "max_tokens": 4096, "output_format": "answer_only",
                })
            else:
                size, port = {"qwen08": ("0.8B", 8000), "qwen2": ("2B", 8001),
                              "qwen4": ("4B", 8002), "qwen9": ("9B", 8003)}[entry["model"]]
                self.assertEqual(config.model.client_kwargs(), {
                    "model": f"Qwen3.5-{size}", "timeout_seconds": 180.0, "think": False,
                    "max_tokens": 4096, "temperature": 0.0, "top_p": 1.0, "seed": 20260925,
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "output_format": "answer_only",
                })
            graph_hashes.add(entry["input_sha256"]["graphs.jsonl"])
            label_hashes.setdefault(entry["task"], set()).add(entry["input_sha256"]["labels.jsonl"])
            request_hashes.setdefault((entry["task"], entry["arm"]), set()).add(entry["input_sha256"]["requests.jsonl"])
        self.assertEqual(len(graph_hashes), 1)
        self.assertTrue(all(len(h) == 1 for h in label_hashes.values()))
        self.assertTrue(all(len(h) == 1 for h in request_hashes.values()))
        self.assertEqual(len({next(iter(h)) for (task, _), h in request_hashes.items() if task == "adjacency"}), 4)
        first = manifest["entries"][0]
        graphs = ablation.read_rows(f.root / first["inputs"] / "graphs.jsonl")
        self.assertEqual(len(graphs), 8)
        blocked = {name: {0} for name in ablation.DATASETS}
        for graph in graphs:
            nodes = set(graph["original_ids_by_anonymous_id"])
            self.assertFalse(nodes & blocked[graph["dataset"]])
            blocked[graph["dataset"]].update(nodes)
        self.assertFalse((f.root / "runs").exists())
        self.assertIn("src/benchmark/prepare.py", manifest["code_sha256"])
        self.assertIn("src/clients/typesafe.py", manifest["code_sha256"])

    def test_new_root_reservation_never_overwrites_even_incomplete_plan(self):
        f = self.fixture
        f.root.mkdir()
        sentinel = f.root / "sentinel"
        sentinel.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            f.plan()
        self.assertEqual(sentinel.read_text(), "unchanged")
        self.assertFalse(f.calls)

    def test_preflight_regenerates_every_config_with_only_four_raw_loads(self):
        f = self.fixture
        f.plan()
        f.calls.clear()
        with patch.object(ablation, "prepare", wraps=ablation.prepare) as prepare, \
                patch.object(ablation, "create_client", side_effect=AssertionError("offline")), \
                patch("httpx.AsyncClient", side_effect=AssertionError("offline")):
            flight = ablation.offline_preflight(f.root, ["qwen9"], loader=f.loader)
        self.assertEqual(prepare.call_count, 56)
        self.assertEqual(len(flight.pending), 8)
        self.assertEqual(f.calls, Counter({name: 1 for name in ablation.DATASETS}))

    def test_historical_five_model_roster_is_preserved(self):
        f = self.fixture
        old = f.plan(models=ablation.LEGACY_MODELS, output_format=None)
        old.pop("model_order")
        old.pop("local_output_format")
        (f.root / "manifest.json").write_text(json.dumps(old))
        (f.root / "manifest.sha256").write_text(ablation.digest(f.root / "manifest.json"))
        _, loaded = ablation.load_plan(f.root, check_code=False)
        self.assertEqual(len(loaded["entries"]), 40)
        self.assertEqual(loaded["budget"]["max_total"], 1600)
        f.root = f.directory / "seven-models"
        new = f.plan()
        for before, after in zip(old["entries"], new["entries"]):
            self.assertEqual(before["id"], after["id"])
            self.assertEqual(before["input_sha256"], after["input_sha256"])

    def test_reproducible_inputs_across_new_roots(self):
        f = self.fixture
        first = f.plan()
        f.root = f.directory / "another-plan"
        second = f.plan()
        self.assertEqual([e["input_sha256"] for e in first["entries"]],
                         [e["input_sha256"] for e in second["entries"]])
        self.assertEqual(first["lane_order"], second["lane_order"])

    def test_code_prior_shared_input_and_last_config_drift_refused(self):
        f = self.fixture
        manifest = f.plan()
        original = ablation.code_hashes()
        with patch.object(ablation, "code_hashes", return_value={**original, "added.py": "0" * 64}):
            with self.assertRaisesRegex(ablation.Refusal, "Source code drift"):
                ablation.offline_preflight(f.root, loader=f.loader)
        paths = [f.prior, f.root / manifest["entries"][-1]["config"],
                 f.root / manifest["entries"][-1]["inputs"] / "requests.jsonl"]
        for path in paths:
            with self.subTest(path=path):
                content = path.read_bytes()
                try:
                    path.write_bytes(content + b"\n")
                    with self.assertRaises(ablation.Refusal):
                        ablation.offline_preflight(f.root, loader=f.loader)
                finally:
                    path.write_bytes(content)

    def test_effective_protocol_cannot_be_changed_by_resealing_manifest(self):
        f = self.fixture
        manifest = f.plan()
        entry = manifest["entries"][-1]
        path = f.root / entry["config"]
        config = load_config(path)
        altered = replace(config, run=replace(config.run, max_calls=999))
        path.write_text(yaml.safe_dump(altered.snapshot()))
        entry.update(effective_config=altered.snapshot(), config_sha256=altered.sha256,
                     yaml_sha256=ablation.digest(path), max_calls=999)
        (f.root / "manifest.json").write_text(json.dumps(manifest))
        (f.root / "manifest.sha256").write_text(ablation.digest(f.root / "manifest.json"))
        with self.assertRaisesRegex(ablation.Refusal, "configuration changed"):
            ablation.offline_preflight(f.root, loader=f.loader)

    def test_regenerated_raw_provenance_mismatch_refused(self):
        f = self.fixture
        f.plan()
        def changed(name, **kwargs):
            return replace(f.loader(name, **kwargs), raw_sha256="0" * 64)
        with self.assertRaisesRegex(ablation.Refusal, "Regenerated"):
            ablation.offline_preflight(f.root, loader=changed)

    def test_node_overlap_is_checked_independently_of_prepare(self):
        f = self.fixture
        manifest = f.plan()
        config = load_config(f.root / manifest["entries"][0]["config"])
        prepared = ablation.prepare(config, f.loader)
        prepared.graphs[0]["original_ids_by_anonymous_id"][0] = 0
        with self.assertRaisesRegex(ablation.Refusal, "overlap"):
            ablation.validate_prepared(config, prepared)

    def test_incomplete_graph_pool_fails_without_resampling_protocol_changes(self):
        f = self.fixture
        f.graph = nx.freeze(nx.path_graph(40))
        with self.assertRaisesRegex(ablation.Refusal, "eight"):
            f.plan()
        self.assertFalse((f.root / "manifest.sha256").exists())
        with self.assertRaises(FileExistsError):
            f.plan()


class CredentialAndHTTPTests(unittest.TestCase):
    def test_missing_credentials_before_plan_or_network(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(ablation, "offline_preflight", side_effect=AssertionError("too late")), \
                patch("httpx.AsyncClient", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(ablation.Refusal, "missing"):
                asyncio.run(ablation.run_live(Path("/does-not-exist")))
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ablation.credential_preflight(["qwen4"]))

    def test_env_only_or_secure_prompt_no_mutation_and_fail_closed(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-secret"}, clear=True), \
                patch.object(getpass, "getpass", side_effect=AssertionError("must not prompt")):
            self.assertIsNone(ablation.credential_preflight(["jev"], True))
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys.stdin, "isatty", return_value=False), \
                patch.object(sys.stderr, "isatty", return_value=True), \
                patch.object(getpass, "getpass") as prompt:
            with self.assertRaisesRegex(ablation.Refusal, "TTY"):
                ablation.credential_preflight(["jev"], True)
            prompt.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(sys.stderr, "isatty", return_value=True), \
                patch.object(getpass, "getpass", return_value="test-secret"):
            self.assertEqual(ablation.credential_preflight(["jev"], True), "test-secret")
            self.assertNotIn("TYPESAFE_API_KEY", os.environ)
        def warn_fallback(*args):
            warnings.warn("no controlling tty", getpass.GetPassWarning)
            raise AssertionError("must never read echoed input")
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(sys.stderr, "isatty", return_value=True), \
                patch.object(getpass, "getpass", side_effect=warn_fallback):
            with self.assertRaisesRegex(ablation.Refusal, "controlling-terminal"):
                ablation.credential_preflight(["jev"], True)

    def test_get_only_exact_identity_idle_all_engines(self):
        configs = [next(c for m, a, t, r, c in ablation.configurations(
            Path("/unused"), Path("/unused-data"), Path("/prior"), "0" * 64) if m == model)
            for model in ("qwen4", "qwen9")]
        seen = []
        def handler(request):
            seen.append((request.method, request.url.port, request.url.path))
            self.assertEqual(request.method, "GET")
            self.assertNotIn("authorization", request.headers)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "Qwen3.5-4B" if request.url.port == 8002 else "Qwen3.5-9B"}]})
            return httpx.Response(200, text='vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting 0\n'
                                  'vllm:num_requests_waiting_by_reason{reason="capacity"} 0\n')
        asyncio.run(ablation.vllm_preflight(configs, transport=httpx.MockTransport(handler)))
        self.assertEqual(seen, [("GET", 8002, "/v1/models"), ("GET", 8002, "/metrics"),
                                ("GET", 8003, "/v1/models"), ("GET", 8003, "/metrics")])

    def test_busy_wrong_missing_malformed_and_http_errors_fail_closed(self):
        config = next(c for m, a, t, r, c in ablation.configurations(
            Path("/unused"), Path("/unused-data"), Path("/prior"), "0" * 64) if m == "qwen4")
        cases = [
            ("wrong-model", "", 200),
            ("Qwen3.5-4B", "vllm:num_requests_running 1\nvllm:num_requests_waiting 0", 200),
            ("Qwen3.5-4B", "vllm:num_requests_running 0\nvllm:num_requests_waiting 1", 200),
            ("Qwen3.5-4B", "vllm:num_requests_running 0", 200),
            ("Qwen3.5-4B", "vllm:num_requests_running NaN\nvllm:num_requests_waiting 0", 200),
            ("Qwen3.5-4B", "vllm:num_requests_running bad line extra extra\nvllm:num_requests_waiting 0", 200),
            ("Qwen3.5-4B", "", 403),
        ]
        for identity, metrics, status in cases:
            def handler(request):
                if request.url.path == "/v1/models":
                    return httpx.Response(200, json={"data": [{"id": identity}]})
                return httpx.Response(status, text=metrics)
            with self.subTest(identity=identity, metrics=metrics, status=status):
                with self.assertRaises(ablation.Refusal):
                    asyncio.run(ablation.vllm_preflight([config], transport=httpx.MockTransport(handler)))


class ExecutionTests(unittest.TestCase):
    def test_single_prompt_parallel_seven_model_report(self):
        self.fixture.root = self.fixture.directory / "single"
        self.manifest = self.fixture.plan(single_prompt=True)
        result, clients, _, _ = self.mock_run(ablation.MODELS, parallel_models=True)
        self.assertEqual(sum(len(c.requests) for c in clients), 560)
        self.assertEqual(len(result["runs"]), 14)
        self.assertEqual(result["versus_baseline"], [])
        self.assertEqual(len(result["cross_model"]), 12)
        self.assertEqual(result["local_output_format"], "answer_only")
        self.assertEqual(ablation.report(self.fixture.root), result)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Fixture(self.tmp.name)
        self.manifest = self.fixture.plan()

    def mock_run(self, models=("jev",), **kwargs):
        clients = []
        def factory(provider, **settings):
            client = MockDecisionClient(settings["model"])
            clients.append(client)
            return client
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                patch.object(ablation, "vllm_preflight", new_callable=AsyncMock) as http, \
                patch.object(ablation, "create_client", side_effect=factory) as create:
            result = asyncio.run(ablation.run_live(self.fixture.root, models, loader=self.fixture.loader, **kwargs))
        return result, clients, http, create

    def test_all_selected_configs_checked_before_any_client_or_get(self):
        f = self.fixture
        late = next(e for e in reversed(self.manifest["entries"]) if e["model"] == "qwen9")
        (f.root / late["config"]).write_text("corrupt")
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                patch.object(ablation, "create_client") as create, \
                patch.object(ablation, "vllm_preflight", new_callable=AsyncMock) as get:
            with self.assertRaises(ablation.Refusal):
                asyncio.run(ablation.run_live(f.root, loader=f.loader))
            create.assert_not_called()
            get.assert_not_called()
        self.assertFalse((f.root / "runs").exists())

    def test_partial_or_failed_output_is_never_resumed(self):
        f = self.fixture
        entry = next(e for e in self.manifest["entries"] if e["model"] == "qwen9")
        output = f.root / "runs" / entry["id"]
        output.mkdir(parents=True)
        for skip in (False, True):
            with self.subTest(skip=skip), self.assertRaises((ablation.Refusal, FileNotFoundError)):
                ablation.offline_preflight(f.root, ["qwen9"], skip_completed=skip, loader=f.loader)
        self.assertEqual(list(output.iterdir()), [])
        with self.assertRaises((ablation.Refusal, FileNotFoundError)):
            ablation.report(f.root, ["qwen9"])

    def test_serial_order_complete_report_read_only_and_verified_skip(self):
        f = self.fixture
        result, clients, http, create = self.mock_run()
        self.assertEqual(len(clients), 8)
        self.assertEqual(sum(len(c.requests) for c in clients), 320)
        expected = [e for e in self.manifest["entries"] if e["model"] == "jev"]
        for client, entry in zip(clients, expected):
            ids = [r["request_id"] for r in ablation.read_rows(f.root / entry["inputs"] / "requests.jsonl")]
            self.assertEqual(client.requests, ids)
            self.assertTrue(client.closed)
            self.assertTrue(ablation.read_json(f.root / "runs" / entry["id"] / "run.json")["injected_client"])
        self.assertEqual(len(result["versus_baseline"]), 6)
        self.assertEqual(result["cross_model"], [])
        self.assertTrue(all(p["counts"]["all"]["correctness_changed"] == 0 for p in result["versus_baseline"]))
        before = {p: (ablation.digest(p), p.stat().st_mtime_ns) for p in f.root.rglob("*") if p.is_file()}
        self.assertEqual(ablation.report(f.root, ["jev"]), result)
        self.assertEqual(before, {p: (ablation.digest(p), p.stat().st_mtime_ns) for p in before})
        with self.assertRaises(ablation.Refusal):
            ablation.offline_preflight(f.root, ["jev"], loader=f.loader)
        skipped, more_clients, _, _ = self.mock_run(skip_completed=True)
        self.assertFalse(more_clients)
        self.assertEqual(skipped, result)
        # One altered count invalidates skip, even though status remains completed.
        path = f.root / "runs" / expected[-1]["id"] / "run.json"
        run = ablation.read_json(path)
        run["completed_calls"] -= 1
        path.write_text(json.dumps(run))
        with self.assertRaisesRegex(ablation.Refusal, "count mismatch"):
            ablation.offline_preflight(f.root, ["jev"], skip_completed=True, loader=f.loader)

    def test_completed_skip_verifies_config_inputs_and_answers(self):
        f = self.fixture
        self.mock_run()
        entry = self.manifest["entries"][0]
        directory = f.root / "runs" / entry["id"]
        for name in ("config.json", "requests.jsonl", "labels.jsonl"):
            path = directory / name
            content = path.read_bytes()
            try:
                path.write_bytes(b"{}\n")
                with self.assertRaises(ablation.Refusal):
                    ablation.completed_rows(f.root, entry)
            finally:
                path.write_bytes(content)
        path = directory / "attempts.jsonl"
        rows = ablation.read_rows(path)
        rows[0]["answer"] = "wrong"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ablation.Refusal, "answers"):
            ablation.completed_rows(f.root, entry)

    def test_first_401_or_403_aborts_runner_and_entire_lane(self):
        f = self.fixture
        for status in (401, 403):
            if status == 403:
                f.root = f.directory / "second-plan"
                self.manifest = f.plan()
            client = MockDecisionClient(auth_status=status)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                    patch.object(ablation, "vllm_preflight", new_callable=AsyncMock), \
                    patch.object(ablation, "create_client", return_value=client) as create:
                with self.assertRaises(ablation.AuthenticationAbort):
                    asyncio.run(ablation.run_live(f.root, ["jev"], loader=f.loader))
            self.assertEqual(create.call_count, 1)
            self.assertEqual(len(client.requests), 1)
            output = next((f.root / "runs").iterdir())
            run = ablation.read_json(output / "run.json")
            self.assertEqual((run["status"], run["completed_calls"]), ("aborted", 1))
            rows = ablation.read_rows(output / "attempts.jsonl")
            self.assertEqual(rows[0]["error_type"], "AuthenticationAbort")
            with self.assertRaises(ablation.Refusal):
                ablation.offline_preflight(f.root, ["jev"], skip_completed=True, loader=f.loader)

    def test_parallel_lanes_never_overlap_same_model_and_report_cross_model(self):
        f = self.fixture
        active, peak, order = Counter(), Counter(), []
        real_run = ablation.run_experiment
        async def tracked(config, **kwargs):
            model = config.model.model
            active[model] += 1
            peak[model] = max(peak[model], active[model])
            order.append(config.run.output_dir.name)
            try:
                return await real_run(config, **kwargs)
            finally:
                active[model] -= 1
        with patch.object(ablation, "run_experiment", side_effect=tracked):
            result, clients, _, _ = self.mock_run(ablation.MODELS, parallel_models=True)
        self.assertEqual(sum(len(c.requests) for c in clients), 2240)
        self.assertEqual(peak, Counter({"jev-1.13.0": 1, "Qwen3.5-0.8B": 1,
                "Qwen3.5-2B": 1, "Qwen3.5-4B": 1, "Qwen3.5-9B": 1,
                "gpt-5.4": 1, "gpt-6-astra": 1}))
        self.assertEqual([c.model for c in clients[:2]], ["gpt-5.4", "gpt-6-astra"])
        self.assertTrue(all(not c.requests and c.closed for c in clients[:2]))
        for model in ablation.MODELS:
            self.assertEqual([r for r in order if r.startswith(model + ".")], self.manifest["lane_order"][model])
        self.assertEqual(len(result["cross_model"]), 48)
        self.assertEqual(len(result["versus_baseline"]), 42)
        local_report = ablation.report(f.root, ["qwen08", "qwen2", "qwen4", "qwen9"])
        self.assertEqual(local_report["local_output_format"], "answer_only")
        self.assertEqual(len(local_report["runs"]), 32)
        self.assertEqual(len(local_report["versus_baseline"]), 24)
        self.assertEqual(local_report["cross_model"], [])

    def test_fatal_auth_cancels_other_inflight_lanes(self):
        f = self.fixture
        clients = []

        async def exercise():
            qwen_started = asyncio.Event()
            never = asyncio.Event()

            class BlockingClient(MockDecisionClient):
                async def predict(self, request):
                    self.requests.append(request.request_id)
                    if self.model == "jev-1.13.0":
                        await qwen_started.wait()
                        raise ProviderHTTPError(401)
                    qwen_started.set()
                    await never.wait()
                    raise AssertionError("A cancelled Qwen call must not return.")

            def factory(provider, **settings):
                client = BlockingClient(settings["model"])
                clients.append(client)
                return client

            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                    patch.object(ablation, "vllm_preflight", new_callable=AsyncMock), \
                    patch.object(ablation, "create_client", side_effect=factory):
                with self.assertRaises(ablation.AuthenticationAbort):
                    await asyncio.wait_for(ablation.run_live(
                        f.root, parallel_models=True, loader=f.loader), timeout=30)

        asyncio.run(exercise())
        self.assertEqual(len(clients), 9)  # Two discovery probes, then seven live lane clients.
        self.assertTrue(all(len(c.requests) <= 1 and c.closed for c in clients))
        self.assertEqual(sum(len(c.requests) for c in clients if c.model == "jev-1.13.0"), 1)
        for output in (f.root / "runs").iterdir():
            self.assertEqual(ablation.read_json(output / "run.json")["status"], "aborted")

    def test_ordinary_http_errors_are_counted_not_retried_or_dropped(self):
        f = self.fixture
        clients = []

        def factory(provider, **settings):
            client = MockDecisionClient(settings["model"], auth_status=500)
            clients.append(client)
            return client

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                patch.object(ablation, "vllm_preflight", new_callable=AsyncMock), \
                patch.object(ablation, "create_client", side_effect=factory):
            result = asyncio.run(ablation.run_live(f.root, ["jev"], loader=f.loader))
        self.assertEqual(sum(len(c.requests) for c in clients), 320)
        self.assertTrue(all(run["overall"]["failure_rate"] == 1 for run in result["runs"]))
        self.assertEqual(sum(run["overall"]["failures"] for run in result["runs"]), 320)

    def test_second_endpoint_preflight_failure_prevents_all_generation(self):
        f = self.fixture
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                patch.object(ablation, "vllm_preflight", new_callable=AsyncMock,
                             side_effect=ablation.Refusal("busy")) as get, \
                patch.object(ablation, "create_client") as create:
            with self.assertRaisesRegex(ablation.Refusal, "busy"):
                asyncio.run(ablation.run_live(f.root, loader=f.loader))
        self.assertEqual({c.model.model for c in get.call_args.args[0]},
                         {"jev-1.13.0", "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B",
                          "gpt-5.4", "gpt-6-astra"})
        create.assert_not_called()

    def test_unavailable_second_copilot_model_prevents_all_generation(self):
        clients = []

        class DiscoveryClient(MockDecisionClient):
            async def initialize(self):
                if self.model == "gpt-6-astra":
                    raise UnsupportedRequestError("unavailable")

        def factory(provider, **settings):
            self.assertEqual(provider, "github_copilot")
            client = DiscoveryClient(settings["model"])
            clients.append(client)
            return client

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                patch.object(ablation, "vllm_preflight", new_callable=AsyncMock), \
                patch.object(ablation, "create_client", side_effect=factory):
            with self.assertRaisesRegex(ablation.Refusal, "Copilot preflight failed"):
                asyncio.run(ablation.run_live(self.fixture.root, parallel_models=True,
                                             loader=self.fixture.loader))
        self.assertEqual([c.model for c in clients], ["gpt-5.4", "gpt-6-astra"])
        self.assertTrue(all(not c.requests and c.closed for c in clients))
        self.assertFalse((self.fixture.root / "runs").exists())

    def test_changed_runner_export_stops_before_first_prediction(self):
        f = self.fixture
        original = ablation.run_experiment
        client = MockDecisionClient()
        async def drifted(config, **kwargs):
            changed = replace(config, presentation=ablation.ARMS[
                next(a for a in ablation.ARMS if ablation.ARMS[a] != config.presentation)])
            return await original(changed, **kwargs)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-not-real"}, clear=True), \
                patch.object(ablation, "vllm_preflight", new_callable=AsyncMock), \
                patch.object(ablation, "create_client", return_value=client), \
                patch.object(ablation, "run_experiment", side_effect=drifted):
            with self.assertRaisesRegex(ablation.Refusal, "Runner inputs"):
                asyncio.run(ablation.run_live(f.root, ["jev"], loader=f.loader))
        self.assertEqual(client.requests, [])

    def test_report_incomplete_selected_lane_is_not_presented_as_final(self):
        f = self.fixture
        self.mock_run(["jev"])
        with self.assertRaises(ablation.Refusal):
            ablation.report(f.root, ablation.MODELS)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            status = ablation.main(["report", "--root", str(f.root)])
        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")


class PairwiseTests(unittest.TestCase):
    def test_lane_lock_refuses_duplicate_gpu_lane_and_releases_after_error(self):
        with ablation.lane_locks(["qwen4"]):
            with self.assertRaisesRegex(ablation.Refusal, "already active"):
                with ablation.lane_locks(["qwen4"]):
                    self.fail("Duplicate lane acquired")
            with ablation.lane_locks(["qwen9"]):
                pass
        with ablation.lane_locks(["qwen4", "qwen9"]):
            pass

    def test_matched_correctness_yes_no_and_request_answer_guards(self):
        def row(index, answer, correct, selected):
            return dict(request_id=str(index), answer=answer, correct=correct, selected_option_id=selected)
        left = [row(0, "yes", True, "yes"), row(1, "no", False, "yes"), row(2, "yes", False, None)]
        right = [row(0, "yes", False, "no"), row(1, "no", True, "no"), row(2, "yes", False, None)]
        counts = ablation.matched_counts(left, right)
        self.assertEqual(counts["all"], dict(n=3, both_correct=0, left_only_correct=1,
                                           right_only_correct=1, neither_correct=1,
                                           correctness_changed=2, selection_changed=2))
        self.assertEqual(counts["yes"]["n"], 2)
        self.assertEqual(counts["no"]["right_only_correct"], 1)
        with self.assertRaises(ablation.Refusal):
            ablation.matched_counts(left, right[:-1])
        with self.assertRaises(ablation.Refusal):
            ablation.matched_counts(left, [dict(right[0], answer="no"), *right[1:]])


if __name__ == "__main__":
    unittest.main()