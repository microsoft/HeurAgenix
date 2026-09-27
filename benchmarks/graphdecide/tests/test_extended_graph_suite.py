"""Offline runner integration tests; fake clients never perform network calls."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import asyncio
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

import httpx
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from src.benchmark import extended_tasks as tasks
from src.benchmark.config import ModelConfig, load_config
from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, ClientTimeoutError, ClientTransportError,
    DecisionOption, DecisionRequest, DecisionResponse, InvalidResponseError,
    ProviderHTTPError, TokenUsage,
)
from src.clients.vllm_client import VLLMClient

REAL_CODE_HASHES = suite.code_hashes


def small_instances():
    graph = {"nodes": list(range(5)), "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
             "directed": False}
    return [
        tasks._record("maxcut_construct", 0, deepcopy(graph)),
        tasks._record("degree_exact", 0, {**deepcopy(graph), "vertex": 0}),
        tasks._record("cycle_detection", 0, deepcopy(graph)),
        tasks._record("pair_connectivity", 0, {**deepcopy(graph), "pair": [0, 4]}),
    ]


class FakeClient(BaseDecisionClient):
    def __init__(self, model, behavior=None, initialize_error=None, close_error=None):
        self.model, self.behavior = model, behavior
        self.initialize_error, self.close_error = initialize_error, close_error
        self.requests, self.initialized, self.closed = [], False, False

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def initialize(self):
        self.initialized = True
        if self.initialize_error:
            raise self.initialize_error

    async def predict(self, request):
        self.requests.append(deepcopy(request))
        if self.behavior:
            result = self.behavior(request, len(self.requests))
            if isinstance(result, BaseException):
                raise result
            if result is not None:
                return result
        await asyncio.sleep(0)
        choice = request.options[-1].id
        return DecisionResponse(request.request_id, choice, self.model,
                                usage=TokenUsage(20, 1, None),
                                raw_output={"secret": "MUST_NOT_APPEAR"})

    async def aclose(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class ExtendedSuiteTests(unittest.TestCase):
    def setUp(self):
        self.workspace = REPO / "output" / ("suite-tests-" + uuid.uuid4().hex)
        self.workspace.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.workspace)
        self.root = self.workspace / "plan"
        self.instances = small_instances()
        self.addCleanup(patch.stopall)
        # Test-only filesystem isolation; production uses the shared namespace unchanged.
        locks = self.workspace / "locks"
        locks.mkdir()
        patch.object(suite.shared.tempfile, "gettempdir", return_value=str(locks)).start()
        patch.object(tasks, "build_instances", return_value=deepcopy(self.instances)).start()
        # Keep tests independent from simultaneous unrelated source edits.
        patch.object(suite, "code_hashes", return_value=REAL_CODE_HASHES()).start()
        self.clients, self.preflights = [], []

    def plan(self, models=("jev",), **kwargs):
        return suite.plan(self.root, models=models, **kwargs)

    async def preflight(self, configs):
        self.preflights.extend(c.model.model for c in configs)

    def factory(self, provider, **kwargs):
        client = FakeClient(kwargs["model"])
        self.clients.append(client)
        return client

    def run_lane(self, models=None, factory=None, preflight=None):
        return asyncio.run(suite.run_live(
            self.root, models, client_factory=factory or self.factory,
            preflight=preflight or self.preflight))

    def attempts(self, name="jev"):
        return suite.read_rows(self.root / "runs" / name / "attempts.jsonl")

    def scores(self, name="jev"):
        return suite.read_rows(self.root / "runs" / name / "scores.jsonl")

    def test_offline_plan_validates_and_freezes(self):
        with patch.object(tasks, "validate_instances", wraps=tasks.validate_instances) as validate:
            result = self.plan(("jev", "qwen9", "qwen9think"))
        validate.assert_called_once()
        self.assertEqual(result["budgets"]["max_calls_per_model"], 7)
        root, manifest, instances, configs = suite.load_plan(self.root)
        self.assertEqual(instances, self.instances)
        self.assertFalse(configs["qwen9"].think)
        self.assertTrue(configs["qwen9think"].think)
        self.assertEqual(configs["qwen9think"].max_tokens, 8192)
        self.assertEqual(configs["qwen9"].base_url, configs["qwen9think"].base_url)
        self.assertIn("private", instances[0])
        self.assertEqual(manifest["validation"]["instances"], 4)
        with self.assertRaises(suite.Refusal):
            self.plan()

    def test_plan_snapshots_exact_hashed_source_files_as_read_only(self):
        self.plan()
        manifest = suite.read_json(self.root / "manifest.json")
        self.assertIn("src/__init__.py", manifest["code_sha256"])
        self.assertEqual(manifest["source_snapshot"], "frozen-source")
        snapshot = self.root / manifest["source_snapshot"]
        self.assertEqual({p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()},
                         set(manifest["code_sha256"]))
        for name, expected in manifest["code_sha256"].items():
            with self.subTest(file=name):
                path = snapshot / name
                self.assertEqual(suite.digest(path), expected)
                self.assertEqual(path.read_bytes(), (REPO / name).read_bytes())
                self.assertEqual(path.stat().st_mode & 0o222, 0)
        self.assertFalse(any(p.suffix in (".json", ".jsonl", ".env") for p in snapshot.rglob("*")))

    def test_snapshot_subprocess_imports_own_src_and_reports_from_workspace(self):
        self.plan()
        snapshot = self.root / "frozen-source"
        script = snapshot / "scripts" / "extended_graph_suite.py"
        probe = (
            "import json,runpy,sys;"
            "runpy.run_path(sys.argv[1],run_name='snapshot_import_check');"
            "print(json.dumps({name:sys.modules[name].__file__ "
            "for name in ('src','src.benchmark.config','src.benchmark.extended_tasks',"
            "'scripts.paired_graph_ablation')}))")
        imported = subprocess.run([sys.executable, "-B", "-c", probe, str(script)],
                                  cwd=REPO, text=True, capture_output=True, timeout=30, check=True)
        for name, path in json.loads(imported.stdout).items():
            with self.subTest(module=name):
                self.assertTrue(Path(path).resolve().is_relative_to(snapshot))
        before = {p: (p.stat().st_mtime_ns, suite.digest(p))
                  for p in self.root.rglob("*") if p.is_file()}
        completed = subprocess.run(
            [sys.executable, str(script), "report", "--root", str(self.root)],
            cwd=REPO, text=True, capture_output=True, timeout=30, check=True)
        report = json.loads(completed.stdout)
        self.assertEqual(report["models"]["jev"]["status"], "not_started")
        self.assertEqual(report["budgets"]["instances_per_model"], len(self.instances))
        after = {p: (p.stat().st_mtime_ns, suite.digest(p))
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_snapshot_tampering_or_missing_package_marker_is_refused(self):
        self.plan()
        path = self.root / "frozen-source" / "src" / "__init__.py"
        original = path.read_bytes()
        path.chmod(0o644)
        with self.assertRaises(suite.Refusal):
            suite.report(self.root)
        path.write_bytes(original + b"\n# unexpected change\n")
        path.chmod(0o444)
        with self.assertRaises(suite.Refusal):
            suite.report(self.root)
        path.unlink()
        with self.assertRaises(suite.Refusal):
            suite.report(self.root)

    def test_answer_only_protocol_frozen_for_all_non_typesafe_models(self):
        self.plan(suite.MODELS)
        _, manifest, _, configs = suite.load_plan(self.root)
        for name, config in configs.items():
            with self.subTest(model=name):
                expected = None if config.provider == "typesafe" else "answer_only"
                self.assertEqual(config.output_format, expected)
                self.assertEqual(manifest["models"][name]["output_format"], expected)
                if config.provider != "typesafe":
                    self.assertEqual(config.client_kwargs()["output_format"], "answer_only")

    def test_no_think_output_budget_frozen_without_changing_other_lanes(self):
        result = self.plan(suite.MODELS, no_think_max_tokens=64)
        _, manifest, _, configs = suite.load_plan(self.root)
        self.assertEqual(result["no_think_max_tokens"], 64)
        self.assertEqual(manifest["no_think_max_tokens"], 64)
        for name in ("qwen08", "qwen2", "qwen4", "qwen9"):
            self.assertFalse(configs[name].think)
            self.assertEqual(configs[name].max_tokens, 64)
            self.assertEqual(manifest["models"][name]["max_tokens"], 64)
        self.assertEqual(configs["qwen9think"].max_tokens, 8192)
        self.assertTrue(configs["qwen9think"].think)
        self.assertEqual(configs["qwen9recommended"].max_tokens, 64)
        self.assertFalse(configs["qwen9recommended"].think)
        self.assertEqual(configs["qwen9thinkrecommended"].max_tokens, 8192)
        self.assertTrue(configs["qwen9thinkrecommended"].think)
        self.assertEqual(configs["gpt54"].max_tokens, 4096)
        self.assertEqual(configs["gpt6astra"].max_tokens, 4096)
        self.assertIsNone(configs["jev"].max_tokens)
        captured = []
        def factory(provider, **kwargs):
            captured.append(kwargs["max_tokens"])
            return FakeClient(kwargs["model"])
        reported = self.run_lane(("qwen9",), factory=factory)
        self.assertEqual(captured, [64])
        self.assertEqual(reported["no_think_max_tokens"], 64)

    def test_legacy_manifest_defaults_to_4096_output_tokens(self):
        self.plan(("qwen9",))
        manifest_path = self.root / "manifest.json"
        manifest = suite.read_json(manifest_path)
        del manifest["no_think_max_tokens"]
        manifest_path.write_text(suite.canonical(manifest) + "\n")
        (self.root / "manifest.sha256").write_text(suite.canonical(suite.digest(manifest_path)) + "\n")
        _, _, _, configs = suite.load_plan(self.root)
        self.assertEqual(configs["qwen9"].max_tokens, 4096)
        self.assertEqual(suite.report(self.root)["no_think_max_tokens"], 4096)

    def test_invalid_no_think_output_budgets_refused(self):
        for value in (0, -1, True, False, "64", 64.0, None):
            with self.subTest(value=value):
                with self.assertRaises(suite.Refusal):
                    self.plan(no_think_max_tokens=value)
                with self.assertRaises(suite.Refusal):
                    suite.model_configs(no_think_max_tokens=value)
        self.assertFalse(self.root.exists())

    def test_no_think_budget_cli_and_frozen_config_consistency(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main([
                "plan", "--root", str(self.root), "--models", "qwen9",
                "--no-think-max-tokens", "64"]), 0)
        path = self.root / "manifest.json"
        manifest = suite.read_json(path)
        self.assertEqual(manifest["no_think_max_tokens"], 64)
        manifest["no_think_max_tokens"] = 65
        path.write_text(suite.canonical(manifest) + "\n")
        (self.root / "manifest.sha256").write_text(suite.canonical(suite.digest(path)) + "\n")
        with self.assertRaises(suite.Refusal):
            suite.load_plan(self.root)

    def test_production_lock_wrapper_preserves_shared_namespace(self):
        from contextlib import contextmanager
        seen = []
        @contextmanager
        def lock_helper(models):
            seen.append(models)
            yield
        before = suite.shared.tempfile.tempdir
        with patch.object(suite.shared, "lane_locks", side_effect=lock_helper):
            with suite.lane_locks(("jev", *sorted(suite.QWEN9_LANES))):
                self.assertEqual(suite.shared.tempfile.tempdir, before)
        self.assertEqual(seen, [["jev", "qwen9"]])

    def test_opt_in_recommended_qwen_sampling_settings_are_frozen(self):
        self.plan(suite.MODELS, no_think_max_tokens=64)
        _, manifest, _, configs = suite.load_plan(self.root)
        for name, expected in (
            ("qwen9recommended", {"think": False, "temperature": 1.0, "top_p": 1.0,
                                  "top_k": 40, "presence_penalty": 2.0,
                                  "max_tokens": 64, "timeout_seconds": 180.0}),
            ("qwen9thinkrecommended", {"think": True, "temperature": 1.0, "top_p": 0.95,
                                       "top_k": 20, "presence_penalty": 1.5,
                                       "max_tokens": 8192, "timeout_seconds": 360.0}),
        ):
            with self.subTest(model=name):
                for field, value in expected.items():
                    self.assertEqual(getattr(configs[name], field), value)
                    self.assertEqual(manifest["models"][name][field], value)
                    self.assertEqual(configs[name].client_kwargs()[field], value)
                self.assertEqual(configs[name].model, configs["qwen9"].model)
                self.assertEqual(configs[name].base_url, configs["qwen9"].base_url)
                self.assertEqual(configs[name].output_format, "answer_only")
        self.assertFalse(set(suite.DEFAULT_MODELS) & {
            "qwen9think", "qwen9recommended", "qwen9thinkrecommended", "qwen9constrained"})

    def test_constrained_reference_alias_is_opt_in_and_frozen(self):
        self.plan(("qwen9", "qwen9constrained"), no_think_max_tokens=64)
        _, manifest, _, configs = suite.load_plan(self.root)
        self.assertEqual(configs["qwen9constrained"],
                         replace(configs["qwen9"], constrain_choices=True))
        self.assertEqual(configs["qwen9constrained"].max_tokens, 64)
        self.assertTrue(manifest["models"]["qwen9constrained"]["constrain_choices"])
        self.assertNotIn("constrain_choices", configs["qwen9"].client_kwargs())
        captured = []
        def factory(provider, **kwargs):
            captured.append(kwargs)
            return FakeClient(kwargs["model"])
        self.run_lane(("qwen9constrained",), factory=factory)
        self.assertTrue(captured[0]["constrain_choices"])
        self.assertEqual(captured[0]["temperature"], 0.0)
        self.assertFalse(captured[0]["think"])
        self.assertEqual(captured[0]["output_format"], "answer_only")

    def test_constrained_config_validation_and_yaml_round_trip(self):
        baseline = load_config(REPO / "configs" / "adjacency.yaml")
        self.assertNotIn("constrain_choices", baseline.snapshot()["model"])
        model = ModelConfig("vllm", "local", 30, think=False, output_format="answer_only")
        config = replace(baseline, model=model)
        self.assertNotIn("constrain_choices", config.snapshot()["model"])
        self.assertNotIn("constrain_choices", model.client_kwargs())
        path = self.workspace / "constrained.yaml"
        for value in (None, False, True):
            with self.subTest(value=value):
                updated = replace(config, model=replace(model, constrain_choices=value))
                path.write_text(yaml.safe_dump(updated.snapshot()))
                loaded = load_config(path)
                self.assertEqual(loaded, updated)
                self.assertEqual(loaded.sha256, updated.sha256)
                if value is not None:
                    self.assertIs(loaded.model.client_kwargs()["constrain_choices"], value)
                    self.assertNotEqual(loaded.sha256, config.sha256)
        invalid = [
            {"provider": provider, "constrain_choices": value}
            for provider in ("typesafe", "github_copilot") for value in (False, True)]
        invalid += [{"constrain_choices": value} for value in (1, 0, "true", [], {})]
        invalid += [{"constrain_choices": True, "think": value} for value in (True, None)]
        invalid += [{"constrain_choices": True, "output_format": value} for value in ("json", None)]
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(model, **changes)
                snapshot = config.snapshot()
                snapshot["model"].update(changes)
                path.write_text(yaml.safe_dump(snapshot))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_constrained_client_validates_modes_before_creating_transport(self):
        cases = [{"constrain_choices": value} for value in (None, 0, 1, "true")]
        cases += [{"constrain_choices": True, "think": value} for value in (True, None)]
        cases += [{"constrain_choices": True, "output_format": "json"}]
        for changes in cases:
            kwargs = {"think": False, "output_format": "answer_only", **changes}
            with self.subTest(changes=changes), patch("httpx.AsyncClient") as transport:
                with self.assertRaises(ValueError):
                    VLLMClient(**kwargs)
                transport.assert_not_called()

    def test_constrained_mock_transport_wire_uses_current_options_without_probabilities(self):
        payloads = []
        def handler(request):
            payload = json.loads(request.content)
            payloads.append(payload)
            selected = payload["structured_outputs"]["choice"][-1]
            return httpx.Response(200, json={"model": "local", "choices": [{
                "finish_reason": "stop", "message": {"role": "assistant", "content": selected}}]})
        async def run():
            client = VLLMClient(model="local", think=False, output_format="answer_only",
                                constrain_choices=True, max_tokens=64,
                                transport=httpx.MockTransport(handler))
            try:
                for options in (("1", "2", "3"), ("1", "2")):
                    request = DecisionRequest("choice", {"nodes": [1, 2, 3]}, "Choose a vertex.",
                                              tuple(DecisionOption(id=o) for o in options))
                    response = await client.predict(request)
                    self.assertEqual(response.selected_option_id, options[-1])
                    self.assertIsNone(response.probabilities)
                    self.assertIsNone(response.probability_kind)
                    self.assertIsNone(response.confidence)
            finally:
                await client.aclose()
        asyncio.run(run())
        self.assertEqual([p["structured_outputs"] for p in payloads],
                         [{"choice": ["1", "2", "3"]}, {"choice": ["1", "2"]}])
        self.assertTrue(all(p["chat_template_kwargs"] == {"enable_thinking": False}
                            and p["max_tokens"] == 64 for p in payloads))
        self.assertTrue(all("response_format" not in p for p in payloads))

    def test_constrained_output_still_rejects_invalid_or_truncated_answers(self):
        request = DecisionRequest("choice", {}, "Choose.", (
            DecisionOption(id="A"), DecisionOption(id="B")))
        async def run():
            for content, finish in (("A because it is optimal", "stop"), ("C", "stop"), ("A", "length")):
                def handler(unused):
                    return httpx.Response(200, json={"model": "local", "choices": [{
                        "finish_reason": finish, "message": {"role": "assistant", "content": content}}]})
                client = VLLMClient(model="local", think=False, output_format="answer_only",
                                    constrain_choices=True, transport=httpx.MockTransport(handler))
                try:
                    with self.subTest(content=content, finish=finish), \
                            self.assertRaises(InvalidResponseError) as error:
                        await client.predict(request)
                    self.assertEqual(error.exception.diagnostic_code,
                                     "finish_length" if finish == "length" else "invalid_choice")
                finally:
                    await client.aclose()
        asyncio.run(run())

    def test_default_and_disabled_client_wire_remain_unconstrained(self):
        payloads = []
        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"model": "local", "choices": [{
                "finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}]})
        async def run():
            request = DecisionRequest("choice", {}, "Choose.", (
                DecisionOption(id="A"), DecisionOption(id="B")))
            for kwargs in ({}, {"constrain_choices": False}):
                client = VLLMClient(model="local", output_format="answer_only",
                                    transport=httpx.MockTransport(handler), **kwargs)
                try:
                    await client.predict(request)
                finally:
                    await client.aclose()
        asyncio.run(run())
        self.assertEqual(payloads[0], payloads[1])
        self.assertNotIn("structured_outputs", payloads[0])

    def test_runtime_curriculum_builder_arguments(self):
        with patch.object(tasks, "build_curriculum", create=True,
                          return_value=deepcopy(self.instances)) as builder:
            self.plan(design="curriculum", seed=42, samples_per_size=3)
        builder.assert_called_once_with(seed=42, node_counts=tuple(range(5, 13)),
                                        samples_per_size=3)
        manifest = suite.read_json(self.root / "manifest.json")
        self.assertEqual(manifest["samples_per_size"], 3)
        self.assertEqual(manifest["node_counts"], list(range(5, 13)))

    def test_curriculum_size_subset_and_default_models(self):
        with patch.object(tasks, "build_curriculum", create=True,
                          return_value=deepcopy(self.instances)) as builder:
            suite.plan(self.root, design="curriculum", node_counts=(6,))
        builder.assert_called_once_with(seed=suite.SEED, node_counts=(6,), samples_per_size=2)
        manifest = suite.read_json(self.root / "manifest.json")
        self.assertEqual(manifest["node_counts"], [6])
        self.assertEqual(set(manifest["models"]), set(suite.shared.MODELS))
        self.assertNotIn("qwen9think", manifest["models"])

    def test_invalid_sizes_refused_before_plan_creation(self):
        for sizes in ((), (6, 6), (4,), (13,), ("6",)):
            with self.subTest(sizes=sizes), self.assertRaises(suite.Refusal):
                self.plan(design="curriculum", node_counts=sizes)
        with self.assertRaises(suite.Refusal):
            self.plan(design="pilot", node_counts=(6,))
        self.assertFalse(self.root.exists())

    def test_node_counts_cli_thinking_probe(self):
        with patch.object(tasks, "build_curriculum", create=True,
                          return_value=deepcopy(self.instances)) as builder, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main([
                "plan", "--root", str(self.root), "--design", "curriculum",
                "--node-counts", "6", "--models", "qwen9think"]), 0)
        builder.assert_called_once_with(seed=suite.SEED, node_counts=(6,), samples_per_size=2)
        self.assertEqual(set(suite.read_json(self.root / "manifest.json")["models"]),
                         {"qwen9think"})

    def test_iterative_requests_use_actual_decisions_only(self):
        self.plan()
        result = self.run_lane()
        self.assertEqual(result["models"]["jev"]["status"], "completed")
        self.assertEqual(len(self.clients[0].requests), 7)
        for index, request in enumerate(self.clients[0].requests[:4]):
            self.assertEqual(request.state["partial_partition"], ["A"] + ["B"] * index)
            self.assertNotIn("private", request.state)
            self.assertNotIn("baseline", request.state)
        serialized = "".join(p.read_text() for p in (self.root / "runs" / "jev").iterdir())
        self.assertNotIn("MUST_NOT_APPEAR", serialized)
        self.assertTrue(self.clients[0].closed)
        rows = self.scores()
        self.assertEqual(rows[0]["decisions"], ["B"] * 4)
        self.assertEqual(rows[0]["objective"], 1)
        self.assertEqual(rows[0]["absolute_gap"], 3)
        self.assertEqual(rows[0]["baseline_absolute_gap"], 0)
        self.assertEqual(rows[0]["node_count"], 5)
        self.assertEqual(rows[0]["edge_count"], 4)
        summary = result["models"]["jev"]
        self.assertNotIn("overall", summary)
        self.assertEqual(summary["by_task_and_size"]["maxcut_construct"]["5"]["instances"], 1)

    def test_failure_stops_instance_without_repair(self):
        self.plan()
        error = InvalidResponseError("SECRET", raw_output={"SECRET": "TOKEN"},
                                     diagnostic_code="finish_length", usage=TokenUsage(5, 8))
        def factory(provider, **kwargs):
            return FakeClient(kwargs["model"], lambda request, count: error if count == 2 else None)
        result = self.run_lane(factory=factory)
        rows = self.scores()
        self.assertEqual(rows[0]["decisions"], ["B"])
        self.assertFalse(rows[0]["feasible"])
        self.assertIsNone(rows[0]["objective"])
        self.assertEqual(rows[0]["query_count"], 2)
        self.assertTrue(all(r["feasible"] for r in rows[1:]))
        self.assertEqual(result["models"]["jev"]["status"], "completed_with_failures")
        attempts = self.attempts()
        self.assertEqual(attempts[1]["usage"]["output_tokens"], 8)
        self.assertNotIn("SECRET", json.dumps(result) + json.dumps(attempts))
        metrics = result["models"]["jev"]["by_task"]["maxcut_construct"]
        self.assertEqual(metrics["feasibility_rate"], 0)
        self.assertEqual(metrics["feasible_mean_denominator"], 0)
        self.assertIsNone(metrics["mean_absolute_gap_feasible"])
        self.assertEqual(metrics["baseline_denominator"], 1)

    def test_fatal_http_aborts_only_one_lane(self):
        self.plan(("jev", "qwen2"))
        def factory(provider, **kwargs):
            return FakeClient(kwargs["model"], lambda request, count:
                              ProviderHTTPError(429) if provider == "typesafe" else None)
        result = self.run_lane(factory=factory)
        self.assertEqual(result["models"]["jev"]["status"], "aborted")
        self.assertEqual(result["models"]["qwen2"]["status"], "completed")
        self.assertEqual(len(self.attempts()), 1)
        rows = self.scores()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(not r["feasible"] for r in rows))
        self.assertEqual(sum(r["query_count"] == 0 for r in rows), 3)

    def test_timeout_continues_next_instance(self):
        self.plan()
        def factory(provider, **kwargs):
            return FakeClient(kwargs["model"], lambda request, count:
                              ClientTimeoutError("TOKEN") if count == 1 else None)
        self.run_lane(factory=factory)
        self.assertEqual(len(self.attempts()), 4)
        self.assertEqual(self.attempts()[0]["diagnostic_code"], "timeout")
        self.assertFalse(self.attempts()[0]["fatal"])

    def test_transport_context_or_quota_errors_fail_closed(self):
        self.plan()
        def factory(provider, **kwargs):
            return FakeClient(kwargs["model"], lambda request, count:
                              ClientTransportError("secret quota/context/auth details"))
        self.run_lane(factory=factory)
        self.assertEqual(len(self.attempts()), 1)
        self.assertEqual(self.attempts()[0]["diagnostic_code"], "client_error")
        self.assertTrue(self.attempts()[0]["fatal"])

    def test_preflight_initialization_and_close_errors_are_sealed(self):
        self.plan(("jev", "qwen2", "qwen4"))
        async def preflight(configs):
            if configs[0].model.model == "Qwen3.5-2B":
                raise RuntimeError("SECRET")
        def factory(provider, **kwargs):
            return FakeClient(kwargs["model"],
                              initialize_error=ValueError("SECRET") if provider == "typesafe" else None,
                              close_error=ValueError("SECRET") if provider == "vllm" else None)
        result = self.run_lane(factory=factory, preflight=preflight)
        self.assertTrue(all(v["status"] == "aborted" for v in result["models"].values()))
        for name, code in (("jev", "initialization_failed"), ("qwen2", "preflight_failed"),
                           ("qwen4", "shutdown_failed")):
            status = suite.read_json(self.root / "runs" / name / "status.json")
            self.assertEqual(status["diagnostic_code"], code)
        self.assertNotIn("SECRET", json.dumps(result))

    def test_disjoint_invocations_and_duplicate_refusal_before_client(self):
        self.plan(("jev", "gpt54"))
        self.run_lane(("jev",))
        self.assertEqual(self.preflights, ["jev-1.13.0"])
        first_hash = suite.digest(self.root / "runs" / "jev" / "attempts.jsonl")
        self.run_lane(("gpt54",))
        self.assertEqual(first_hash, suite.digest(self.root / "runs" / "jev" / "attempts.jsonl"))
        count = len(self.clients)
        with self.assertRaises(suite.Refusal):
            self.run_lane(("jev", "gpt54"))
        self.assertEqual(len(self.clients), count)
        with self.assertRaises(suite.Refusal):
            self.run_lane(("qwen9",))

    def test_frozen_code_inputs_manifest_and_config_checked(self):
        self.plan()
        with patch.object(suite, "code_hashes", return_value={"changed.py": "xyz"}):
            with self.assertRaises(suite.Refusal):
                self.run_lane()
        instance_path = self.root / "instances.jsonl"
        instance_path.write_text(instance_path.read_text() + "\n")
        with self.assertRaises(suite.Refusal):
            self.run_lane()
        self.assertFalse(self.clients)
        instance_path.write_text("".join(suite.canonical(i) + "\n" for i in self.instances))
        manifest_path = self.root / "manifest.json"
        manifest = suite.read_json(manifest_path)
        manifest["models"]["jev"]["model"] = "unplanned"
        manifest_path.write_text(suite.canonical(manifest) + "\n")
        with self.assertRaises(suite.Refusal):
            self.run_lane()
        (self.root / "manifest.sha256").write_text(suite.canonical(suite.digest(manifest_path)) + "\n")
        with self.assertRaises(suite.Refusal):
            self.run_lane()

    def test_report_is_read_only_and_detects_tampering(self):
        self.plan()
        self.run_lane()
        before = {p: (p.stat().st_mtime_ns, suite.digest(p)) for p in self.root.rglob("*") if p.is_file()}
        suite.report(self.root)
        after = {p: (p.stat().st_mtime_ns, suite.digest(p)) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        path = self.root / "runs" / "jev" / "scores.jsonl"
        path.write_text(path.read_text().replace('"objective":1', '"objective":99'))
        with self.assertRaises(suite.Refusal):
            suite.report(self.root)
        status_path = self.root / "runs" / "jev" / "status.json"
        status = suite.read_json(status_path)
        status["artifact_sha256"]["scores.jsonl"] = suite.digest(path)
        status_path.write_text(suite.canonical(status))
        with self.assertRaises(suite.Refusal):
            suite.report(self.root)

    def test_replay_rejects_requests_skips_extra_calls_and_bad_decisions(self):
        self.plan()
        self.run_lane()
        attempts = self.attempts()
        config = suite.model_configs(("jev",))["jev"]
        for mutate in (
            lambda a: a[1]["request"]["state"].update(partial_partition=["A", "A"]),
            lambda a: a[0].update(selected_option_id="not-an-option"),
            lambda a: a.pop(1),
            lambda a: a.append(deepcopy(a[-1])),
            lambda a: a[0].update(raw_output="SECRET"),
        ):
            with self.subTest(mutate=mutate):
                changed = deepcopy(attempts)
                mutate(changed)
                with self.assertRaises(suite.Refusal):
                    suite.replay(self.instances, changed, config)

    def test_invalid_response_identity_selection_usage_do_not_leak(self):
        self.plan(("jev", "qwen2", "qwen4"))
        def factory(provider, **kwargs):
            def behavior(request, count):
                if kwargs["model"] == "jev-1.13.0":
                    return DecisionResponse(request.request_id, "SECRET", kwargs["model"])
                if kwargs["model"] == "Qwen3.5-2B":
                    return DecisionResponse(request.request_id, request.options[0].id, "SECRET")
                return DecisionResponse(request.request_id, request.options[0].id, kwargs["model"],
                                        usage=TokenUsage(-1, "SECRET", True))
            return FakeClient(kwargs["model"], behavior)
        result = self.run_lane(factory=factory)
        self.assertNotIn("SECRET", json.dumps(result))
        for name in ("jev", "qwen2", "qwen4"):
            attempts = self.attempts(name)
            self.assertEqual(len(attempts), 4)
            self.assertNotIn("SECRET", json.dumps(attempts))
            self.assertTrue(all(a["status"] == "failure" for a in attempts))
        self.assertTrue(all(a["diagnostic_code"] == "invalid_model"
                            for a in self.attempts("qwen2")))

    def test_alias_locks_conflict_and_alias_lanes_serialize(self):
        with suite.lane_locks(("qwen9",)):
            for alias in suite.QWEN9_LANES - {"qwen9"}:
                with self.subTest(alias=alias), self.assertRaises(suite.Refusal):
                    with suite.lane_locks((alias,)):
                        self.fail("alias lock failed")
        self.plan(tuple(sorted(suite.QWEN9_LANES)))
        active = set()
        class CheckingClient(FakeClient):
            async def predict(self, request):
                self.assertion()
                active.add(id(self))
                try:
                    return await super().predict(request)
                finally:
                    active.remove(id(self))
            def assertion(self):
                if active:
                    raise RuntimeError("overlapping alias generations")
        def factory(provider, **kwargs):
            return CheckingClient(kwargs["model"])
        result = self.run_lane(factory=factory)
        self.assertTrue(all(v["status"] == "completed" for v in result["models"].values()))

    def test_cancellation_seals_partial_history(self):
        self.plan()
        def factory(provider, **kwargs):
            return FakeClient(kwargs["model"], lambda request, count: asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            self.run_lane(factory=factory)
        result = suite.report(self.root)
        self.assertEqual(result["models"]["jev"]["status"], "interrupted")
        self.assertEqual(self.attempts()[0]["diagnostic_code"], "interrupted")
        self.assertEqual(len(self.scores()), 4)

    def test_cli_sigterm_handler_seals_without_signaling_a_process(self):
        self.plan()
        original_run = suite.run_live
        handlers = {}
        async def exercise():
            loop = asyncio.get_running_loop()
            def factory(provider, **kwargs):
                def behavior(request, count):
                    handlers[suite.signal.SIGTERM]()
                    handlers[suite.signal.SIGTERM]()  # Repeated TERM must not disrupt cleanup.
                return FakeClient(kwargs["model"], behavior)
            async def invoke(root, models):
                return await original_run(root, models, client_factory=factory,
                                          preflight=self.preflight)
            with patch.object(loop, "add_signal_handler",
                              side_effect=lambda sig, handler: handlers.update({sig: handler})), \
                    patch.object(loop, "remove_signal_handler") as remove, \
                    patch.object(suite, "run_live", side_effect=invoke):
                with self.assertRaises(asyncio.CancelledError):
                    await suite.run_cli(self.root, ("jev",))
                remove.assert_called_once_with(suite.signal.SIGTERM)
        asyncio.run(exercise())
        result = suite.report(self.root)
        self.assertEqual(result["models"]["jev"]["status"], "interrupted")
        self.assertEqual(self.attempts()[0]["diagnostic_code"], "interrupted")
        self.assertEqual(len(self.scores()), 4)

    def test_cli_cancellation_has_sanitized_exit_143(self):
        output = io.StringIO()
        with patch.object(suite, "run_cli", side_effect=asyncio.CancelledError()), \
                redirect_stderr(output):
            self.assertEqual(suite.main(["run", "--root", str(self.root)]), 143)
        self.assertEqual(json.loads(output.getvalue())["status"], "interrupted")

    def test_all_optimization_tasks_replay_dynamic_options(self):
        graph = {"nodes": list(range(5)), "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
                 "directed": False}
        points = [[0, 0], [1, 0], [2, 1], [3, 0], [4, 2]]
        tsp = {**deepcopy(graph), "points": points, "metric": "Manhattan",
               "edges": [[u, v, sum(abs(a - b) for a, b in zip(points[u], points[v]))]
                         for u in range(5) for v in range(u + 1, 5)]}
        lt = {**deepcopy(graph), "directed": True, "threshold": {"numerator": 1, "denominator": 2},
              "seed_count": 2,
              "weights": [{"source": u, "target": v, "numerator": 1, "denominator": 1}
                          for u, v in graph["edges"]]}
        instances = self.instances + [
            tasks._record("tsp_construct", 0, tsp),
            tasks._record("lt_influence_construct", 0, lt),
            tasks._record("community_bipartition_construct", 0, deepcopy(graph))]
        with patch.object(tasks, "build_instances", return_value=instances):
            self.plan()
        result = self.run_lane()
        self.assertEqual(len(result["models"]["jev"]["by_task"]), 7)
        rows = {row["task"]: row for row in self.scores()}
        self.assertEqual(rows["tsp_construct"]["decisions"], ["4", "3", "2"])
        self.assertEqual(rows["tsp_construct"]["solution"], [0, 4, 3, 2, 1, 0])
        self.assertEqual(rows["lt_influence_construct"]["solution"], [4, 3])
        self.assertTrue(all(r["feasible"] for r in rows.values()))
        self.assertEqual(rows["tsp_construct"]["query_count"], 3)

    def test_slow_gpt_initialization_does_not_gate_jev(self):
        self.plan(("gpt54", "jev"))
        async def run():
            jev_called = asyncio.Event()
            class ConcurrentClient(FakeClient):
                async def initialize(self):
                    if self.model == "gpt-5.4":
                        await asyncio.wait_for(jev_called.wait(), timeout=2)
                    await super().initialize()
                async def predict(self, request):
                    if self.model == "jev-1.13.0":
                        jev_called.set()
                    return await super().predict(request)
            def factory(provider, **kwargs):
                return ConcurrentClient(kwargs["model"])
            return await suite.run_live(self.root, client_factory=factory, preflight=self.preflight)
        result = asyncio.run(run())
        self.assertTrue(all(v["status"] == "completed" for v in result["models"].values()))

    def test_macro_size_accuracy_weights_sizes_equally(self):
        self.plan()
        self.run_lane()
        degree = next(row for row in self.scores() if row["task"] == "degree_exact")
        small = {**degree, "node_count": 5, "correct": True, "prediction": degree["answer"]}
        large = {**degree, "node_count": 6, "correct": False}
        summary = suite.summarize([small, large, deepcopy(large)])
        exact = summary["by_task"]["degree_exact"]
        self.assertEqual(exact["accuracy"], 1 / 3)
        self.assertEqual(exact["macro_size_accuracy"], 1 / 2)
        self.assertEqual(exact["macro_size_denominator"], 2)
        self.assertEqual(summary["by_task_and_size"]["degree_exact"]["5"]["accuracy"], 1)
        self.assertEqual(summary["by_task_and_size"]["degree_exact"]["6"]["accuracy"], 0)

    def test_http_auth_context_quota_failures_are_fatal(self):
        for code in (400, 401, 403, 404, 413, 422, 429, 451):
            self.assertTrue(suite.failure(ProviderHTTPError(code))["fatal"])
        for code in (408, 500, 502, 503, 504, 599):
            self.assertFalse(suite.failure(ProviderHTTPError(code))["fatal"])

    def test_http_503_stops_instance_then_continues_without_retry(self):
        self.plan()
        def factory(provider, **kwargs):
            return FakeClient(kwargs["model"], lambda request, count:
                              ProviderHTTPError(503) if count == 2 else None)
        result = self.run_lane(factory=factory)
        self.assertEqual(result["models"]["jev"]["status"], "completed_with_failures")
        attempts = self.attempts()
        self.assertEqual(len(attempts), 5)
        self.assertEqual(attempts[1]["http_status"], 503)
        self.assertFalse(attempts[1]["fatal"])
        self.assertEqual(attempts[2]["instance_id"], self.instances[1]["id"])
        rows = self.scores()
        self.assertEqual(rows[0]["decisions"], ["B"])
        self.assertEqual(rows[0]["query_count"], 2)
        self.assertFalse(rows[0]["feasible"])
        self.assertTrue(all(row["feasible"] for row in rows[1:]))

    def test_stdout_and_errors_are_sanitized(self):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            self.assertEqual(suite.main(["plan", "--root", str(self.root), "--models", "jev"]), 0)
        self.assertNotIn("private", output.getvalue())
        with patch.object(suite, "load_plan", side_effect=RuntimeError("SECRET")), \
                redirect_stdout(output), redirect_stderr(error):
            self.assertEqual(suite.main(["report", "--root", str(self.root)]), 2)
        self.assertNotIn("SECRET", output.getvalue() + error.getvalue())


if __name__ == "__main__":
    unittest.main()
