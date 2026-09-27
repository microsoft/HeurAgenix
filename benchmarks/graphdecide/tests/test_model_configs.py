"""Offline checks for the task/model matrix and YAML-to-provider integration."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
import yaml

from src.benchmark.__main__ import main
from src.benchmark.config import load_config, ModelConfig
from src.benchmark.prepare import prepare
from src.benchmark.runner import run_experiment
from src.clients._llm_choice import ANSWER_ONLY_SYSTEM_MESSAGE, prompt_for
from src.clients.base import UnsupportedRequestError
from test_benchmark import config_for, FakeClient, path_loader
from test_llm_clients import FakeRuntime, model_info


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIGS = ROOT / "output" / "configs"
QWEN_TARGETS = {
    "0.6B": ("Qwen3.5-0.8B", 8000),
    "1.7B": ("Qwen3.5-2B", 8001),
    "4B": ("Qwen3.5-4B", 8002),
    "8B": ("Qwen3.5-9B", 8003),
}
QWEN_VARIANTS = {
    f"Qwen3-{size}-{mode}" for size in QWEN_TARGETS for mode in ("think", "no-think")
}


class ModelConfigTests(unittest.TestCase):
    def test_answer_only_round_trip_and_default_snapshot_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            for provider, model in (("vllm", "local"), ("github_copilot", "gpt-5.4"),
                                    ("github_copilot", "gpt-6-astra")):
                base = replace(config_for(Path(tmp)), model=ModelConfig(provider, model, 30))
                self.assertNotIn("output_format", base.snapshot()["model"])
                self.assertNotIn("output_format", base.model.client_kwargs())
                expected = {"model": model, "timeout_seconds": 30, "think": None, "max_tokens": 4096}
                if provider == "github_copilot":
                    self.assertEqual(base.model.client_kwargs(), {**expected, "reasoning_effort": None})
                for output_format in (None, "json", "answer_only"):
                    with self.subTest(provider=provider, model=model, output_format=output_format):
                        changed = replace(base, model=replace(base.model, output_format=output_format))
                        path = Path(tmp) / "answer.yaml"
                        path.write_text(yaml.safe_dump(changed.snapshot()))
                        loaded = load_config(path)
                        self.assertEqual(loaded, changed)
                        self.assertEqual(loaded.sha256, changed.sha256)
                        kwargs = loaded.model.client_kwargs()
                        if output_format is None:
                            self.assertEqual(kwargs, base.model.client_kwargs())
                        else:
                            self.assertEqual(kwargs, {**base.model.client_kwargs(), "output_format": output_format})
                            self.assertNotEqual(base.sha256, changed.sha256)
                        self.assertEqual(prepare(base, path_loader()).examples,
                                         prepare(loaded, path_loader()).examples)

    @unittest.skipUnless(LOCAL_CONFIGS.is_dir(), "Local output/configs matrix is not versioned")
    def test_matrix_preserves_tasks_sampling_paths_and_separate_outputs(self):
        outputs = set()
        for task in ("adjacency", "distance_threshold"):
            original = load_config(ROOT / "configs" / f"{task}.yaml")
            paths = sorted((LOCAL_CONFIGS / task).glob("*.yaml"))
            self.assertEqual({p.stem for p in paths if p.stem.startswith("Qwen3")}, QWEN_VARIANTS)
            for path in paths:
                with self.subTest(config=path.name, task=task):
                    config = load_config(path)
                    self.assertEqual(config.data, original.data)
                    self.assertEqual(config.sampling, original.sampling)
                    self.assertEqual(config.tasks, original.tasks)
                    self.assertEqual(config.run.seed, original.run.seed)
                    self.assertEqual(config.run.repetitions, original.run.repetitions)
                    self.assertEqual(config.run.max_calls, original.run.max_calls)
                    self.assertEqual(config.planned_graphs, 240)
                    self.assertEqual(config.planned_questions, original.planned_questions)
                    self.assertEqual(config.run.output_dir.parent, ROOT / "output")
                    self.assertNotIn(config.run.output_dir, outputs)
                    outputs.add(config.run.output_dir)
                    self.assertEqual(config.model.timeout_seconds, 180)
                    self.assertEqual(config.model.max_tokens, 4096)
                    if path.stem.startswith("Qwen3"):
                        size = path.stem.split("-")[1]
                        alias, port = QWEN_TARGETS[size]
                        mode = "no-think" if path.stem.endswith("no-think") else "think"
                        self.assertEqual(config.model.provider, "vllm")
                        self.assertEqual(config.model.model, alias)
                        self.assertEqual(config.model.think, not path.stem.endswith("no-think"))
                        self.assertEqual(config.model.base_url, f"http://127.0.0.1:{port}/v1")
                        self.assertEqual(config.run.output_dir, ROOT / "output" / f"{task}.{alias}-{mode}")
                        self.assertEqual((config.model.temperature, config.model.top_p, config.model.seed),
                                         (0.0, 1.0, 20260924))
                    else:
                        self.assertEqual(config.model.provider, "github_copilot")
                        self.assertEqual(config.model.model, path.stem)
                        self.assertIsNone(config.model.think)
                        self.assertIsNone(config.model.reasoning_effort)
        self.assertEqual(len(outputs), 20)

    @unittest.skipUnless(LOCAL_CONFIGS.is_dir(), "Local output/configs matrix is not versioned")
    def test_model_changes_do_not_change_fixture_graphs_requests_or_labels(self):
        fixture = config_for(Path("/unused"))
        for task in ("adjacency", "distance_threshold"):
            baseline = load_config(ROOT / "configs" / f"{task}.yaml")
            fixture_config = replace(fixture, tasks=baseline.tasks)
            expected = prepare(fixture_config, path_loader())
            self.assertTrue(expected.examples)
            for path in (LOCAL_CONFIGS / task).glob("*.yaml"):
                with self.subTest(path=path.name, task=task):
                    config = load_config(path)
                    actual = prepare(replace(fixture_config, model=config.model), path_loader())
                    self.assertEqual(actual.graphs, expected.graphs)
                    self.assertEqual(actual.examples, expected.examples)

    def test_config_snapshots_round_trip_and_record_effective_settings(self):
        paths = [ROOT / "configs" / "adjacency.yaml"] + list((LOCAL_CONFIGS / "adjacency").glob("*.yaml"))
        with tempfile.TemporaryDirectory() as tmp:
            for path in paths:
                config = load_config(path)
                snapshot = Path(tmp) / "snapshot.yaml"
                snapshot.write_text(yaml.safe_dump(config.snapshot()))
                self.assertEqual(config, load_config(snapshot))
                if config.model.provider == "typesafe":
                    self.assertEqual(set(config.snapshot()["model"]), {"provider", "model", "timeout_seconds"})
        zero = ModelConfig("vllm", "test", 30, think=False, temperature=0.0, seed=0)
        self.assertIs(zero.client_kwargs()["think"], False)
        self.assertEqual(zero.client_kwargs()["seed"], 0)
        self.assertEqual(zero.client_kwargs()["temperature"], 0.0)

    def test_invalid_provider_options_and_secrets_rejected_during_loading(self):
        original = load_config(ROOT / "configs" / "adjacency.yaml").snapshot()
        variants = [
            {"output_format": "answer_only"},
            {"output_format": "json"},
            {"provider": "github_copilot", "output_format": "unknown"},
            {"provider": "github_copilot", "output_format": []},
            {"provider": "github_copilot", "output_format": False},
            {"provider": "vllm", "output_format": "unknown"},
            {"provider": "vllm", "output_format": []},
            {"provider": "missing"}, {"provider": []}, {"provider": None},
            {"think": True}, {"max_tokens": 4096},
            {"provider": "vllm", "reasoning_effort": "medium"},
            {"provider": "github_copilot", "base_url": "http://localhost:8000/v1"},
            {"provider": "github_copilot", "temperature": 0.0},
            {"provider": "github_copilot", "seed": 0},
            {"provider": "github_copilot", "think": False, "reasoning_effort": "low"},
            {"provider": "github_copilot", "reasoning_effort": "none"},
            {"provider": "github_copilot", "reasoning_effort": []},
            {"provider": "vllm", "think": "false"}, {"provider": "vllm", "think": 0},
            {"provider": "vllm", "max_tokens": True}, {"provider": "vllm", "max_tokens": 0},
            {"provider": "vllm", "temperature": float("nan")},
            {"provider": "vllm", "top_p": 0}, {"provider": "vllm", "top_p": True},
            {"provider": "vllm", "seed": -1}, {"provider": "vllm", "seed": True},
            {"provider": "vllm", "base_url": "http://user:PRIVATE@localhost:8000/v1"},
            {"provider": "vllm", "base_url": "http://localhost:8000/v1?token=PRIVATE"},
            {"provider": "vllm", "base_url": "http://localhost:8000/v1#PRIVATE"},
            {"provider": "vllm", "base_url": "http://[PRIVATE"},
            {"provider": "vllm", "base_url": 42},
        ]
        for provider in ("typesafe", "vllm", "github_copilot"):
            for key in ("api_key", "github_token", "extra_body", "headers"):
                variants.append({"provider": provider, key: "PRIVATE_CREDENTIAL"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            for change in variants:
                data = {**original, "model": {**original["model"], **change}}
                path.write_text(yaml.safe_dump(data))
                with self.subTest(change=change):
                    with self.assertRaises(ValueError) as caught:
                        load_config(path)
                    self.assertNotIn("PRIVATE", str(caught.exception))


@unittest.skipUnless((ROOT / "output" / "support" / "run_qwen_matrix.py").is_file(),
                     "Local matrix runner is not versioned")
class LocalMatrixRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "local_qwen_matrix", ROOT / "output" / "support" / "run_qwen_matrix.py")
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.runner.ROOT = self.root
        (self.root / "output").mkdir()
        fixture = config_for(self.root)
        self.configs = {}
        for task in ("adjacency", "distance_threshold"):
            for size, (alias, port) in QWEN_TARGETS.items():
                path = self.root / "output" / "configs" / task / f"Qwen3-{size}-no-think.yaml"
                self.configs[path] = replace(
                    fixture,
                    model=ModelConfig("vllm", alias, 180, think=False,
                                      base_url=f"http://127.0.0.1:{port}/v1"),
                    run=replace(fixture.run, output_dir=self.root / "output" / f"{task}.{alias}-no-think"),
                )
        loader = patch.object(self.runner, "load_config", side_effect=self.configs.__getitem__)
        loader.start()
        self.addCleanup(loader.stop)

    def test_preflight_accepts_all_legacy_entrypoints_with_new_targets(self):
        configs = self.runner.preflight_configs()
        self.assertEqual(self.runner.SIZES, tuple(QWEN_TARGETS))
        self.assertEqual(len(configs), 8)
        self.assertEqual({path for path, _ in configs.values()}, set(self.configs))

    async def assert_preflight_rejected(self, error, message):
        with patch.object(self.runner.asyncio, "create_subprocess_exec") as spawn:
            with self.assertRaisesRegex(error, message):
                await self.runner.main()
            spawn.assert_not_called()
        self.assertFalse((self.root / "output" / "qwen35-no-think-matrix-20260923").exists())

    async def test_preflight_rejects_wrong_alias_endpoint_provider_and_mode_before_calls(self):
        path = list(self.configs)[-1]
        config = self.configs[path]
        for change in ({"model": "Qwen3-8B"}, {"base_url": "http://127.0.0.1:8000/v1"},
                       {"provider": "typesafe", "base_url": None, "think": None}, {"think": True}):
            with self.subTest(change=change):
                self.configs[path] = replace(config, model=replace(config.model, **change))
                await self.assert_preflight_rejected(ValueError, "Expected")

    async def test_preflight_rejects_duplicate_and_legacy_outputs_before_calls(self):
        first, *_, last = self.configs
        config = self.configs[last]
        duplicate = self.configs[first].run.output_dir
        for output, message in (
            (duplicate, "Duplicate output directory"),
            (duplicate.parent / "unused" / ".." / duplicate.name, "Duplicate output directory"),
            (self.root / "output" / "distance_threshold.Qwen3-8B-no-think", "Expected new output"),
        ):
            with self.subTest(output=output):
                self.configs[last] = replace(config, run=replace(config.run, output_dir=output))
                await self.assert_preflight_rejected(ValueError, message)

    async def test_preflight_rejects_existing_outputs_before_calls(self):
        config = list(self.configs.values())[-1]
        config.run.output_dir.mkdir()
        await self.assert_preflight_rejected(FileExistsError, "distance_threshold.Qwen3.5-9B")

    async def test_two_four_parallel_stages_use_new_models_and_log_directory(self):
        started = []
        finished = set()

        async def spawn(*args, **kwargs):
            path = Path(args[3])
            task = path.parent.name
            if task == "distance_threshold":
                self.assertEqual(len([p for p in finished if p.parent.name == "adjacency"]), 4)
            self.assertEqual(args[:3], (self.runner.sys.executable, "run_benchmark.py", "--config"))
            self.assertEqual(kwargs["cwd"], self.root)
            for name in ("TYPESAFE_API_KEY", "COPILOT_GITHUB_TOKEN", "VLLM_API_KEY", "PYTHONPATH"):
                self.assertNotIn(name, kwargs["env"])
            started.append(path)
            proc = NS(pid=len(started), returncode=None)

            async def wait():
                self.assertEqual(len([p for p in started if p.parent.name == task]), 4)
                finished.add(path)
                proc.returncode = 0
                return 0

            proc.wait = wait
            return proc

        summary = json.dumps({"run_status": "completed", "overall": {
            "attempts": 1, "correct": 1, "accuracy": 1, "failures": 0,
            "latency_all": {}, "token_usage": {},
        }})
        with patch.object(self.runner.asyncio, "create_subprocess_exec", side_effect=spawn), \
             patch.object(Path, "read_text", return_value=summary), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            await self.runner.main()
        self.assertEqual(started, list(self.configs))
        logs = self.root / "output" / "qwen35-no-think-matrix-20260923"
        self.assertEqual(len(list(logs.glob("*.log"))), 16)
        for task in ("adjacency", "distance_threshold"):
            rows = json.loads((logs / f"{task}.summary.json").read_text())
            self.assertEqual([row["model"] for row in rows], [alias for alias, _ in QWEN_TARGETS.values()])
            for alias, _ in QWEN_TARGETS.values():
                self.assertIn(f"Started {task} {alias}: PID", stdout.getvalue())


class MultiProviderCliTests(unittest.TestCase):
    def test_answer_only_yaml_reaches_adapter_and_records_protocol(self):
        def handler(wire):
            body = json.loads(wire.content)
            self.assertIn("one option ID", body["messages"][0]["content"])
            return httpx.Response(200, json={"model": "local", "choices": [
                {"finish_reason": "stop", "message": {"role": "assistant", "content": " YES\n"}}
            ]})

        def factory(**kwargs):
            return self.original_http(**{**kwargs, "transport": httpx.MockTransport(handler)})

        with patch("src.clients.vllm_client.httpx.AsyncClient", side_effect=factory), \
                patch.dict(os.environ, {}, clear=True):
            self.execute(ModelConfig("vllm", "local", 5, think=False, output_format="answer_only"))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original_http = httpx.AsyncClient

    def execute(self, model):
        config = replace(config_for(self.root), model=model)
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config.snapshot()))
        with patch("src.benchmark.runner.prepare", side_effect=lambda config, loader: prepare(config, path_loader())), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            main(["--config", str(path), "--no-progress"])
        summary = json.loads(stdout.getvalue())
        run = json.loads((config.run.output_dir / "run.json").read_text())
        self.assertEqual(run["model_settings"], model.client_kwargs())
        self.assertEqual(run["response_protocol"],
                 "answer-only-v1" if model.output_format == "answer_only" else "json-choice-v1")
        self.assertFalse(run["injected_client"])
        self.assertEqual(run["adapter_retries"], 0)
        self.assertEqual(summary["completed_calls"], 8)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertEqual(summary["overall"]["probability_calls"], 0)
        self.assertIsNone(summary["overall"]["brier_yes"])
        self.assertIsNone(summary["overall"]["cost"])
        for artifact in config.run.output_dir.iterdir():
            self.assertNotIn("PRIVATE", artifact.read_text())
        return run

    def test_vllm_yaml_think_and_no_think_reach_real_adapter(self):
        bodies = []
        for think in (False, True):
            self.root = Path(self.tmp.name) / str(think)
            self.root.mkdir()
            model = ModelConfig("vllm", "Qwen/Qwen3-0.6B", 5, think=think, max_tokens=512,
                                base_url="http://127.0.0.1:8000/v1", temperature=0.0, top_p=1.0, seed=0)

            def handler(wire):
                body = json.loads(wire.content)
                bodies.append(body)
                self.assertIs(body["chat_template_kwargs"]["enable_thinking"], think)
                self.assertEqual(body["max_tokens"], 512)
                self.assertEqual(body["seed"], 0)
                self.assertEqual(body["model"], model.model)
                return httpx.Response(200, json={"model": model.model, "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": '{"choice":"yes"}'}}
                ], "usage": {"prompt_tokens": 30, "completion_tokens": 10,
                              "completion_tokens_details": {"reasoning_tokens": 5 if think else 0}}})

            def factory(**kwargs):
                return self.original_http(**{**kwargs, "transport": httpx.MockTransport(handler)})

            with patch("src.clients.vllm_client.httpx.AsyncClient", side_effect=factory), \
                 patch.dict(os.environ, {"VLLM_API_KEY": "PRIVATE_FAKE_KEY"}, clear=True):
                run = self.execute(model)
            self.assertEqual(run["retries"], 0)
        self.assertEqual(len(bodies), 16)

    def test_copilot_yaml_reaches_real_adapter_and_records_retry_limits(self):
        runtime = FakeRuntime()
        with patch("src.clients.github_copilot_client._sdk_types", return_value=(
            lambda **kwargs: runtime, NS, NS, lambda: NS(kind="denied")
        )), patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "PRIVATE_FAKE_KEY"}, clear=True):
            run = self.execute(ModelConfig("github_copilot", "test-gpt", 5, think=True,
                                           max_tokens=512, reasoning_effort="medium"))
        self.assertIsNone(run["retries"])
        self.assertEqual(run["sdk_retries"], "not_controlled")
        self.assertIn("github-copilot-sdk", run["versions"])
        self.assertGreaterEqual(run["initialization_seconds"], 0)
        runtime.start.assert_awaited_once()
        runtime.list_models.assert_awaited_once()
        runtime.stop.assert_awaited_once()
        self.assertEqual(len(runtime.sessions), 8)
        self.assertTrue(all(s.options["model_capabilities"].limits.max_output_tokens == 512 for s in runtime.sessions))

    def test_copilot_answer_only_yaml_uses_same_protocol_for_both_gpt_models(self):
        prompts = []
        for model in ("gpt-5.4", "gpt-6-astra"):
            with self.subTest(model=model):
                self.root = Path(self.tmp.name) / model
                self.root.mkdir()
                runtime = FakeRuntime()
                runtime.list_models.return_value = [NS(**{**vars(model_info()), "id": model})]
                runtime.answer.data.content = " YES\n"
                with patch("src.clients.github_copilot_client._sdk_types", return_value=(
                    lambda **kwargs: runtime, NS, NS, lambda: NS(kind="denied")
                )), patch.dict(os.environ, {}, clear=True):
                    self.execute(ModelConfig("github_copilot", model, 5, output_format="answer_only"))
                self.assertEqual(len(runtime.sessions), 8)
                for session in runtime.sessions:
                    self.assertEqual(session.options["model"], model)
                    self.assertIsNone(session.options["reasoning_effort"])
                    self.assertEqual(session.options["system_message"], {
                        "mode": "replace", "content": ANSWER_ONLY_SYSTEM_MESSAGE})
                config = replace(config_for(self.root), model=ModelConfig("github_copilot", model, 5))
                self.assertEqual(runtime.prompts, [prompt_for(example.request)
                                                  for example in prepare(config, path_loader()).examples])
                prompts.append(runtime.prompts)
                runtime.stop.assert_awaited_once()
        self.assertEqual(prompts[0], prompts[1])

    def test_copilot_unavailable_model_aborts_before_any_scored_question(self):
        runtime = FakeRuntime()
        runtime.list_models.return_value = [model_info()]
        with patch("src.clients.github_copilot_client._sdk_types", return_value=(
            lambda **kwargs: runtime, NS, NS, lambda: NS(kind="denied")
        )), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(UnsupportedRequestError):
                self.execute(ModelConfig("github_copilot", "unavailable-test-model", 5))
        run = json.loads((self.root / "output" / "run.json").read_text())
        summary = json.loads((self.root / "output" / "summary.json").read_text())
        self.assertEqual(run["status"], "aborted")
        self.assertEqual(run["completed_calls"], 0)
        self.assertEqual(summary["completed_calls"], 0)
        self.assertEqual(runtime.prompts, [])
        runtime.create_session.assert_not_called()
        runtime.stop.assert_awaited_once()


class ProviderRunnerTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(LOCAL_CONFIGS.is_dir(), "Local output/configs matrix is not versioned")
    async def test_every_template_is_forwarded_and_saved_without_real_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = config_for(Path(tmp))
            for task in ("adjacency", "distance_threshold"):
                for path in (LOCAL_CONFIGS / task).glob("*.yaml"):
                    template = load_config(path)
                    output = Path(tmp) / f"{task}-{path.stem}"
                    config = replace(fixture, model=template.model,
                                     run=replace(fixture.run, output_dir=output))
                    with self.subTest(path=path), patch("src.benchmark.runner.create_client", return_value=FakeClient()) as factory:
                        await run_experiment(config, loader=path_loader())
                        factory.assert_called_once_with(template.model.provider, **template.model.client_kwargs())
                        recorded = json.loads((output / "run.json").read_text())
                        self.assertEqual(recorded["model_settings"], template.model.client_kwargs())
                        self.assertEqual(recorded["provider"], template.model.provider)

    async def test_preflight_timeout_closes_client_and_retains_zero_attempt_summary(self):
        class HangingClient(FakeClient):
            async def initialize(self):
                await asyncio.Future()

        with tempfile.TemporaryDirectory() as tmp:
            config = config_for(Path(tmp))
            config = replace(config, model=replace(config.model, timeout_seconds=0.01))
            client = HangingClient()
            with self.assertRaises(TimeoutError):
                await run_experiment(config, client=client, loader=path_loader())
            self.assertTrue(client.closed)
            self.assertEqual(client.calls, 0)
            summary = json.loads((config.run.output_dir / "summary.json").read_text())
            self.assertEqual(summary["run_status"], "aborted")
            self.assertEqual(summary["completed_calls"], 0)


if __name__ == "__main__":
    unittest.main()