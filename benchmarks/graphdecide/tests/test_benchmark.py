"""Offline pipeline tests: no credentials, downloaded archives, or live requests."""

from __future__ import annotations

import asyncio
import io
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import httpx
import networkx as nx
import yaml

from src.benchmark.__main__ import main
from src.benchmark.config import (
    BenchmarkConfig, DataConfig, ModelConfig, RunConfig, SamplingConfig, TaskConfig, load_config,
)
from src.benchmark.prepare import anonymous_induced_graph, prepare, verify_examples
from src.benchmark.runner import _percentile, run_experiment, summarize
from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, DecisionResponse, ProviderHTTPError, TokenUsage,
)
from src.clients.typesafe import TypeSafeClient
from src.datasets import GraphDataset, LoadStats, get_source


TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml"


def config_for(root: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        1, RunConfig(123, 1, 20, root / "output"), DataConfig(("facebook",), root / "data"),
        SamplingConfig((8,), 2, 10), TaskConfig(2, 2, (2,)),
        ModelConfig("typesafe", "test-only-model", 1.0),
    )


def fixture_loader(graph: nx.Graph):
    def load(name, **kwargs):
        stats = LoadStats(graph.number_of_edges(), 0, 0, len(graph), graph.number_of_edges(),
                          nx.number_of_isolates(graph), nx.number_connected_components(graph))
        return GraphDataset(get_source(name), graph, stats, "offline-fixture-sha")
    return load


def path_loader():
    graph = nx.path_graph(range(10000, 10030))
    graph.graph["title"] = "PRIVATE_GRAPH_TITLE"
    nx.set_node_attributes(graph, "PRIVATE_NODE_NAME", "name")
    nx.set_edge_attributes(graph, "PRIVATE_EDGE_ATTRIBUTE", "description")
    return fixture_loader(graph)


class ConfigTests(unittest.TestCase):
    def test_single_task_templates_preserve_shared_sampling_and_budgets(self):
        adjacency = load_config(TEMPLATE.with_name("adjacency.yaml"))
        distance = load_config(TEMPLATE.with_name("distance_threshold.yaml"))
        self.assertEqual(adjacency.data, distance.data)
        self.assertEqual(adjacency.sampling, distance.sampling)
        self.assertEqual(adjacency.model, distance.model)
        self.assertEqual(adjacency.run.seed, distance.run.seed)
        self.assertEqual(adjacency.run.repetitions, distance.run.repetitions)
        for config in (adjacency, distance):
            self.assertEqual(config.run.max_calls, config.planned_questions)
        self.assertEqual(adjacency.tasks, TaskConfig(4, 0, ()))
        self.assertEqual(distance.tasks, TaskConfig(0, 4, (2, 3, 4, 6)))
        self.assertEqual(adjacency.planned_questions, 960)
        self.assertEqual(distance.planned_questions, 3840)
        self.assertEqual(adjacency.planned_questions + distance.planned_questions, 4800)
        self.assertNotEqual(adjacency.run.output_dir, distance.run.output_dir)

    def test_example_and_path_resolution(self):
        config = load_config(TEMPLATE)
        self.assertEqual(config.planned_graphs, 60)
        self.assertEqual(config.planned_questions, 240)
        self.assertEqual(config.run.repetitions, 1)
        self.assertEqual(config.data.data_dir, TEMPLATE.parent.parent / "data" / "real")
        self.assertEqual(config.sha256, load_config(TEMPLATE).sha256)
        self.assertEqual(len(config.sha256), 64)

    def test_invalid_configs_fail_early(self):
        original = yaml.safe_load(TEMPLATE.read_text())
        changes = [
            (None, "schema_version", True),
            ("run", "seed", -1), ("run", "repetitions", 0), ("run", "max_calls", 239),
            ("tasks", "adjacency_questions", 3), ("tasks", "adjacency_questions", True),
            ("tasks", "distance_thresholds", [1]), ("tasks", "distance_thresholds", []),
            ("sampling", "node_counts", [16, 16]), ("sampling", "node_counts", []),
            ("sampling", "method", "bfs"), ("model", "provider", "mock"),
            ("model", "model", ""), ("model", "timeout_seconds", float("nan")),
            ("model", "timeout_seconds", True),
            ("data", "datasets", ["facebook", "facebook"]), ("data", "datasets", ["missing"]),
            ("model", "api_key", "not-allowed"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            for section, key, value in changes:
                with self.subTest(section=section, key=key, value=value):
                    data = json.loads(json.dumps(original))
                    (data if section is None else data[section])[key] = value
                    path.write_text(yaml.safe_dump(data))
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_task_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adjacency.yml"
            path.write_text(TEMPLATE.read_text().replace("distance_questions_per_threshold: 2", "distance_questions_per_threshold: 0")
                            .replace("distance_thresholds: [2]", "distance_thresholds: []"))
            self.assertEqual(load_config(path).planned_questions, 120)

    def test_yaml_duplicate_keys_rejected_at_all_levels(self):
        original = TEMPLATE.read_text()
        variants = [original + "\nschema_version: 1\n",
                    original.replace("  seed: 20260923", "  seed: 20260923\n  seed: 1")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.yaml"
            for text in variants:
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
                    load_config(path)

    def test_unsafe_malformed_and_nonmapping_yaml_rejected(self):
        variants = ["", "[]", "null", "[PRIVATE_INVALID_VALUE", "a: 1\n---\nb: 2",
                    "!!python/object/apply:os.system ['PRIVATE_DO_NOT_EXECUTE']",
                    "1: value", "? [key, list]\n: value", "<<: {seed: 1}"]
        with tempfile.TemporaryDirectory() as tmp, patch("os.system", side_effect=AssertionError("No execution")):
            path = Path(tmp) / "invalid.yaml"
            for text in variants:
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError) as caught:
                        load_config(path)
                    self.assertNotIn("PRIVATE", str(caught.exception))

    def test_distance_yaml_and_legacy_format_rejection(self):
        config = load_config(TEMPLATE.with_name("distance_threshold.yaml"))
        self.assertEqual(config.planned_graphs, 240)
        self.assertEqual(config.planned_questions, 3840)
        self.assertEqual(config.run.max_calls, 3840)
        self.assertEqual(config.sampling.node_counts, (32, 64, 128))
        self.assertEqual(config.tasks.distance_thresholds, (2, 3, 4, 6))
        self.assertEqual(config.model.provider, "typesafe")
        self.assertEqual(config.model.model, "jev-1.13.0")
        with self.assertRaisesRegex(ValueError, ".yaml or .yml"):
            load_config(TEMPLATE.with_suffix(".toml"))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.config = config_for(Path("/unused"))

    def test_split_tasks_match_combined_graphs_requests_and_labels(self):
        combined_config = replace(self.config, tasks=TaskConfig(4, 4, (2, 3)))
        combined = prepare(combined_config, path_loader())
        union = []
        for task, tasks in (("adjacency", TaskConfig(4, 0, ())),
                            ("distance_threshold", TaskConfig(0, 4, (2, 3)))):
            with self.subTest(task=task):
                split = prepare(replace(combined_config, tasks=tasks), path_loader())
                self.assertEqual(split.graphs, combined.graphs)
                self.assertEqual(split.examples, [e for e in combined.examples if e.task == task])
                self.assertTrue(all(cell["task"] == task for cell in split.summary()["strata"]))
                union.extend(split.examples)
        key = lambda e: e.request.request_id
        self.assertEqual(sorted(union, key=key), sorted(combined.examples, key=key))

    def test_anonymization_removes_all_attributes_and_preserves_induced_edges(self):
        original = nx.Graph(title="private")
        original.add_nodes_from([("Alice", {"name": "Alice"}), ("Bob", {"age": 30}), ("Carol", {})])
        original.add_edge("Alice", "Bob", relation="private")
        original.add_edge("Bob", "Carol", weight=99)
        graph, mapping = anonymous_induced_graph(original, list(original), random.Random(1))
        self.assertEqual(set(graph), {0, 1, 2})
        self.assertEqual(graph.graph, {})
        self.assertTrue(all(not attrs for _, attrs in graph.nodes(data=True)))
        self.assertTrue(all(not attrs for _, _, attrs in graph.edges(data=True)))
        self.assertEqual({frozenset((mapping[u], mapping[v])) for u, v in graph.edges()},
                         {frozenset(e) for e in original.edges()})
        self.assertEqual(original["Bob"]["Carol"]["weight"], 99)
        self.assertTrue(nx.is_frozen(graph))

    def test_deterministic_balanced_generation_and_independent_verification(self):
        first = prepare(self.config, path_loader())
        second = prepare(self.config, path_loader())
        self.assertEqual(first, second)
        self.assertEqual(len(first.graphs), 2)
        self.assertEqual(len(first.examples), 8)
        self.assertFalse(first.issues)
        verify_examples(first.examples)
        for graph in first.graphs:
            selected = [e for e in first.examples if e.graph_id == graph["graph_id"]]
            for task in ("adjacency", "distance_threshold"):
                self.assertEqual(sorted(e.answer for e in selected if e.task == task), ["no", "yes"])
            self.assertEqual(graph["edge_count"], 7)
            self.assertEqual(len(graph["original_ids_by_anonymous_id"]), 8)
        for example in first.examples:
            payload = json.dumps({"state": example.request.state, "question": example.request.question})
            self.assertNotIn("PRIVATE", payload)
            self.assertNotIn("facebook", payload)
            self.assertNotIn("10000", payload)
            self.assertEqual(set(example.request.state), {"nodes", "edges"})
            if example.task == "distance_threshold":
                self.assertEqual(example.distance, 2 if example.answer == "yes" else 3)

    def test_changing_seed_changes_samples(self):
        other = replace(self.config, run=replace(self.config.run, seed=456))
        self.assertNotEqual(prepare(self.config, path_loader()).graphs, prepare(other, path_loader()).graphs)

    def test_duplicates_bounded_and_missing_components_reported(self):
        prepared = prepare(self.config, fixture_loader(nx.path_graph(8)))
        self.assertEqual(len(prepared.graphs), 1)
        self.assertEqual(prepared.issues[0]["reason"], "duplicate_sample_limit")
        self.assertEqual(prepared.issues[0]["attempts"], 10)
        empty = prepare(self.config, fixture_loader(nx.path_graph(4)))
        self.assertFalse(empty.graphs)
        self.assertEqual(len(empty.issues), 2)
        self.assertEqual(empty.issues[0]["reason"], "no_eligible_component")

    def test_no_forced_labels_or_question_driven_resampling(self):
        prepared = prepare(self.config, fixture_loader(nx.complete_graph(30)))
        self.assertEqual(len(prepared.graphs), 2)
        self.assertFalse(prepared.examples)
        self.assertEqual(len(prepared.issues), 4)
        self.assertTrue(all(g["sampling_attempts"] == 1 for g in prepared.graphs))
        self.assertEqual(prepared.summary()["question_coverage"], 0)

    def test_missing_strata_remain_visible_with_null_accuracy(self):
        prepared = prepare(self.config, fixture_loader(nx.complete_graph(30)))
        summary = summarize([], expected_calls=0, preparation=prepared.summary())
        self.assertEqual(len(summary["strata"]), 2)
        for cell in summary["strata"]:
            self.assertIsNone(cell["accuracy"])
            self.assertEqual(cell["attempts"], 0)
            self.assertEqual(cell["generation"]["planned_questions"], 4)
            self.assertEqual(cell["generation"]["missing_questions"], 4)
            self.assertEqual(cell["generation"]["question_coverage"], 0)

    def test_generation_coverage_accounts_for_missing_graphs(self):
        prepared = prepare(self.config, fixture_loader(nx.path_graph(8)))
        for cell in prepared.summary()["strata"]:
            self.assertEqual(cell["planned_graphs"], 2)
            self.assertEqual(cell["generated_graphs"], 1)
            self.assertEqual(cell["generated_questions"], 2)
            self.assertEqual(cell["missing_questions"], 2)
            self.assertEqual(cell["question_coverage"], 0.5)

    def test_multiple_thresholds_have_separate_balanced_quotas(self):
        config = replace(self.config, tasks=TaskConfig(0, 4, (2, 3)))
        result = prepare(config, path_loader())
        self.assertEqual(len(result.examples), 16)
        for graph in result.graphs:
            for threshold in (2, 3):
                labels = [e.answer for e in result.examples if e.graph_id == graph["graph_id"] and e.threshold == threshold]
                self.assertEqual(sorted(labels), ["no", "no", "yes", "yes"])

    def test_verifier_rejects_wrong_labels_and_metadata(self):
        example = prepare(self.config, path_loader()).examples[0]
        for changed in (replace(example, answer="no" if example.answer == "yes" else "yes"),
                        replace(example, distance=999), replace(example, edge_count=999),
                        replace(example, request=replace(example.request, state={"nodes": [], "edges": [], "answer": "yes"}))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                verify_examples([changed])


class FakeClient(BaseDecisionClient):
    """No CLI exposure: fixture accuracy must never be mistaken for Jev results."""

    def __init__(self, behavior="yes"):
        self.behavior = behavior
        self.calls = 0
        self.closed = False

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        self.calls += 1
        if self.behavior == "http" and self.calls == 1:
            raise ProviderHTTPError(503)
        if self.behavior == "timeout" and self.calls == 1:
            await asyncio.Future()
        if self.behavior == "bug" and self.calls == 2:
            raise RuntimeError("PRIVATE_ERROR_TEXT")
        if self.behavior == "cancel" and self.calls == 1:
            raise asyncio.CancelledError()
        return DecisionResponse(request.request_id, "invalid" if self.behavior == "invalid" else "yes",
                                "offline-test-only", probabilities={"yes": 0.75, "no": 0.25},
                                usage=TokenUsage(input_tokens=10))

    async def aclose(self):
        self.closed = True


class JevYamlCliTests(unittest.TestCase):
    """Exercise YAML -> CLI -> real registry/adapter -> mocked HTTP -> metrics."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "jev.yaml"
        # Keep the small CLI fixture independent of user-facing run templates.
        data = config_for(self.root).snapshot()
        data["data"]["datasets"] = ["power"]
        data["sampling"]["node_counts"] = [16]
        data["model"]["model"] = "jev-offline-requested"
        self.config_path.write_text(yaml.safe_dump(data))
        self.output = self.root / "results"
        # Only replace raw-data access; keep the CLI, config, registry and adapter.
        preparation = patch("src.benchmark.runner.prepare", side_effect=lambda config, loader: prepare(config, path_loader()))
        preparation.start()
        self.addCleanup(preparation.stop)

    def args(self):
        return ["--config", str(self.config_path), "--output", str(self.output)]

    def test_yaml_cli_executes_jev_through_real_factory(self):
        bodies = []

        def handler(request):
            self.assertEqual(str(request.url), "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(request.headers["authorization"], "Bearer FAKE_YAML_CLI_KEY")
            body = json.loads(request.content)
            bodies.append(body)
            self.assertEqual(body["model"], "jev-offline-requested")
            self.assertEqual(set(body["state"]), {"nodes", "edges"})
            self.assertEqual(body["state"]["nodes"], list(range(16)))
            self.assertEqual(set(body["questions"]["decision"]["criteria"]), {"yes", "no"})
            for forbidden in ("PRIVATE", "power", "original_ids", "answer", "distance_threshold"):
                self.assertNotIn(forbidden, json.dumps(body))
            return httpx.Response(200, json={"model": "jev-offline-resolved", "answers": {
                "decision": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.8, "no": 0.2}, "confidence": 0.6}
            }, "usage": {"input_tokens": 100, "output_tokens": 1}})

        original_http_client = httpx.AsyncClient

        def mock_http_client(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return original_http_client(**kwargs)

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "FAKE_YAML_CLI_KEY"}), \
             patch("src.clients.typesafe.httpx.AsyncClient", side_effect=mock_http_client), \
                         patch("sys.stderr", new_callable=io.StringIO) as stderr, \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            main(self.args())
        self.assertEqual(len(bodies), 8)
        self.assertEqual(json.loads(stdout.getvalue())["overall"]["attempts"], 8)
        self.assertIn("Preparing", stderr.getvalue())
        self.assertIn("8/8", stderr.getvalue())
        self.assertIn("accuracy=50.00%", stderr.getvalue())
        self.assertIn("Run completed", stderr.getvalue())
        self.assertNotIn("FAKE_YAML_CLI_KEY", stderr.getvalue())
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["run_status"], "completed")
        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertEqual(summary["overall"]["resolved_models"], {"jev-offline-resolved": 8})
        self.assertGreater(summary["overall"]["latency_all"]["total_seconds"], 0)
        run = json.loads((self.output / "run.json").read_text())
        self.assertFalse(run["injected_client"])
        self.assertEqual(run["requested_model"], "jev-offline-requested")
        for path in self.output.iterdir():
            self.assertNotIn("FAKE_YAML_CLI_KEY", path.read_text())

    def test_no_progress_flag_disables_display(self):
        with patch("src.benchmark.runner.create_client", return_value=FakeClient()), \
             patch("sys.stderr", new_callable=io.StringIO) as stderr, \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            main(self.args() + ["--no-progress"])
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue())["completed_calls"], 8)

    def test_removed_execute_flag_is_rejected(self):
        with patch("sys.stderr", new_callable=io.StringIO), \
             patch("src.benchmark.__main__.run_experiment") as run:
            with self.assertRaises(SystemExit) as caught:
                main(self.args() + ["--execute"])
        self.assertEqual(caught.exception.code, 2)
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_yaml_cli_live_without_key_fails_before_http(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("sys.stderr", new_callable=io.StringIO) as stderr, \
             patch("src.clients.typesafe.httpx.AsyncClient", side_effect=AssertionError("No HTTP")):
            with self.assertRaisesRegex(ValueError, "TYPESAFE_API_KEY"):
                main(self.args())
        run = json.loads((self.output / "run.json").read_text())
        self.assertEqual(run["status"], "aborted")
        self.assertEqual(run["completed_calls"], 0)
        self.assertIn("Run aborted: 0 attempts, accuracy=n/a", stderr.getvalue())


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = config_for(Path(self.tmp.name))

    def read(self, name):
        return json.loads((self.config.run.output_dir / name).read_text())

    async def test_json_artifacts_are_indented_but_jsonl_stays_one_record_per_line(self):
        await run_experiment(self.config, client=FakeClient(), loader=path_loader())
        output = self.config.run.output_dir
        for path in output.glob("*.json"):
            with self.subTest(artifact=path.name):
                text = path.read_text(encoding="utf-8")
                expected = json.dumps(json.loads(text), ensure_ascii=False, allow_nan=False,
                                      sort_keys=True, indent=2) + "\n"
                self.assertEqual(text, expected)
                self.assertGreater(len(text.splitlines()), 1)
        counts = {"requests.jsonl": 8, "labels.jsonl": 8, "graphs.jsonl": 2, "attempts.jsonl": 8}
        for name, count in counts.items():
            with self.subTest(artifact=name):
                lines = (output / name).read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), count)
                self.assertTrue(all(isinstance(json.loads(line), dict) for line in lines))

    async def test_progress_uses_generated_calls_times_repetitions_and_counts_failures(self):
        config = replace(self.config, run=replace(self.config.run, repetitions=2))
        # Only one distinct subgraph is available, so planned calls exceed actual calls.
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            result = await run_experiment(config, client=FakeClient("http"),
                                          loader=fixture_loader(nx.path_graph(8)), progress=True)
        text = stderr.getvalue()
        self.assertEqual(result["expected_calls"], 8)
        self.assertIn("Prepared 1 graphs, 4 questions x 2 repetitions.", text)
        self.assertIn("8/8", text)
        metrics = result["overall"]
        self.assertIn(f"accuracy={metrics['accuracy']:.2%}, failures=1", text)
        self.assertIn("Run completed: 8 attempts", text)

    async def test_progress_retains_partial_counts_on_abort_or_cancellation(self):
        for behavior, error, attempts in (("bug", RuntimeError, 2),
                                           ("cancel", asyncio.CancelledError, 1)):
            with self.subTest(behavior=behavior):
                config = replace(self.config, run=replace(
                    self.config.run, output_dir=self.config.run.output_dir / behavior))
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(error):
                        await run_experiment(config, client=FakeClient(behavior),
                                             loader=path_loader(), progress=True)
                text = stderr.getvalue()
                self.assertIn(f"{attempts}/8", text)
                self.assertIn(f"Run aborted: {attempts} attempts", text)
                self.assertIn("failures=1", text)
                self.assertNotIn("PRIVATE_ERROR_TEXT", text)
                self.assertNotIn("100%", text)

    async def test_progress_handles_preparation_failure_without_fake_completion(self):
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaisesRegex(ValueError, "No questions"):
                await run_experiment(self.config, client=FakeClient(),
                                     loader=fixture_loader(nx.complete_graph(30)), progress=True)
        self.assertIn("Preparing", stderr.getvalue())
        self.assertIn("Run aborted: 0 attempts, accuracy=n/a", stderr.getvalue())
        self.assertNotIn("Evaluating", stderr.getvalue())

    async def test_default_runs_client_and_exports_separate_requests_and_labels(self):
        client = FakeClient()
        with patch("src.benchmark.runner.create_client", return_value=client) as factory, \
             patch("sys.stderr", new_callable=io.StringIO) as stderr:
            result = await run_experiment(self.config, loader=path_loader())
        self.assertEqual(stderr.getvalue(), "")
        factory.assert_called_once_with("typesafe", model="test-only-model", timeout_seconds=1.0)
        self.assertEqual(client.calls, 8)
        self.assertTrue(client.closed)
        self.assertEqual(self.read("run.json")["status"], "completed")
        self.assertEqual(result["expected_calls"], 8)
        self.assertTrue((self.config.run.output_dir / "summary.json").exists())
        self.assertTrue((self.config.run.output_dir / "attempts.jsonl").exists())
        self.assertEqual(len(self.read("run.json")["artifact_sha256"]), 5)
        requests = (self.config.run.output_dir / "requests.jsonl").read_text()
        self.assertNotIn('"answer"', requests)
        self.assertNotIn("PRIVATE", requests)

    async def test_execute_repeats_accuracy_and_latency(self):
        self.config = replace(self.config, run=replace(self.config.run, repetitions=2))
        client = FakeClient()
        await run_experiment(self.config, client=client, loader=path_loader())
        result = self.read("summary.json")
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(client.calls, 16)
        self.assertTrue(client.closed)
        self.assertEqual(result["overall"]["accuracy"], 0.5)
        self.assertEqual(result["overall"]["unique_questions"], 8)
        self.assertEqual(result["overall"]["unique_graphs"], 2)
        self.assertEqual(result["overall"]["token_usage"]["input_tokens"]["known_sum"], 160)
        self.assertIsNone(result["overall"]["token_usage"]["output_tokens"]["known_sum"])
        self.assertEqual(result["overall"]["brier_yes"], 0.3125)
        self.assertEqual(result["overall"]["latency_all"]["count"], 16)
        self.assertGreater(result["execution_seconds"], 0)
        self.assertIsNone(result["overall"]["cost"])
        self.assertEqual(set(result["by_repetition"]), {"0", "1"})
        self.assertEqual(result["by_dataset"]["facebook"]["attempts"], 16)
        self.assertEqual(result["by_node_count"]["8"]["attempts"], 16)
        self.assertEqual(result["by_threshold"]["2"]["attempts"], 8)

    async def test_http_failure_counted_without_retry(self):
        client = FakeClient("http")
        await run_experiment(self.config, client=client, loader=path_loader())
        result = self.read("summary.json")["overall"]
        self.assertEqual(client.calls, 8)
        self.assertEqual(result["attempts"], 8)
        self.assertEqual(result["successes"], 7)
        self.assertEqual(result["failure_rate"], 1 / 8)
        self.assertEqual(result["accuracy"], result["correct"] / 8)
        self.assertEqual(result["accuracy_on_success"], result["correct"] / 7)

    async def test_total_timeout_and_continued_execution(self):
        self.config = replace(self.config, model=replace(self.config.model, timeout_seconds=0.01))
        await run_experiment(self.config, client=FakeClient("timeout"), loader=path_loader())
        self.assertEqual(self.read("summary.json")["overall"]["error_counts"], {"TimeoutError": 1})

    async def test_invalid_responses_count_as_failures(self):
        await run_experiment(self.config, client=FakeClient("invalid"), loader=path_loader())
        metrics = self.read("summary.json")["overall"]
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["failures"], 8)

    async def test_unexpected_error_aborts_with_partial_logs_not_secret_text(self):
        client = FakeClient("bug")
        with self.assertRaises(RuntimeError):
            await run_experiment(self.config, client=client, loader=path_loader())
        self.assertTrue(client.closed)
        self.assertEqual(self.read("run.json")["status"], "aborted")
        self.assertEqual(self.read("summary.json")["completed_calls"], 2)
        self.assertEqual(self.read("summary.json")["execution_coverage"], 2 / 8)
        for path in self.config.run.output_dir.iterdir():
            self.assertNotIn("PRIVATE_ERROR_TEXT", path.read_text())

    async def test_cancellation_retains_inflight_attempt_and_propagates(self):
        client = FakeClient("cancel")
        with self.assertRaises(asyncio.CancelledError):
            await run_experiment(self.config, client=client, loader=path_loader())
        self.assertTrue(client.closed)
        self.assertEqual(self.read("run.json")["status"], "aborted")
        self.assertEqual(self.read("summary.json")["completed_calls"], 1)
        row = json.loads((self.config.run.output_dir / "attempts.jsonl").read_text())
        self.assertEqual(row["status"], "aborted")
        self.assertEqual(row["error_type"], "CancelledError")

    async def test_output_directory_never_overwritten(self):
        await run_experiment(self.config, client=FakeClient(), loader=path_loader())
        before = (self.config.run.output_dir / "run.json").read_bytes()
        with patch("src.benchmark.runner.create_client", side_effect=AssertionError("No client")), self.assertRaises(FileExistsError):
            await run_experiment(self.config, loader=path_loader())
        self.assertEqual(before, (self.config.run.output_dir / "run.json").read_bytes())

    async def test_call_cap_precedes_files_and_clients(self):
        config = replace(self.config, run=replace(self.config.run, max_calls=1))
        with self.assertRaises(ValueError):
            await run_experiment(config, loader=path_loader())
        self.assertFalse(config.run.output_dir.exists())

    async def test_no_questions_prevents_client_creation(self):
        with patch("src.benchmark.runner.create_client", side_effect=AssertionError("No client")):
            with self.assertRaisesRegex(ValueError, "No questions"):
                await run_experiment(self.config, loader=fixture_loader(nx.complete_graph(30)))
        self.assertEqual(self.read("run.json")["status"], "aborted")

    async def test_end_to_end_typesafe_wire_contains_only_anonymous_graph(self):
        bodies = []

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            self.assertEqual(set(body), {"model", "state", "questions"})
            self.assertEqual(set(body["state"]), {"nodes", "edges"})
            self.assertEqual(body["state"]["nodes"], list(range(8)))
            wire = json.dumps(body)
            for forbidden in ("PRIVATE", "facebook", "raw_sha256", "answer", "graph_id", "distance_threshold", "10000"):
                self.assertNotIn(forbidden, wire)
            return httpx.Response(200, json={"model": "offline-wire-test", "answers": {
                "decision": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.75, "no": 0.25}, "confidence": 0.5}
            }, "usage": {}})

        client = TypeSafeClient(api_key="FAKE_TEST_KEY", transport=httpx.MockTransport(handler))
        await run_experiment(self.config, client=client, loader=path_loader())
        self.assertEqual(len(bodies), 8)
        self.assertEqual(self.read("summary.json")["overall"]["accuracy"], 0.5)
        for path in self.config.run.output_dir.iterdir():
            self.assertNotIn("FAKE_TEST_KEY", path.read_text())

    def test_percentiles(self):
        self.assertIsNone(_percentile([], 0.95))
        self.assertEqual(_percentile([4], 0.95), 4)
        self.assertEqual(_percentile([0, 10], 0.5), 5)
        self.assertEqual(_percentile([0, 10], 0.95), 9.5)


if __name__ == "__main__":
    unittest.main()