"""Offline fake-client tests; all scratch artifacts stay under the project."""
import asyncio
from pathlib import Path
import shutil
from unittest import TestCase
from uuid import uuid4

from scripts import run_paper_gpt_anchor as anchor
from src.clients.base import ClientTimeoutError, ProviderHTTPError
from test_extended_graph_suite import FakeClient, small_instances


class GPTAnchorTests(TestCase):
    def setUp(self):
        self.base = anchor.REPO / "output" / ("gpt-anchor-tests-" + uuid4().hex)
        self.bundle = self.base / "bundle"
        self.output = self.base / "run"
        self.bundle.mkdir(parents=True)
        instances = small_instances()
        for name, rows in (
            ("public/development.jsonl", [{k: v for k, v in i.items() if k != "private"}
                                          for i in instances]),
            ("references/development.jsonl", [{"id": i["id"], "private": i["private"]}
                                              for i in instances]),
        ):
            path = self.bundle / name
            path.parent.mkdir()
            with path.open("x") as stream:
                for row in rows:
                    anchor.suite.append_row(stream, row)
        (self.bundle / "analysis").mkdir()
        anchor.suite.write_json(self.bundle / "analysis/development.json",
                                {"cases": {i["id"]: {} for i in instances}})
        manifest = {"schema_version": 1, "protocols": ["explicit_actions"],
                    "splits": {"development": {
                        "validation": anchor.evaluation.tasks.validate_instances(instances)}},
                    "sha256": {str(p.relative_to(self.bundle)): anchor.suite.digest(p)
                               for p in self.bundle.rglob("*") if p.is_file()}}
        anchor.suite.write_json(self.bundle / "manifest.json", manifest)
        anchor.suite.write_json(self.bundle / "manifest.sha256",
                                anchor.suite.digest(self.bundle / "manifest.json"))
        anchor.register(self.bundle, self.output, test_only=True, split="development")

    def tearDown(self):
        shutil.rmtree(self.base)

    def run_fake(self, **kwargs):
        client = FakeClient("gpt-5.4", **kwargs)
        factories = []
        def factory(provider, **settings):
            factories.append((provider, settings["model"]))
            return client
        result = asyncio.run(anchor.execute(self.output, client_factory=factory))
        self.assertTrue(client.closed)
        self.assertEqual(factories, [("github_copilot", "gpt-5.4")])
        return client, result

    def test_only_standard_gpt54_resource_is_requested(self):
        from contextlib import contextmanager
        ownership = []
        @contextmanager
        def fake_shared_locks(models):
            ownership.append(("acquire", models))
            yield
            ownership.append(("release", models))
        with anchor.resource_locks(lock_factory=fake_shared_locks):
            self.assertEqual(ownership, [("acquire", ("gpt54",))])
        self.assertEqual(ownership, [
            ("acquire", ("gpt54",)), ("release", ("gpt54",))])
        manifest = anchor.suite.read_json(self.output / "manifest.json")
        self.assertEqual(manifest["resource_models"], ["gpt54"])
        self.assertNotIn("serialize both", manifest["policy"]["shared_cli_limit"])

    def test_snapshot_scope_excludes_unrelated_helpers(self):
        paths = anchor.sources()
        self.assertIn("scripts/evaluate_graph_release.py", paths)
        self.assertIn("src/clients/github_copilot_client.py", paths)
        self.assertNotIn("scripts/construction_prefix_controls.py", paths)

    def test_resource_amendment_pins_original_and_rejects_dependency_changes(self):
        from copy import deepcopy
        from unittest.mock import patch
        original = anchor.suite.read_json(self.output / "manifest.json")
        prior = self.base / "fake-v2"
        prior.mkdir()
        original["resource_models"] = ["gpt54", "gpt6astra"]
        original["policy"]["shared_cli_limit"] = "serialize both GPT lanes for full runtime lifetime"
        anchor.suite.write_json(prior / "manifest.json", original)
        digest = anchor.suite.digest(prior / "manifest.json")
        anchor.suite.write_json(prior / "manifest.sha256", digest)
        anchor.suite.write_json(prior / "status.json", {"status": "waiting", "calls": 0})
        current = anchor.suite.read_json(self.output / "manifest.json")
        with patch.object(anchor, "V2_MANIFEST_SHA256", digest):
            amendment = anchor.resource_amendment(prior, current)
            self.assertEqual(amendment["prior_manifest_sha256"], digest)
            self.assertEqual(amendment["new_resource_models"], ["gpt54"])
            self.assertEqual(amendment["prior_calls"], 0)
            changed = deepcopy(current)
            changed["code_sha256"]["scripts/evaluate_graph_release.py"] = "bad"
            with self.assertRaises(anchor.suite.Refusal):
                anchor.resource_amendment(prior, changed)
            (prior / "attempts.jsonl").touch()
            with self.assertRaises(anchor.suite.Refusal):
                anchor.resource_amendment(prior, current)

    def test_success_own_history_metadata_seal(self):
        client, result = self.run_fake()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["calls"], 7)
        self.assertTrue(client.initialized)
        self.assertEqual(anchor.verify(self.output)["status"], "verified")
        manifest = anchor.suite.read_json(self.output / "manifest.json")
        self.assertTrue(manifest["test_only"])
        self.assertTrue(manifest["injected_client"])
        self.assertIsNone(manifest["model"]["seed"])
        self.assertEqual(manifest["model"]["max_tokens"], 4096)
        text = (self.output / "attempts.jsonl").read_text()
        self.assertNotIn("MUST_NOT_APPEAR", text)
        self.assertNotIn('"private"', text)
        attempts = anchor.suite.read_rows(self.output / "attempts.jsonl")
        self.assertEqual([r["step"] for r in attempts[:4]], [0, 1, 2, 3])
        self.assertTrue(all(o["description"] for r in attempts for o in r["request"]["options"]))
        summary = anchor.suite.read_json(self.output / "summary.json")
        self.assertEqual(summary["usage"]["input_tokens"]["known_total"], 140)
        self.assertEqual(summary["usage"]["reasoning_tokens"]["unknown_calls"], 7)

    def test_timeout_no_retry_ends_episode(self):
        _, result = self.run_fake(behavior=lambda request, call:
                                 ClientTimeoutError("timeout") if call == 1 else None)
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["calls"], 4)
        rows = anchor.suite.read_rows(self.output / "episodes.jsonl")
        self.assertEqual(rows[0]["query_count"], 1)
        self.assertEqual(rows[0]["failure_count"], 1)
        anchor.verify(self.output)

    def test_invalid_mapping_and_model_rejected(self):
        def behavior(request, call):
            from src.clients.base import DecisionResponse, TokenUsage
            return DecisionResponse(request.request_id, "UNKNOWN", "gpt-5.4",
                                    usage=TokenUsage(1, 1))
        _, result = self.run_fake(behavior=behavior)
        self.assertEqual(result["status"], "completed_with_failures")
        attempts = anchor.suite.read_rows(self.output / "attempts.jsonl")
        self.assertTrue(all(r["diagnostic_code"] == "invalid_choice" for r in attempts))
        anchor.verify(self.output)

    def test_fatal_preserves_denominator(self):
        _, result = self.run_fake(behavior=lambda *_: ProviderHTTPError(401))
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["calls"], 1)
        rows = anchor.suite.read_rows(self.output / "episodes.jsonl")
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r["query_count"] == 0 for r in rows), 3)
        anchor.verify(self.output)

    def test_quota_error_never_retries_or_falls_back(self):
        client, result = self.run_fake(behavior=lambda *_: ProviderHTTPError(429))
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(len(client.requests), 1)
        attempts = anchor.suite.read_rows(self.output / "attempts.jsonl")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["http_status"], 429)
        self.assertTrue(attempts[0]["fatal"])
        anchor.verify(self.output)

    def test_initialization_and_shutdown_failures(self):
        _, result = self.run_fake(initialize_error=ProviderHTTPError(401),
                                  close_error=RuntimeError("secret"))
        self.assertEqual(result["calls"], 0)
        self.assertEqual(result["diagnostic_code"], "shutdown_failed")
        events = anchor.suite.read_rows(self.output / "events.jsonl")
        self.assertEqual([e["phase"] for e in events], ["initialization", "shutdown"])
        self.assertNotIn("secret", (self.output / "events.jsonl").read_text())
        anchor.verify(self.output)

    def test_cancel_attempt_is_explicit_and_sealed(self):
        client = FakeClient("gpt-5.4", behavior=lambda *_: asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(anchor.execute(self.output, client_factory=lambda *_a, **_k: client))
        self.assertTrue(client.closed)
        self.assertEqual(anchor.verify(self.output)["run_status"], "interrupted")
        attempts = anchor.suite.read_rows(self.output / "attempts.jsonl")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["diagnostic_code"], "interrupted")

    def test_repeated_cancel_does_not_abandon_cleanup(self):
        async def exercise():
            closing = asyncio.Event()
            released = asyncio.Event()

            class SlowClose(FakeClient):
                async def aclose(self):
                    closing.set()
                    await released.wait()
                    self.closed = True

            client = SlowClose("gpt-5.4", behavior=lambda *_: asyncio.CancelledError())
            task = asyncio.create_task(anchor.execute(
                self.output, client_factory=lambda *_a, **_k: client))
            await closing.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            released.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(client.closed)
        asyncio.run(exercise())
        self.assertEqual(anchor.verify(self.output)["run_status"], "interrupted")

    def test_replay_rejects_metadata_and_history_tampering(self):
        self.run_fake()
        manifest, instances, config = anchor.load(self.output)
        original = anchor.suite.read_rows(self.output / "attempts.jsonl")
        from copy import deepcopy
        for field, value in (("latency_seconds", -1), ("usage", {"input_tokens": -1}),
                             ("resolved_model", "other"), ("step", 99)):
            rows = deepcopy(original)
            rows[0][field] = value
            with self.assertRaises(anchor.suite.Refusal):
                anchor.replay(instances, rows, config)
        with (self.output / "attempts.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(anchor.suite.Refusal):
            anchor.verify(self.output)

    def test_registration_exclusive_and_fake_marker_enforced(self):
        with self.assertRaises(anchor.suite.Refusal):
            anchor.register(self.bundle, self.output, test_only=True, split="development")
        with self.assertRaises(anchor.suite.Refusal):
            asyncio.run(anchor.execute(self.output))
        self.assertFalse((self.output / "attempts.jsonl").exists())

    def test_production_rejects_small_fixture(self):
        with self.assertRaises(anchor.suite.Refusal):
            anchor.register(self.bundle, self.base / "production", split="development")
