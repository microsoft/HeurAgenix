"""Offline canonical Astra checks; no external calls or out-of-project scratch."""
import asyncio
from dataclasses import asdict
import shutil
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

from scripts import run_canonical_astra as astra
from src.clients.base import ClientTimeoutError, ProviderHTTPError
from test_extended_graph_suite import FakeClient, small_instances


class CanonicalAstraTests(TestCase):
    def setUp(self):
        self.base = astra.REPO / "output" / ("astra-tests-" + uuid4().hex)
        self.bundle, self.output = self.base / "bundle", self.base / "run"
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
                    astra.suite.append_row(stream, row)
        (self.bundle / "analysis").mkdir()
        astra.suite.write_json(self.bundle / "analysis/development.json",
                              {"cases": {i["id"]: {} for i in instances}})
        manifest = {"schema_version": 1, "protocols": ["explicit_actions"],
                    "splits": {"development": {
                        "validation": astra.evaluation.tasks.validate_instances(instances)}},
                    "sha256": {str(p.relative_to(self.bundle)): astra.suite.digest(p)
                               for p in self.bundle.rglob("*") if p.is_file()}}
        astra.suite.write_json(self.bundle / "manifest.json", manifest)
        astra.suite.write_json(self.bundle / "manifest.sha256",
                              astra.suite.digest(self.bundle / "manifest.json"))
        astra.register(self.bundle, self.output, test_only=True, split="development")

    def tearDown(self):
        shutil.rmtree(self.base)

    def run_fake(self, **kwargs):
        client = FakeClient("gpt-6-astra", **kwargs)
        result = asyncio.run(astra.execute(
            self.output, client_factory=lambda *_a, **_k: client))
        self.assertTrue(client.closed)
        astra.verify(self.output)
        return client, result

    def test_protocol_config_and_reference_untouched(self):
        self.assertEqual(astra.LOCK_MODELS, ("gpt6astra",))
        self.assertEqual(astra.reference.LOCK_MODELS, ("gpt54",))
        expected = asdict(astra.reference.configuration())
        expected["model"] = "gpt-6-astra"
        self.assertEqual(asdict(astra.configuration()), expected)
        self.assertEqual(astra.TOTAL_SECONDS, 172800)
        self.assertEqual(astra.core.resource_locks(
            lock_factory=lambda models: models), ("gpt6astra",))

    def test_success_own_history_and_no_sensitive_bodies(self):
        _, result = self.run_fake()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["calls"], 7)
        rows = astra.suite.read_rows(self.output / "attempts.jsonl")
        self.assertEqual([r["step"] for r in rows[:4]], [0, 1, 2, 3])
        text = (self.output / "attempts.jsonl").read_text()
        self.assertNotIn("MUST_NOT_APPEAR", text)
        self.assertNotIn('"private"', text)
        self.assertNotIn('"reasoning"', text)
        self.assertFalse(astra.suite.read_json(self.output / "manifest.json")[
            "provider_sampling_seed_supported"])

    def test_timeout_no_retry(self):
        _, result = self.run_fake(behavior=lambda request, call:
                                 ClientTimeoutError("timeout") if call == 1 else None)
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["calls"], 4)

    def test_fatal_no_fallback_and_full_denominator(self):
        client, result = self.run_fake(behavior=lambda *_: ProviderHTTPError(429))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result["status"], "aborted")
        rows = astra.suite.read_rows(self.output / "episodes.jsonl")
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(r["query_count"] == 0 for r in rows), 3)

    def test_initialization_failure_is_sealed(self):
        _, result = self.run_fake(initialize_error=ProviderHTTPError(401))
        self.assertEqual(result["calls"], 0)
        self.assertEqual(result["status"], "aborted")

    def test_interrupted_attempt_is_preserved(self):
        client = FakeClient("gpt-6-astra", behavior=lambda *_: asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(astra.execute(
                self.output, client_factory=lambda *_a, **_k: client))
        self.assertTrue(client.closed)
        checked = astra.verify(self.output)
        self.assertEqual(checked["calls"], 1)
        self.assertEqual(checked["run_status"], "interrupted")

    def test_registration_and_no_resume_guards(self):
        for name in ("config.json", "manifest.json", "manifest.sha256",
                     "artifact-manifest.json", "protocol-check.json"):
            self.assertEqual((self.output / name).stat().st_mode & 0o222, 0)
        with self.assertRaises(astra.suite.Refusal):
            astra.register(self.bundle, self.output, test_only=True, split="development")
        self.run_fake()
        with self.assertRaises(astra.suite.Refusal):
            asyncio.run(astra.execute(self.output, client_factory=lambda *_a, **_k: None))

    def test_expired_resource_wait_keeps_all_scheduled_instances(self):
        astra.finalize_unstarted(self.output, "wall_budget_exhausted")
        self.assertEqual(astra.verify(self.output)["calls"], 0)
        self.assertEqual(len(astra.suite.read_rows(self.output / "episodes.jsonl")), 4)

    def test_offline_recovery_preserves_raw_inflight_and_denominator(self):
        _, instances, _ = astra.load(self.output)
        request = {"instance_id": instances[0]["id"], "step": 0,
                   "request": astra.suite.json_value(asdict(
                       astra.evaluation.public_request(instances[0], [])))}
        with (self.output / "requests.jsonl").open("x") as stream:
            astra.suite.append_row(stream, request)
        (self.output / "attempts.jsonl").touch()
        digest = astra.suite.digest(self.output / "requests.jsonl")
        result = astra.offline_summary(self.output)
        self.assertEqual(result["unresolved_inflight_requests"], 1)
        self.assertEqual(result["calls"], 0)
        self.assertEqual(len(astra.suite.read_rows(
            self.output / "offline-finalized/episodes.jsonl")), 4)
        self.assertEqual(astra.suite.digest(self.output / "requests.jsonl"), digest)

    def test_offline_recovery_rejects_live_worker(self):
        import os
        astra.suite.write_json(self.output / "worker-process.json",
                              astra.process_identity(os.getpid()))
        with self.assertRaises(astra.suite.Refusal):
            astra.offline_summary(self.output)

    def test_worker_arms_budget_timer_with_cleanup_reserve(self):
        import time
        astra.suite.write_json(self.output / "dispatch-owner.json",
                              {"deadline_epoch": time.time() + astra.CLEANUP_SECONDS - 1})

        async def slow_execute(_):
            await asyncio.sleep(10)

        with patch.object(astra, "execute", slow_execute):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(astra.worker(self.output))

    def test_system_lock_authorization_is_explicit_and_parent_only(self):
        import sys
        for flags, expected in (([], False), (["--allow-system-shared-locks"], True)):
            with patch.object(sys, "argv", ["astra", "dispatch", "--output",
                                           str(self.output), *flags]):
                with patch.object(astra, "dispatch", return_value=0) as launch:
                    self.assertEqual(astra.main(), 0)
                    launch.assert_called_once_with(
                        self.output.resolve(), allow_system_shared_locks=expected)

    def test_extra_registration_artifact_tampering_is_rejected(self):
        path = self.output / "artifact-manifest.json"
        path.chmod(0o644)
        path.write_text("{}")
        with self.assertRaises(astra.suite.Refusal):
            astra.load(self.output)

    def test_full_reference_request_equality(self):
        bundle = astra.REPO / "output/graphbench-portable-release-20260924-v2"
        instances = astra.release.labeled_instances(
            bundle, "structural_challenge", astra.release.manifest_at(bundle))
        checked = astra.protocol_check(instances, astra.REPO / astra.REFERENCE)
        self.assertEqual(checked["equal_requests"], 960)
        self.assertEqual(checked["different_requests"], 0)
        changed = [dict(i) for i in instances]
        changed[0]["id"] += "-changed"
        with self.assertRaises(astra.suite.Refusal):
            astra.protocol_check(changed, astra.REPO / astra.REFERENCE)
