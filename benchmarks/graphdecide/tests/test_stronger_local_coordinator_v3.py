import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from scripts import stronger_local_coordinator_v3 as c
from scripts import stronger_local_replay_v3 as replay
from tests.test_stronger_local_execution_v1 import ProjectFixture


class IPCAndStorageTests(ProjectFixture):
    def setUp(self):
        super().setUp()
        self.ipc = c.ipc_directory(self.root)
        self.addCleanup(lambda: shutil.rmtree(self.ipc) if self.ipc.exists() else None)

    def test_actual_zmq_bind_with_uuid_under_sockaddr_limit(self):
        result = c.ipc_probe(self.root)
        self.assertTrue(result["bound"])
        self.assertLess(result["path_bytes"], result["ipc_limit"])
        self.assertFalse(Path(result["path"]).exists())
        self.assertTrue(result["parent_tmp_environment_unchanged"])

    def test_child_tmpdir_short_and_parent_namespace_unchanged(self):
        before = dict(os.environ)
        environment = c.child_environment(self.root, CUDA_VISIBLE_DEVICES="")
        self.assertEqual(environment["TMPDIR"], str(self.ipc))
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertLess(c.validate_ipc_length(Path(environment["TMPDIR"])), 107)

    def test_v2_recorded_path_shape_is_rejected(self):
        long = c.REPO / "output/stronger-local-execution-20260926-v2/work"
        with self.assertRaisesRegex(c.suite.shared.Refusal, "IPC socket path"):
            c.validate_ipc_length(long)

    def test_existing_scratch_must_belong_to_this_execution(self):
        self.ipc.mkdir()
        c.suite.write_json(self.ipc / "owner.json", {"execution_root": "another"})
        with self.assertRaisesRegex(c.suite.shared.Refusal, "another execution"):
            c.prepare_ipc_directory(self.root)

    def test_existing_200gb_payload_not_charged_as_new_bytes(self):
        artifacts = {lane: {"files": {"shard": {"bytes": 100_000_000_000}}}
                     for lane in c.run.SPECS}
        result = c.reuse_storage(89_000_000_000, artifacts)
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["new_checkpoint_bytes_required"], 0)
        self.assertEqual(result["verified_existing_artifact_bytes"], 200_000_000_000)

    def test_reuse_still_requires_disk_headroom_and_both_models(self):
        artifacts = {lane: {"files": {"shard": {"bytes": 1}}} for lane in c.run.SPECS}
        with self.assertRaisesRegex(c.suite.shared.Refusal, "headroom"):
            c.reuse_storage(9, artifacts, headroom=10)
        self.assertTrue(c.reuse_storage(10, artifacts, headroom=10)["sufficient"])
        with self.assertRaisesRegex(c.suite.shared.Refusal, "Both locally verified"):
            c.reuse_storage(100, {}, headroom=10)

    def test_verified_reuse_hashes_each_checkpoint_before_storage_credit(self):
        with patch.object(c.run, "verify_artifacts", side_effect=ValueError("hash drift")) as verify, \
                patch.object(c, "reuse_storage") as storage:
            with self.assertRaisesRegex(ValueError, "hash drift"):
                c.verified_reuse(self.root)
        verify.assert_called_once()
        storage.assert_not_called()


class StandaloneReplayTests(ProjectFixture):
    def test_real_subprocess_replays_preserved_zero_attempt_960(self):
        source = c.v2.ROOT / "qwen38_27b_bf16"
        self.assertEqual((source / "run/attempts.jsonl").stat().st_size, 0)
        before = c.suite.digest(source / "run/seal.json")
        output = self.root / "recovered"
        result = subprocess.run(
            c.replay_command(source, output), cwd=c.REPO, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual((report["episodes"], report["calls"]), (960, 0))
        self.assertEqual(report["provider_modules_imported"], [])
        self.assertEqual(set(report["bundled_contract_modules"]), set(replay.CONTRACTS))
        self.assertEqual(sum(m["unattempted"] for m in report["by_task"].values()), 960)
        self.assertIsNone(report["by_task"]["tsp_construct"]["mean_absolute_gap_feasible"])
        self.assertEqual(c.suite.digest(source / "run/seal.json"), before)

    def test_provider_guard_rejects_backend_import_in_actual_subprocess(self):
        code = (
            "import sys; from scripts.stronger_local_replay_v3 import NoProviders;"
            "sys.meta_path.insert(0,NoProviders()); import src.clients.vllm_client")
        result = subprocess.run(
            [c.runtime.PYTHON, "-B", "-c", code], cwd=c.REPO,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Provider import prohibited", result.stderr)

    def test_actual_provider_names_remain_prohibited(self):
        for name in ("src.clients.vllm_client", "src.clients.registry", "src.clients.typesafe",
                     "vllm", "transformers", "httpx", "openai", "huggingface_hub"):
            self.assertTrue(replay.is_provider(name), name)
        self.assertFalse(replay.is_provider("src.clients"))
        self.assertFalse(replay.is_provider("src.clients.base"))

    def test_contract_exception_requires_exact_bundled_origin(self):
        modules = {}
        for name in replay.CONTRACTS:
            module = ModuleType(name)
            module.__file__ = str(c.REPO / "src/clients/base.py")
            modules[name] = module
        manifest = {"sha256": {relative: "incorrect" for relative in replay.CONTRACTS.values()}}
        with self.assertRaisesRegex(ValueError, "outside the sealed bundle"):
            replay.import_audit(c.run.base.BUNDLE, manifest, modules)


if __name__ == "__main__":
    unittest.main()
