import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import signal
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

from scripts import stronger_local_coordinator_v1 as c
from scripts import stronger_local_replay_v1 as independent
from scripts import stronger_local_runtime_v1 as r
from src.clients.base import DecisionResponse, InvalidResponseError, ProviderHTTPError, TokenUsage


class FakeClient:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]

    async def initialize(self):
        pass

    async def aclose(self):
        pass

    async def predict(self, request):
        return DecisionResponse(
            request_id=request.request_id, selected_option_id=request.options[0].id,
            resolved_model=self.model, usage=TokenUsage(), raw_output={"synthetic_test": True})


class ProjectFixture(unittest.TestCase):
    def setUp(self):
        self.root = r.REPO / "output" / ("stronger-local-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.root))


class RuntimeTests(ProjectFixture):
    @classmethod
    def setUpClass(cls):
        cls.instances = r.release.labeled_instances(
            r.base.BUNDLE, r.base.SPLIT, r.release.manifest_at(r.base.BUNDLE))

    def execute(self, factory=FakeClient, lane="qwen38_27b_bf16"):
        (self.root / "registration.json").write_text("{}")
        self.registration = {"lane": lane, "model": asdict(r.configuration(lane))}
        with patch.object(r, "load", return_value=(self.registration, self.instances)), \
                patch.object(r, "check_sources"):
            return asyncio.run(r.run(self.root, client_factory=factory, preflight=AsyncMock()))

    def verify(self, **kwargs):
        with patch.object(r, "load", return_value=(self.registration, self.instances)):
            return r.verify(self.root, **kwargs)

    def test_both_configs_exact_budgets_no_reasoning(self):
        for lane in r.SPECS:
            config = r.configuration(lane)
            self.assertFalse(config.think)
            self.assertTrue(config.constrain_choices)
            self.assertEqual((config.max_tokens, config.timeout_seconds), (64, 180))
            self.assertEqual(config.output_format, "answer_only")
        r.base.check_instances(self.instances)

    def test_full_960_own_history_and_independent_replay(self):
        result = self.execute()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["calls"], 1920)
        with self.assertRaisesRegex(r.suite.shared.Refusal, "Injected"):
            self.verify()
        self.assertTrue(self.verify(allow_injected=True)["verified"])
        # Direct unit injection is never written as independent model evidence.
        api = Mock()
        api.tasks = r.canonical.tasks
        api.canonical_graph_request = r.canonical.prompts.canonical_graph_request
        audit = independent.replay(
            self.instances, r.suite.read_rows(self.root / "run/attempts.jsonl"),
            r.suite.read_rows(self.root / "run/scores.jsonl"), api,
            self.registration["model"]["model"])
        self.assertEqual(audit["episodes"], 960)
        self.assertEqual(audit["by_task"]["tsp_construct"]["gap_denominator"], 40)

    def test_fatal_failure_preserves_all_unattempted(self):
        class Fatal(FakeClient):
            async def predict(self, request):
                raise ProviderHTTPError(401)
        result = self.execute(Fatal, lane="qwen25_72b_bf16")
        self.assertEqual((result["status"], result["calls"]), ("aborted", 1))
        scores = r.suite.read_rows(self.root / "run/scores.jsonl")
        self.assertEqual(sum(row["query_count"] == 0 for row in scores), 959)

    def test_invalid_response_no_repair_or_retry(self):
        class Invalid(FakeClient):
            async def predict(self, request):
                raise InvalidResponseError("test", diagnostic_code="finish_length")
        result = self.execute(Invalid)
        self.assertEqual(result["calls"], 960)
        self.assertEqual(result["status"], "completed_with_failures")
        scores = r.suite.read_rows(self.root / "run/scores.jsonl")
        self.assertEqual(sum(s["feasible"] for s in scores), 0)
        api = Mock(tasks=r.canonical.tasks,
                   canonical_graph_request=r.canonical.prompts.canonical_graph_request)
        audit = independent.replay(
            self.instances, r.suite.read_rows(self.root / "run/attempts.jsonl"), scores,
            api, self.registration["model"]["model"])
        self.assertIsNone(audit["by_task"]["tsp_construct"]["mean_absolute_gap_feasible"])
        self.assertEqual(audit["by_task"]["tsp_construct"]["feasible_coverage"], 0)

    def test_cancellation_persists_inflight_failure(self):
        class Cancel(FakeClient):
            async def predict(self, request):
                raise asyncio.CancelledError()
        result = self.execute(Cancel)
        self.assertEqual((result["status"], result["calls"]), ("interrupted", 1))

    def test_deadline_is_bounded_and_explicit(self):
        class Slow(FakeClient):
            async def predict(self, request):
                await asyncio.sleep(10)
        with patch.object(r, "MAX_SECONDS", .03):
            result = self.execute(Slow)
        self.assertEqual(result["diagnostic_code"], "run_deadline")

    def test_initialization_failure_no_calls(self):
        class Init(FakeClient):
            async def initialize(self):
                raise RuntimeError("test")
        result = self.execute(Init)
        self.assertEqual(result["calls"], 0)
        self.assertEqual(len(r.suite.read_rows(self.root / "run/scores.jsonl")), 960)

    def test_seal_tampering_refused(self):
        class Init(FakeClient):
            async def initialize(self):
                raise RuntimeError("test")
        self.execute(Init)
        (self.root / "run/summary.json").write_text("{}")
        with self.assertRaisesRegex(r.suite.shared.Refusal, "seal"):
            self.verify(allow_injected=True)

    def test_existing_run_never_retried(self):
        (self.root / "run").mkdir()
        with self.assertRaisesRegex(r.suite.shared.Refusal, "retry/resume"):
            self.execute()

    def test_registered_unstarted_lane_preserves_960_without_model_calls(self):
        (self.root / "registration.json").write_text("{}")
        registration = {"lane": "qwen25_72b_bf16"}
        with patch.object(r, "load", return_value=(registration, self.instances)), \
                patch.object(r.runtime, "RecordingClient") as client:
            result = r.preserve_unattempted(self.root, "Previous lane aborted")
        client.assert_not_called()
        self.assertEqual(result["status"], "unattempted_due_coordinator_failure")
        scores = r.suite.read_rows(self.root / "run/scores.jsonl")
        self.assertEqual(len(scores), 960)
        self.assertTrue(all(row["query_count"] == 0 for row in scores))


class ArtifactTests(ProjectFixture):
    def test_mutable_process_ledger_is_atomic_and_keeps_all_identities(self):
        path = self.root / "process.json"
        r.atomic_json(path, {"known": [1]})
        r.atomic_json(path, {"known": [1, 2]})
        self.assertEqual(r.suite.read_json(path), {"known": [1, 2]})
        self.assertEqual(list(self.root.glob("*.writing-*")), [])

    def test_metadata_repeat_verifies_instead_of_overwriting(self):
        path = self.root / "metadata.json"
        r.stable_json(path, {"at": "first", "sha": "pinned"})
        r.stable_json(path, {"at": "later", "sha": "pinned"}, ignore=("at",))
        self.assertEqual(r.suite.read_json(path)["at"], "first")
        with self.assertRaisesRegex(r.suite.shared.Refusal, "no overwrite"):
            r.stable_json(path, {"at": "later", "sha": "changed"}, ignore=("at",))

    def test_local_existing_blob_checked_without_network(self):
        payload = b"test checkpoint"
        path = self.root / "config.json"
        path.write_bytes(payload)
        remote = {"rfilename": "config.json", "size": len(payload),
                  "blobId": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}
        with patch.object(r.urllib.request, "urlopen") as request:
            result = r.fetch_file(r.SPECS["qwen38_27b_bf16"], remote, self.root)
        request.assert_not_called()
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())

    def test_wrong_lfs_checksum_and_symlinks_rejected(self):
        payload = self.root / "model.safetensors"
        payload.write_bytes(b"123")
        with self.assertRaisesRegex(r.suite.shared.Refusal, "LFS SHA"):
            r.check_file(payload, {"size": 3, "lfs": {"sha256": "0" * 64}})
        link = self.root / "link"
        link.symlink_to(payload)
        with self.assertRaises(r.suite.shared.Refusal):
            r.check_file(link, {"size": 3})

    def test_remote_code_refused_before_network(self):
        with patch.object(r.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(r.suite.shared.Refusal, "Remote code"):
                r.fetch_file(r.SPECS["qwen38_27b_bf16"], {"rfilename": "evil.py"}, self.root)
        request.assert_not_called()

    def test_partial_download_requires_exact_range_continuation(self):
        partial = self.root / "model.safetensors.partial"
        partial.write_bytes(b"123")
        response = Mock(status=200, headers={})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(r.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(r.suite.shared.Refusal, "continuation"):
                r.fetch_file(r.SPECS["qwen38_27b_bf16"],
                             {"rfilename": "model.safetensors", "size": 10}, self.root)
        self.assertEqual(partial.read_bytes(), b"123")


class ResourceTests(ProjectFixture):
    def test_launch_access_gate_precedes_any_lock_or_process(self):
        with patch.object(c.suite, "lane_locks") as locks, \
                patch.object(c, "originals") as originals:
            with self.assertRaisesRegex(r.suite.shared.Refusal, "permitted access"):
                c.launch(self.root)
        locks.assert_not_called()
        originals.assert_not_called()

    def test_busy_standard_lock_wait_is_bounded_no_service_touch(self):
        lock = Mock()
        lock.__enter__ = Mock(side_effect=r.suite.shared.Refusal("busy"))
        with patch.object(c.suite, "lane_locks", return_value=lock), \
                patch.object(c, "snapshot_originals") as snapshot:
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                with c.acquire(self.root / "resources", wait_seconds=0):
                    self.fail("Busy lock admitted.")
        snapshot.assert_not_called()

    def test_unknown_occupant_never_stopped_for_restoration(self):
        services = {}
        for lane, (gpu, _, _) in c.LANES.items():
            services[lane] = {"pid": 100 + gpu, "start_ticks": "1"}
        lease = c.Lease(self.root, services, [f"GPU-{i}" for i in range(4)])
        inventory = {"processes": [["999", f"GPU-{i}", "10"] for i in range(4)]}
        with patch.object(c.runtime, "alive", return_value=False), \
                patch.object(c.runtime, "gpu_inventory", return_value=inventory), \
                patch.object(c.subprocess, "Popen") as popen, \
                patch.object(c.runtime, "stop_tree") as stop:
            with self.assertRaisesRegex(r.suite.shared.Refusal, "Restoration incomplete"):
                lease.restore()
        popen.assert_not_called()
        stop.assert_not_called()
        self.assertFalse(r.suite.read_json(self.root / "restoration.json")["original_services_restored"])

    def test_restore_all_four_exact_captured_settings(self):
        services, launched, current = {}, [], {}
        for lane, (gpu, port, _) in c.LANES.items():
            services[lane] = {
                "pid": 100 + gpu, "start_ticks": "1", "argv": ["original", str(port)],
                "cwd": str(r.REPO), "cuda_environment": {"CUDA_VISIBLE_DEVICES": str(gpu)},
                "restore_environment": {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "7"}}
            current[lane] = {**services[lane], "pid": 200 + gpu, "start_ticks": "2",
                             "process_tree": [{"pid": 200 + gpu}]}
        def popen(command, **kwargs):
            gpu = len(launched)
            self.assertEqual(command, services[list(c.LANES)[gpu]]["argv"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(gpu))
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "7")
            launched.append(gpu)
            return Mock(pid=200 + gpu)
        def inventory():
            return {"gpus": [[str(i), f"GPU-{i}"] for i in range(4)],
                    "processes": [[str(200 + i), f"GPU-{i}", "10"] for i in launched]}
        lease = c.Lease(self.root, services, [f"GPU-{i}" for i in range(4)])
        with patch.object(c.runtime, "alive", side_effect=lambda x: x["pid"] >= 200), \
                patch.object(c.runtime, "process", side_effect=lambda pid:
                             current[list(c.LANES)[pid - 200]]), \
                patch.object(c.runtime, "gpu_inventory", side_effect=inventory), \
                patch.object(c.resources, "port_free"), \
                patch.object(c, "idle_originals"), \
                patch.object(c, "originals", return_value=current), \
                patch.object(c.subprocess, "Popen", side_effect=popen):
            lease.restore()
        self.assertEqual(launched, [0, 1, 2, 3])
        self.assertTrue(r.suite.read_json(self.root / "restoration.json")["original_services_restored"])

    def test_failed_cleanup_does_not_skip_restoration(self):
        lock = Mock()
        lock.__enter__ = Mock()
        lock.__exit__ = Mock()
        originals = {lane: {"pid": i + 1, "process_tree": []}
                     for i, lane in enumerate(c.LANES)}
        lease = Mock()
        lease.cleanup.side_effect = RuntimeError("owned worker cleanup failed")
        with patch.object(c.suite, "lane_locks", return_value=lock), \
                patch.object(c, "lock_ownership", return_value={}), \
                patch.object(c, "snapshot_originals", return_value=(originals, ["a", "b", "c", "d"])), \
                patch.object(c, "idle_originals"), \
                patch.object(c.resources, "port_free"), \
                patch.object(c.resources, "service"), \
                patch.object(c.resources, "same_identity", return_value=True), \
                patch.object(c.admission, "four_gpu_mapping"), \
                patch.object(c.runtime, "gpu_inventory"), \
                patch.object(c.runtime, "stop_tree", return_value=True), \
                patch.object(c, "Lease", return_value=lease):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                with c.acquire(self.root / "resources"):
                    pass
        lease.restore.assert_called_once()
        lock.__exit__.assert_called_once()

    def test_partial_shutdown_still_restores_and_unlocks(self):
        lock = Mock()
        lock.__enter__ = Mock()
        lock.__exit__ = Mock()
        originals = {lane: {"pid": i + 1, "process_tree": []}
                     for i, lane in enumerate(c.LANES)}
        lease = Mock()
        previous = signal.getsignal(signal.SIGTERM)
        with patch.object(c.suite, "lane_locks", return_value=lock), \
                patch.object(c, "lock_ownership", return_value={}), \
                patch.object(c, "snapshot_originals", return_value=(originals, ["a", "b", "c", "d"])), \
                patch.object(c, "idle_originals"), \
                patch.object(c.resources, "port_free"), \
                patch.object(c.resources, "service"), \
                patch.object(c.resources, "same_identity", return_value=True), \
                patch.object(c.admission, "four_gpu_mapping"), \
                patch.object(c.runtime, "gpu_inventory"), \
                patch.object(c.runtime, "stop_tree", return_value=False), \
                patch.object(c, "Lease", return_value=lease):
            with self.assertRaisesRegex(r.suite.shared.Refusal, "shutdown incomplete"):
                with c.acquire(self.root / "resources"):
                    self.fail("Incomplete shutdown admitted.")
        lease.cleanup.assert_called_once()
        lease.restore.assert_called_once()
        lock.__exit__.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_body_cancellation_restores_and_unlocks(self):
        lock = Mock()
        lock.__enter__ = Mock()
        lock.__exit__ = Mock()
        originals = {lane: {"pid": i + 1, "process_tree": []}
                     for i, lane in enumerate(c.LANES)}
        lease = Mock()
        with patch.object(c.suite, "lane_locks", return_value=lock), \
                patch.object(c, "lock_ownership", return_value={}), \
                patch.object(c, "snapshot_originals", return_value=(originals, ["a", "b", "c", "d"])), \
                patch.object(c, "idle_originals"), \
                patch.object(c.resources, "port_free"), \
                patch.object(c.resources, "service"), \
                patch.object(c.resources, "same_identity", return_value=True), \
                patch.object(c.admission, "four_gpu_mapping"), \
                patch.object(c.runtime, "gpu_inventory"), \
                patch.object(c.runtime, "stop_tree", return_value=True), \
                patch.object(c, "Lease", return_value=lease):
            with self.assertRaises(KeyboardInterrupt):
                with c.acquire(self.root / "resources"):
                    raise KeyboardInterrupt("fake SIGTERM")
        lease.cleanup.assert_called_once()
        lease.restore.assert_called_once()
        lock.__exit__.assert_called_once()

    def test_child_deadline_is_bounded(self):
        children = c.Children(self.root)
        entry = {"child": Mock(), "identity": {"pid": 99}, "label": "fake"}
        entry["child"].poll.return_value = None
        with patch.object(children, "capture"):
            with self.assertRaisesRegex(r.suite.shared.Refusal, "deadline"):
                children.wait(entry, 0)

    def test_recovery_requires_dead_coordinator(self):
        (self.root / "start.json").write_text(json.dumps({"identity": {"pid": 99}}))
        with patch.object(c, "require_standard_lock_access"), \
                patch.object(c, "verify_registration"), \
                patch.object(c.runtime, "alive", return_value=True), \
                patch.object(c.suite, "lane_locks") as lock:
            with self.assertRaisesRegex(r.suite.shared.Refusal, "still alive"):
                c.recover(self.root, allow_standard_lock_access=True)
        lock.assert_not_called()

    def test_server_commands_are_dense_bf16_not_quantized(self):
        for lane, spec in r.SPECS.items():
            command = c.server_command(lane, self.root / "snapshot")
            self.assertNotIn("--trust-remote-code", command)
            self.assertNotIn("--quantization", command)
            self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1],
                             str(spec["tensor_parallel_size"]))
        self.assertNotIn("--reasoning-parser",
                         c.server_command("qwen25_72b_bf16", self.root / "snapshot"))

    def test_served_tokenization_mismatch_rejected(self):
        lane = "qwen38_27b_bf16"
        model = self.root / lane
        model.mkdir()
        (model / "tokenizer-preflight.json").write_text(json.dumps(
            {"messages": [], "token_ids": [1, 2]}))
        response = Mock(status_code=200)
        response.json.return_value = {"tokens": [1, 3]}
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = response
        with patch.object(c.suite.shared, "vllm_preflight", new=AsyncMock()), \
                patch("httpx.AsyncClient", return_value=client):
            with self.assertRaisesRegex(r.suite.shared.Refusal, "tokenization differs"):
                asyncio.run(c.served_tokenization(lane, self.root))


if __name__ == "__main__":
    unittest.main()
