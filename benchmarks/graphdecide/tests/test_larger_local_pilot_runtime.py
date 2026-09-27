import asyncio
import hashlib
import json
import shutil
import unittest
from unittest.mock import AsyncMock, patch
import uuid
from types import SimpleNamespace

from scripts import larger_local_pilot_runtime as runtime
from src.clients.base import DecisionResponse, InvalidResponseError, ProviderHTTPError, TokenUsage


class FakeClient:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]

    async def initialize(self):
        pass

    async def aclose(self):
        pass

    async def predict(self, request):
        return DecisionResponse(request_id=request.request_id,
                                selected_option_id=request.options[0].id,
                                resolved_model=self.model, usage=TokenUsage(),
                                raw_output={"fake_client_test": True})


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.instances = runtime.load_panel(runtime.panel.DEFAULT_ROOT)

    def setUp(self):
        self.root = runtime.REPO / "output" / ("larger-runtime-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        (self.root / "execution.json").write_text("{}")

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_profile(self, factory):
        with patch.object(runtime, "execution", return_value=({}, self.instances)):
            return asyncio.run(runtime.run_profile(self.root, "qwen27_grammar",
                               client_factory=factory, preflight=AsyncMock()))

    def test_full_own_history_is_bounded_and_replays(self):
        state = self.run_profile(FakeClient)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["calls"], 160)
        scores = runtime.suite.read_rows(self.root / "qwen27_grammar/scores.jsonl")
        self.assertEqual(len(scores), 64)
        self.assertTrue(all(row["feasible"] for row in scores))
        attempts = runtime.suite.read_rows(self.root / "qwen27_grammar/attempts.jsonl")
        self.assertTrue(all(row["response_diagnostics"] == {"response_is_object": True}
                            and "raw_output" not in row for row in attempts))

    def test_fatal_failure_preserves_all_denominators(self):
        class Fatal(FakeClient):
            async def predict(self, request):
                raise ProviderHTTPError(401)
        state = self.run_profile(Fatal)
        self.assertEqual(state["status"], "aborted")
        self.assertEqual(state["calls"], 1)
        rows = runtime.suite.read_rows(self.root / "qwen27_grammar/scores.jsonl")
        self.assertEqual(len(rows), 64)
        self.assertEqual(sum(r["query_count"] == 0 for r in rows), 63)

    def test_nonfatal_parse_failure_no_repair_or_retry(self):
        class Invalid(FakeClient):
            async def predict(self, request):
                exc = InvalidResponseError("length", diagnostic_code="finish_length")
                exc.raw_output = {"content": "unfinished"}
                raise exc
        state = self.run_profile(Invalid)
        self.assertEqual(state["status"], "completed_with_failures")
        self.assertEqual(state["calls"], 64)
        rows = runtime.suite.read_rows(self.root / "qwen27_grammar/attempts.jsonl")
        self.assertTrue(all(r["exception_type"] == "InvalidResponseError"
                            and r["diagnostic_code"] == "finish_length"
                            and "raw_output" not in r for r in rows))

    def test_initialization_failure_has_all_zero_call_denominators(self):
        class BadInit(FakeClient):
            async def initialize(self):
                raise RuntimeError("injected initialization failure")
        state = self.run_profile(BadInit)
        self.assertEqual(state["calls"], 0)
        rows = runtime.suite.read_rows(self.root / "qwen27_grammar/scores.jsonl")
        self.assertEqual(len(rows), 64)

    def test_existing_profile_cannot_resume(self):
        self.run_profile(FakeClient)
        with self.assertRaises(runtime.suite.shared.Refusal):
            self.run_profile(FakeClient)

    def test_service_and_physical_mapping_require_owned_processes(self):
        services = {
            "qwen08": {"cuda_environment": {"CUDA_VISIBLE_DEVICES": "0"},
                       "process_tree": [{"pid": 11}]},
            "qwen2": {"cuda_environment": {"CUDA_VISIBLE_DEVICES": "1"},
                      "process_tree": [{"pid": 22}]}}
        inventory = {"gpus": [["0", "A"], ["1", "B"], ["2", "C"], ["3", "D"]],
                     "processes": [["11", "A", "100"], ["22", "B", "100"], ["99", "C", "100"]]}
        self.assertEqual(runtime.validate_mapping(services, inventory), ["A", "B"])
        inventory["processes"].append(["666", "A", "1"])
        with self.assertRaises(runtime.suite.shared.Refusal):
            runtime.validate_mapping(services, inventory)
        inventory["processes"] = [["11", "B", "100"], ["22", "A", "100"]]
        with self.assertRaises(runtime.suite.shared.Refusal):
            runtime.validate_mapping(services, inventory)

    def test_server_uses_pinned_local_bf16_tp2_without_remote_code(self):
        command = runtime.server_command(self.root)
        for flag, value in (("--tensor-parallel-size", "2"), ("--dtype", "bfloat16"),
                            ("--max-model-len", "16384"), ("--max-num-seqs", "1"),
                            ("--port", "8010"), ("--host", "127.0.0.1")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--trust-remote-code", command)
        self.assertEqual(len(runtime.REVISION), 40)

    def test_parser_failure_keeps_only_allowlisted_diagnostics(self):
        async def check():
            client = runtime.RecordingClient(**runtime.panel.configurations()["qwen27_grammar"].client_kwargs())
            request = runtime.canonical.public_request(self.instances[0], [])
            body = {"model": "Qwen3.5-27B", "choices": [{"finish_reason": "length",
                    "message": {"role": "assistant", "content": "unfinished"}}]}
            try:
                with self.assertRaises(InvalidResponseError) as raised:
                    client._parse_response(request, body)
                self.assertIsNone(raised.exception.raw_output)
                diagnostic = raised.exception.response_diagnostics
                self.assertEqual(diagnostic["finish_reason"], "length")
                self.assertEqual(diagnostic["final_content_characters"], len("unfinished"))
            finally:
                await client.aclose()
        asyncio.run(check())

    def test_success_and_failure_artifacts_never_include_hidden_reasoning_or_full_body(self):
        body = {"private_provider_field": "PRIVATE_TOP_LEVEL_SENTINEL", "choices": [{
            "finish_reason": "stop", "message": {"role": "assistant",
                "content": "UNTRUSTED_FINAL_TEXT_SENTINEL",
                "reasoning": "HIDDEN_REASONING_SENTINEL",
                "reasoning_content": "HIDDEN_REASONING_CONTENT_SENTINEL"}}]}
        class Success(FakeClient):
            async def predict(self, request):
                response = await super().predict(request)
                return DecisionResponse(request_id=response.request_id,
                    selected_option_id=response.selected_option_id, resolved_model=self.model,
                    usage=TokenUsage(), raw_output=body)
        class Failure(FakeClient):
            async def predict(self, request):
                exc = InvalidResponseError("PRIVATE_EXCEPTION_MESSAGE_SENTINEL",
                                           diagnostic_code="finish_length")
                exc.raw_output = body
                raise exc
        for factory in (Success, Failure):
            self.run_profile(factory)
            output = self.root / "qwen27_grammar"
            text = "\n".join(p.read_text() for p in output.iterdir() if p.is_file())
            self.assertNotIn("SENTINEL", text)
            self.assertNotIn('"raw_output"', text)
            rows = runtime.suite.read_rows(output / "attempts.jsonl")
            self.assertTrue(all(r["response_diagnostics"]["reasoning_present"] for r in rows))
            self.assertTrue(all("final_content_sha256" in r["response_diagnostics"] for r in rows))
            shutil.rmtree(output)

    def test_readiness_timeout_persists_last_concrete_preflight_error(self):
        clocks = iter((0, 2))
        concrete = "vLLM served model identity must match exactly."
        with self.assertRaisesRegex(RuntimeError, concrete):
            runtime.wait_ready(SimpleNamespace(poll=lambda: None), self.root, timeout_seconds=1,
                preflight=AsyncMock(side_effect=runtime.suite.shared.Refusal(concrete)),
                clock=lambda: next(clocks), sleep=lambda _: None)
        state = runtime.suite.read_json(self.root / "status.json")
        self.assertEqual(state["last_preflight_error"],
                         {"error_type": "Refusal", "message": concrete})

    def test_stopper_never_signals_reused_or_dead_pid(self):
        records = [{"pid": 71, "start_ticks": "old"}]
        with patch.object(runtime, "alive", return_value=False), patch.object(runtime.os, "kill") as kill:
            self.assertTrue(runtime.stop_tree(records))
            kill.assert_not_called()

    def test_lock_refusal_precedes_process_access_or_shutdown(self):
        with patch.object(runtime, "execution", return_value=({"artifacts": str(self.root)}, [])), \
                patch.object(runtime, "verify_artifacts", return_value={}), \
                patch.object(runtime, "port_free"), \
                patch.object(runtime.suite, "lane_locks",
                             side_effect=runtime.suite.shared.Refusal("already owned")), \
                patch.object(runtime, "process") as process, \
                patch.object(runtime, "stop_tree") as stop:
            with self.assertRaises(runtime.suite.shared.Refusal):
                runtime.supervisor(self.root)
            process.assert_not_called()
            stop.assert_not_called()

    def test_unattempted_later_profile_retains_all_planned_episodes(self):
        with patch.object(runtime, "execution", return_value=({}, self.instances)):
            runtime.unattempted_profile(self.root, "qwen27_thinking", "earlier profile aborted")
        rows = runtime.suite.read_rows(self.root / "qwen27_thinking/scores.jsonl")
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(r["query_count"] == 0 for r in rows))

    def test_download_is_revision_pinned_checksums_verified_and_repeatable(self):
        files = {"config.json": b"{}"}
        shards = [f"model-{i}.safetensors" for i in range(11)]
        index = {"weight_map": {f"weight{i}": name for i, name in enumerate(shards)}}
        files["model.safetensors.index.json"] = json.dumps(index).encode()
        header = json.dumps({"weight": {"dtype": "BF16"}}).encode()
        files.update({name: len(header).to_bytes(8, "little") + header for name in shards})
        siblings = [SimpleNamespace(rfilename=name, size=len(body),
                                   lfs=SimpleNamespace(sha256=hashlib.sha256(body).hexdigest()))
                    for name, body in files.items()]

        def snapshot_download(**kwargs):
            self.assertEqual(kwargs["revision"], runtime.REVISION)
            kwargs["local_dir"].mkdir()
            for name, body in files.items():
                (kwargs["local_dir"] / name).write_bytes(body)

        api = SimpleNamespace(model_info=lambda *args, **kwargs:
                              SimpleNamespace(sha=runtime.REVISION, siblings=siblings))
        hub = SimpleNamespace(HfApi=lambda: api, snapshot_download=snapshot_download)
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            first = runtime.download(self.root / "checkpoint")
            second = runtime.download(self.root / "checkpoint")
            self.assertEqual(first, second)
            path = self.root / "checkpoint/snapshot" / shards[0]
            path.write_bytes(b"corrupted")
            with self.assertRaises(runtime.suite.shared.Refusal):
                runtime.download(self.root / "checkpoint")

    def test_interrupted_download_never_gets_completed_manifest(self):
        def fail(**kwargs):
            raise OSError("injected network interruption")
        api = SimpleNamespace(model_info=lambda *args, **kwargs:
                              SimpleNamespace(sha=runtime.REVISION, siblings=[]))
        hub = SimpleNamespace(HfApi=lambda: api, snapshot_download=fail)
        with patch.dict("sys.modules", {"huggingface_hub": hub}):
            with self.assertRaises(OSError):
                runtime.download(self.root / "checkpoint")
        self.assertFalse((self.root / "checkpoint/artifacts.json").exists())

    def test_standard_preflight_rejects_nonzero_running_requests(self):
        import httpx
        def respond(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "Qwen3.5-0.8B"}]})
            return httpx.Response(200, text="vllm:num_requests_running 1\n"
                                           "vllm:num_requests_waiting 0\n")
        config = runtime.suite.shared.model_configs(models=("qwen08",))["qwen08"]
        with self.assertRaises(runtime.suite.shared.Refusal):
            asyncio.run(runtime.suite.shared.vllm_preflight(
                [SimpleNamespace(model=config)], transport=httpx.MockTransport(respond)))


if __name__ == "__main__":
    unittest.main()
