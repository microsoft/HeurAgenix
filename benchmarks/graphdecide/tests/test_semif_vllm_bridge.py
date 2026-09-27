import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import httpx


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "semif_vllm_bridge", ROOT / "scripts/semif_vllm_bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def request():
    return bridge.DecisionRequest(
        "public-case-7", {"vertices": ["10", "2", "none"]}, "Choose the next node.",
        tuple(bridge.DecisionOption(i, f"Choose node {i}.") for i in ["10", "2", "none"]))


def response_for(payload, scores=(-3.0, -1.0, -5.0)):
    choices = []
    for index, prompt in enumerate(payload["prompt"]):
        logs = [None] + [{str(token): {"logprob": -2.0}} for token in prompt[1:]]
        logs[-1] = {str(prompt[-1]): {"logprob": scores[index]}}
        choices.append({"index": index, "text": "", "finish_reason": "length",
                        "prompt_token_ids": prompt, "prompt_logprobs": logs,
                        "token_ids": [999]})
    inputs = sum(len(p) for p in payload["prompt"])
    return {"id": "cmpl-" + payload["request_id"], "model": bridge.MODEL_ID,
            "choices": choices,
            "usage": {"prompt_tokens": inputs, "completion_tokens": len(choices),
                      "total_tokens": inputs + len(choices)}}


def requested_response_for(payload, scores=(-3.0, -1.0, -5.0)):
    table = {f"token_id:{token}": score
             for token, score in zip(payload["logprob_token_ids"], scores)}
    table["token_id:999"] = -0.5
    return {"id": "cmpl-" + payload["request_id"], "model": bridge.MODEL_ID,
            "choices": [{"index": 0, "text": "ignored", "finish_reason": "length",
                         "prompt_token_ids": payload["prompt"], "token_ids": [999],
                         "logprobs": {"top_logprobs": [table]}}],
            "usage": {"prompt_tokens": len(payload["prompt"]), "completion_tokens": 1,
                      "total_tokens": len(payload["prompt"]) + 1}}


def encoder(tokenizer, row, maximum):
    if maximum < 3:
        raise ValueError("No truncation allowed")
    return [10, 20, 30], [100 + i for i in range(len(row["options"]))], "canonical-hash"


class Guard:
    def __init__(self):
        self.entered = self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *args):
        self.exited += 1


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.request = request()
        self.payload = bridge.make_payload([10, 20, 30], [100, 101, 102], "case-token")
        self.body = response_for(self.payload)

    def test_numeric_nonordered_ids(self):
        result = bridge.parse_response(self.body, self.request, self.payload)
        self.assertEqual(result["selected_option_id"], "2")
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["prefill_tokens"], 12)
        self.assertEqual(result["generated_tokens"], 3)

    def test_batch_indices_not_response_order(self):
        self.body["choices"].reverse()
        result = bridge.parse_response(self.body, self.request, self.payload)
        self.assertEqual(result["selected_option_id"], "2")

    def test_tie_uses_supplied_order(self):
        body = response_for(self.payload, (-1.0, -1.0, -3.0))
        self.assertEqual(bridge.parse_response(body, self.request, self.payload)
                         ["selected_option_id"], "10")

    def test_actual_candidate_not_top_k_is_used(self):
        for choice in self.body["choices"]:
            choice["prompt_logprobs"][-1]["444"] = {"logprob": -0.01}
        result = bridge.parse_response(self.body, self.request, self.payload)
        self.assertEqual(result["selected_option_id"], "2")
        self.assertEqual(self.payload["prompt_logprobs"], 0)

    def test_generated_tokens_have_no_effect_on_selection(self):
        for choice in self.body["choices"]:
            choice["token_ids"] = [100 + (2 - choice["index"])]
        self.assertEqual(bridge.parse_response(self.body, self.request, self.payload)
                         ["selected_option_id"], "2")

    def test_structural_and_numeric_failures(self):
        mutations = [
            lambda b: b.update(id="cmpl-wrong-case"),
            lambda b: b.update(model="wrong-model"),
            lambda b: b["choices"].pop(),
            lambda b: b["choices"][1].update(index=0),
            lambda b: b["choices"][1].update(index=True),
            lambda b: b["choices"][1].update(index=7),
            lambda b: b["choices"][0].update(prompt_token_ids=[20, 30, 100]),
            lambda b: b["choices"][0].update(prompt_logprobs=[None]),
            lambda b: b["choices"][0]["prompt_logprobs"][-1].clear(),
            lambda b: b["choices"][0]["prompt_logprobs"][-1]["100"].update(logprob=float("nan")),
            lambda b: b["choices"][0]["prompt_logprobs"][-1]["100"].update(logprob=float("inf")),
            lambda b: b["choices"][0]["prompt_logprobs"][-1]["100"].update(logprob=True),
            lambda b: b["choices"][0].update(token_ids=[7, 8]),
            lambda b: b["choices"][0].update(finish_reason="abort"),
            lambda b: b["usage"].update(prompt_tokens=99),
            lambda b: b["usage"].update(completion_tokens=0),
        ]
        for change in mutations:
            with self.subTest(mutation=change):
                body = copy.deepcopy(self.body)
                change(body)
                with self.assertRaises(bridge.InvalidResponseError):
                    bridge.parse_response(body, self.request, self.payload)

    def test_option_and_generation_bounds(self):
        for slots in [[1], list(range(17)), [1, 1]]:
            with self.assertRaises(bridge.UnsupportedRequestError):
                bridge.make_payload([10], slots, "x")
        with self.assertRaises(bridge.UnsupportedRequestError):
            bridge.make_payload([10], [1, 2], "x", max_generated=2)
        self.assertEqual(bridge.make_payload([10], [1, 2], "x", 1)["max_tokens"], 1)

    def test_empty_descriptions_use_exact_id(self):
        req = bridge.DecisionRequest("id", "public", "choose",
                                     (bridge.DecisionOption("10"), bridge.DecisionOption("2")))
        self.assertEqual([o["description"] for o in bridge.row_for(req)["options"]], ["10", "2"])

    def test_prepare_uses_all_twelve_cpu_exact_probes(self):
        rows, references = bridge.calibration_inputs(bridge.DEFAULT_CPU)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["task"] for row in rows}), 6)
        self.assertTrue(all(row["id"] in references for row in rows))
        self.assertTrue(all("expected" not in row for row in rows))


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, mode="ok", max_prefix_tokens=2048, scoring_mode="teacher_forced"):
        self.posts = []
        self.guard = Guard()

        def handler(req):
            if req.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": bridge.MODEL_ID}]})
            if req.url.path == "/metrics":
                value = "1" if mode == "busy" else "0"
                return httpx.Response(200, text=f"vllm:num_requests_running {value}\n"
                                      "vllm:num_requests_waiting 0\n")
            self.posts.append(json.loads(req.content))
            if mode == "http_error":
                return httpx.Response(503, text="not persisted")
            if mode == "timeout":
                raise httpx.ReadTimeout("timeout", request=req)
            if mode == "invalid_json":
                return httpx.Response(200, text="not JSON")
            body = (requested_response_for(self.posts[-1])
                    if scoring_mode == "requested" else response_for(self.posts[-1]))
            body["choices"].reverse()
            return httpx.Response(200, json=body)

        return bridge.SemIfVLLMBridge(
            object(), encoder, max_prefix_tokens=max_prefix_tokens,
            transport=httpx.MockTransport(handler), test_guard=self.guard,
            scoring_mode=scoring_mode)

    async def test_requested_single_prefix_predict(self):
        client = await self.make_client(scoring_mode="requested")
        try:
            await client.initialize()
            answer = await client.predict(request())
            self.assertEqual(answer.selected_option_id, "2")
            self.assertEqual((answer.usage.input_tokens, answer.usage.output_tokens), (3, 1))
            self.assertEqual(self.posts[0]["prompt"], [10, 20, 30])
            self.assertEqual(self.posts[0]["logprob_token_ids"], [100, 101, 102])
            self.assertEqual(self.posts[0]["logprobs"], 3)
            self.assertEqual(self.posts[0]["max_tokens"], 1)
            self.assertEqual(self.posts[0]["temperature"], 1.0)
            self.assertNotIn("prompt_logprobs", self.posts[0])
            self.assertFalse(self.posts[0]["echo"])
        finally:
            await client.aclose()

    async def test_full_predict_maps_and_accounts(self):
        client = await self.make_client()
        try:
            await client.initialize()
            result = await client.predict(request())
            self.assertEqual(result.request_id, "public-case-7")
            self.assertEqual(result.selected_option_id, "2")
            self.assertEqual(result.usage.input_tokens, 12)
            self.assertEqual(result.usage.output_tokens, 3)
            self.assertEqual(len(self.posts), 1)
            self.assertEqual(self.posts[0]["prompt"], [
                [10, 20, 30, 100], [10, 20, 30, 101], [10, 20, 30, 102]])
            self.assertFalse(self.posts[0]["add_special_tokens"])
            self.assertIsNone(self.posts[0]["truncate_prompt_tokens"])
        finally:
            await client.aclose()
        self.assertEqual((self.guard.entered, self.guard.exited), (1, 1))

    async def test_endpoint_failures_no_retry(self):
        for mode, error in [
            ("http_error", bridge.ProviderHTTPError),
            ("timeout", bridge.ClientTimeoutError),
            ("invalid_json", bridge.InvalidResponseError),
        ]:
            client = await self.make_client(mode)
            try:
                await client.initialize()
                with self.assertRaises(error):
                    await client.predict(request())
                self.assertEqual(len(self.posts), 1)
            finally:
                await client.aclose()

    async def test_busy_preflight_sends_no_inference(self):
        client = await self.make_client("busy")
        with self.assertRaises(Exception):
            await client.initialize()
        self.assertEqual(len(self.posts), 0)
        self.assertEqual((self.guard.entered, self.guard.exited), (1, 1))
        await client.aclose()

    async def test_overlength_rejects_before_endpoint(self):
        client = await self.make_client(max_prefix_tokens=2)
        try:
            await client.initialize()
            with self.assertRaises(bridge.UnsupportedRequestError):
                await client.predict(request())
            self.assertEqual(len(self.posts), 0)
        finally:
            await client.aclose()

    async def test_initialize_required(self):
        client = await self.make_client()
        try:
            with self.assertRaises(bridge.ClientClosedError):
                await client.predict(request())
            self.assertEqual(len(self.posts), 0)
        finally:
            await client.aclose()


class ExplicitTokenTests(unittest.TestCase):
    def setUp(self):
        self.request = request()
        self.payload = bridge.make_requested_payload([10, 20, 30], [100, 101, 102], "direct")
        self.body = requested_response_for(self.payload)

    def test_explicit_ids_not_natural_rank(self):
        result = bridge.parse_requested_response(self.body, self.request, self.payload)
        self.assertEqual(result["selected_option_id"], "2")
        self.assertEqual(result["candidate_prompts"], 1)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["generated_tokens"], 1)

    def test_coverage_clamping_and_prefix_validation(self):
        mutations = [
            lambda b: b["choices"][0]["logprobs"]["top_logprobs"][0].pop("token_id:101"),
            lambda b: b["choices"][0]["logprobs"]["top_logprobs"][0].update({"token_id:333": -1}),
            lambda b: b["choices"][0]["logprobs"]["top_logprobs"][0].update({"token_id:100": -9999}),
            lambda b: b["choices"][0]["logprobs"]["top_logprobs"][0].update({"token_id:100": float("nan")}),
            lambda b: b["choices"][0]["logprobs"]["top_logprobs"][0].update({"token_id:100": float("-inf")}),
            lambda b: b["choices"][0].update(prompt_token_ids=[20, 30]),
            lambda b: b["choices"].append(copy.deepcopy(b["choices"][0])),
            lambda b: b["choices"][0].update(index=1),
            lambda b: b["choices"][0].update(token_ids=[]),
            lambda b: b["usage"].update(completion_tokens=0),
        ]
        for change in mutations:
            with self.subTest(mutation=change):
                body = copy.deepcopy(self.body)
                change(body)
                with self.assertRaises(bridge.InvalidResponseError):
                    bridge.parse_requested_response(body, self.request, self.payload)

    def test_sampled_option_is_not_duplicated(self):
        choice = self.body["choices"][0]
        choice["token_ids"] = [100]
        del choice["logprobs"]["top_logprobs"][0]["token_id:999"]
        result = bridge.parse_requested_response(self.body, self.request, self.payload)
        self.assertEqual(result["selected_option_id"], "2")


if __name__ == "__main__":
    unittest.main()
