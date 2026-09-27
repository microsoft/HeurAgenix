"""Sanitized failure diagnostics; HTTP mocks and tiny offline graph fixtures only."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import httpx
import networkx as nx

from src.benchmark.config import (
    BenchmarkConfig, DataConfig, ModelConfig, RunConfig, SamplingConfig, TaskConfig,
)
from src.benchmark.runner import run_experiment, summarize
from src.clients._llm_choice import parse_choice, token_usage
from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, DecisionOption, DecisionRequest,
    InvalidResponseError, ProviderHTTPError, TokenUsage,
)
from src.clients.vllm_client import VLLMClient
from src.datasets import GraphDataset, LoadStats, get_source


PRIVATE = "PRIVATE_RESPONSE_SENTINEL"


def request():
    return DecisionRequest("offline-question", {"nodes": [0, 1], "edges": [[0, 1]]},
                           "Are nodes 0 and 1 adjacent?",
                           (DecisionOption("yes"), DecisionOption("no")))


def completion(content='{"choice":"yes"}', *, finish="stop", output_tokens=12):
    return {
        "model": "offline-served-model",
        "choices": [{"finish_reason": finish, "message": {
            "role": "assistant", "content": content,
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": output_tokens,
                  "completion_tokens_details": {"reasoning_tokens": 0}},
        "ignored_raw_field": PRIVATE,
    }


class DiagnosticContractTests(unittest.TestCase):
    def test_old_constructor_and_new_safe_fields(self):
        body = {"text": PRIVATE}
        old = InvalidResponseError("original message", raw_output=body)
        self.assertEqual(str(old), "original message")
        self.assertIs(old.raw_output, body)
        self.assertEqual(old.diagnostic_code, "invalid_response")
        self.assertIsNone(old.finish_reason)
        self.assertIsNone(old.usage)
        usage = TokenUsage(20, 4096, None)
        new = InvalidResponseError("message", diagnostic_code="finish_length",
                                   finish_reason="length", usage=usage)
        self.assertEqual(new.diagnostic_code, "finish_length")
        self.assertEqual(new.finish_reason, "length")
        self.assertEqual(new.usage, usage)

    def test_untrusted_diagnostic_values_are_not_accepted(self):
        class StringSubclass(str):
            pass

        for value in (PRIVATE, "finish_length\n" + PRIVATE, [], {}, True, 7, None,
                      StringSubclass("finish_length")):
            with self.subTest(value=value):
                error = InvalidResponseError(PRIVATE, diagnostic_code=value, finish_reason=value)
                self.assertEqual(error.diagnostic_code, "invalid_response")
                self.assertIsNone(error.finish_reason)

    def test_usage_requires_actual_nonnegative_integers_and_safe_container(self):
        for bad in (True, False, -1, 1.5, "4096", PRIVATE, float("nan"), float("inf"), [], {}):
            for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
                with self.subTest(bad=bad, field=field):
                    error = InvalidResponseError("bad", usage=TokenUsage(**{field: bad}))
                    self.assertIsNone(error.usage)
        for bad in (PRIVATE, {"output_tokens": 4096}, [], object()):
            self.assertIsNone(InvalidResponseError("bad", usage=bad).usage)
        self.assertEqual(InvalidResponseError("ok", usage=TokenUsage(0, None, 0)).usage,
                         TokenUsage(0, None, 0))
        with self.assertRaises(InvalidResponseError) as caught:
            token_usage(output_tokens=True)
        self.assertEqual(caught.exception.diagnostic_code, "invalid_usage")

    def test_parse_choice_classifies_without_repair_or_echo(self):
        cases = [
            (None, "missing_answer"), ("  ", "missing_answer"),
            ("<think>" + PRIVATE + "</think>", "missing_answer"),
            ("```json\n\n```", "missing_answer"),
            (PRIVATE, "invalid_json"), ('{"choice":', "invalid_json"),
            ('{"choice":"yes","choice":"no"}', "invalid_json"),
            ('{"choice":"yes"} {"choice":"no"}', "invalid_json"),
            ('{"choice":"' + PRIVATE + '"}', "invalid_choice"),
            ('{"choice":true}', "invalid_choice"), ("[]", "invalid_choice"),
            ('{"choice":"yes","extra":1}', "invalid_choice"),
            ("<think>" + PRIVATE, "incomplete_reasoning"),
            ('<think><think>x</think>{"choice":"yes"}', "incomplete_reasoning"),
        ]
        for text, code in cases:
            with self.subTest(code=code, text=text), self.assertRaises(InvalidResponseError) as caught:
                parse_choice(text, request())
            self.assertEqual(caught.exception.diagnostic_code, code)
            self.assertNotIn(PRIVATE, str(caught.exception))
        for text in ('{"choice":"yes"}', '```json\n{"choice":"yes"}\n```',
                     '<think>reasoning</think>{"choice":"yes"}'):
            self.assertEqual(parse_choice(text, request()), "yes")


class VLLMDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def failure(self, body, **kwargs):
        calls = []

        def handler(wire):
            calls.append(wire)
            payload = json.loads(wire.content)
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertEqual(payload["temperature"], 0.0)
            self.assertEqual(payload["top_p"], 1.0)
            self.assertEqual(payload["n"], 1)
            self.assertFalse(payload["stream"])
            return httpx.Response(200, content=json.dumps(body))

        async with VLLMClient(api_key="", transport=httpx.MockTransport(handler), **kwargs) as client:
            with self.assertRaises(InvalidResponseError) as caught:
                await client.predict(request())
        self.assertEqual(len(calls), 1)
        self.assertNotIn(PRIVATE, str(caught.exception))
        self.assertIsNone(caught.exception.raw_output)
        return caught.exception

    async def test_length_retains_4096_tokens_even_for_invalid_or_valid_answer(self):
        for content in (PRIVATE, '<think>' + PRIVATE, '{"choice":"yes"}'):
            error = await self.failure(completion(content, finish="length", output_tokens=4096))
            self.assertEqual(error.diagnostic_code, "finish_length")
            self.assertEqual(error.finish_reason, "length")
            self.assertEqual(error.usage, TokenUsage(20, 4096, 0))

    async def test_stop_parse_failures_retain_usage_and_reason(self):
        for content, code in ((PRIVATE, "invalid_json"), (None, "missing_answer"),
                              ('{"choice":"maybe"}', "invalid_choice"),
                              ('<think>' + PRIVATE, "incomplete_reasoning")):
            with self.subTest(code=code):
                error = await self.failure(completion(content))
                self.assertEqual(error.diagnostic_code, code)
                self.assertEqual(error.finish_reason, "stop")
                self.assertEqual(error.usage, TokenUsage(20, 12, 0))

    async def test_unexpected_reasoning_retains_metadata(self):
        for field in ("reasoning_content", "reasoning", "content", "reasoning_tokens"):
            body = completion()
            if field == "reasoning_tokens":
                body["usage"]["completion_tokens_details"][field] = 8
            elif field == "content":
                body["choices"][0]["message"][field] = '<think>' + PRIVATE + '</think>{"choice":"yes"}'
            else:
                body["choices"][0]["message"][field] = PRIVATE
            error = await self.failure(body, think=False)
            self.assertEqual(error.diagnostic_code, "unexpected_reasoning")
            self.assertEqual(error.finish_reason, "stop")
            self.assertEqual(error.usage.output_tokens, 12)

    async def test_invalid_usage_is_discarded_without_masking_answer_failures(self):
        invalid = [[], PRIVATE, {"prompt_tokens": True}, {"completion_tokens": -1},
                   {"completion_tokens": PRIVATE}, {"completion_tokens": 1.5},
                   {"completion_tokens": float("nan")}, {"completion_tokens": float("inf")},
                   {"completion_tokens_details": []},
                   {"completion_tokens_details": {"reasoning_tokens": PRIVATE}}]
        for usage in invalid:
            for content, finish, code in (('{"choice":"yes"}', "stop", "invalid_usage"),
                                          (PRIVATE, "stop", "invalid_json"),
                                          (PRIVATE, "length", "finish_length")):
                with self.subTest(usage=usage, code=code):
                    body = completion(content, finish=finish)
                    body["usage"] = usage
                    error = await self.failure(body)
                    self.assertEqual(error.diagnostic_code, code)
                    self.assertEqual(error.finish_reason, finish)
                    self.assertIsNone(error.usage)

    async def test_untrusted_finish_reason_is_not_echoed(self):
        for finish in (PRIVATE, [PRIVATE], {PRIVATE: "stop"}, True, None):
            error = await self.failure(completion(PRIVATE, finish=finish))
            self.assertEqual(error.diagnostic_code, "invalid_finish_reason")
            self.assertIsNone(error.finish_reason)
            self.assertEqual(error.usage, TokenUsage(20, 12, 0))
        for finish in ("tool_calls", "function_call", "content_filter", "error", "abort"):
            error = await self.failure(completion(PRIVATE, finish=finish))
            self.assertEqual(error.diagnostic_code, "finish_" + finish)
            self.assertEqual(error.finish_reason, finish)

    async def test_envelope_failures_retain_only_unambiguous_metadata(self):
        for change, finish in (({"model": ""}, "stop"), ({"choices": []}, None),
                               ({"choices": [{}, {}]}, None)):
            error = await self.failure(completion() | change)
            self.assertEqual(error.diagnostic_code, "invalid_response")
            self.assertEqual(error.finish_reason, finish)
            self.assertEqual(error.usage, TokenUsage(20, 12, 0))
        for body in (None, [], PRIVATE):
            error = await self.failure(body)
            self.assertEqual(error.diagnostic_code, "invalid_response")
            self.assertIsNone(error.usage)

    async def test_missing_usage_stays_unknown_on_failure_and_success(self):
        body = completion(PRIVATE)
        body.pop("usage")
        error = await self.failure(body)
        self.assertEqual(error.usage, TokenUsage())
        body["choices"][0]["message"]["content"] = '{"choice":"yes"}'
        async with VLLMClient(api_key="", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=body)
        )) as client:
            response = await client.predict(request())
        self.assertEqual(response.selected_option_id, "yes")
        self.assertEqual(response.usage, TokenUsage())

    async def test_invalid_http_json_has_no_invented_metadata(self):
        async with VLLMClient(api_key="", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text=PRIVATE)
        )) as client:
            with self.assertRaises(InvalidResponseError) as caught:
                await client.predict(request())
        self.assertEqual(caught.exception.diagnostic_code, "invalid_json")
        self.assertIsNone(caught.exception.finish_reason)
        self.assertIsNone(caught.exception.usage)
        self.assertNotIn(PRIVATE, str(caught.exception))


def fixture_loader(name, **kwargs):
    graph = nx.path_graph(30)
    return GraphDataset(get_source(name), graph, LoadStats(29, 0, 0, 30, 29, 0, 1),
                        "offline-fixture-sha")


class FailingClient(BaseDecisionClient):
    def __init__(self, error):
        self.error = error

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        raise self.error


class RunnerDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = BenchmarkConfig(
            1, RunConfig(123, 1, 8, self.root / "output"),
            DataConfig(("facebook",), self.root / "data"), SamplingConfig((8,), 2, 10),
            TaskConfig(2, 2, (2,)), ModelConfig("vllm", "offline-model", 1.0),
        )

    def artifacts(self):
        output = self.config.run.output_dir
        for path in output.iterdir():
            text = path.read_text()
            self.assertNotIn(PRIVATE, text)
            self.assertNotIn('"raw_output"', text)
            self.assertNotIn('"message"', text)
        rows = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
        return rows, json.loads((output / "summary.json").read_text())

    async def test_http_failures_log_codes_usage_not_bodies_and_keep_denominator(self):
        calls = []

        def handler(wire):
            calls.append(wire)
            body = (completion(PRIVATE, finish="length", output_tokens=4096) if len(calls) == 1
                    else completion(PRIVATE) if len(calls) == 2 else completion())
            return httpx.Response(200, json=body)

        client = VLLMClient(api_key="", transport=httpx.MockTransport(handler))
        result = await run_experiment(self.config, client=client, loader=fixture_loader)
        rows, summary = self.artifacts()
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["diagnostic_code"], "finish_length")
        self.assertEqual(rows[0]["finish_reason"], "length")
        self.assertEqual(rows[0]["usage"], asdict(TokenUsage(20, 4096, 0)))
        self.assertEqual(rows[1]["diagnostic_code"], "invalid_json")
        self.assertEqual(rows[1]["finish_reason"], "stop")
        self.assertEqual(rows[1]["usage"], asdict(TokenUsage(20, 12, 0)))
        for row in rows[:2]:
            self.assertEqual(row["error_type"], "InvalidResponseError")
            self.assertEqual(row["status"], "failure")
            self.assertFalse(row["correct"])
            self.assertIsNone(row["selected_option_id"])
        for row in rows[2:]:
            self.assertEqual(row["status"], "success")
            self.assertIsNone(row["diagnostic_code"])
            self.assertIsNone(row["finish_reason"])
        overall = summary["overall"]
        self.assertEqual(result["overall"], overall)
        self.assertEqual(overall["attempts"], 8)
        self.assertEqual(overall["failures"], 2)
        self.assertEqual(overall["failure_rate"], 2 / 8)
        self.assertEqual(overall["accuracy"], sum(row["correct"] for row in rows) / 8)
        self.assertEqual(overall["diagnostic_counts"], {"finish_length": 1, "invalid_json": 1})
        self.assertEqual(overall["finish_reason_counts"], {"length": 1, "stop": 1})
        self.assertEqual(overall["error_counts"], {"InvalidResponseError": 2})
        # Preserve the existing success-only token aggregate semantics.
        self.assertEqual(overall["token_usage"]["output_tokens"]["known_sum"], 6 * 12)
        legacy = [{k: v for k, v in row.items() if k not in ("diagnostic_code", "finish_reason")}
                  for row in rows]
        old_summary = summarize(legacy, expected_calls=8, preparation={})
        self.assertEqual(old_summary["overall"]["diagnostic_counts"], {})
        self.assertEqual(old_summary["overall"]["accuracy"], overall["accuracy"])

    async def test_runner_revalidates_mutated_metadata_and_drops_message_rawbody(self):
        error = InvalidResponseError(PRIVATE, raw_output={"body": PRIVATE})
        error.diagnostic_code = PRIVATE
        error.finish_reason = {"text": PRIVATE}
        error.usage = TokenUsage(20, PRIVATE, 0)
        await run_experiment(self.config, client=FailingClient(error), loader=fixture_loader)
        rows, summary = self.artifacts()
        for row in rows:
            self.assertEqual(row["diagnostic_code"], "invalid_response")
            self.assertIsNone(row["finish_reason"])
            self.assertEqual(row["usage"], asdict(TokenUsage()))
        self.assertEqual(summary["overall"]["accuracy"], 0)
        self.assertEqual(summary["overall"]["diagnostic_counts"], {"invalid_response": 8})

    async def test_other_typed_failure_preserves_error_type_without_new_metadata(self):
        await run_experiment(self.config, client=FailingClient(ProviderHTTPError(503, PRIVATE)),
                             loader=fixture_loader)
        rows, summary = self.artifacts()
        for row in rows:
            self.assertEqual(row["error_type"], "ProviderHTTPError")
            self.assertEqual(row["http_status"], 503)
            self.assertIsNone(row["diagnostic_code"])
            self.assertIsNone(row["finish_reason"])
        self.assertEqual(summary["overall"]["failures"], 8)

    async def test_cancellation_still_persists_aborted_attempt_and_propagates(self):
        with self.assertRaises(asyncio.CancelledError):
            await run_experiment(self.config, client=FailingClient(asyncio.CancelledError(PRIVATE)),
                                 loader=fixture_loader)
        rows, summary = self.artifacts()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "aborted")
        self.assertEqual(rows[0]["error_type"], "CancelledError")
        self.assertEqual(summary["overall"]["accuracy"], 0)
        self.assertEqual(summary["run_status"], "aborted")


if __name__ == "__main__":
    unittest.main()