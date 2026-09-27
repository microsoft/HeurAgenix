"""Generative client tests with mocked HTTP/SDK only; no login or model calls."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from src.clients._llm_choice import (
    ANSWER_ONLY_SYSTEM_MESSAGE, SYSTEM_MESSAGE, parse_answer_only, parse_choice, prompt_for,
)
from src.clients.base import (
    ClientClosedError, ClientTimeoutError, ClientTransportError, DecisionOption,
    DecisionRequest, InvalidResponseError, ProviderHTTPError, UnsupportedRequestError,
)
from src.clients.github_copilot_client import GitHubCopilotClient
from src.clients.registry import available_clients, create_client
from src.clients.vllm_client import VLLMClient


def request(request_id="PRIVATE_SOURCE-42"):
    return DecisionRequest(request_id, {"nodes": [0, 1], "edges": [[0, 1]]},
                           "Are nodes 0 and 1 adjacent?",
                           (DecisionOption("yes", "An edge exists"), DecisionOption("no")))


def completion(content='{"choice":"yes"}', *, thinking=False):
    return {"model": "served-model", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content,
        "reasoning_content": 'PRIVATE_REASONING {"choice":"no"}' if thinking else None,
    }}], "usage": {"prompt_tokens": 20, "completion_tokens": 12,
                   "completion_tokens_details": {"reasoning_tokens": 8 if thinking else 0}}}


def model_info(reasoning=True):
    return NS(id="test-gpt", supported_reasoning_efforts=["low", "medium", "high"] if reasoning else [],
              default_reasoning_effort="medium" if reasoning else None,
              capabilities=NS(supports=NS(reasoning_effort=reasoning)))


class ChoiceProtocolTests(unittest.TestCase):
    def test_answer_only_normalization_and_strict_boundaries(self):
        for text, expected in (("yes", "yes"), (" NO\n", "no"), ("YeS", "yes"),
                               ("<think>no is not right</think>yes", "yes")):
            self.assertEqual(parse_answer_only(text, request()), expected)
        for text in (None, "", "yes.", '"yes"', '{"choice":"yes"}', "yes no",
                     "The answer is yes", "```\nyes\n```", "yes\nno", "maybe",
                     "<think>yes", "<think>yes</think>", "<think><think>x</think>yes"):
            with self.subTest(text=text), self.assertRaises(InvalidResponseError):
                parse_answer_only(text, request())
        custom = replace(request(), options=(DecisionOption("node_7"), DecisionOption("node_91")))
        self.assertEqual(parse_answer_only("node_91", custom), "node_91")
        with self.assertRaises(InvalidResponseError):
            parse_answer_only("NODE_91", custom)

    def test_prompt_has_only_model_visible_fields(self):
        content = prompt_for(request())
        value = json.loads(content)
        self.assertEqual(set(value), {"state", "question", "options"})
        self.assertEqual(value["state"], request().state)
        self.assertEqual(value["options"], [{"id": "yes", "description": "An edge exists"},
                                             {"id": "no", "description": None}])
        self.assertNotIn("PRIVATE", content)
        self.assertNotIn("choice-v1", content)

    def test_exact_final_answer_and_safe_wrappers(self):
        for text in (' {"choice":"yes"} ', '```json\n{"choice":"yes"}\n```',
                     '<think>{"choice":"no"}</think>\n{"choice":"yes"}',
                     '<think></think>\n```json\n{"choice":"yes"}\n```'):
            with self.subTest(text=text):
                self.assertEqual(parse_choice(text, request()), "yes")
        custom = replace(request(), options=(DecisionOption("node_7"), DecisionOption("node_91")))
        self.assertEqual(parse_choice('{"choice":"node_91"}', custom), "node_91")

    def test_malformed_ambiguous_or_truncated_answers_are_not_repaired(self):
        for text in (None, "", "yes", "YES", "The answer is yes.", '[]', '{"choice":"YES"}',
                     '{"choice":"maybe"}', '{"choice":true}', '{"choice":["yes"]}',
                     '{"choice":"yes","explanation":"x"}', '{"choice":"no","choice":"yes"}',
                     '{"choice":"yes"} {"choice":"no"}', '<think>{"choice":"yes"}',
                     '<think><think>x</think>{"choice":"yes"}',
                     '<think>{"choice":"yes"}</think>', '{"choice":"yes"}\nMore text'):
            with self.subTest(text=text), self.assertRaises(InvalidResponseError):
                parse_choice(text, request())


class VLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_only_wire_and_diagnostics(self):
        for text, code in ((" YES\n", None), ("<think></think>no", None),
                           ('{"choice":"yes"}', "invalid_choice"),
                           ("yes because ...", "invalid_choice"),
                           ("<think>reasoning</think>yes", "unexpected_reasoning")):
            def handler(wire):
                body = json.loads(wire.content)
                self.assertEqual(body["messages"][0]["content"], ANSWER_ONLY_SYSTEM_MESSAGE)
                self.assertEqual(body["messages"][1]["content"], prompt_for(request()))
                self.assertNotIn("response_format", body)
                self.assertNotIn("output_format", body)
                return httpx.Response(200, json=completion(text))

            async with VLLMClient(output_format="answer_only", think=False,
                                  transport=httpx.MockTransport(handler)) as client:
                if code is None:
                    result = await client.predict(request())
                    self.assertEqual(result.selected_option_id, "no" if text.endswith("no") else "yes")
                    self.assertEqual(result.usage.output_tokens, 12)
                else:
                    with self.assertRaises(InvalidResponseError) as caught:
                        await client.predict(request())
                    self.assertEqual(caught.exception.diagnostic_code, code)
                    self.assertEqual(caught.exception.finish_reason, "stop")
                    self.assertEqual(caught.exception.usage.output_tokens, 12)
        for think in (True, False):
            async with VLLMClient(output_format="answer_only", think=think,
                                  transport=httpx.MockTransport(lambda _: httpx.Response(
                                      200, json=completion("yes", thinking=True)))) as client:
                if think:
                    self.assertEqual((await client.predict(request())).selected_option_id, "yes")
                else:
                    with self.assertRaises(InvalidResponseError):
                        await client.predict(request())

    async def test_think_and_no_think_wire_and_usage(self):
        for think in (True, False, None):
            bodies = []

            def handler(wire):
                self.assertEqual(str(wire.url), "http://127.0.0.1:8000/v1/chat/completions")
                self.assertEqual(wire.headers["authorization"], "Bearer FAKE_LOCAL_KEY")
                body = json.loads(wire.content)
                bodies.append(body)
                self.assertEqual(body["model"], "local-model")
                self.assertEqual(body["max_tokens"], 1024)
                self.assertEqual(body["seed"], 7)
                self.assertEqual(body["temperature"], 0.1)
                self.assertEqual(body["top_p"], 0.9)
                self.assertFalse(body["stream"])
                self.assertEqual(len(body["messages"]), 2)
                self.assertEqual(body["messages"][0]["content"], SYSTEM_MESSAGE)
                self.assertEqual(json.loads(body["messages"][1]["content"]), json.loads(prompt_for(request())))
                if think is None:
                    self.assertNotIn("chat_template_kwargs", body)
                else:
                    self.assertIs(body["chat_template_kwargs"]["enable_thinking"], think)
                self.assertNotIn("extra_body", body)
                self.assertNotIn("tools", body)
                self.assertNotIn("PRIVATE_SOURCE", json.dumps(body))
                return httpx.Response(200, json=completion(thinking=think is not False))

            with self.subTest(think=think):
                async with create_client("vllm", model="local-model", think=think, max_tokens=1024,
                                         seed=7, temperature=0.1, top_p=0.9, api_key="FAKE_LOCAL_KEY",
                                         transport=httpx.MockTransport(handler)) as client:
                    result = await client.predict(request())
                    await client.predict(request("different-id"))
                    self.assertTrue(client.capabilities.supports_thinking_control)
                    self.assertFalse(client.capabilities.supports_probabilities)
                self.assertEqual(len(bodies), 2)
                self.assertEqual(bodies[0], bodies[1])
                self.assertEqual(result.request_id, request().request_id)
                self.assertEqual(result.selected_option_id, "yes")
                self.assertEqual(result.resolved_model, "served-model")
                self.assertEqual((result.usage.input_tokens, result.usage.output_tokens,
                                  result.usage.reasoning_tokens), (20, 12, 0 if think is False else 8))
                self.assertIsNone(result.probabilities)
                self.assertIsNone(result.probability_kind)
                self.assertIsNone(result.confidence)

    async def test_optional_auth_and_unknown_usage(self):
        for key in (None, "FAKE_ENV_KEY"):
            def handler(wire):
                if key:
                    self.assertEqual(wire.headers["authorization"], f"Bearer {key}")
                else:
                    self.assertNotIn("authorization", wire.headers)
                body = completion()
                body.pop("usage")
                return httpx.Response(200, json=body)

            with patch.dict(os.environ, {"VLLM_API_KEY": key} if key else {}, clear=True):
                async with VLLMClient(transport=httpx.MockTransport(handler)) as client:
                    result = await client.predict(request())
                    self.assertIsNone(result.usage.input_tokens)
                    self.assertIsNone(result.usage.reasoning_tokens)

    async def test_no_think_rejects_observed_reasoning_but_allows_empty_block(self):
        for body in (completion(thinking=True),
                     completion('<think>PRIVATE_REASONING</think>{"choice":"yes"}'),
                     completion() | {"usage": {"completion_tokens_details": {"reasoning_tokens": 1}}}):
            async with VLLMClient(think=False, transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=body)
            )) as client:
                with self.assertRaisesRegex(InvalidResponseError, "no-think"):
                    await client.predict(request())
        async with VLLMClient(think=False, transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=completion('<think></think>{"choice":"yes"}'))
        )) as client:
            self.assertEqual((await client.predict(request())).selected_option_id, "yes")

    async def test_http_errors_no_retry_no_redirect_no_body_leak(self):
        for status in (302, 400, 401, 429, 503):
            calls = []

            def handler(wire):
                calls.append(wire)
                return httpx.Response(status, text="PRIVATE_SERVER_BODY",
                                      headers={"retry-after": "2", "location": "http://elsewhere/"})

            async with VLLMClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(ProviderHTTPError) as caught:
                    await client.predict(request())
            self.assertEqual(len(calls), 1)
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(caught.exception.retry_after, "2")
            self.assertNotIn("PRIVATE", str(caught.exception))

    async def test_transport_timeout_and_cancellation(self):
        for error, expected in ((httpx.ReadTimeout, ClientTimeoutError),
                                (httpx.ConnectError, ClientTransportError),
                                (asyncio.CancelledError, asyncio.CancelledError)):
            def handler(wire):
                raise error("PRIVATE_TRANSPORT")

            async with VLLMClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(expected) as caught:
                    await client.predict(request())
                if expected is not asyncio.CancelledError:
                    self.assertNotIn("PRIVATE", str(caught.exception))

        async def hanging(wire):
            await asyncio.Future()

        async with VLLMClient(timeout_seconds=0.01, transport=httpx.MockTransport(hanging)) as client:
            with self.assertRaises(ClientTimeoutError):
                await client.predict(request())

    async def test_bad_responses_are_typed_failures(self):
        bodies = [None, [], {}, completion() | {"model": ""}, completion() | {"choices": []},
                  completion() | {"usage": []}, completion() | {"usage": {"prompt_tokens": -1}},
                  completion() | {"usage": {"completion_tokens_details": []}},
                  completion() | {"usage": {"completion_tokens": True}},
                  completion() | {"usage": {"completion_tokens_details": {"reasoning_tokens": "8"}}}]
        for finish in ("length", "tool_calls", "content_filter", None):
            body = completion()
            body["choices"][0]["finish_reason"] = finish
            bodies.append(body)
        for change in ({"content": None}, {"tool_calls": [{}]}, {"refusal": "refused"},
                       {"role": "user"}, {"content": '<think>{"choice":"yes"}'}):
            body = completion()
            body["choices"][0]["message"].update(change)
            bodies.append(body)
        for body in bodies:
            with self.subTest(body=body):
                async with VLLMClient(transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, content=json.dumps(body))
                )) as client:
                    with self.assertRaises(InvalidResponseError):
                        await client.predict(request())
        async with VLLMClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text="PRIVATE_NOT_JSON")
        )) as client:
            with self.assertRaises(InvalidResponseError) as caught:
                await client.predict(request())
            self.assertNotIn("PRIVATE", str(caught.exception))

    async def test_closed_and_protocol_checks_precede_http(self):
        def fail(_):
            raise AssertionError("No HTTP expected")

        client = VLLMClient(transport=httpx.MockTransport(fail))
        with self.assertRaises(UnsupportedRequestError):
            await client.predict(replace(request(), protocol_version="future"))
        await client.aclose()
        await client.aclose()
        with self.assertRaises(ClientClosedError):
            await client.predict(request())

    def test_invalid_settings(self):
        for kwargs in ({"output_format": "unknown"}, {"output_format": []}, {"output_format": None},
                       {"think": "false"}, {"max_tokens": True}, {"max_tokens": 0},
                       {"model": ""}, {"timeout_seconds": float("nan")}, {"timeout_seconds": True},
                       {"temperature": float("inf")}, {"temperature": -1}, {"top_p": 0},
                       {"top_p": 2}, {"seed": True}, {"seed": -1}, {"api_key": "bad\nkey"},
                       {"base_url": "http://user:pass@localhost:8000/v1"},
                       {"base_url": "http://localhost:8000/v1?token=secret"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VLLMClient(**kwargs)


class FakeSession:
    def __init__(self, runtime, options):
        self.runtime, self.options = runtime, options
        self.session_id = f"session-{len(runtime.sessions)}"
        self.abort = AsyncMock()
        self.disconnect = AsyncMock()

    async def send_and_wait(self, prompt, *, timeout):
        self.runtime.prompts.append(prompt)
        for event in self.runtime.events:
            self.options["on_event"](event)
        if self.runtime.error:
            raise self.runtime.error
        return self.runtime.answer


class FakeRuntime:
    def __init__(self):
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.get_auth_status = AsyncMock(return_value=NS(isAuthenticated=True))
        self.list_models = AsyncMock(return_value=[model_info()])
        self.delete_session = AsyncMock()
        self.create_session = AsyncMock(side_effect=self.new_session)
        self.sessions, self.prompts = [], []
        self.error = None
        self.answer = NS(type="assistant.message", data=NS(content='{"choice":"yes"}',
                                                         model="resolved-gpt", tool_requests=[]))
        self.events = [NS(type="assistant.usage", data=NS(model="resolved-gpt", input_tokens=100,
                          output_tokens=20, reasoning_tokens=14, finish_reason="stop",
                          reasoning_effort="medium"))]

    async def new_session(self, **options):
        session = FakeSession(self, options)
        self.sessions.append(session)
        return session


class CopilotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.factory_options = []

        def factory(**kwargs):
            self.factory_options.append(kwargs)
            return self.runtime

        self.patch = patch("src.clients.github_copilot_client._sdk_types",
                           return_value=(factory, NS, NS, lambda: NS(kind="denied")))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def client(self, **kwargs):
        return GitHubCopilotClient(**({"model": "test-gpt", "github_token": "FAKE_GITHUB_KEY"} | kwargs))

    async def test_output_format_wire_matches_vllm_for_both_gpt_models(self):
        for output_format in (None, "json", "answer_only"):
            settings = {} if output_format is None else {"output_format": output_format}
            content = " YES\n" if output_format == "answer_only" else '{"choice":"yes"}'
            messages = []

            def handler(wire):
                messages.extend(json.loads(wire.content)["messages"])
                return httpx.Response(200, json=completion(content))

            async with VLLMClient(**settings, transport=httpx.MockTransport(handler)) as client:
                await client.predict(request())
            for model in ("gpt-5.4", "gpt-6-astra"):
                with self.subTest(output_format=output_format, model=model):
                    self.runtime.list_models.return_value = [NS(**{**vars(model_info()), "id": model})]
                    self.runtime.answer.data.content = content
                    async with self.client(model=model, **settings) as client:
                        self.assertIsNone(client.think)
                        self.assertEqual(client.output_format, output_format or "json")
                        result = await client.predict(request())
                    session = self.runtime.sessions[-1]
                    opts = session.options
                    self.assertEqual(opts["system_message"], {"mode": "replace", "content": messages[0]["content"]})
                    self.assertEqual(self.runtime.prompts[-1], messages[1]["content"])
                    self.assertEqual(opts["model"], model)
                    self.assertIsNone(opts["reasoning_effort"])
                    self.assertEqual(opts["model_capabilities"].limits.max_output_tokens, 4096)
                    self.assertNotIn("output_format", opts)
                    self.assertNotIn("response_format", opts)
                    self.assertEqual(opts["tools"], [])
                    self.assertEqual(result.selected_option_id, "yes")
                    self.assertEqual(result.usage.reasoning_tokens, 14)
                    self.assertEqual(result.raw_output["content"], content)
                    self.assertIsNone(result.probabilities)
                    session.abort.assert_not_called()
                    session.disconnect.assert_awaited_once()

    async def test_answer_only_parser_acceptance_rejection_and_cleanup(self):
        cases = [("yes", "yes"), (" NO\n", "no"), ("YeS", "yes"),
                 ("<think>no is not right</think>yes", "yes")]
        cases.extend((text, None) for text in (
            None, "", "yes.", '"yes"', '{"choice":"yes"}', "yes no", "The answer is yes",
            "```\nyes\n```", "yes\nno", "maybe", "ｙｅｓ", "<think>yes",
            "<think>yes</think>", "<think><think>x</think>yes",
        ))
        for text, expected in cases:
            with self.subTest(text=text):
                self.runtime.answer.data.content = text
                before = len(self.runtime.prompts)
                async with self.client(output_format="answer_only") as client:
                    if expected is None:
                        with self.assertRaises(InvalidResponseError) as shared:
                            parse_answer_only(text, request())
                        with self.assertRaises(InvalidResponseError) as caught:
                            await client.predict(request())
                        self.assertEqual(caught.exception.diagnostic_code, shared.exception.diagnostic_code)
                    else:
                        self.assertEqual((await client.predict(request())).selected_option_id, expected)
                self.assertEqual(len(self.runtime.prompts), before + 1)
                session = self.runtime.sessions[-1]
                if expected is None:
                    session.abort.assert_awaited_once()
                else:
                    session.abort.assert_not_called()
                session.disconnect.assert_awaited_once()
                self.runtime.delete_session.assert_any_await(session.session_id)

    async def test_answer_only_preserves_reasoning_controls_and_contradiction_checks(self):
        self.runtime.answer.data.content = "yes"
        async with self.client(output_format="answer_only", think=False) as client:
            with self.assertRaises(UnsupportedRequestError):
                await client.predict(request())
        self.runtime.create_session.assert_not_called()
        self.runtime.events[0].data.reasoning_effort = "high"
        async with self.client(output_format="answer_only", think=True, reasoning_effort="high") as client:
            self.assertEqual((await client.predict(request())).usage.reasoning_tokens, 14)
        self.assertEqual(self.runtime.sessions[-1].options["reasoning_effort"], "high")
        async with self.client(output_format="answer_only", think=True, reasoning_effort="low") as client:
            with self.assertRaisesRegex(InvalidResponseError, "different reasoning"):
                await client.predict(request())

        self.runtime.list_models.return_value = [model_info(False)]
        usage = self.runtime.events[0].data
        usage.reasoning_effort, usage.reasoning_tokens = None, 0
        for field, value in (("content", "<think>reasoning</think>yes"),
                             ("reasoning_text", "reasoning"), ("reasoning_blocks", ["reasoning"]),
                             ("reasoning_tokens", 1), ("reasoning_effort", "low")):
            target = usage if field in ("reasoning_tokens", "reasoning_effort") else self.runtime.answer.data
            with self.subTest(field=field), patch.object(target, field, value, create=True):
                async with self.client(output_format="answer_only", think=False) as client:
                    with self.assertRaises(InvalidResponseError) as caught:
                        await client.predict(request())
                    self.assertEqual(caught.exception.diagnostic_code, "unexpected_reasoning")
        self.runtime.answer.data.content = "<think></think>no"
        async with self.client(output_format="answer_only", think=False) as client:
            self.assertEqual((await client.predict(request())).selected_option_id, "no")
        self.assertIsNone(self.runtime.sessions[-1].options["reasoning_effort"])

    async def test_lazy_isolated_tool_free_sessions_and_normalized_result(self):
        self.runtime.events[0].data.reasoning_effort = "high"
        client = self.client(think=True, reasoning_effort="high", max_tokens=2048)
        self.assertFalse(self.factory_options)
        async with client:
            first = await client.predict(request())
            await client.predict(request("different-local-id"))
            self.assertEqual(first.resolved_model, "resolved-gpt")
            self.assertEqual(first.request_id, request().request_id)
            self.assertEqual(first.selected_option_id, "yes")
            self.assertEqual(first.usage.reasoning_tokens, 14)
            self.assertIsNone(first.probabilities)
            self.assertIsNone(first.confidence)
            self.assertEqual(len(self.runtime.sessions), 2)
            self.assertEqual(self.runtime.prompts[0], self.runtime.prompts[1])
            self.assertNotIn("PRIVATE", self.runtime.prompts[0])
            self.assertEqual(self.factory_options[0]["mode"], "empty")
            self.assertFalse(self.factory_options[0]["use_logged_in_user"])
            directory = Path(self.factory_options[0]["base_directory"])
            self.assertTrue(directory.is_dir())
            for session in self.runtime.sessions:
                opts = session.options
                self.assertEqual(opts["available_tools"], [])
                self.assertEqual(opts["tools"], [])
                self.assertEqual(opts["reasoning_effort"], "high")
                self.assertEqual(opts["model_capabilities"].limits.max_output_tokens, 2048)
                self.assertEqual(opts["system_message"]["mode"], "replace")
                self.assertEqual(opts["system_message"]["content"], SYSTEM_MESSAGE)
                self.assertEqual(opts["memory"], {"enabled": False})
                self.assertEqual(opts["infinite_sessions"], {"enabled": False})
                self.assertFalse(opts["enable_config_discovery"])
                self.assertFalse(opts["enable_skills"])
                self.assertFalse(opts["enable_file_hooks"])
                self.assertEqual(opts["working_directory"], str(directory))
                self.assertEqual(opts["on_permission_request"](None, None).kind, "denied")
                self.assertEqual(opts["on_auto_mode_switch_request"](), "no")
                session.disconnect.assert_awaited_once()
                session.abort.assert_not_called()
                self.runtime.delete_session.assert_any_await(session.session_id)
        self.assertFalse(directory.exists())
        self.runtime.start.assert_awaited_once()
        self.runtime.stop.assert_awaited_once()
        await client.aclose()
        self.runtime.stop.assert_awaited_once()

    async def test_optional_dependency_api_signatures_without_runtime(self):
        if importlib.util.find_spec("copilot") is None:
            self.skipTest("Optional SDK not installed")
        from copilot import CopilotClient as SDKClient
        from copilot.session import ModelCapabilitiesOverride, ModelLimitsOverride
        async with self.client() as client:
            await client.predict(request())
        inspect.signature(SDKClient).bind(**self.factory_options[0])
        opts = self.runtime.sessions[0].options.copy()
        opts["model_capabilities"] = ModelCapabilitiesOverride(limits=ModelLimitsOverride(max_output_tokens=4096))
        inspect.signature(SDKClient.create_session).bind(None, **opts)

    async def test_env_token_and_existing_login_without_device_flow(self):
        for token in (None, "FAKE_ENV_TOKEN"):
            with patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": token} if token else {}, clear=True):
                async with create_client("github_copilot", model="test-gpt") as client:
                    await client.predict(request())
            self.assertEqual(self.factory_options[-1]["github_token"], token)
            self.assertEqual(self.factory_options[-1]["use_logged_in_user"], token is None)

    async def test_supported_thinking_controls_and_default(self):
        for think, explicit, expected in ((True, None, "medium"), (None, None, None),
                                          (None, "low", "low")):
            self.runtime.events[0].data.reasoning_effort = expected
            async with self.client(think=think, reasoning_effort=explicit) as client:
                await client.predict(request())
            self.assertEqual(self.runtime.sessions[-1].options["reasoning_effort"], expected)
        self.runtime.list_models.return_value = [model_info(False)]
        self.runtime.events[0].data.reasoning_effort = None
        self.runtime.events[0].data.reasoning_tokens = 0
        async with self.client(think=False) as client:
            await client.predict(request())
        self.assertIsNone(self.runtime.sessions[-1].options["reasoning_effort"])

    async def test_unsupported_thinking_never_sends_a_question(self):
        for info, settings in ((model_info(), {"think": False}),
                               (model_info(False), {"think": True}),
                               (model_info(), {"reasoning_effort": "max"})):
            self.runtime.list_models.return_value = [info]
            async with self.client(**settings) as client:
                with self.assertRaises(UnsupportedRequestError):
                    await client.predict(request())
        self.runtime.create_session.assert_not_called()
        self.assertEqual(self.runtime.prompts, [])

    async def test_model_and_auth_rejected_before_question(self):
        async with self.client(model="not-available") as client:
            with self.assertRaises(UnsupportedRequestError):
                await client.predict(request())
        self.runtime.get_auth_status.return_value = NS(isAuthenticated=False, statusMessage="PRIVATE_AUTH")
        async with self.client() as client:
            with self.assertRaises(ClientTransportError) as caught:
                await client.predict(request())
            self.assertNotIn("PRIVATE", str(caught.exception))
        self.runtime.create_session.assert_not_called()

    async def test_timeout_cancellation_and_sdk_failure_abort_and_clean_up(self):
        for error, expected in ((TimeoutError("PRIVATE"), ClientTimeoutError),
                                (RuntimeError("PRIVATE_SDK"), ClientTransportError),
                                (asyncio.CancelledError(), asyncio.CancelledError)):
            self.runtime.error = error
            async with self.client() as client:
                with self.assertRaises(expected) as caught:
                    await client.predict(request())
                self.assertNotIn("PRIVATE", str(caught.exception))
            session = self.runtime.sessions[-1]
            session.abort.assert_awaited_once()
            session.disconnect.assert_awaited_once()
            self.runtime.delete_session.assert_any_await(session.session_id)
        self.assertEqual(len(self.runtime.prompts), 3)

    async def test_wall_clock_timeout_during_send(self):
        async def hanging(*args, **kwargs):
            await asyncio.Future()
        with patch.object(FakeSession, "send_and_wait", new=hanging):
            async with self.client(timeout_seconds=0.01) as client:
                with self.assertRaises(ClientTimeoutError):
                    await client.predict(request())
        self.runtime.sessions[-1].abort.assert_awaited_once()

    async def test_external_and_repeated_cancellation_still_cleans_session(self):
        entered, cleaning, finish_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def hanging(*args, **kwargs):
            entered.set()
            await asyncio.Future()

        async def slow_abort():
            cleaning.set()
            await finish_cleanup.wait()

        with patch.object(FakeSession, "send_and_wait", new=hanging):
            async with self.client() as client:
                task = asyncio.create_task(client.predict(request()))
                await asyncio.wait_for(entered.wait(), timeout=1)
                session = self.runtime.sessions[-1]
                session.abort.side_effect = slow_abort
                task.cancel()
                await asyncio.wait_for(cleaning.wait(), timeout=1)
                task.cancel()
                finish_cleanup.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                session.abort.assert_awaited_once()
                session.disconnect.assert_awaited_once()
                self.runtime.delete_session.assert_any_await(session.session_id)

    async def test_reported_model_effort_or_no_think_contradictions_are_rejected(self):
        usage = self.runtime.events[0].data
        usage.model = "different-model"
        async with self.client() as client:
            with self.assertRaisesRegex(InvalidResponseError, "inconsistent"):
                await client.predict(request())
        usage.model = "resolved-gpt"
        async with self.client(think=True, reasoning_effort="high") as client:
            with self.assertRaisesRegex(InvalidResponseError, "different reasoning"):
                await client.predict(request())
        self.runtime.list_models.return_value = [model_info(False)]
        async with self.client(think=False) as client:
            with self.assertRaisesRegex(InvalidResponseError, "no-think"):
                await client.predict(request())

    async def test_initialization_failure_is_sanitized_and_resources_closed(self):
        self.runtime.start.side_effect = RuntimeError("PRIVATE_START_FAILURE")
        async with self.client() as client:
            with self.assertRaises(ClientTransportError) as caught:
                await client.predict(request())
            self.assertNotIn("PRIVATE", str(caught.exception))
            directory = Path(self.factory_options[0]["base_directory"])
        self.assertFalse(directory.exists())
        self.runtime.stop.assert_awaited_once()

    async def test_invalid_final_outputs_no_repair_or_retry(self):
        for content in ("", 'yes', '{"choice":"not-an-option"}', '<think>{"choice":"yes"}'):
            self.runtime.answer.data.content = content
            async with self.client() as client:
                with self.assertRaises(InvalidResponseError):
                    await client.predict(request())
        self.assertEqual(len(self.runtime.prompts), 4)

    async def test_sdk_internal_retry_or_multiple_usage_events_are_rejected(self):
        usage = self.runtime.events[0]
        for events in ([usage, usage], [usage, NS(type="assistant.turn_retry", data=NS())]):
            self.runtime.events = events
            async with self.client() as client:
                with self.assertRaises(InvalidResponseError):
                    await client.predict(request())
        self.assertEqual(len(self.runtime.prompts), 2)

    async def test_missing_usage_is_unknown_not_invented(self):
        self.runtime.events = []
        async with self.client() as client:
            result = await client.predict(request())
        self.assertEqual(result.resolved_model, "resolved-gpt")
        self.assertIsNone(result.usage.input_tokens)
        self.assertIsNone(result.usage.reasoning_tokens)

    async def test_malformed_metadata_is_rejected(self):
        usage = self.runtime.events[0].data
        for key, value in (("finish_reason", "length"), ("input_tokens", True),
                           ("reasoning_tokens", -1), ("content_filter_triggered", True)):
            previous = getattr(usage, key, None)
            setattr(usage, key, value)
            async with self.client() as client:
                with self.assertRaises(InvalidResponseError):
                    await client.predict(request())
            setattr(usage, key, previous)
        self.runtime.answer.data.tool_requests = [{}]
        async with self.client() as client:
            with self.assertRaises(InvalidResponseError):
                await client.predict(request())
        self.runtime.answer = None
        async with self.client() as client:
            with self.assertRaises(InvalidResponseError):
                await client.predict(request())

    async def test_close_unused_and_protocol_validation_are_side_effect_free(self):
        client = self.client()
        with self.assertRaises(UnsupportedRequestError):
            await client.predict(replace(request(), protocol_version="future"))
        await client.aclose()
        with self.assertRaises(ClientClosedError):
            await client.predict(request())
        self.assertFalse(self.factory_options)
        self.runtime.start.assert_not_called()

    def test_registry_listing_and_constructor_do_not_load_sdk(self):
        with patch("src.clients.github_copilot_client._sdk_types", side_effect=AssertionError("No import")):
            self.assertIn("github_copilot", available_clients())
            self.assertIsInstance(create_client("github_copilot"), GitHubCopilotClient)

    def test_invalid_settings(self):
        for kwargs in ({"output_format": "unknown"}, {"output_format": []}, {"output_format": None},
                       {"output_format": False}, {"think": "no"}, {"reasoning_effort": "none"},
                       {"think": False, "reasoning_effort": "low"},
                       {"max_tokens": -1}, {"model": ""}, {"timeout_seconds": 0},
                       {"github_token": ""}, {"github_token": "bad\nkey"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.client(**kwargs)


if __name__ == "__main__":
    unittest.main()