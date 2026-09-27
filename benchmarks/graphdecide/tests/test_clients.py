"""Offline client contract tests; all HTTP calls use httpx.MockTransport.

Install once from the repository root:
    python -m pip install -e .
Then run either:
    python tests/test_clients.py -v
    python -m unittest discover -s tests -v
No API key, model download, or live service is required.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx

from src.clients import registry
from src.clients.base import (
    BaseDecisionClient,
    ClientCapabilities,
    ClientClosedError,
    ClientTimeoutError,
    ClientTransportError,
    DecisionOption,
    DecisionRequest,
    DecisionResponse,
    InvalidResponseError,
    ProviderHTTPError,
    UnsupportedRequestError,
)
from src.clients.typesafe import TypeSafeClient


def make_request(request_id: str = "graph-1") -> DecisionRequest:
    return DecisionRequest(
        request_id=request_id,
        state={"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]},
        question="Is node 2 reachable from node 0?",
        options=(DecisionOption("yes", "Reachable"), DecisionOption("no")),
    )


def make_body() -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.8, "no": 0.2},
                # Vendor confidence is NOT necessarily the top probability.
                "confidence": 0.6,
            }
        },
        "usage": {"input_tokens": 123, "output_tokens": 7},
    }


class StubClient(BaseDecisionClient):
    """Test-only fixture, not a benchmark model."""

    def __init__(self, model: str = "test-stub") -> None:
        self.model = model
        self.seen = []
        self.closed = False

    @property
    def capabilities(self) -> ClientCapabilities:
        return ClientCapabilities(max_options=2)

    async def predict(self, request: DecisionRequest) -> DecisionResponse:
        self.validate_request(request)
        self.seen.append(request.request_id)
        return DecisionResponse(
            request_id=request.request_id,
            selected_option_id=request.options[0].id,
            resolved_model=self.model,
        )

    async def aclose(self) -> None:
        self.closed = True


class RequestTests(unittest.TestCase):
    def test_base_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            BaseDecisionClient()

    def test_option_validation(self) -> None:
        for option_id in ("", "  ", None, 1):
            with self.subTest(option_id=option_id), self.assertRaises(ValueError):
                DecisionOption(option_id)
        with self.assertRaises(ValueError):
            DecisionOption("yes", 123)

    def test_invalid_requests(self) -> None:
        changes = [
            {"request_id": ""},
            {"question": " "},
            {"protocol_version": ""},
            {"state": 123},
            {"state": {"bad": {1, 2}}},
            {"state": {"bad": float("nan")}},
            {"options": ()},
            {"options": (DecisionOption("yes"),)},
            {"options": (DecisionOption("yes"), DecisionOption("yes"))},
            {"options": ("yes", "no")},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(make_request(), **change)

    def test_ordered_options_are_copied(self) -> None:
        options = [DecisionOption("no"), DecisionOption("yes")]
        request = replace(make_request(), options=options)
        options.reverse()
        self.assertEqual([o.id for o in request.options], ["no", "yes"])

    def test_structured_or_string_state(self) -> None:
        for state in ("0 -- 1", {"edges": [[0, 1]]}, [[0, 1]]):
            with self.subTest(state=state):
                self.assertEqual(replace(make_request(), state=state).state, state)


class BaseClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequential_fallback_and_context_manager(self) -> None:
        async with StubClient() as client:
            results = await client.predict_many([make_request("a"), make_request("b")])
            self.assertEqual(client.seen, ["a", "b"])
            self.assertEqual([r.request_id for r in results], ["a", "b"])
            self.assertIsNone(results[0].probabilities)
            self.assertIsNone(results[0].usage.input_tokens)
            self.assertEqual(await client.predict_many([]), [])
            self.assertFalse(client.capabilities.supports_native_batch)
        self.assertTrue(client.closed)

    async def test_batch_prevalidation_avoids_partial_calls(self) -> None:
        async with StubClient() as client:
            bad = replace(make_request(), protocol_version="future")
            with self.assertRaises(UnsupportedRequestError):
                await client.predict_many([make_request(), bad])
            self.assertEqual(client.seen, [])


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Isolate decorator registration so tests never depend on execution order.
        self.registrations = patch.dict(registry._REGISTERED, {}, clear=True)
        self.registrations.start()
        self.addCleanup(self.registrations.stop)

    async def test_custom_registration_and_constructor(self) -> None:
        self.assertIs(registry.register_client(" Stub ")(StubClient), StubClient)
        async with registry.create_client("STUB", model="fixture") as client:
            self.assertIsInstance(client, StubClient)
            self.assertEqual(client.model, "fixture")
        self.assertEqual(registry.available_clients(), ("github_copilot", "stub", "typesafe", "vllm"))

    def test_duplicate_and_invalid_registration(self) -> None:
        registry.register_client("stub")(StubClient)
        for name in ("stub", "typesafe", " TYPESAFE "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                registry.register_client(name)(StubClient)
        with self.assertRaises(TypeError):
            registry.register_client("invalid")(object)

    def test_unknown_and_empty_names(self) -> None:
        for name in ("", " ", None, "missing"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                registry.create_client(name)

    def test_listing_does_not_import_backends(self) -> None:
        with patch.object(registry, "import_module") as import_module:
            self.assertIn("typesafe", registry.available_clients())
            import_module.assert_not_called()

    async def test_builtin_factory(self) -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=make_body()))
        async with registry.create_client(
            "typesafe", api_key="test-key", transport=transport
        ) as client:
            self.assertIsInstance(client, TypeSafeClient)
            self.assertEqual((await client.predict(make_request())).resolved_model, "jev-1.13.0")


class TypeSafeTests(unittest.IsolatedAsyncioTestCase):
    async def test_wire_contract_and_normalization(self) -> None:
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.assertEqual(request.headers["content-type"], "application/json")
            body = json.loads(request.content)
            self.assertEqual(body, {
                "state": make_request().state,
                "model": "jev-pinned",
                "questions": {"decision": {
                    "type": "choice",
                    "instructions": make_request().question,
                    "criteria": {"yes": "Reachable", "no": None},
                }},
            })
            self.assertEqual(list(body["questions"]["decision"]["criteria"]), ["yes", "no"])
            return httpx.Response(200, json=make_body())

        async with TypeSafeClient(
            api_key="test-key", model="jev-pinned", transport=httpx.MockTransport(handler)
        ) as client:
            result = await client.predict(make_request())
            self.assertEqual(result.request_id, "graph-1")
            self.assertEqual(result.selected_option_id, "yes")
            self.assertEqual(result.resolved_model, "jev-1.13.0")
            self.assertEqual(result.probabilities, {"yes": 0.8, "no": 0.2})
            self.assertEqual(result.probability_kind, "native")
            self.assertEqual(result.confidence, 0.6)
            self.assertEqual(result.usage.input_tokens, 123)
            self.assertEqual(result.usage.output_tokens, 7)
            self.assertIsNone(result.usage.reasoning_tokens)
            self.assertEqual(result.raw_output, make_body())
            self.assertEqual(result.status, "success")
            self.assertIsNone(client.capabilities.max_context_tokens)
            self.assertFalse(client.capabilities.supports_shared_state_questions)
        self.assertEqual(len(requests), 1)

    async def test_candidate_ids_and_permutations_are_preserved(self) -> None:
        request = replace(make_request(), options=(DecisionOption("node_91"), DecisionOption("node_7")))

        def handler(wire_request: httpx.Request) -> httpx.Response:
            criteria = json.loads(wire_request.content)["questions"]["decision"]["criteria"]
            self.assertEqual(list(criteria), ["node_91", "node_7"])
            body = make_body()
            body["answers"]["decision"].update(
                choice="node_7", probabilities={"node_7": 0.8, "node_91": 0.2}
            )
            return httpx.Response(200, json=body)

        async with TypeSafeClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
            result = await client.predict(request)
            self.assertEqual(result.selected_option_id, "node_7")
            self.assertEqual(list(result.probabilities), ["node_91", "node_7"])

    async def test_environment_key_and_custom_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer env-key")
            self.assertEqual(str(request.url), "https://gateway.example/v1/systemone")
            return httpx.Response(200, json=make_body())

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "env-key"}):
            async with TypeSafeClient(
                base_url="https://gateway.example/v1/", transport=httpx.MockTransport(handler)
            ) as client:
                await client.predict(make_request())

    def test_missing_key_and_invalid_settings(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TYPESAFE_API_KEY"):
                TypeSafeClient()
        for kwargs in (
            {"api_key": ""}, {"api_key": "bad\nkey"}, {"model": ""},
            {"timeout_seconds": 0}, {"timeout_seconds": float("inf")},
            {"timeout_seconds": float("nan")}, {"timeout_seconds": True},
            {"base_url": "http://api.typesafe.ai/v1"},
            {"base_url": "https://user:pass@example.com/v1"},
            {"base_url": "https://example.com/v1?key=secret"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TypeSafeClient(**({"api_key": "test-key"} | kwargs))

    async def test_limit_and_protocol_fail_before_http(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json=make_body())

        async with TypeSafeClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
            excessive = replace(make_request(), options=tuple(DecisionOption(str(i)) for i in range(256)))
            for request in (excessive, replace(make_request(), protocol_version="v2")):
                with self.assertRaises(UnsupportedRequestError):
                    await client.predict(request)
        self.assertEqual(calls, [])

    async def test_exactly_255_candidates(self) -> None:
        request = replace(make_request(), options=tuple(DecisionOption(str(i)) for i in range(255)))

        def handler(wire_request: httpx.Request) -> httpx.Response:
            self.assertEqual(len(json.loads(wire_request.content)["questions"]["decision"]["criteria"]), 255)
            body = make_body()
            body["answers"]["decision"].update(
                choice="0", probabilities={str(i): 1.0 if i == 0 else 0.0 for i in range(255)}
            )
            return httpx.Response(200, json=body)

        async with TypeSafeClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
            self.assertEqual((await client.predict(request)).selected_option_id, "0")

    async def test_http_failures_never_retry_or_expose_body(self) -> None:
        for status in (302, 401, 422, 429, 500, 529):
            calls = []

            def handler(request: httpx.Request) -> httpx.Response:
                calls.append(request)
                return httpx.Response(
                    status,
                    headers={"retry-after": "3", "location": "https://other.example"},
                    json={"error": "sensitive-provider-body"},
                )

            with self.subTest(status=status):
                async with TypeSafeClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
                    with self.assertRaises(ProviderHTTPError) as caught:
                        await client.predict(make_request())
                error = caught.exception
                self.assertEqual(error.status_code, status)
                self.assertEqual(error.retry_after, "3")
                self.assertEqual(error.retryable, status in (429, 500, 529))
                self.assertNotIn("sensitive-provider-body", str(error))
                self.assertNotIn("test-key", str(error))
                self.assertEqual(len(calls), 1)

    async def test_transport_failures_are_typed(self) -> None:
        for original, expected in (
            (httpx.ReadTimeout, ClientTimeoutError),
            (httpx.ConnectError, ClientTransportError),
        ):
            def handler(request: httpx.Request) -> httpx.Response:
                raise original("sensitive-transport-message", request=request)

            with self.subTest(original=original):
                async with TypeSafeClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
                    with self.assertRaises(expected) as caught:
                        await client.predict(make_request())
                self.assertNotIn("sensitive", str(caught.exception))

    async def test_cancellation_propagates(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise asyncio.CancelledError()

        async with TypeSafeClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(asyncio.CancelledError):
                await client.predict(make_request())

    async def test_non_json_response_is_preserved(self) -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text="not JSON"))
        async with TypeSafeClient(api_key="test-key", transport=transport) as client:
            with self.assertRaises(InvalidResponseError) as caught:
                await client.predict(make_request())
            self.assertEqual(caught.exception.raw_output, "not JSON")

    async def test_malformed_choice_responses_are_rejected(self) -> None:
        answer_changes = [
            {"type": "noul"}, {"choice": "unknown"}, {"choice": ["yes"]},
            {"choice": "no"}, {"probabilities": {"yes": 1.0}},
            {"probabilities": {"yes": 0.8, "no": 0.2, "other": 0}},
            {"probabilities": {"yes": 0.6, "no": 0.2}},
            {"probabilities": {"yes": True, "no": 0}},
            {"probabilities": {"yes": 1.1, "no": -0.1}},
            {"probabilities": {"yes": "0.8", "no": 0.2}},
            {"confidence": None}, {"confidence": True}, {"confidence": 2},
        ]
        bodies = [None, [], {}, {"model": "jev", "answers": []}]
        for change in answer_changes:
            body = make_body()
            body["answers"]["decision"].update(change)
            bodies.append(body)
        for change in (
            {"model": ""}, {"usage": None}, {"usage": {"input_tokens": -1}},
            {"usage": {"input_tokens": True}}, {"usage": {"output_tokens": 1.5}},
            {"answers": {"wrong_question": make_body()["answers"]["decision"]}},
        ):
            bodies.append(make_body() | change)
        for body in bodies:
            with self.subTest(body=body):
                # httpx's json=None means no body, not the JSON literal null.
                transport = httpx.MockTransport(
                    lambda _: httpx.Response(
                        200, content=json.dumps(body),
                        headers={"content-type": "application/json"},
                    )
                )
                async with TypeSafeClient(api_key="test-key", transport=transport) as client:
                    with self.assertRaises(InvalidResponseError) as caught:
                        await client.predict(make_request())
                    self.assertEqual(caught.exception.raw_output, body)

    def test_nonfinite_probabilities_are_rejected(self) -> None:
        # JSON decoders may accept NaN/Infinity even though they are not JSON numbers.
        for value in (float("nan"), float("inf"), float("-inf")):
            body = make_body()
            body["answers"]["decision"]["probabilities"]["yes"] = value
            with self.subTest(value=value), self.assertRaises(InvalidResponseError):
                TypeSafeClient._parse_response(make_request(), body)

    def test_native_failure_diagnostics_distinguish_validation_stages(self) -> None:
        cases = (
            ({"model": ""}, {}, "invalid_model"),
            ({"answers": {}}, {}, "invalid_native_answer"),
            ({}, {"choice": "unknown"}, "invalid_choice"),
            ({}, {"probabilities": {"yes": 1}}, "invalid_probabilities"),
            ({}, {"probabilities": {"yes": 0.6, "no": 0.2}}, "invalid_probability_sum"),
            ({}, {"choice": "no"}, "choice_probability_mismatch"),
            ({}, {"confidence": None}, "invalid_confidence"),
            ({"usage": None}, {}, "invalid_usage"),
        )
        for body_change, answer_change, expected in cases:
            body = make_body()
            body["answers"]["decision"].update(answer_change)
            body.update(body_change)
            with self.subTest(expected=expected), self.assertRaises(InvalidResponseError) as caught:
                TypeSafeClient._parse_response(make_request(), body)
            self.assertEqual(caught.exception.diagnostic_code, expected)
            self.assertEqual(caught.exception.raw_output, body)

    async def test_missing_counts_are_unknown_and_ties_are_allowed(self) -> None:
        body = make_body()
        body["usage"] = {}
        body["answers"]["decision"].update(choice="no", probabilities={"yes": 0.5, "no": 0.5})
        original = copy.deepcopy(body)
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        async with TypeSafeClient(api_key="test-key", transport=transport) as client:
            result = await client.predict(make_request())
            self.assertIsNone(result.usage.input_tokens)
            self.assertIsNone(result.usage.output_tokens)
            self.assertEqual(result.selected_option_id, "no")
            self.assertEqual(result.raw_output, original)

    async def test_close_is_idempotent_and_blocks_further_requests(self) -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=make_body()))
        client = TypeSafeClient(api_key="test-key", transport=transport)
        await client.aclose()
        await client.aclose()
        with self.assertRaises(ClientClosedError):
            await client.predict(make_request())


if __name__ == "__main__":
    unittest.main()