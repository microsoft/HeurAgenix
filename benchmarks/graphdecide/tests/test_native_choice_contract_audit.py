"""Offline native contract cases; no credentials, network, or production edits."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from scripts import native_choice_contract_audit as audit
from src.benchmark.config import ModelConfig
from src.clients.base import (
    DecisionOption, DecisionRequest, InvalidResponseError,
    ProviderHTTPError,
)
from src.clients.typesafe import TypeSafeClient

PRIVATE = "DO_NOT_SAVE_PROVIDER_TEXT"
CONFIG = ModelConfig(provider="typesafe", model="jev-1.13.0", timeout_seconds=30)
REQUEST = DecisionRequest("test", {"nodes": [0, 1], "edges": [[0, 1]]}, "Adjacent?",
                          (DecisionOption("yes"), DecisionOption("no")))
CASE = {"instance_id": "test", "task": "adjacency", "node_count": 8, "label": "yes"}


def body():
    return {
        "model": CONFIG.model,
        "answers": {"decision": {"type": "choice", "choice": "yes",
                                 "probabilities": {"yes": .6, "no": .4}, "confidence": .6}},
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "reasoning": PRIVATE,
    }


class NativeObservationTests(unittest.IsolatedAsyncioTestCase):
    async def check(self, payload, *, status=200, malformed=False):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(status, text=PRIVATE if malformed else json.dumps(payload))

        async with TypeSafeClient(api_key="offline-placeholder", model=CONFIG.model,
                                  transport=httpx.MockTransport(handler)) as client:
            row = await audit.observe(client, REQUEST, CASE, CONFIG)
        self.assertEqual(len(calls), 1)
        self.assertNotIn(PRIVATE, json.dumps(row))
        self.assertNotIn("raw_output", row)
        return row

    async def test_success_paired_from_same_response(self):
        row = await self.check(body())
        self.assertEqual(row["strict"]["status"], "success")
        self.assertTrue(row["strict"]["correct"])
        self.assertTrue(row["action_observation"]["eligible"])
        self.assertTrue(row["action_observation"]["correct"])

    async def test_sum_099_strict_error_unchanged_and_observation_eligible(self):
        payload = body()
        payload["answers"]["decision"]["probabilities"] = {"yes": .6, "no": .39}
        before = deepcopy(payload)
        with self.assertRaises(InvalidResponseError) as caught:
            TypeSafeClient._parse_response(REQUEST, payload)
        self.assertEqual(str(caught.exception), "probabilities must sum to 1")
        self.assertEqual(caught.exception.diagnostic_code, "invalid_probability_sum")
        self.assertIs(caught.exception.raw_output, payload)
        row = await self.check(payload)
        self.assertEqual(row["strict"]["diagnostic_code"], "invalid_probability_sum")
        self.assertFalse(row["strict"]["correct"])
        self.assertTrue(row["action_observation"]["correct"])
        self.assertAlmostEqual(row["action_observation"]["native_diagnostics"]["probability_sum"], .99)
        self.assertEqual(payload, before)

    async def test_argmax_mismatch_not_inferred_from_probabilities(self):
        payload = body()
        payload["answers"]["decision"]["probabilities"] = {"yes": .4, "no": .6}
        with self.assertRaises(InvalidResponseError) as caught:
            TypeSafeClient._parse_response(REQUEST, payload)
        self.assertEqual(str(caught.exception), "choice does not have the highest probability")
        row = await self.check(payload)
        self.assertEqual(row["strict"]["diagnostic_code"], "choice_probability_mismatch")
        self.assertEqual(row["action_observation"]["selected_option_id"], "yes")
        self.assertTrue(row["action_observation"]["correct"])

    async def test_supplied_but_wrong_choice_is_not_correct(self):
        payload = body()
        payload["answers"]["decision"]["choice"] = "no"
        row = await self.check(payload)
        self.assertTrue(row["action_observation"]["eligible"])
        self.assertFalse(row["action_observation"]["correct"])

    async def test_invalid_choice_never_eligible_or_echoed(self):
        for choice in (PRIVATE, 1, True, None, [], {}):
            with self.subTest(choice=type(choice).__name__):
                payload = body()
                payload["answers"]["decision"]["choice"] = choice
                row = await self.check(payload)
                self.assertEqual(row["strict"]["diagnostic_code"], "invalid_choice")
                self.assertFalse(row["action_observation"]["eligible"])
                self.assertIsNone(row["action_observation"]["selected_option_id"])

    async def test_model_identity_checked_even_for_parser_success(self):
        for model in (PRIVATE, "", None, 7):
            payload = body()
            payload["model"] = model
            row = await self.check(payload)
            self.assertEqual(row["strict"]["diagnostic_code"], "invalid_model")
            self.assertFalse(row["action_observation"]["eligible"])
            self.assertFalse(row["action_observation"]["native_diagnostics"]["model_matches"])

    async def test_shape_and_answer_type_checked(self):
        for answers in (None, [], {}, {"other": {}},
                        {"decision": {"type": "text", "choice": "yes"}},
                        {"decision": None}, {"decision": body()["answers"]["decision"], "extra": {}}):
            payload = body()
            payload["answers"] = answers
            row = await self.check(payload)
            self.assertEqual(row["strict"]["diagnostic_code"], "invalid_native_answer")
            self.assertFalse(row["action_observation"]["eligible"])

    async def test_nonprobability_failure_and_hidden_failure_not_salvaged(self):
        for field, value, reason in (
                ("confidence", True, "invalid_confidence"),
                ("confidence", None, "invalid_confidence"),
                ("usage", {"output_tokens": -1}, "invalid_usage"),
                ("usage", None, "invalid_usage")):
            for bad_sum in (False, True):
                payload = body()
                target = payload if field == "usage" else payload["answers"]["decision"]
                target[field] = value
                if bad_sum:
                    payload["answers"]["decision"]["probabilities"]["no"] = .39
                row = await self.check(payload)
                self.assertFalse(row["action_observation"]["eligible"])
                self.assertIn(reason, row["action_observation"]["ineligibility_reasons"])
                self.assertEqual(row["strict"]["diagnostic_code"],
                                 "invalid_probability_sum" if bad_sum else reason)

    async def test_invalid_probabilities_not_blanket_salvaged(self):
        for probabilities in (None, {}, {"yes": 1}, {"yes": True, "no": 0},
                              {"yes": -1, "no": 2}, {"yes": "0.6", "no": .4},
                              {"yes": float("nan"), "no": .4},
                              {"yes": float("inf"), "no": .4}):
            payload = body()
            payload["answers"]["decision"]["probabilities"] = probabilities
            row = await self.check(payload)
            self.assertFalse(row["action_observation"]["eligible"])
            self.assertEqual(row["strict"]["diagnostic_code"], "invalid_probabilities")
            json.dumps(row, allow_nan=False)

    async def test_malformed_missing_bodies_explicit(self):
        for payload in (None, [], PRIVATE, {}):
            row = await self.check(payload)
            self.assertFalse(row["action_observation"]["eligible"])
            self.assertIn(row["strict"]["diagnostic_code"], {"invalid_response", "invalid_model"})
        row = await self.check(None, malformed=True)
        self.assertEqual(row["strict"]["diagnostic_code"], "invalid_response")
        self.assertIn("malformed_response", row["action_observation"]["ineligibility_reasons"])

    async def test_http_statuses_use_existing_fatal_conventions(self):
        for status in (401, 403, 408, 429, 500, 503):
            row = await self.check(None, status=status, malformed=True)
            self.assertEqual(row["strict"]["http_status"], status)
            self.assertEqual(row["strict"]["fatal"], audit.suite.failure(ProviderHTTPError(status))["fatal"])
            self.assertFalse(row["action_observation"]["eligible"])

    async def test_timeout_transport_are_explicit_and_never_retried(self):
        for error, code, fatal in ((httpx.ReadTimeout(PRIVATE), "timeout", False),
                                   (httpx.ConnectError(PRIVATE), "client_error", True)):
            calls = []

            def handler(request):
                calls.append(request)
                raise error

            async with TypeSafeClient(api_key="offline-placeholder", model=CONFIG.model,
                                      transport=httpx.MockTransport(handler)) as client:
                row = await audit.observe(client, REQUEST, CASE, CONFIG)
            self.assertEqual(len(calls), 1)
            self.assertEqual(row["strict"]["diagnostic_code"], code)
            self.assertEqual(row["strict"]["fatal"], fatal)
            self.assertFalse(row["action_observation"]["eligible"])
            self.assertNotIn(PRIVATE, json.dumps(row))


def instances():
    rows = []
    for task in audit.TASKS:
        for size in audit.SIZES:
            labels = range(size) if task == "degree_exact" else ("yes", "no")
            for label in labels:
                for repeat in range(4):
                    rows.append({"id": f"{task}:{size}:{label}:{repeat}", "task": task,
                                 "kind": "exact", "state": {"nodes": list(range(size))},
                                 "private": {"answer": label}})
    return rows


class SelectionAndMetricTests(unittest.TestCase):
    def test_existing_output_refused_before_reading_inputs_or_creating_client(self):
        with patch.object(audit, "create_client", side_effect=AssertionError("No inference")), \
                patch.object(audit.suite, "read_rows", side_effect=AssertionError("No input read")):
            with self.assertRaises(audit.suite.Refusal):
                asyncio.run(audit.run(audit.REPO, audit.REPO))

    def test_exact_quotas_balance_distinct_degrees(self):
        selected = audit.select_cases(instances())
        self.assertEqual(len(selected), 120)
        self.assertEqual(len({i["id"] for i in selected}), 120)
        for task in audit.TASKS:
            for size in audit.SIZES:
                cell = [i for i in selected if i["task"] == task and len(i["state"]["nodes"]) == size]
                self.assertEqual(len(cell), 4)
                labels = [i["private"]["answer"] for i in cell]
                if task == "degree_exact":
                    self.assertEqual(len(set(labels)), 4)
                else:
                    self.assertEqual(labels.count("yes"), 2)
                    self.assertEqual(labels.count("no"), 2)

    def test_selection_deterministic_order_invariant_and_ignores_predictions(self):
        original = instances()
        expected = [i["id"] for i in audit.select_cases(original)]
        changed = deepcopy(original)
        for i in changed:
            i.update(prediction=PRIVATE, status="failure", diagnostic_code=PRIVATE)
        with patch.object(audit.suite, "read_rows", side_effect=AssertionError("No run artifacts")):
            self.assertEqual([i["id"] for i in audit.select_cases(changed[::-1])], expected)

    def test_short_or_duplicate_pool_refused(self):
        for values in (instances()[10:], instances() + [instances()[0]]):
            with self.assertRaises(audit.suite.Refusal):
                audit.select_cases(values)

    def test_partial_fatal_run_keeps_planned_denominator(self):
        cases = [CASE] * 120
        row = {"strict": {"status": "failure", "correct": False, "diagnostic_code": "client_error"},
               "action_observation": {"eligible": False, "correct": None,
                                      "ineligibility_reasons": ["missing_response"]}}
        result = audit.metrics(cases, [row])
        self.assertEqual(result["planned"], 120)
        self.assertEqual(result["unattempted"], 119)
        self.assertIsNone(result["action_observation_accuracy_eligible"])
        self.assertIsNone(result["strict_accuracy_successful"])

    def test_paired_metrics_separate_strict_and_action(self):
        row = {"strict": {"status": "failure", "correct": False,
                          "diagnostic_code": "invalid_probability_sum"},
               "action_observation": {"eligible": True, "correct": True,
                                      "ineligibility_reasons": []}}
        result = audit.metrics([CASE] * 120, [row])
        self.assertEqual(result["strict_correct"], 0)
        self.assertEqual(result["action_observation_correct"], 1)
        self.assertEqual(result["action_observation_accuracy_planned"], 1 / 120)
        self.assertEqual(result["strict_failure_action_eligible"], 1)


class LockedRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def exercise_runner(self, fatal_at=None):
        source_root = audit.REPO / "source-unit"
        output = audit.REPO / "audit-unit"
        source = {"models": {"jev": asdict(CONFIG)}, "input_sha256": {}, "code_sha256": {}}
        lock_events, captured = [], {}
        held = False

        @contextmanager
        def lock(models):
            nonlocal held
            self.assertEqual(models, ("jev",))
            lock_events.append("enter")
            held = True
            try:
                yield
            finally:
                held = False
                lock_events.append("exit")

        client = AsyncMock()
        calls = []

        async def predict(request):
            self.assertTrue(held)
            calls.append(request.request_id)
            if len(calls) == fatal_at:
                raise ProviderHTTPError(429)
            return TypeSafeClient._parse_response(request, body())

        client.predict.side_effect = predict

        def factory(provider, **kwargs):
            self.assertTrue(held)
            self.assertEqual(provider, "typesafe")
            self.assertEqual(kwargs, CONFIG.client_kwargs())
            return client

        def write_json(path, value):
            captured[path.name] = deepcopy(value)

        with ExitStack() as stack:
            for target, name, kwargs in (
                (audit, "verified_inputs", {"return_value": (source_root, source, instances(), {})}),
                (audit, "source_stable", {}),
                (audit, "create_client", {"side_effect": factory}),
                (audit.tasks, "next_request", {"side_effect": lambda i, _: replace(REQUEST, request_id=i["id"])}),
                (audit.suite, "read_rows", {"return_value": instances()}),
                (audit.suite, "read_json", {"return_value": "hash"}),
                (audit.suite, "digest", {"return_value": "hash"}),
                (audit.suite, "write_json", {"side_effect": write_json}),
                (audit.suite, "append_row", {}),
                (audit.suite, "lane_locks", {"side_effect": lock}),
                (audit.suite.shared, "credential_preflight", {}),
                (Path, "exists", {"return_value": False}),
                (Path, "mkdir", {}),
                (Path, "chmod", {}),
                (Path, "iterdir", {"return_value": []}),
                (Path, "open", {"return_value": StringIO()}),
            ):
                stack.enter_context(patch.object(target, name, **kwargs))
            result = await audit.run(source_root, output)
        self.assertEqual(lock_events, ["enter", "exit"])
        client.initialize.assert_awaited_once()
        client.aclose.assert_awaited_once()
        self.assertEqual(captured["manifest.json"]["planned_calls"], 120)
        self.assertEqual(captured["summary.json"], result)
        return result, calls

    async def test_all_120_calls_under_standard_suite_lock_without_retries(self):
        result, calls = await self.exercise_runner()
        self.assertEqual(len(calls), 120)
        self.assertEqual(len(set(calls)), 120)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["overall"]["attempted"], 120)
        self.assertEqual(result["overall"]["unattempted"], 0)

    async def test_fatal_http_stops_calls_and_keeps_all_planned_cases(self):
        result, calls = await self.exercise_runner(fatal_at=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["diagnostic_code"], "http_error")
        self.assertEqual(result["overall"]["planned"], 120)
        self.assertEqual(result["overall"]["unattempted"], 117)


if __name__ == "__main__":
    unittest.main()
