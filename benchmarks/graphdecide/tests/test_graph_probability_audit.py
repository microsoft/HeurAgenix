"""Offline arithmetic, response-contract, privacy, and bounded-execution checks."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from scripts import graph_probability_audit as audit
from src.clients.base import (
    ClientTimeoutError, ClientTransportError, DecisionOption, DecisionRequest,
    DecisionResponse, InvalidResponseError, ProviderHTTPError, TokenUsage,
)
from src.clients.typesafe import TypeSafeClient

PRIVATE = "PRIVATE_PROVIDER_OR_REFERENCE_SENTINEL"
CONFIG = audit.evaluation.configuration("jev_action")[0]
REQUEST = DecisionRequest("case", {"nodes": [0, 1], "edges": [[0, 1]]}, "Adjacent?",
                          (DecisionOption("yes"), DecisionOption("no")))


def case_for(request=REQUEST, label="yes", task="adjacency"):
    public = audit.suite.json_value(asdict(request))
    return {"instance_id": request.request_id, "task": task, "node_count": 8, "label": label,
            "request": public, "request_sha256": audit.value_hash(public)}


def body():
    return {"model": CONFIG.model, "usage": {"input_tokens": 12, "output_tokens": 2},
            "reasoning": PRIVATE,
            "answers": {"decision": {"type": "choice", "choice": "yes",
                                     "probabilities": {"yes": .6, "no": .4}, "confidence": .7}}}


def row_for(case, probabilities, choice, *, eligible=True):
    return {
        "instance_id": case["instance_id"], "request_sha256": case["request_sha256"],
        "strict": {"status": "success", "diagnostic_code": None, "fatal": False},
        "native_diagnostics": None, "selected_option_id": choice if eligible else None,
        "action_eligible": eligible, "action_ineligibility_reasons": [] if eligible else ["strict_failure"],
        "confidence": None, "latency_seconds": 0,
        **audit.vector_record(audit.request_from(case), probabilities, eligible),
    }


class VectorAndScoringTests(unittest.TestCase):
    def test_summed_multiclass_brier_and_natural_log_loss(self):
        request = replace(REQUEST, options=tuple(DecisionOption(v) for v in ("0", "1", "2")))
        case = case_for(request, "1", "degree_exact")
        score = audit.score_row(case, row_for(case, {"0": .2, "1": .3, "2": .5}, "2"))
        self.assertAlmostEqual(score["brier"], .78)
        self.assertAlmostEqual(score["log_loss"], -math.log(.3))
        self.assertAlmostEqual(score["uniform_brier"], 2 / 3)
        self.assertAlmostEqual(score["uniform_log_loss"], math.log(3))
        self.assertFalse(score["action_correct"])

    def test_binary_brier_not_divided_by_class_count(self):
        case = case_for()
        score = audit.score_row(case, row_for(case, {"yes": .6, "no": .4}, "yes"))
        self.assertAlmostEqual(score["brier"], .32)
        self.assertEqual(score["uniform_brier"], .5)

    def test_zero_true_probability_explicit_infinite_without_epsilon(self):
        case = case_for()
        row = row_for(case, {"yes": 0.0, "no": 1.0}, "no")
        scores, result = audit.replay([case], [row])
        self.assertEqual(scores[0]["brier"], 2)
        self.assertTrue(scores[0]["log_loss_infinite"])
        self.assertIsNone(scores[0]["log_loss"])
        task = result["by_task"]["adjacency"]
        self.assertEqual(task["log_loss_infinite_count"], 1)
        self.assertTrue(task["mean_log_loss_is_infinite"])
        self.assertIsNone(task["mean_log_loss"])
        self.assertEqual(task["finite_log_loss_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_perfect_vector_and_subnormal_probability_not_clipped(self):
        case = case_for()
        for p in (1.0, 5e-324):
            score = audit.score_row(case, row_for(case, {"yes": p, "no": 1 - p}, "yes"))
            self.assertEqual(score["log_loss"], -math.log(p))
            self.assertFalse(score["log_loss_infinite"])
        self.assertEqual(audit.score_row(case, row_for(case, {"yes": 1, "no": 0}, "yes"))["brier"], 0)

    def test_invalid_keys_types_nan_infinity_range_and_sums_omitted(self):
        for vector, reason in (
            (None, "missing_probabilities"), ({}, "invalid_probability_keys"),
            ({"yes": 1}, "invalid_probability_keys"),
            ({"yes": .6, "no": .4, PRIVATE: 0}, "invalid_probability_keys"),
            ({"yes": True, "no": 0}, "invalid_probability_values"),
            ({"yes": float("nan"), "no": .4}, "invalid_probability_values"),
            ({"yes": float("inf"), "no": .4}, "invalid_probability_values"),
            ({"yes": -.1, "no": 1.1}, "invalid_probability_values"),
            ({"yes": ".6", "no": .4}, "invalid_probability_values"),
            ({"yes": .6, "no": .39}, "invalid_probability_sum"),
        ):
            with self.subTest(reason=reason):
                row = audit.vector_record(REQUEST, vector, True)
                self.assertIsNone(row["probabilities"])
                self.assertIn(reason, row["vector_omission_reasons"])
                self.assertNotIn(PRIVATE, json.dumps(row, allow_nan=False))

    def test_within_tolerance_values_are_retained_without_normalization(self):
        vector = {"yes": .6000004, "no": .4}
        result = audit.vector_record(REQUEST, vector, True)
        self.assertEqual(result["probabilities"], vector)
        self.assertNotEqual(sum(result["probabilities"].values()), 1)
        self.assertIsNone(audit.vector_record(REQUEST, {"yes": .600002, "no": .4}, True)["probabilities"])

    def test_reliability_fixed_edges_and_final_closed_bin(self):
        case = case_for()
        for p, expected in ((0, 0), (.19999, 0), (.2, 1), (.4, 2), (.6, 3), (.8, 4), (1, 4)):
            score = audit.score_row(case, row_for(case, {"yes": p, "no": 1 - p}, "yes"))
            self.assertEqual(score["reliability_bin"], expected)
            self.assertEqual(score["selected_probability"], p)

    def test_reliability_uses_selected_actual_action_not_maximum(self):
        case = case_for()
        scores, result = audit.replay([case], [row_for(case, {"yes": .2, "no": .8}, "yes")])
        self.assertTrue(scores[0]["choice_argmax_disagreement"])
        self.assertTrue(scores[0]["action_correct"])
        bins = result["by_task"]["adjacency"]["reliability_bins"]
        self.assertEqual(bins[1]["count"], 1)
        self.assertEqual(bins[1]["mean_selected_probability"], .2)
        self.assertEqual(bins[1]["actual_action_accuracy"], 1)
        self.assertEqual(bins[4]["count"], 0)
        self.assertIsNone(bins[4]["actual_action_accuracy"])

    def test_uniform_reference_uses_identical_valid_rows_and_scheduled_accuracy(self):
        cases = [case_for(replace(REQUEST, request_id=str(i))) for i in range(3)]
        rows = [row_for(cases[0], {"yes": .6, "no": .4}, "yes"),
                row_for(cases[1], {"yes": .6, "no": .39}, "yes")]
        _, result = audit.replay(cases, rows)
        task = result["by_task"]["adjacency"]
        self.assertEqual(task["action_correct"], 2)
        self.assertEqual(task["action_accuracy_all_scheduled"], 2 / 3)
        self.assertEqual(task["vector_denominator"], 1)
        self.assertEqual(task["uniform_reference_same_rows"]["denominator"], 1)
        self.assertEqual(task["omitted_vectors_attempted"], 1)
        self.assertEqual(task["unattempted"], 1)
        self.assertNotIn("mean_log_loss", result)
        self.assertNotIn("action_accuracy_all_scheduled", result)

    def test_task_summaries_not_pooled_and_unattempted_empty_bins(self):
        cases = [case_for(task="adjacency"), case_for(replace(REQUEST, request_id="second"),
                                                    task="cycle_detection")]
        _, result = audit.replay(cases, [])
        self.assertEqual(set(result["by_task"]), {"adjacency", "cycle_detection"})
        for task in result["by_task"].values():
            self.assertEqual(task["action_accuracy_all_scheduled"], 0)
            self.assertIsNone(task["mean_log_loss"])
            self.assertFalse(task["mean_log_loss_is_infinite"])
            self.assertEqual([b["count"] for b in task["reliability_bins"]], [0] * 5)

    def test_raw_replay_rejects_bad_vector_request_or_order(self):
        case = case_for()
        good = row_for(case, {"yes": .6, "no": .4}, "yes")
        for key, value in (("request_sha256", "bad"), ("instance_id", "other"),
                           ("probabilities", {"yes": .6, "no": .39}),
                           ("selected_option_id", "other"), ("probability_sum", .99)):
            row = {**good, key: value}
            with self.assertRaises(audit.suite.Refusal):
                audit.replay([case], [row])
        with self.assertRaises(audit.suite.Refusal):
            audit.replay([case], [good, good])

    def test_private_reference_never_enters_canonical_request(self):
        instance = {"id": "degree", "task": "degree_exact", "kind": "exact", "query_budget": 1,
                    "state": {"nodes": [0, 1, 2], "edges": [[0, 1]], "vertex": 0, "directed": False},
                    "private": {"answer": 1, "oracle_notes": PRIVATE}}
        with patch.object(audit, "COUNTS", {"degree_exact": 1}), patch.object(audit, "CALLS", 1):
            case = audit.cases_for([instance])[0]
        text = json.dumps(case["request"])
        self.assertNotIn(PRIVATE, text)
        self.assertNotIn("private", text)
        self.assertNotIn("oracle", text)
        self.assertEqual(case["label"], "1")
        public = {k: instance[k] for k in audit.release.PUBLIC_KEYS}
        self.assertEqual(case["request"], audit.suite.json_value(asdict(audit.evaluation.public_request(public, []))))

    def test_missing_confirmation_cases_refused(self):
        with self.assertRaises(audit.suite.Refusal):
            audit.cases_for([])


class ResponseTests(unittest.IsolatedAsyncioTestCase):
    async def native(self, payload, response_transform=None):
        class Client:
            async def predict(self, request):
                response = TypeSafeClient._parse_response(request, payload)
                return response_transform(response) if response_transform else response
        before = deepcopy(payload)
        row = await audit.observe(Client(), REQUEST, case_for(), CONFIG, "jev_action")
        self.assertEqual(before, payload)
        self.assertNotIn(PRIVATE, json.dumps(row, allow_nan=False))
        self.assertNotIn("raw_output", row)
        return row

    async def test_valid_strict_response_uses_decisionresponse_vector(self):
        row = await self.native(body(), lambda r: replace(r, probabilities={"yes": .8, "no": .2}))
        self.assertEqual(row["probabilities"], {"yes": .8, "no": .2})
        self.assertEqual(row["confidence"], .7)
        self.assertEqual(row["strict"]["usage"]["input_tokens"], 12)

    async def test_sum_099_preserves_action_but_omits_vector(self):
        payload = body()
        payload["answers"]["decision"]["probabilities"]["no"] = .39
        row = await self.native(payload)
        self.assertEqual(row["strict"]["diagnostic_code"], "invalid_probability_sum")
        self.assertTrue(row["action_eligible"])
        self.assertEqual(row["selected_option_id"], "yes")
        self.assertEqual(row["confidence"], .7)
        self.assertIsNone(row["probabilities"])
        self.assertAlmostEqual(row["probability_sum"], .99)
        self.assertEqual(row["vector_omission_reasons"], ["invalid_probability_sum"])

    async def test_argmax_mismatch_vector_retained_original_choice(self):
        payload = body()
        payload["answers"]["decision"]["probabilities"] = {"yes": .2, "no": .8}
        row = await self.native(payload)
        self.assertEqual(row["strict"]["diagnostic_code"], "choice_probability_mismatch")
        self.assertEqual(row["selected_option_id"], "yes")
        self.assertEqual(row["probabilities"], {"yes": .2, "no": .8})
        self.assertTrue(audit.score_row(case_for(), row)["choice_argmax_disagreement"])

    async def test_hidden_confidence_or_usage_error_never_rescued(self):
        for field in ("confidence", "usage"):
            payload = body()
            payload["answers"]["decision"]["probabilities"]["no"] = .39
            if field == "confidence":
                payload["answers"]["decision"]["confidence"] = True
            else:
                payload["usage"] = {"input_tokens": -1}
            row = await self.native(payload)
            self.assertFalse(row["action_eligible"])
            self.assertIsNone(row["selected_option_id"])
            self.assertIsNone(row["probabilities"])
            self.assertIn("invalid_" + field, row["action_ineligibility_reasons"])

    async def test_invalid_model_choice_and_answer_shape_ineligible(self):
        for field, value in (("model", PRIVATE), ("choice", 1), ("choice", PRIVATE),
                             ("type", "text"), ("answers", {})):
            payload = body()
            if field in ("model", "answers"):
                payload[field] = value
            else:
                payload["answers"]["decision"][field] = value
            row = await self.native(payload)
            self.assertFalse(row["action_eligible"])
            self.assertIsNone(row["probabilities"])
            self.assertIsNone(row["selected_option_id"])

    async def test_invalid_missing_or_malformed_native_body(self):
        for payload in (None, PRIVATE, [], {}):
            row = await self.native(payload)
            self.assertFalse(row["action_eligible"])
            self.assertIsNone(row["probabilities"])

    async def token_row(self, probabilities):
        config = audit.evaluation.configuration("qwen4_token_scores")[0]
        metadata = {k: None for k in (
            "shared_prefix_tokens", "prefill_tokens", "requested_max_tokens", "scoring_rule",
            "prompt_sha256", "candidate_logprobs", "option_ids")}
        metadata.update(selected_option_id="yes", candidate_count=2, candidate_prompts=1,
                        generated_tokens=1, generated_tokens_used_for_decision=False,
                        scoring_mode="requested", probabilities=probabilities)
        response = DecisionResponse(
            request_id=REQUEST.request_id, selected_option_id="yes", resolved_model=config.model,
            probabilities=probabilities, probability_kind="option_logits", usage=TokenUsage(12, 1),
            raw_output={"bridge": metadata, "provider": {"reasoning": PRIVATE}})
        return await audit.observe(AsyncMock(predict=AsyncMock(return_value=response)),
                                   REQUEST, case_for(), config, "qwen4_token_scores")

    async def test_token_scores_keep_vector_and_do_not_invent_confidence(self):
        row = await self.token_row({"yes": .7, "no": .3})
        self.assertTrue(row["action_eligible"])
        self.assertEqual(row["probabilities"], {"yes": .7, "no": .3})
        self.assertIsNone(row["confidence"])
        self.assertIsNone(row["native_diagnostics"])
        self.assertNotIn(PRIVATE, json.dumps(row))

    async def test_invalid_token_vector_not_rescued(self):
        row = await self.token_row({"yes": .7})
        self.assertFalse(row["action_eligible"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["selected_option_id"])
        self.assertIn("invalid_score_vector", row["action_ineligibility_reasons"])

    async def test_http_timeout_transport_and_unexpected_errors_allowlisted(self):
        for error, code, fatal in (
            (ProviderHTTPError(429, PRIVATE), "http_error", True),
            (ProviderHTTPError(500), "http_error", False),
            (ClientTimeoutError(PRIVATE), "timeout", False),
            (ClientTransportError(PRIVATE), "client_error", True),
            (RuntimeError(PRIVATE), "internal_error", True),
            (InvalidResponseError(PRIVATE, raw_output=PRIVATE), "invalid_response", False),
        ):
            client = AsyncMock(predict=AsyncMock(side_effect=error))
            row = await audit.observe(client, REQUEST, case_for(), CONFIG, "jev_action")
            client.predict.assert_awaited_once()
            self.assertEqual(row["strict"]["diagnostic_code"], code)
            self.assertEqual(row["strict"]["fatal"], fatal)
            self.assertFalse(row["action_eligible"])
            self.assertNotIn(PRIVATE, json.dumps(row))

    async def test_fatal_stops_collection_nonfatal_advances_without_retry(self):
        cases = [case_for(replace(REQUEST, request_id=str(i))) for i in range(3)]
        for error, count in ((ProviderHTTPError(429), 1), (ClientTimeoutError(PRIVATE), 3)):
            client = AsyncMock(predict=AsyncMock(side_effect=error))
            with patch.object(audit.suite, "append_row") as append:
                rows = await audit.collect(client, cases, CONFIG, "jev_action", StringIO())
            self.assertEqual(client.predict.await_count, count)
            self.assertEqual(append.call_count, count)
            _, summary = audit.replay(cases, rows)
            self.assertEqual(summary["unattempted"], 3 - count)
            self.assertEqual(summary["by_task"]["adjacency"]["scheduled"], 3)

    async def test_collection_cap_no_extra_call(self):
        client = AsyncMock(predict=AsyncMock(side_effect=ClientTimeoutError("timeout")))
        cases = [case_for(replace(REQUEST, request_id=str(i))) for i in range(2)]
        with patch.object(audit, "CALLS", 1), patch.object(audit.suite, "append_row"):
            with self.assertRaises(audit.suite.Refusal):
                await audit.collect(client, cases, CONFIG, "jev_action", StringIO())
        client.predict.assert_awaited_once()

    async def test_existing_profile_refused_no_resume(self):
        with patch.object(audit, "load_plan", return_value=({}, [])), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(audit, "make_client") as factory:
            with self.assertRaises(audit.suite.Refusal):
                await audit.run(audit.REPO, "jev_action")
            factory.assert_not_called()


class LockAndFactoryTests(unittest.TestCase):
    def test_native_uses_standard_lock_qwen_has_no_duplicate_acquisition(self):
        with patch.object(audit.suite, "lane_locks", return_value=nullcontext()) as lock:
            with audit.outer_lock("qwen4_token_scores"):
                pass
            lock.assert_not_called()
            with audit.outer_lock("jev_action"):
                pass
            lock.assert_called_once_with(("jev",))
        with self.assertRaises(audit.suite.Refusal):
            audit.outer_lock("qwen9_grammar")

    def test_native_real_registry_and_existing_auth(self):
        with patch.object(audit.suite.shared, "credential_preflight") as auth, \
                patch.object(audit, "create_client") as factory:
            audit.make_client("jev_action", CONFIG)
            auth.assert_called_once_with(("jev",))
            factory.assert_called_once_with(CONFIG.provider, **CONFIG.client_kwargs())

    def test_token_factory_requested_mode_uses_bridge_owned_lock(self):
        with patch.object(audit.bridge, "upstream_functions", return_value=("tokenizer", "encoder")), \
                patch.object(audit.bridge, "SemIfVLLMBridge") as factory, \
                patch.object(audit.suite, "lane_locks") as lock:
            audit.make_client("qwen4_token_scores", audit.evaluation.configuration("qwen4_token_scores")[0])
            factory.assert_called_once_with("tokenizer", "encoder", max_prefix_tokens=4096,
                                            scoring_mode="requested", timeout=120.0)
            lock.assert_not_called()


class ArtifactReplayTests(unittest.TestCase):
    def test_seal_replays_raw_observations_and_writes_exclusively(self):
        output = audit.REPO / "audit-test"
        case = case_for()
        row = row_for(case, {"yes": .6, "no": .4}, "yes")
        writes = {}
        with patch.object(audit.suite, "read_rows", return_value=[row]), \
                patch.object(audit.suite, "environment_versions", return_value={"python": "offline-test"}), \
                patch.object(audit.suite, "write_json", side_effect=lambda p, v: writes.update({p.name: v})), \
                patch.object(audit.suite, "digest", return_value="hash"), \
                patch.object(audit.suite, "append_row") as append, \
                patch.object(Path, "open", return_value=StringIO()) as opened, \
                patch.object(Path, "iterdir", return_value=[]), patch.object(Path, "chmod"):
            result = audit.seal_lane(output, "jev_action", [case], "completed", None)
        opened.assert_called_once_with("x", encoding="utf-8")
        self.assertEqual(append.call_args.args[1], audit.score_row(case, row))
        self.assertEqual(writes["summary.json"], audit.replay([case], [row])[1])
        self.assertEqual(result["attempted_calls"], 1)
        self.assertEqual(set(result["artifact_sha256"]), {"attempts.jsonl", "scores.jsonl", "summary.json"})

    def replay_report(self, *, tamper_summary=False, tamper_seal=False):
        output = audit.REPO / "audit-test"
        case = case_for()
        attempts = [row_for(case, {"yes": .6, "no": .4}, "yes")]
        scores, summary = audit.replay([case], attempts)
        status = {"profile": "jev_action", "status": "completed", "manifest_sha256": "hash",
                  "planned_calls": 1, "attempted_calls": 1,
                  "artifact_sha256": {n: "hash" for n in ("attempts.jsonl", "scores.jsonl", "summary.json")}}
        seal = {"manifest_sha256": "hash",
                "profile_status_sha256": {p: "hash" for p in audit.PROFILES}}
        if tamper_seal:
            seal["manifest_sha256"] = "other"
        files = {"status.json": status, "summary.json": {} if tamper_summary else summary,
                 "seal.json": seal}
        with ExitStack() as stack:
            for target, name, kwargs in (
                (audit, "load_plan", {"return_value": ({}, [case])}),
                (audit.suite, "read_rows", {"side_effect": lambda p: attempts if p.name == "attempts.jsonl" else scores}),
                (audit.suite, "read_json", {"side_effect": lambda p: files[p.name]}),
                (audit.suite, "digest", {"return_value": "hash"}),
                (Path, "is_symlink", {"return_value": False}),
                (Path, "iterdir", {"return_value": []}),
                (Path, "stat", {"return_value": SimpleNamespace(st_mode=0o444)}),
                (Path, "exists", {"return_value": True}),
            ):
                stack.enter_context(patch.object(target, name, **kwargs))
            return audit.report(output, "jev_action")

    def test_report_recomputes_summary_and_verifies_root_seal(self):
        self.assertEqual(self.replay_report()["summary"]["attempted"], 1)
        for kwargs in ({"tamper_summary": True}, {"tamper_seal": True}):
            with self.assertRaises(audit.suite.Refusal):
                self.replay_report(**kwargs)


if __name__ == "__main__":
    unittest.main()
