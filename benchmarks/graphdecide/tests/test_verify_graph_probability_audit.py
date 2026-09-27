"""Offline recovery tests: immutable input, roundoff bookkeeping, and unchanged gates."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import verify_graph_probability_audit as verifier
from src.clients.base import DecisionOption, DecisionRequest, InvalidResponseError, TokenUsage
from src.clients.typesafe import TypeSafeClient

old = verifier.original
CONFIGS = {p: asdict(old.evaluation.configuration(p)[0]) for p in old.PROFILES}


def fixture(vector=None, selected="2", label="2", ident="example", profile="jev_action"):
    vector = {"2": .7, "1": .2, "0": .1} if vector is None else vector
    request = DecisionRequest(ident, {"nodes": [0, 1], "edges": []}, "Choose?",
                              tuple(DecisionOption(key) for key in vector))
    public = old.suite.json_value(asdict(request))
    case = {"instance_id": ident, "request": public, "request_sha256": old.value_hash(public),
            "label": label, "task": "degree_exact", "node_count": 8}
    body = {"model": CONFIGS["jev_action"]["model"], "usage": {"input_tokens": 12, "output_tokens": 3},
            "answers": {"decision": {"type": "choice", "choice": selected,
                                     "probabilities": vector, "confidence": .8}}}
    config = old.ModelConfig(**CONFIGS["jev_action"])
    try:
        response = TypeSafeClient._parse_response(request, body)
        strict = old.suite.response_record(response, request, config)
    except InvalidResponseError as exc:
        strict = {"status": "failure", "selected_option_id": None, "resolved_model": None,
                  "usage": old.suite.safe_usage(TokenUsage(12, 3)), **old.suite.failure(exc)}
    observation = old.native.action_observation(request, body, config.model, label, strict)
    row = {
        "instance_id": ident, "request_sha256": case["request_sha256"],
        "strict": strict, "native_diagnostics": observation["native_diagnostics"],
        "action_eligible": observation["eligible"],
        "action_ineligibility_reasons": observation["ineligibility_reasons"],
        "selected_option_id": observation["selected_option_id"], "confidence": .8,
        **old.vector_record(request, vector, observation["eligible"]), "latency_seconds": .1,
    }
    if profile == "qwen4_token_scores":
        row["native_diagnostics"] = None
        row["confidence"] = None
        row["strict"].update(status="success", diagnostic_code=None, fatal=False,
                             resolved_model=CONFIGS[profile]["model"], selected_option_id=selected)
    return case, row


class ArithmeticRecoveryTests(unittest.TestCase):
    def test_json_reordering_reproduces_old_failure_and_corrected_replay_preserves_input(self):
        case, row = fixture()
        # Reproduce Python 3.11's original left-to-right summation even under Python 3.12.
        stored = 0.0
        for value in row["probabilities"].values():
            stored += value
        row["probability_sum"] = stored
        row["native_diagnostics"]["probability_sum"] = stored
        canonical = json.loads(json.dumps(row, sort_keys=True))
        self.assertNotEqual(sum(canonical["probabilities"].values()), canonical["probability_sum"])
        before = deepcopy(canonical)
        with self.assertRaises(old.suite.Refusal):
            old.score_row(case, canonical)
        score, validation = verifier.score_row(case, canonical, "jev_action", CONFIGS["jev_action"])
        self.assertAlmostEqual(score["brier"], .14)
        self.assertTrue(score["valid_vector"])
        self.assertGreater(validation["bookkeeping_abs_difference"], 0)
        self.assertLess(validation["bookkeeping_abs_difference"], 1e-12)
        self.assertEqual(canonical, before)

    def test_bookkeeping_tamper_above_tolerance_refused(self):
        case, row = fixture()
        row["probability_sum"] += 2e-12
        with self.assertRaises(verifier.VerificationError) as error:
            verifier.score_row(case, row, "jev_action", CONFIGS["jev_action"])
        self.assertEqual(error.exception.code, "derived_sum_bookkeeping_mismatch")

    def test_bookkeeping_has_no_relative_tolerance(self):
        with self.assertRaises(verifier.VerificationError):
            verifier.check_sum_bookkeeping(1000.0, 1000.0 + 1e-9)
        self.assertEqual(verifier.check_sum_bookkeeping(1, 1), 0)

    def test_within_original_gate_not_renormalized(self):
        case, row = fixture({"yes": .6000004, "no": .4}, selected="yes", label="yes")
        before = deepcopy(row)
        score, _ = verifier.score_row(case, row, "jev_action", CONFIGS["jev_action"])
        self.assertTrue(score["valid_vector"])
        self.assertNotEqual(math.fsum(row["probabilities"].values()), 1)
        self.assertEqual(row, before)

    def test_1e6_gate_not_loosened_by_bookkeeping_fix(self):
        case, row = fixture({"yes": .6, "no": .4}, selected="yes", label="yes")
        row["probabilities"]["yes"] = .600002
        row["probability_sum"] = math.fsum(row["probabilities"].values())
        with self.assertRaises(verifier.VerificationError) as error:
            verifier.score_row(case, row, "jev_action", CONFIGS["jev_action"])
        self.assertEqual(error.exception.code, "vector_gate")
        self.assertEqual(verifier.VECTOR_ATOL, 1e-6)

    def test_invalid_keys_nan_range_and_boolean_refused(self):
        case, row = fixture()
        for vector in ({"2": .7}, {"2": float("nan"), "1": .2, "0": .1},
                       {"2": True, "1": 0, "0": 0}, {"2": 1.1, "1": -.1, "0": 0}):
            changed = {**row, "probabilities": vector}
            with self.assertRaises(verifier.VerificationError) as error:
                verifier.score_row(case, changed, "jev_action", CONFIGS["jev_action"])
            self.assertEqual(error.exception.code, "vector_schema")

    def test_original_099_omission_and_action_are_preserved(self):
        case, row = fixture({"yes": .6, "no": .39}, selected="yes", label="yes")
        before = deepcopy(row)
        score, _ = verifier.score_row(case, row, "jev_action", CONFIGS["jev_action"])
        self.assertTrue(score["action_correct"])
        self.assertFalse(score["valid_vector"])
        self.assertIsNone(score["brier"])
        self.assertEqual(row, before)
        row["probability_sum"] = 1
        with self.assertRaises(verifier.VerificationError):
            verifier.score_row(case, row, "jev_action", CONFIGS["jev_action"])

    def test_zero_true_probability_explicit_positive_infinity_not_finite_only_mean(self):
        case, row = fixture({"yes": 0, "no": 1}, selected="no", label="yes")
        result = verifier.replay([case], [row], "jev_action", CONFIGS["jev_action"])
        score = result["scores"][0]
        summary = result["summary"]["by_task"]["degree_exact"]
        self.assertEqual(score["brier"], 2)
        self.assertIsNone(score["log_loss"])
        self.assertTrue(score["log_loss_infinite"])
        self.assertTrue(summary["mean_log_loss_is_infinite"])
        self.assertIsNone(summary["mean_log_loss"])
        self.assertEqual(summary["log_loss_infinite_count"], 1)
        self.assertEqual(summary["finite_log_loss_count"], 0)
        json.dumps(result, allow_nan=False)

    def test_actual_choice_and_summed_brier_are_not_argmax_replacements(self):
        case, row = fixture({"yes": .2, "no": .8}, selected="yes", label="yes")
        score, _ = verifier.score_row(case, row, "jev_action", CONFIGS["jev_action"])
        self.assertEqual(score["selected_probability"], .2)
        self.assertEqual(score["reliability_bin"], 1)
        self.assertTrue(score["choice_argmax_disagreement"])
        self.assertTrue(score["action_correct"])
        self.assertAlmostEqual(score["brier"], 1.28)
        self.assertEqual(row["strict"]["diagnostic_code"], "choice_probability_mismatch")

    def test_scores_and_summaries_stable_under_dict_reordering(self):
        case, row = fixture()
        left = verifier.replay([case], [row], "jev_action", CONFIGS["jev_action"])
        right = verifier.replay([case], [json.loads(json.dumps(row, sort_keys=True))],
                                "jev_action", CONFIGS["jev_action"])
        self.assertEqual(left, right)

    def test_supplied_choice_tamper_and_hidden_nonprobability_error_rejected(self):
        case, row = fixture()
        changed = deepcopy(row)
        changed["selected_option_id"] = "1"
        with self.assertRaises(verifier.VerificationError) as error:
            verifier.score_row(case, changed, "jev_action", CONFIGS["jev_action"])
        self.assertEqual(error.exception.code, "choice_changed")
        changed = deepcopy(row)
        changed["native_diagnostics"]["contract_issues"].append("invalid_usage")
        with self.assertRaises(verifier.VerificationError) as error:
            verifier.score_row(case, changed, "jev_action", CONFIGS["jev_action"])
        self.assertEqual(error.exception.code, "native_eligibility")


class MatchedAnalysisTests(unittest.TestCase):
    def test_empty_common_rows_explicitly_undefined(self):
        case, _ = fixture()
        result = verifier.common_valid([case], {p: [] for p in old.PROFILES}, CONFIGS)
        self.assertEqual(result, {"denominator": 0, "status": "undefined_no_common_valid_rows",
                                  "profiles": None})

    def test_matching_is_by_id_and_validity_with_same_row_uniform_reference(self):
        first, native = fixture(ident="shared")
        _, token = fixture(ident="shared", profile="qwen4_token_scores")
        second, omitted = fixture({"yes": .6, "no": .39}, selected="yes", label="yes", ident="omitted")
        _, second_token = fixture({"yes": .6, "no": .4}, selected="yes", label="yes",
                                  ident="omitted", profile="qwen4_token_scores")
        attempts = {"jev_action": [native, omitted], "qwen4_token_scores": [second_token, token]}
        common = verifier.common_valid([first, second], attempts, CONFIGS)
        self.assertEqual(common["denominator"], 1)
        for metrics in common["profiles"].values():
            self.assertEqual(metrics["uniform_reference_same_rows"]["denominator"], 1)
            self.assertEqual(metrics["vector_denominator"], 1)
        native_result = verifier.replay([first, second], [native, omitted],
                                       "jev_action", CONFIGS["jev_action"])
        metrics = native_result["summary"]["by_task"]["degree_exact"]
        self.assertEqual(metrics["scheduled"], 2)
        self.assertEqual(metrics["action_accuracy_all_scheduled"], 1)
        self.assertEqual(metrics["vector_denominator"], 1)

    def test_common_infinity_not_changed_to_zero_or_finite_only(self):
        case, native = fixture({"yes": .6, "no": .4}, selected="yes", label="yes")
        _, token = fixture({"yes": 0, "no": 1}, selected="no", label="yes", profile="qwen4_token_scores")
        result = verifier.common_valid([case], {"jev_action": [native], "qwen4_token_scores": [token]}, CONFIGS)
        metric = result["profiles"]["qwen4_token_scores"]
        self.assertTrue(metric["mean_log_loss_is_infinite"])
        self.assertIsNone(metric["mean_log_loss"])
        self.assertEqual(metric["log_loss_infinite_count"], 1)

    def test_pooled_tasks_and_duplicate_rows_refused(self):
        case, row = fixture()
        other = {**case, "task": "adjacency"}
        with self.assertRaises(verifier.VerificationError):
            verifier.common_valid([case, other], {p: [row] for p in old.PROFILES}, CONFIGS)
        with self.assertRaises(verifier.VerificationError):
            verifier.common_valid([case], {p: [row, row] for p in old.PROFILES}, CONFIGS)


class PreservationAndDiagnosticTests(unittest.TestCase):
    def test_changed_raw_digest_refused_without_any_write(self):
        source = old.REPO / "offline-source"
        path = source / "jev_action/attempts.jsonl"
        with patch.object(verifier, "original_paths", return_value=[path]), \
                patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=0o444)), \
                patch.object(verifier.suite, "digest", return_value="changed"), \
                patch.object(Path, "open", side_effect=AssertionError("No writes")), \
                patch.object(Path, "chmod", side_effect=AssertionError("No permission changes")):
            with self.assertRaises(verifier.VerificationError) as error:
                verifier.check_source_seal(source, {"artifact_sha256": {"jev_action/attempts.jsonl": "original"}})
        self.assertEqual(error.exception.code, "original_digest_changed")

    def test_existing_sidecar_refused_before_source_access(self):
        with patch.object(Path, "exists", return_value=True), \
                patch.object(verifier, "seal_original", side_effect=AssertionError("No source changes")):
            with self.assertRaises(verifier.VerificationError) as error:
                verifier.recover(old.REPO, old.REPO / "already-exists")
        self.assertEqual(error.exception.code, "output_exists")

    def test_failed_producer_subprocess_has_explicit_error(self):
        with patch.object(verifier.subprocess, "run",
                          return_value=SimpleNamespace(returncode=2, stdout="", stderr="failure")):
            with self.assertRaises(verifier.VerificationError) as error:
                verifier.run_producer_check(old.REPO, "jev_action")
        self.assertEqual(error.exception.code, "producer_runtime_check_failed")

    def test_discrete_legacy_changes_not_dismissed_as_float_roundoff(self):
        with self.assertRaises(verifier.VerificationError):
            verifier.numeric_differences({"correct": True}, {"correct": False})
        self.assertEqual(verifier.numeric_differences({"loss": .1}, {"loss": .1 + 1e-16}),
                         [abs(.1 - (.1 + 1e-16))])

    def test_scoring_never_calls_clients_or_modifies_raw_input(self):
        case, row = fixture()
        before = json.dumps(row, sort_keys=True)
        with patch.object(old, "make_client", side_effect=AssertionError("No inference")), \
                patch.object(old, "create_client", side_effect=AssertionError("No inference")):
            verifier.replay([case], [row], "jev_action", CONFIGS["jev_action"])
        self.assertEqual(json.dumps(row, sort_keys=True), before)

    def test_exporter_api_checks_sidecar_and_returns_requested_shape(self):
        output = old.REPO / "analysis-fixture"
        with patch.object(verifier, "check", return_value={"status": "verified"}) as check, \
                patch.object(verifier.suite, "read_json", side_effect=lambda p: {"file": p.name}), \
                patch.object(verifier.suite, "read_rows", return_value=[{"score": 1}]):
            result = verifier.report(output, "jev_action")
        check.assert_called_once_with(output)
        self.assertEqual(set(result), {"status", "summary", "scores"})
        self.assertEqual(result["scores"], [{"score": 1}])


if __name__ == "__main__":
    unittest.main()
