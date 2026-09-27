import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from scripts import export_expanded_model_pilots as export


class ExpandedExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence = export.prior.Evidence()
        cls.registration, cls.instances, _, _ = export.prior.load_panel(
            export.prior.panel.DEFAULT_ROOT, evidence)
        cls.manifest, cls.rows, cls.attempts, _ = export.prior.load_candidate(
            export.prior.DEFAULT_EXECUTION / "qwen27_grammar", cls.registration,
            export.prior.panel.DEFAULT_ROOT, cls.instances, evidence)
        cls.model = cls.manifest["model"]["model"]

    def setUp(self):
        self.root = export.REPO / "output" / ("expanded-export-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def sealed_fixture(self):
        run = self.root / "qwen9_thinking"
        for path in export.required_inputs("qwen9_thinking", run).values():
            self.write(path, {})
        status = {"status": "completed_with_failures", "calls": 64, "profile": "qwen9_thinking"}
        self.write(run / "status.json", status)
        proof = {
            "schema_version": 1, "status": "verified_terminal_expanded_pilot",
            "profile": "qwen9_thinking", "run": str(run), "source_status": status["status"],
            "calls": 64, "episodes": 64, "independent_replay_verified": True,
            "new_model_calls": 0, "panel_registration_sha256": "panel",
            "input_sha256": {name: export.suite.digest(path)
                             for name, path in export.required_inputs("qwen9_thinking", run).items()},
        }
        attestation = self.root / "parent-attestation.json"
        self.write(attestation, proof)
        return run, attestation, proof

    def test_pending_rejected_before_attestation_or_correctness_read(self):
        for state in ("running", "loading", "unattempted", "ready"):
            with self.subTest(state=state), patch.object(
                    export.suite, "read_json", return_value={"status": state}) as read, \
                    patch.object(export.suite, "read_rows") as rows:
                with self.assertRaises(export.suite.shared.Refusal):
                    export.admit("kev", self.root / "run-kev", self.root / "absent.json",
                                 export.prior.Evidence())
                self.assertEqual(read.call_count, 1)
                rows.assert_not_called()

    def test_pending_export_does_not_load_panel_or_create_output(self):
        run = self.root / "qwen9_thinking"
        self.write(run / "status.json", {"status": "running"})
        target = self.root / "export"
        with patch.object(export.prior, "load_panel") as panel:
            with self.assertRaises(export.suite.shared.Refusal):
                export.export("qwen9_thinking", run, self.root / "absent.json", target)
            panel.assert_not_called()
        self.assertFalse(target.exists())

    def test_explicit_attestation_required_for_terminal_runtime(self):
        run = self.root / "run-kev"
        self.write(run / "status.json", {"status": "completed"})
        with self.assertRaises(export.suite.shared.Refusal):
            export.admit("kev", run, self.root / "missing.json", export.prior.Evidence())

    def test_seal_checks_every_byte_before_replay(self):
        run, attestation, _ = self.sealed_fixture()
        status, _, legacy = export.admit("qwen9_thinking", run, attestation, export.prior.Evidence())
        self.assertEqual(status["calls"], 64)
        self.assertFalse(legacy)
        for filename in ("scores.jsonl", "attempts.jsonl", "manifest.json"):
            path = run / filename
            original = path.read_text()
            path.write_text(original + "\n")
            with self.subTest(file=filename), patch.object(export.suite, "read_rows") as reader:
                with self.assertRaises(export.suite.shared.Refusal):
                    export.admit("qwen9_thinking", run, attestation, export.prior.Evidence())
                reader.assert_not_called()
            path.write_text(original)

    def test_attestation_wrong_profile_missing_input_and_extra_input_refused(self):
        run, attestation, original = self.sealed_fixture()
        for case in ("profile", "missing", "extra", "not_independent"):
            proof = copy.deepcopy(original)
            if case == "profile":
                proof["profile"] = "qwen27_thinking"
            elif case == "missing":
                proof["input_sha256"].pop(next(iter(proof["input_sha256"])))
            elif case == "extra":
                proof["input_sha256"]["output/other-pending/scores.jsonl"] = "not-read"
            else:
                proof["independent_replay_verified"] = False
            self.write(attestation, proof)
            with self.subTest(case=case), self.assertRaises(export.suite.shared.Refusal):
                export.admit("qwen9_thinking", run, attestation, export.prior.Evidence())

    def test_zero_attempt_aborted_retains_all_denominators_and_null_gaps(self):
        rows = export.canonical.replay(
            self.instances, [], SimpleNamespace(model=self.model), native_action=False)
        replayed = export.replay_terminal(self.instances, [], rows, {
            "status": "aborted", "calls": 0}, self.model)
        metrics = export.metrics({"candidate": replayed})
        self.assertEqual(len(replayed), 64)
        for metric in metrics:
            self.assertEqual(metric["unattempted_episodes"], metric["episodes"])
            self.assertEqual(metric["feasible"], 0)
            self.assertEqual(metric["hits"], 0)
            self.assertEqual(metric["objective_gap_denominator"], 0)
            self.assertIsNone(metric["mean_absolute_gap_feasible"])
            self.assertIsNone(metric["mean_objective_feasible"])

    def failure_attempts(self):
        return [{"instance_id": i["id"], "step": 0,
                 "request": export.suite.json_value(asdict(export.canonical.public_request(i, []))),
                 "status": "failure", "selected_option_id": None, "fatal": False,
                 "resolved_model": None, "diagnostic_code": "timeout"}
                for i in self.instances]

    def test_all_failed_completed_retains_64_and_64_actual_calls(self):
        attempts = self.failure_attempts()
        rows = export.canonical.replay(
            self.instances, attempts, SimpleNamespace(model=self.model), native_action=False)
        result = export.replay_terminal(self.instances, attempts, rows, {
            "status": "completed_with_failures", "calls": 64}, self.model)
        self.assertEqual(len(result), 64)
        for metric in export.metrics({"candidate": result}):
            self.assertEqual(metric["failed_episodes"], metric["episodes"])
            self.assertEqual(metric["attempted_episodes"], metric["episodes"])
            self.assertEqual(metric["unattempted_episodes"], 0)
            self.assertIsNone(metric["mean_absolute_gap_feasible"])

    def test_zero_attempt_cannot_be_promoted_to_completed(self):
        rows = export.canonical.replay(
            self.instances, [], SimpleNamespace(model=self.model), native_action=False)
        with self.assertRaises(export.suite.shared.Refusal):
            export.replay_terminal(self.instances, [], rows, {
                "status": "completed", "calls": 0}, self.model)

    def test_illegal_action_and_malformed_history_refused(self):
        for change in ("illegal", "history", "request"):
            attempts = copy.deepcopy(self.attempts)
            if change == "illegal":
                attempts[0]["selected_option_id"] = "not-an-option"
            elif change == "history":
                next(a for a in attempts if a["step"] == 1)["step"] = 8
            else:
                attempts[0]["request"]["question"] = "Different public information"
            with self.subTest(change=change), self.assertRaises(export.suite.shared.Refusal):
                export.replay_terminal(self.instances, attempts, self.rows, {
                    "status": "completed", "calls": 160}, self.model)

    def test_recomputed_score_mismatch_refused(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["correct"] = not rows[0]["correct"]
        with self.assertRaises(export.suite.shared.Refusal):
            export.replay_terminal(self.instances, self.attempts, rows, {
                "status": "completed", "calls": 160}, self.model)

    def native_attempt(self):
        attempt = copy.deepcopy(self.attempts[0])
        ids = [o["id"] for o in attempt["request"]["options"]]
        selected = attempt["selected_option_id"]
        probabilities = {i: float(i == selected) for i in ids}
        attempt.update(probabilities=probabilities, probability_kind="native",
                       strict_status="success",
                       action_observation={"eligible": True, "selected_option_id": selected},
                       usage={"input_tokens": 12, "output_tokens": 0, "reasoning_tokens": None},
                       backend={"status": "success", "result": {
                           "selected_option_id": selected, "probabilities": probabilities,
                           "input_tokens": 12, "logits": None}})
        return attempt

    def test_probability_shape_and_choice_eligibility_are_separate(self):
        success = self.native_attempt()
        result = export.probability_audit([success], self.model)[0]
        self.assertTrue(result["choice_eligible"])
        self.assertTrue(result["probability_shape_valid"])
        failure = copy.deepcopy(success)
        failure.update(status="failure", selected_option_id=None, fatal=True,
                       strict_status="failure", diagnostic_code="invalid_probability_sum")
        failure.pop("backend")
        failure.pop("probabilities")
        result = export.probability_audit([failure], self.model)[0]
        self.assertTrue(result["choice_eligible"])
        self.assertFalse(result["probability_shape_valid"])
        rows = export.canonical.replay(
            self.instances, [failure], SimpleNamespace(model=self.model), native_action=False)
        self.assertFalse(rows[0]["feasible"])
        self.assertEqual(rows[0]["decisions"], [])
        self.assertEqual(export.prior.hit(rows[0]), 0)

    def test_native_malformed_probability_or_eligibility_refused(self):
        for field in ("probabilities", "eligibility", "choice"):
            attempt = self.native_attempt()
            if field == "probabilities":
                attempt["backend"]["result"]["probabilities"] = {"wrong": 1}
            elif field == "eligibility":
                attempt["action_observation"]["selected_option_id"] = "wrong"
            else:
                attempt["selected_option_id"] = "wrong"
            with self.subTest(field=field), self.assertRaises(Exception):
                export.probability_audit([attempt], self.model)

    def test_failure_taxonomy_not_incorrectness(self):
        for diagnostic, category in (
                ("UnsupportedRequestError", "unsupported"), ("ContextOverflow", "unsupported"),
                ("ClientTimeoutError", "timeout"), ("TimeoutError", "timeout"),
                ("missing_answer", "parse_or_contract"), ("incomplete_reasoning", "parse_or_contract"),
                ("invalid_choice", "parse_or_contract"), ("RuntimeError", "backend_or_other")):
            self.assertEqual(export.failure_category({
                "status": "failure", "diagnostic_code": diagnostic}), category)
        self.assertIsNone(export.failure_category({"status": "success"}))

    def test_real_grammar_objectives_independently_verified(self):
        audits = export.independent_objectives(self.instances, self.rows)
        self.assertEqual(len(audits), 16)
        self.assertEqual(sum(a["optimal"] for a in audits if a["task"] == "lt_influence_construct"), 2)
        bad = copy.deepcopy(self.rows)
        next(r for r in bad if r["kind"] == "optimization")["objective"] += 1
        with self.assertRaises(export.suite.shared.Refusal):
            export.independent_objectives(self.instances, bad)

    def test_legacy_grammar_only_with_its_explicit_proof(self):
        run = export.prior.DEFAULT_EXECUTION / "qwen27_grammar"
        proof = run.parent / "qwen27-grammar-parent-verification.json"
        _, _, legacy = export.admit("qwen27_grammar", run, proof, export.prior.Evidence())
        self.assertTrue(legacy)
        with self.assertRaises(export.suite.shared.Refusal):
            export.admit("qwen27_thinking", run, proof, export.prior.Evidence())

    def test_thinking_producer_profile_configuration_is_exact(self):
        for profile in ("qwen27_thinking", "qwen9_thinking"):
            run = self.root / profile
            execution = export.suite.read_json(export.prior.DEFAULT_EXECUTION / "execution.json")
            self.write(run.parent / "execution.json", execution)
            execution_hash = export.suite.digest(run.parent / "execution.json")
            self.write(run.parent / "execution.json.sha256", execution_hash)
            configuration = asdict(export.prior.panel.configurations()[profile])
            manifest = {"execution_sha256": execution_hash, "fresh": True, "maximum_calls": 160,
                        "model": configuration, "planned_episodes": 64, "profile": profile}
            self.write(run / "manifest.json", manifest)
            model, _ = export.qwen_identity(
                profile, run, self.registration, export.prior.panel.DEFAULT_ROOT, export.prior.Evidence())
            self.assertEqual(model, configuration["model"])
            manifest["model"]["think"] = False
            self.write(run / "manifest.json", manifest)
            with self.assertRaises(export.suite.shared.Refusal):
                export.qwen_identity(
                    profile, run, self.registration, export.prior.panel.DEFAULT_ROOT,
                    export.prior.Evidence())

    def trained_fixture(self, profile="kev"):
        root = self.root / "trained"
        run = root / f"run-{profile}"
        artifacts = {"source_revisions": export.SOURCES, "models": {
            name: {"repository": repository, "revision": revision,
                   "path": str(root / "models" / name), "sha256": {"model.safetensors": "a" * 64}}
            for name, (repository, revision) in export.PINS.items()}}
        original = {
            "selection": str(export.prior.panel.DEFAULT_ROOT),
            "selection_sha256": self.registration["selection_sha256"],
            "bundle_manifest_sha256": self.registration["bundle_manifest_sha256"],
            "protocol": "explicit_actions", "scheduled_episodes_per_model": 64,
            "maximum_calls_per_model": 160, "context_cap": 8192, "inference_timeout_seconds": 180,
            "policy": {"retry": False, "repair": False, "truncate": False, "own_history": True,
                       "all_scheduled_in_denominator": True, "probability_calibration_claim": False,
                       "private_labels_sent": False},
            "code_sha256": {"scripts/run_trained_jevlike_pilot.py": "old"}, "models": artifacts}
        self.write(root / "benchmark-registration.json", original)
        original_hash = export.suite.digest(root / "benchmark-registration.json")
        self.write(root / "benchmark-registration.sha256", original_hash)
        active = copy.deepcopy(original)
        active["code_sha256"]["scripts/run_trained_jevlike_pilot.py"] = "new"
        active["resource_amendment"] = {
            "prior_registration_sha256": original_hash, "benchmark_calls_before_amendment": 0,
            "changed_source": "scripts/run_trained_jevlike_pilot.py",
            "selection_models_inference_scoring_unchanged": True,
            "prior_source_sha256": "old", "new_source_sha256": "new"}
        self.write(root / "benchmark-registration-parent-locks.json", active)
        self.write(root / "benchmark-registration-parent-locks.sha256",
                   export.suite.digest(root / "benchmark-registration-parent-locks.json"))
        self.write(root / "artifacts.json", artifacts)
        self.write(root / "panel.json", [i["id"] for i in self.instances])
        self.write(root / "initial-requests.json",
                   [export.suite.json_value(asdict(export.canonical.public_request(i, [])))
                    for i in self.instances])
        self.write(run / "runtime.json", {"status": "ready", "runtime": {
            "candidate": profile, "device": "cuda", "temperature": 0.75, "context_limit": 8192,
            "cpu_threads": 4, "dtype": "float32" if profile == "kev" else "bfloat16",
            "cuda_initialized": True, "calibration_claim": False,
            "base_metadata": {"base": export.PINS["kev-base"][0],
                              "base_revision": export.PINS["kev-base"][1]} if profile == "kev" else None}})
        proof = {"effective_temperature": 0.75, "temperature_independently_verified": True,
                 "checkpoint_inventory_independently_verified": True}
        return run, proof

    def test_actual_trained_producer_schema_pins_and_temperatures(self):
        for profile in ("decider", "kev"):
            with self.subTest(profile=profile):
                run, proof = self.trained_fixture(profile)
                with patch.object(export, "source_hashes"):
                    model, identity = export.trained_identity(
                        profile, run, self.registration, export.prior.panel.DEFAULT_ROOT,
                        self.instances, proof, export.prior.Evidence())
                self.assertEqual(model, "@".join(export.PINS[profile]))
                self.assertFalse(identity["checkpoint_payloads_read_by_exporter"])
                proof["effective_temperature"] = 1.0
                with patch.object(export, "source_hashes"), self.assertRaises(export.suite.shared.Refusal):
                    export.trained_identity(
                        profile, run, self.registration, export.prior.panel.DEFAULT_ROOT,
                        self.instances, proof, export.prior.Evidence())

    def test_checkpoint_metadata_change_not_accepted(self):
        run, proof = self.trained_fixture()
        artifacts = export.suite.read_json(run.parent / "artifacts.json")
        artifacts["models"]["kev"]["revision"] = "wrong"
        self.write(run.parent / "artifacts.json", artifacts)
        with patch.object(export, "source_hashes"), self.assertRaises(export.suite.shared.Refusal):
            export.trained_identity(
                "kev", run, self.registration, export.prior.panel.DEFAULT_ROOT,
                self.instances, proof, export.prior.Evidence())

    def test_trained_scheduled_unattempted_score_fields_replayed(self):
        rows = export.canonical.replay(
            self.instances, [], SimpleNamespace(model=self.model), native_action=False)
        for row in rows:
            row.update(scheduled=True, unattempted=True)
        export.replay_terminal(self.instances, [], rows, {
            "status": "aborted", "calls": 0, "fatal": "startup failure",
            "scheduled_episodes": 64, "unattempted_episodes": 64, "failure_count": 0},
            self.model, trained=True)


if __name__ == "__main__":
    unittest.main()
