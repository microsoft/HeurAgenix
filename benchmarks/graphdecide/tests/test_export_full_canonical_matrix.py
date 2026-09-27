import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from scripts import export_full_canonical_matrix as exporter


class FullCanonicalExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = exporter.release.labeled_instances(
            exporter.BUNDLE, exporter.SPLIT, exporter.release.manifest_at(exporter.BUNDLE))
        cls.empty = exporter.canonical.replay(
            cls.instances, [], SimpleNamespace(model="fixture"), native_action=False)

    def setUp(self):
        self.root = exporter.REPO / "output" / ("full-canonical-export-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def fixture_lane(self, name="qwen27_grammar", kind="qwen27"):
        return exporter.Lane(name, "Qwen", kind, self.root / name)

    def decorated(self):
        return [{**row, "profile": "fixture", "family": "Qwen", "group": "decision_matrix",
                 "coverage_outcome": "unattempted", "strict_contract_failures": 0,
                 "strict_contract_observed_calls": 0}
                for row in copy.deepcopy(self.empty)]

    def failed_attempts(self, diagnostic="timeout", growing=False):
        attempts = []
        chosen = next(i["id"] for i in self.instances if i["kind"] == "optimization")
        for instance in self.instances:
            history = []
            if growing and instance["id"] == chosen:
                request = exporter.canonical.public_request(instance, history)
                selected = request.options[0].id
                attempts.append({
                    "instance_id": instance["id"], "step": 0,
                    "request": exporter.suite.json_value(asdict(request)),
                    "status": "success", "selected_option_id": selected,
                    "resolved_model": "fixture", "fatal": False})
                history.append(selected)
            request = exporter.canonical.public_request(instance, history)
            attempts.append({
                "instance_id": instance["id"], "step": len(history),
                "request": exporter.suite.json_value(asdict(request)),
                "status": "failure", "selected_option_id": None,
                "resolved_model": None, "fatal": False, "diagnostic_code": diagnostic})
        return attempts, chosen

    def replay_failed(self, diagnostic="timeout", growing=False, initial_coverage=False):
        attempts, chosen = self.failed_attempts(diagnostic, growing)
        rows = exporter.canonical.replay(
            self.instances, attempts, SimpleNamespace(model="fixture"), native_action=False)
        if initial_coverage:
            for row in rows:
                row["initial_context_compatible"] = False
        kind = "laya" if initial_coverage else "reference"
        lane = self.fixture_lane(kind=kind)
        output, audits = exporter.replay_lane(lane, self.instances, attempts, rows, "fixture", False, {
            "status": "completed" if initial_coverage else "completed_with_failures", "calls": len(attempts)})
        return output, audits, chosen

    def test_fixed_inventory_has_all_families_and_separate_strong_anchors(self):
        lanes = exporter.lanes(self.root / "astra")
        self.assertEqual(len(lanes), 10)
        self.assertEqual({lane.family for lane in lanes}, {"Jev", "Jev-like", "Qwen", "GPT"})
        self.assertEqual(sum(lane.group == "strong_anchors" for lane in lanes), 2)
        self.assertEqual(sum(lane.reused for lane in lanes), 5)
        self.assertFalse(any("thinking" in lane.name for lane in lanes))

    def test_full_panel_counts_and_nonuniform_degree_strata(self):
        exporter.check_panel(self.instances)
        metrics = exporter.task_metrics(self.decorated())
        degree = [r for r in metrics if r["task"] == "degree_exact" and r["node_count"] != "all"]
        self.assertEqual([r["scheduled"] for r in degree], [64, 72, 80, 88, 96])

    def test_pilot_confirmation_or_duplicate_panel_refused(self):
        for instances in (self.instances[:64], self.instances[:952],
                          self.instances[:-1] + [self.instances[0]]):
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.check_panel(instances)

    def test_macro_size_accuracy_not_degree_micro_accuracy(self):
        rows = self.decorated()
        for row in rows:
            if row["task"] == "degree_exact" and row["node_count"] == 8:
                row.update(feasible=True, correct=True, coverage_outcome="feasible")
        metric = next(r for r in exporter.task_metrics(rows)
                      if r["task"] == "degree_exact" and r["node_count"] == "all")
        self.assertEqual(metric["hits"], 64)
        self.assertAlmostEqual(metric["all_scheduled_hit_rate"], .16)
        self.assertAlmostEqual(metric["macro_size_accuracy"], .2)

    def test_all_scheduled_zero_attempt_metrics_keep_null_gaps(self):
        metrics = exporter.task_metrics(self.decorated())
        self.assertEqual(len(metrics), 60)
        all_rows = [r for r in metrics if r["node_count"] == "all"]
        self.assertEqual(sum(r["scheduled"] for r in all_rows), 960)
        self.assertEqual(sum(r["unattempted"] for r in all_rows), 960)
        for row in metrics:
            self.assertIsNone(row["mean_absolute_gap_feasible"])
            self.assertIsNone(row["mean_objective_feasible"])
            self.assertEqual(row["conditional_objective_denominator"], 0)

    def test_all_failed_completed_replay_retains_every_episode(self):
        rows, audits, _ = self.replay_failed()
        self.assertEqual(len(rows), 960)
        self.assertEqual(audits, [])
        metrics = [r for r in exporter.task_metrics(rows) if r["node_count"] == "all"]
        self.assertEqual(sum(r["execution_failures"] for r in metrics), 960)
        self.assertEqual(sum(r["failure_calls"] for r in metrics), 960)
        self.assertTrue(all(r["hits"] == 0 and r["mean_absolute_gap_feasible"] is None for r in metrics))

    def test_initial_unsupported_is_not_execution_failure(self):
        rows, _, _ = self.replay_failed("unsupported", initial_coverage=True)
        metrics = [r for r in exporter.task_metrics(rows) if r["node_count"] == "all"]
        self.assertEqual(sum(r["initial_unsupported"] for r in metrics), 960)
        self.assertEqual(sum(r["execution_failures"] for r in metrics), 0)
        self.assertEqual(sum(r["failure_calls"] for r in metrics), 960)

    def test_growing_context_unsupported_is_separate_from_initial_unsupported(self):
        rows, _, chosen = self.replay_failed("unsupported", growing=True)
        row = next(r for r in rows if r["instance_id"] == chosen)
        self.assertEqual(row["coverage_outcome"], "execution_unsupported")
        self.assertEqual(row["successful_queries"], 1)
        self.assertFalse(row["feasible"])

    def test_completed_cannot_conceal_unattempted_tail(self):
        attempts, _ = self.failed_attempts()
        attempts.pop()
        saved = exporter.canonical.replay(
            self.instances, attempts, SimpleNamespace(model="fixture"), native_action=False)
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.replay_lane(self.fixture_lane(), self.instances, attempts, saved, "fixture", False, {
                "status": "completed_with_failures", "calls": len(attempts)})

    def test_malformed_history_and_modified_score_refused(self):
        attempts, _ = self.failed_attempts()
        saved = exporter.canonical.replay(
            self.instances, attempts, SimpleNamespace(model="fixture"), native_action=False)
        changed = copy.deepcopy(saved)
        changed[0]["query_count"] = 0
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.replay_lane(self.fixture_lane(), self.instances, attempts, changed, "fixture", False, {
                "status": "completed_with_failures", "calls": len(attempts)})
        attempts[0]["request"]["request_id"] = "different"
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.replay_lane(self.fixture_lane(), self.instances, attempts, saved, "fixture", False, {
                "status": "completed_with_failures", "calls": len(attempts)})

    def test_pending_refused_before_import_or_correctness_reads(self):
        lane = self.fixture_lane()
        self.write(lane.terminal, {"status": "running", "calls": 18})
        with patch.object(exporter, "import_verifier") as imported, \
                patch.object(exporter.suite, "read_rows") as read:
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.source_data(lane, exporter.Evidence())
            imported.assert_not_called()
            read.assert_not_called()

    def test_full_export_preflights_all_lanes_before_reading_completed_reference(self):
        source_lanes = [self.fixture_lane(), self.fixture_lane("gpt", "gpt54")]
        self.write(source_lanes[0].terminal, {"status": "completed"})
        self.write(source_lanes[0].seal, {})
        self.write(source_lanes[1].terminal, {"status": "running"})
        destination = self.root / "export"
        with patch.object(exporter, "lanes", return_value=source_lanes), \
                patch.object(exporter, "load_panel") as panel, \
                patch.object(exporter, "source_data") as source:
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.export(self.root / "astra", destination)
            panel.assert_not_called()
            source.assert_not_called()
        self.assertFalse(destination.exists())

    def test_abort_and_missing_seal_are_not_completed_matrix_sources(self):
        for kind in ("qwen27", "laya", "trained", "gpt54", "astra", "reference"):
            lane = self.fixture_lane(name=kind, kind=kind)
            self.write(lane.terminal, {"status": "aborted"})
            self.write(lane.seal, {})
            self.assertFalse(exporter.readiness([lane])[lane.name]["ready"])
            lane.seal.unlink()
            self.write(lane.terminal, {"status": "completed"})
            self.assertFalse(exporter.readiness([lane])[lane.name]["ready"])

    def test_laya_missing_offline_proof_never_calls_writing_fallback(self):
        lane = self.fixture_lane("laya", "laya")
        self.write(lane.terminal, {"status": "completed", "scheduled_episodes": 960})
        with patch.object(exporter, "import_verifier") as imported:
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.source_data(lane, exporter.Evidence())
            imported.assert_not_called()

    def test_explicit_mock_markers_refused(self):
        for key in ("test_only", "injected_worker", "injected_client"):
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.no_test_results({key: True})
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.no_test_results({key: 0})

    def test_injected_qwen_manifest_rejected_before_source_replay(self):
        lane = self.fixture_lane()
        self.write(lane.terminal, {"status": "completed"})
        for name in ("manifest.json", "attempts.jsonl", "scores.jsonl", "summary.json"):
            self.write(lane.run / name, {"injected_client": True} if name == "manifest.json" else {})
        self.write(lane.seal, {name: exporter.suite.digest(lane.run / name) for name in (
            "manifest.json", "terminal.json", "attempts.jsonl", "scores.jsonl", "summary.json")})
        producer = Mock()
        with patch.object(exporter, "import_verifier", return_value=producer), \
                patch.object(exporter.suite, "read_rows") as rows:
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.source_data(lane, exporter.Evidence())
            producer.verify.assert_not_called()
            rows.assert_not_called()

    def test_source_byte_tamper_and_incomplete_seal_refused(self):
        file = self.root / "input.json"
        self.write(file, {"sealed": True})
        evidence = exporter.Evidence()
        evidence.track(file)
        self.write(file, {"sealed": False})
        with self.assertRaises(exporter.suite.shared.Refusal):
            evidence.recheck()
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.Evidence().inventory(self.root, {}, {"attempts.jsonl"})
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.Evidence().inventory(self.root, {"../outside.json": "not-read"})

    def test_probability_conformance_counts_have_explicit_observation_denominator(self):
        rows = self.decorated()
        rows[0].update(strict_contract_observed_calls=1, strict_contract_failures=1)
        own = next(r for r in exporter.task_metrics(rows)
                   if r["task"] == rows[0]["task"] and r["node_count"] == "all")
        self.assertEqual(own["strict_contract_observed_calls"], 1)
        self.assertEqual(own["strict_contract_failure_calls"], 1)
        self.assertEqual(own["failure_calls"], 0)

    def test_existing_output_is_not_overwritten(self):
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.export(self.root / "astra", self.root)

    def test_gpt_fallback_does_not_ignore_non_summary_verifier_failure(self):
        producer = Mock()
        producer.verify.side_effect = exporter.suite.shared.Refusal("Episode replay mismatch.")
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.verify_gpt(producer, self.root)
        producer.load.assert_not_called()

    def test_gpt_aggregate_compatibility_refuses_real_changes_and_nan(self):
        for changed in ({"value": .2}, {"value": float("nan")}):
            self.assertFalse(exporter.matrix.same_aggregates({"value": .1}, changed))
        self.assertTrue(exporter.matrix.same_aggregates({"value": .1}, {"value": .1 + 1e-15}))

    def test_generated_tex_preserves_null_gaps_and_taskwise_denominators(self):
        metrics = exporter.task_metrics(self.decorated())
        path = self.root / "tables.tex"
        exporter.write_tex(path, metrics)
        text = path.read_text()
        self.assertEqual(text.count(r"\begin{table}"), 10)
        self.assertEqual(text.count("n/a & n/a"), 4)
        self.assertIn("fixture & 400 & 0 & 0 & 0.00 & 0.00", text)
        self.assertIn(r"community\_bipartition\_construct", text)
        with self.assertRaises(FileExistsError):
            exporter.write_tex(path, metrics)

    def test_laya_terminal_adapter_with_full_failed_fixture_and_process_identity(self):
        lane = self.fixture_lane("laya", "laya")
        registration = {"model": "fixture", "protocol": "explicit_actions",
                        "split": exporter.SPLIT, "bundle": str(exporter.BUNDLE), "files_sha256": {}}
        self.write(lane.root / "registration.json", registration)
        digest = exporter.suite.digest(lane.root / "registration.json")
        self.write(lane.root / "registration.sha256", digest)
        attempts, _ = self.failed_attempts("unsupported")
        lane.run.mkdir()
        with (lane.run / "attempts.jsonl").open("w") as stream:
            for row in attempts:
                stream.write(json.dumps(row) + "\n")
        rows = exporter.canonical.replay(
            self.instances, attempts, SimpleNamespace(model="fixture"), native_action=False)
        self.write(lane.run / "scores.json", rows)
        self.write(lane.run / "summary.json", {"scheduled_episodes": 960})
        proof = {"verified": True, "model_calls": 0, **{
            name + "_sha256": exporter.suite.digest(lane.run / filename)
            for name, filename in (("attempts", "attempts.jsonl"), ("scores", "scores.json"),
                                   ("summary", "summary.json"))}}
        self.write(lane.seal, proof)
        self.write(lane.terminal, {"status": "completed", "fatal": None,
                                  "sealed_replay": proof, "scheduled_episodes": 960,
                                  "registration_sha256": digest})
        self.write(lane.run / "runtime.json", {
            "status": "ready", "dtype": "float32", "threads": 4,
            "interop_threads": 1, "cuda_initialized": False})
        self.write(lane.run / "supervisor-process.json", {
            "pid": 123, "startticks": 456, "device": "cpu", "cuda_visible_devices": ""})
        self.write(lane.run / "worker-identity.json", {"pid": 124, "startticks": 457})
        self.write(lane.run / "status.json", {"status": "completed"})
        producer = Mock()
        producer.verify_results.return_value = proof
        with patch.object(exporter, "import_verifier", return_value=producer):
            _, terminal, actual, saved, model, native = exporter.source_data(lane, exporter.Evidence())
            checked, audits = exporter.replay_lane(
                lane, self.instances, actual, saved, model, native, terminal)
            self.assertEqual(len(checked), 960)
            self.assertEqual(audits, [])
            self.write(lane.run / "worker-identity.json", {"pid": 124})
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.source_data(lane, exporter.Evidence())


class ExistingReferenceIntegrationTest(unittest.TestCase):
    def test_all_five_real_reused_sources_full_replay_and_independent_objectives(self):
        evidence = exporter.Evidence()
        instances = exporter.release.labeled_instances(
            exporter.BUNDLE, exporter.SPLIT, exporter.release.manifest_at(exporter.BUNDLE))
        lanes = exporter.lanes(exporter.REPO / "output/unused-astra-not-read")
        for lane in (lane for lane in lanes if lane.reused):
            with self.subTest(profile=lane.name):
                manifest, terminal, attempts, rows, model_id, native = exporter.source_data(lane, evidence)
                checked, objectives = exporter.replay_lane(
                    lane, instances, attempts, rows, model_id, native, terminal)
                self.assertEqual(manifest["profile"], lane.name)
                self.assertEqual(len(checked), 960)
                self.assertEqual(len(objectives), sum(r["feasible"] for r in checked
                                                    if r["task"] in exporter.CONSTRUCTION))
                self.assertTrue(all(r["reused_reference"] for r in checked))
        evidence.recheck()


if __name__ == "__main__":
    unittest.main()
