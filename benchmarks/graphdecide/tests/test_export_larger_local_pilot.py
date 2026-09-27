import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import export_larger_local_pilot as export


class CompletedPilotExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = export.Evidence()
        cls.registration, cls.instances, cls.cases, _ = export.load_panel(
            export.panel.DEFAULT_ROOT, cls.evidence)
        cls.manifest, cls.rows, cls.attempts, cls.proof = export.load_candidate(
            export.DEFAULT_EXECUTION / export.CANDIDATE, cls.registration,
            export.panel.DEFAULT_ROOT, cls.instances, cls.evidence)

    def test_real_panel_replay_and_independent_objectives(self):
        self.assertEqual(len(self.instances), 64)
        self.assertEqual(len(self.attempts), 160)
        self.assertTrue(all(r["feasible"] for r in self.rows))
        audits = export.anomaly_audit(self.instances, self.rows, self.proof)
        self.assertEqual(len(audits), 16)
        lt = [r for r in audits if r["task"] == "lt_influence_construct"]
        self.assertEqual(sum(r["optimal"] for r in lt), 2)
        community = [r for r in audits if r["task"] == "community_bipartition_construct"]
        self.assertTrue(all(abs(r["independent_objective"]) < 1e-12 for r in community))

    def test_parent_proof_must_cover_all_eight_cut_community_episodes(self):
        for mutate in ("missing", "duplicate", "wrong_objective"):
            with self.subTest(mutate=mutate):
                proof = copy.deepcopy(self.proof)
                checks = proof["independent_networkx_checks"]
                if mutate == "missing":
                    checks.pop()
                elif mutate == "duplicate":
                    checks.append(checks[0])
                else:
                    checks[0]["independent_objective"] += 1
                with self.assertRaises(export.suite.shared.Refusal):
                    export.anomaly_audit(self.instances, self.rows, proof)

    def test_partial_lane_refused_before_reading_manifest_or_scores(self):
        for status in ("running", "aborted", "unattempted", "registered"):
            with self.subTest(status=status), patch.object(
                    export.suite, "read_json", return_value={"status": status}) as reader, \
                    patch.object(export.matrix, "verified_run") as canonical:
                with self.assertRaises(export.suite.shared.Refusal):
                    export.load_sealed_reference(
                        Path("unused"), Path("unused"), "unused", [], export.Evidence())
                self.assertEqual(reader.call_args_list[0].args[0], Path("unused/status.json"))
                self.assertEqual(reader.call_count, 1)
                canonical.assert_not_called()

    def test_terminal_without_complete_seal_is_not_generic_candidate(self):
        with patch.object(export.suite, "read_json", return_value={
                "status": "completed", "artifact_sha256": {"scores.jsonl": "fake"}}), \
                patch.object(export.matrix, "verified_run") as canonical:
            with self.assertRaises(export.suite.shared.Refusal):
                export.load_sealed_reference(
                    Path("unused"), Path("unused"), "unused", [], export.Evidence())
            canonical.assert_not_called()

    def test_candidate_partial_refused_before_proof_or_attempts(self):
        with patch.object(export.suite, "read_json", return_value={"status": "running"}) as reader:
            with self.assertRaises(export.suite.shared.Refusal):
                export.load_candidate(
                    Path("unused"), {}, Path("unused"), [], export.Evidence())
            self.assertEqual(reader.call_count, 1)

    def test_tampered_initial_request_rejected(self):
        attempts = copy.deepcopy(self.attempts)
        attempts[0]["request"]["request_id"] = "wrong"
        with self.assertRaises(export.suite.shared.Refusal):
            export.check_initial(self.instances, attempts)
        with self.assertRaises(export.suite.shared.Refusal):
            export.canonical.replay(self.instances, attempts,
                                    export.matrix.ModelConfig(**self.manifest["model"]),
                                    native_action=False)

    def test_own_history_action_tampering_rejected(self):
        attempts = copy.deepcopy(self.attempts)
        construction = next(a for a in attempts if a["step"] == 1)
        construction["step"] = 7
        with self.assertRaises(export.suite.shared.Refusal):
            export.canonical.replay(self.instances, attempts,
                                    export.matrix.ModelConfig(**self.manifest["model"]),
                                    native_action=False)

    def test_terminal_label_cannot_hide_missing_final_episode(self):
        last_id = self.instances[-1]["id"]
        attempts = [a for a in self.attempts if a["instance_id"] != last_id]
        rows = export.canonical.replay(
            self.instances, attempts, export.matrix.ModelConfig(**self.manifest["model"]),
            native_action=False)
        with self.assertRaises(export.suite.shared.Refusal):
            export.complete_coverage(
                self.instances, attempts, rows, {"calls": len(attempts), "status": "completed"})

    def test_real_failure_remains_in_planned_denominator(self):
        attempts = copy.deepcopy(self.attempts)
        attempts[0].update(status="failure", selected_option_id=None,
                           diagnostic_code="timeout", fatal=False)
        rows = export.canonical.replay(
            self.instances, attempts, export.matrix.ModelConfig(**self.manifest["model"]),
            native_action=False)
        export.complete_coverage(self.instances, attempts, rows, {
            "calls": len(attempts), "status": "completed_with_failures"})
        metrics = export.task_metrics({"failed": rows})
        adjacency = next(r for r in metrics if r["task"] == "adjacency"
                         and r["node_count"] == "all")
        self.assertEqual((adjacency["episodes"], adjacency["hits"], adjacency["failed_episodes"]),
                         (8, 7, 1))
        pairs, summary = export.paired_differences({"jev_action": self.rows, "failed": rows})
        failed_pair = next(r for r in pairs if r["instance_id"] == self.instances[0]["id"])
        self.assertEqual(failed_pair["hit_difference"], 1)
        self.assertFalse(failed_pair["right_feasible"])
        self.assertEqual(len(summary), 30)
        with self.assertRaises(export.suite.shared.Refusal):
            export.complete_coverage(self.instances, attempts, rows, {
                "calls": len(attempts), "status": "completed"})

    def test_objective_failure_excluded_only_from_objective_pairs_not_hits(self):
        left = copy.deepcopy(next(r for r in self.rows if r["kind"] == "optimization"))
        right = copy.deepcopy(left)
        right.update(feasible=False, objective=None, absolute_gap=None, failure_count=1)
        pairs, aggregates = export.paired_differences({"jev_action": [left], "other": [right]})
        self.assertEqual(len(pairs), 1)
        self.assertIsNone(pairs[0]["objective_difference"])
        self.assertEqual(aggregates[0]["episodes"], 1)
        self.assertEqual(aggregates[0]["objective_pairs"], 0)

    def test_task_size_denominators_no_cross_task_pooling(self):
        metrics = export.task_metrics({"candidate": self.rows})
        self.assertEqual(len(metrics), 30)
        for row in metrics:
            expected = 4 if row["hit_definition"] == "optimum" else 8
            self.assertEqual(row["episodes"], expected if row["node_count"] == "all" else expected // 2)
        self.assertEqual({r["task"] for r in metrics}, {i["task"] for i in self.instances})

    def test_pairing_by_id_not_input_order(self):
        pairs, aggregates = export.paired_differences({
            "jev_action": self.rows, "other": list(reversed(self.rows))})
        self.assertEqual(len(pairs), 64)
        self.assertTrue(all(r["hit_difference"] == 0 for r in pairs))
        self.assertTrue(all(r["mean_hit_difference"] == 0 for r in aggregates))

    def test_missing_or_duplicate_selected_pair_refused(self):
        for rows in (self.rows[:-1], self.rows[:-1] + [self.rows[0]]):
            with self.assertRaises(export.suite.shared.Refusal):
                export.paired_differences({"jev_action": self.rows, "other": rows})

    def test_incomplete_generic_manifest_refused_before_replay(self):
        with patch.object(export, "terminal_status", return_value={
                "status": "completed", "artifact_sha256": dict.fromkeys(export.ARTIFACTS, "hash")}), \
                patch.object(export.Evidence, "track", return_value="hash"), \
                patch.object(export.suite, "read_json", return_value={}), \
                patch.object(export.matrix, "verified_run") as canonical:
            with self.assertRaises(export.suite.shared.Refusal):
                export.load_sealed_reference(
                    Path("unused"), Path("unused"), "unused", [], export.Evidence())
            canonical.assert_not_called()

    def test_export_refuses_existing_destination(self):
        with self.assertRaises(export.suite.shared.Refusal):
            export.export(export.panel.DEFAULT_ROOT)

    def test_source_hash_checks_fail_closed(self):
        with patch.object(export.suite, "digest", return_value="changed"):
            with self.assertRaises(export.suite.shared.Refusal):
                export.Evidence().track(export.REPO / "source.py", "expected")
            with self.assertRaises(export.suite.shared.Refusal):
                self.evidence.recheck()


if __name__ == "__main__":
    unittest.main()
