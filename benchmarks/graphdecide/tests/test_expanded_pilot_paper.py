import copy
import json
from pathlib import Path
import shutil
import unittest
import uuid

from paper import build_expanded_pilot_table as builder


class ExpandedPilotPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = builder.suite.read_rows(builder.prior.panel.DEFAULT_ROOT / "pilot-cases.jsonl")
        evidence = builder.Evidence()
        cls.reports, cls.sources = {}, []
        for profile, path in (("bounded", builder.BOUNDED),
                              *((p, builder.FINALIZATION / "exports" / p) for p in builder.NEW)):
            report, models, _ = builder.verified(path, profile, cls.cases, evidence)
            cls.reports[profile] = report
            cls.sources.append(models)
        cls.models = builder.combine(cls.sources)
        cls.metrics = builder.expanded.metrics(cls.models)
        cls.audit = builder.diagnostics(cls.models, cls.reports, evidence)
        evidence.recheck()

    def setUp(self):
        self.root = builder.REPO / "paper/expanded-pilot-table-test-fixtures" / uuid.uuid4().hex
        self.parent_owned = not self.root.parent.exists()
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        shutil.rmtree(self.root)
        if self.parent_owned:
            try:
                self.root.parent.rmdir()
            except OSError:
                pass

    def test_actual_exact_counts_and_all_returned_thinking_answers(self):
        counts = [self.audit[p]["exact_correct"] for p in builder.PROFILES]
        self.assertEqual(counts, [33, 30, 47, 40, 24, 23])
        for profile, calls, failures in (("qwen27_thinking", 66, 17), ("qwen9_thinking", 64, 24)):
            self.assertEqual(self.audit[profile]["exact_correct"], self.audit[profile]["exact_returned"])
            self.assertEqual(self.audit[profile]["actual_calls"], calls)
            self.assertEqual(self.audit[profile]["failed_calls"], failures)
            self.assertEqual(self.audit[profile]["construction_feasible"], 0)
            self.assertEqual(self.audit[profile]["construction_failed"], 16)

    def test_each_profile_retains_full_task_and_size_denominators(self):
        builder.verify_panel_rows(self.models, self.cases)
        for metric in self.metrics:
            count = 8 if metric["task"] in builder.EXACT else 4
            if metric["node_count"] != "all":
                count //= 2
            self.assertEqual(metric["episodes"], count)

    def test_missing_duplicate_or_wrong_size_episode_refused(self):
        for mutation in ("missing", "duplicate", "size"):
            models = copy.deepcopy(self.models)
            rows = models["jev_action"]
            if mutation == "missing":
                rows.pop()
            elif mutation == "duplicate":
                rows[-1] = rows[0]
            else:
                rows[0]["node_count"] = 12
            with self.subTest(mutation=mutation), self.assertRaises(builder.suite.shared.Refusal):
                builder.verify_panel_rows(models, self.cases)

    def test_reference_trajectories_not_just_accuracy_must_match(self):
        sources = copy.deepcopy(self.sources)
        sources[1]["jev_action"][0]["decisions"] = ["different"]
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.combine(sources)

    def test_reference_missing_profile_cannot_hide_coverage(self):
        models = copy.deepcopy(self.sources[0])
        models["jev_action"] = models["jev_action"][:-1]
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.verify_panel_rows(models, self.cases)

    def test_summary_metric_mismatch_refused(self):
        report = copy.deepcopy(self.reports["bounded"])
        report["metrics"][0]["hits"] += 1
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.check_metrics(report, self.sources[0])

    def test_source_call_count_mismatch_refused(self):
        report = copy.deepcopy(self.reports["bounded"])
        report["provenance"]["jev_action"]["selected_calls"] -= 1
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.check_metrics(report, self.sources[0])

    def test_zero_feasible_gaps_render_missing_not_zero(self):
        tables = builder.tables(self.metrics)
        self.assertEqual(tables["construction-gaps.tex"].count(r"\textemdash"), 8)
        self.assertIn("0/4", tables["construction-feasible.tex"])
        self.assertIn("47/48", tables["exact.tex"])
        self.assertIn("0.113657", tables["construction-gaps.tex"])
        self.assertNotIn("resizebox", "".join(tables.values()))

    def test_invented_zero_gap_or_wrong_conditional_denominator_refused(self):
        for field, value in (("mean_absolute_gap_feasible", 0.0), ("objective_gap_denominator", 4)):
            metrics = copy.deepcopy(self.metrics)
            row = next(r for r in metrics if r["profile"] == "qwen27_thinking"
                       and r["task"] == "tsp_construct" and r["node_count"] == "all")
            row[field] = value
            with self.subTest(field=field), self.assertRaises(builder.suite.shared.Refusal):
                builder.tables(metrics)

    def test_duplicate_table_cells_refused(self):
        metric = next(r for r in self.metrics if r["node_count"] == "all")
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.tables([*self.metrics, metric])

    def test_native_degree_and_constant_construction_diagnostics(self):
        for profile in ("decider", "kev"):
            degree = self.audit[profile]["degree"]
            self.assertTrue(degree["full_8_and_12_class_coverage"])
            self.assertTrue(degree["varied_predictions"])
            self.assertFalse(degree["global_plus_one"])
            self.assertFalse(degree["global_minus_one"])
            self.assertTrue(self.audit[profile]["all_B_cut_decisions"])
        self.assertTrue(self.audit["decider"]["all_A_community_decisions"])
        self.assertFalse(self.audit["kev"]["all_A_community_decisions"])

    def sealed_fixture(self):
        source = self.root / "sealed"
        source.mkdir()
        upstream = self.root / "upstream.json"
        upstream.write_text("{}\n")
        report = copy.deepcopy(self.reports["bounded"])
        report["source_sha256"] = {builder.expanded.relative(upstream): builder.suite.digest(upstream)}
        (source / "verified-summary.json").write_text(json.dumps(report))
        shutil.copyfile(builder.BOUNDED / "episodes.csv", source / "episodes.csv")
        seal = {"status": "completed", "artifact_sha256": {
            name: builder.suite.digest(source / name) for name in ("episodes.csv", "verified-summary.json")}}
        (source / "export-seal.json").write_text(json.dumps(seal))
        return source, upstream

    def test_sealed_artifact_byte_tamper_refused(self):
        source, _ = self.sealed_fixture()
        with (source / "episodes.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.verified(source, "bounded", self.cases, builder.Evidence())

    def test_original_source_hash_tamper_refused(self):
        source, upstream = self.sealed_fixture()
        upstream.write_text('{"changed":true}\n')
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.verified(source, "bounded", self.cases, builder.Evidence())

    def test_cached_conflicting_source_hash_claims_refused(self):
        path = self.root / "evidence.json"
        path.write_text("{}\n")
        evidence = builder.Evidence()
        evidence.track(path)
        with self.assertRaises(builder.suite.shared.Refusal):
            evidence.track(path, "0" * 64)

    def test_paper_numbers_generated_from_audited_counts(self):
        text = builder.numbers(self.metrics, self.audit)
        self.assertIn(r"\newcommand{\EPLargeThinkingExact}{47/48}", text)
        self.assertIn(r"\newcommand{\EPDeciderOptima}{0/16}", text)
        self.assertIn(r"\newcommand{\EPKevOptima}{1/16}", text)
        self.assertIn(r"\newcommand{\EPKevCommunityGap}{0.113657}", text)

    def test_existing_output_refused(self):
        with self.assertRaises(builder.suite.shared.Refusal):
            builder.export(self.root)


if __name__ == "__main__":
    unittest.main()
