"""Run with PYTHONPATH=tests python3 -m unittest test_website_data."""

from copy import deepcopy
import csv
import io
import json
import shutil
import unittest
from unittest.mock import patch
import uuid

from scripts import export_website_data as website
from scripts.paired_graph_ablation import Refusal
import test_paper_full_matrix


class WebsiteDataTests(unittest.TestCase):
    def setUp(self):
        self.root = website.REPO / "output" / ("website-tests-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)

    def fixture(self):
        return test_paper_full_matrix.FullMatrixPaperTests().fixture()

    def provenance(self):
        return {
            "source_summary_sha256": "a" * 64, "source_seal_sha256": "b" * 64,
            "source_completion_sha256": "c" * 64, "verified_input_hashes": 1,
            "new_model_calls": 0, "source": "/private/credential-path",
        }

    def seal(self, summary=None):
        source = self.root / "sealed" / "export"
        source.mkdir(parents=True)
        original = self.root / "original.json"
        original.write_text('{"private": "do not publish"}\n')
        summary = deepcopy(summary if summary is not None else self.fixture())
        summary["source_sha256"] = {"original.json": website.build.suite.digest(original)}
        (source / "verified-summary.json").write_text(json.dumps(summary))
        seal = {"status": "completed", "artifact_sha256": {
            "verified-summary.json": website.build.suite.digest(source / "verified-summary.json"),
        }}
        (source / "export-seal.json").write_text(json.dumps(seal))
        (source.parent / "completion.json").write_text(json.dumps({
            "status": "completed_with_source_failures",
            "export_seal_sha256": website.build.suite.digest(source / "export-seal.json"),
        }))
        return source

    def test_whitelist_and_all_600_mappings(self):
        source = self.fixture()
        source["secret"] = "credential response prompt source_map /private/"
        source["profiles"]["jev_action"]["response"] = source["secret"]
        source["metrics"][0]["prompt"] = source["secret"]
        before = deepcopy(source)
        data = website.public_data(source, self.provenance())
        self.assertEqual(source, before)
        self.assertEqual(set(data), {
            "schemaVersion", "title", "version", "evidenceDate", "scope", "provenance",
            "models", "tasks", "sizes", "metrics",
        })
        self.assertEqual(set(data["provenance"]), {
            "summarySha256", "sealSha256", "completionSha256", "verifiedInputHashes", "modelCalls",
        })
        for model in data["models"]:
            self.assertEqual(set(model), {"id", "name", "family", "group", "parameters", "protocol", "note"})
        for task in data["tasks"]:
            self.assertEqual(set(task), {"id", "name", "kind", "description", "unit", "direction", "scheduled"})
        self.assertEqual(len(data["metrics"]), 600)
        index = website.build.validated_metrics(source)
        mapped = {}
        for row in data["metrics"]:
            self.assertEqual(set(row), set(website.METRIC_FIELDS))
            key = row["model"], row["task"], row["nodeCount"]
            self.assertNotIn(key, mapped)
            mapped[key] = row
            original = index[key]
            exact = key[1] in website.build.EXACT
            rate = original["hits"] / original["scheduled"]
            self.assertEqual(row, dict(
                model=key[0], task=key[1], nodeCount=key[2],
                scheduled=original["scheduled"], feasible=original["feasible"], hits=original["hits"],
                unsupported=original["initial_unsupported"], failures=original["execution_failures"],
                unattempted=original["unattempted"],
                accuracy=(original["macro_size_accuracy"] if key[2] == "all" else rate) if exact else None,
                optimalRate=None if exact else rate,
                meanGap=None if exact else original["mean_absolute_gap_feasible"],
                meanObjective=None if exact else original["mean_objective_feasible"],
            ))
        self.assertEqual(set(mapped), set(index))
        encoded = website.encoded_files(data)
        for content in encoded.values():
            for private in (b"credential", b"/private/", b"source_map", b"do not publish"):
                self.assertNotIn(private, content)
        csv_rows = list(csv.DictReader(io.StringIO(encoded["benchmark.csv"].decode())))
        self.assertEqual(len(csv_rows), 600)
        for row, csv_row in zip(data["metrics"], csv_rows):
            self.assertEqual(csv_row, {k: "" if v is None else str(v) for k, v in row.items()})

    def test_macro_accuracy_coverage_and_nulls(self):
        source = self.fixture()
        for row in source["metrics"]:
            profile, task, size = row["profile"], row["task"], row["node_count"]
            if profile == "jev_action" and task == "degree_exact":
                if size == 8:
                    row.update(hits=64, all_scheduled_hit_rate=1)
                elif size == "all":
                    row.update(hits=232, all_scheduled_hit_rate=.58, macro_size_accuracy=.6)
            if (profile, task) in (("laya", "lt_influence_construct"),
                                    ("gpt54_default_reasoning", "maxcut_construct")):
                missing = 9 if size == "all" else 8 if size == 8 else 1 if size == 9 else 0
                field = "initial_unsupported" if profile == "laya" else "execution_failures"
                row[field] = missing
                row["feasible"] -= missing
                row["conditional_objective_denominator"] -= missing
                if row["feasible"] == 0:
                    row.update(mean_objective_feasible=None, mean_absolute_gap_feasible=None)
        data = website.public_data(source, self.provenance())
        index = {(r["model"], r["task"], r["nodeCount"]): r for r in data["metrics"]}
        self.assertEqual(index["jev_action", "degree_exact", "all"]["accuracy"], .6)
        self.assertEqual(index["jev_action", "degree_exact", 8]["accuracy"], 1)
        self.assertEqual(index["laya", "lt_influence_construct", "all"]["unsupported"], 9)
        self.assertEqual(index["gpt54_default_reasoning", "maxcut_construct", "all"]["failures"], 9)
        self.assertIsNone(index["gpt54_default_reasoning", "maxcut_construct", 8]["meanGap"])
        self.assertIsNone(index["laya", "lt_influence_construct", 8]["meanObjective"])

    def test_invalid_metrics_rejected_before_publication(self):
        source = self.fixture()
        source["metrics"].pop()
        sealed = self.seal(source)
        target = self.root / "public"
        with patch.object(website.build, "REPO", self.root), self.assertRaises(Refusal):
            website.export(sealed, target)
        self.assertFalse(target.exists())

    def test_invalid_numbers_and_provenance_rejected(self):
        for number in (float("nan"), float("inf"), float("-inf")):
            source = self.fixture()
            next(r for r in source["metrics"] if r["task"] == "maxcut_construct")[
                "mean_objective_feasible"] = number
            with self.assertRaises(Refusal):
                website.public_data(source, self.provenance())
        proof = self.provenance()
        proof["source_summary_sha256"] = "/private/not-a-digest"
        with self.assertRaises(Refusal):
            website.public_data(self.fixture(), proof)
        with self.assertRaises(ValueError):
            website.encoded_files({"metrics": [], "unexpected": float("nan")})

    def test_verified_export_is_deterministic_and_public_only(self):
        source = self.seal()
        outputs = [self.root / name for name in ("first", "second")]
        with patch.object(website.build, "REPO", self.root):
            proofs = [website.export(source, output) for output in outputs]
        self.assertEqual(proofs[0], proofs[1])
        self.assertEqual(proofs[0]["verifiedInputHashes"], 1)
        self.assertEqual(proofs[0]["modelCalls"], 0)
        for output in outputs:
            self.assertEqual({p.name for p in output.iterdir()}, {"benchmark.json", "benchmark.csv"})
        for name in ("benchmark.json", "benchmark.csv"):
            self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())
        summary = self.fixture()
        reversed_summary = deepcopy(summary)
        reversed_summary["metrics"].reverse()
        self.assertEqual(website.encoded_files(website.public_data(summary, self.provenance())),
                         website.encoded_files(website.public_data(reversed_summary, self.provenance())))

    def test_existing_files_directories_and_symlinks_are_never_overwritten(self):
        targets = [self.root / name for name in ("file", "directory", "link", "broken-link")]
        targets[0].write_text("keep")
        targets[1].mkdir()
        targets[2].symlink_to(targets[0])
        targets[3].symlink_to(self.root / "missing")
        for target in targets:
            with self.subTest(target=target.name), patch.object(website.build, "export") as verify:
                with self.assertRaises(Refusal):
                    website.export(self.root / "absent-source", target)
                verify.assert_not_called()
        self.assertEqual(targets[0].read_text(), "keep")
        self.assertTrue(targets[3].is_symlink())

    def test_original_evidence_drift_rejected_without_output(self):
        source = self.seal()
        (self.root / "original.json").write_text("modified")
        with patch.object(website.build, "REPO", self.root), self.assertRaises(Refusal):
            website.export(source, self.root / "public")
        self.assertFalse((self.root / "public").exists())

    def test_seal_drift_rejected_without_output(self):
        source = self.seal()
        with (source / "verified-summary.json").open("a") as stream:
            stream.write("\n")
        with patch.object(website.build, "REPO", self.root), self.assertRaises(Refusal):
            website.export(source, self.root / "public")
        self.assertFalse((self.root / "public").exists())

    def test_output_inside_source_rejected(self):
        source = self.seal()
        with patch.object(website.build, "export") as verify, self.assertRaises(Refusal):
            website.export(source, source / "public")
        verify.assert_not_called()

    def test_failed_publication_cleans_partial_output(self):
        source = self.seal()
        target = self.root / "public"
        original_open = type(target).open

        def fail_csv(path, *args, **kwargs):
            if path == target / "benchmark.csv":
                raise OSError("simulated publication failure")
            return original_open(path, *args, **kwargs)

        with patch.object(website.build, "REPO", self.root), \
                patch.object(type(target), "open", fail_csv), self.assertRaises(OSError):
            website.export(source, target)
        self.assertFalse(target.exists())

    def test_original_cells_preserved_in_current_public_artifacts(self):
        directory = website.REPO / "web" / "data"
        data = json.loads((directory / "benchmark.json").read_text())
        self.assertEqual(data["schemaVersion"], 1)
        self.assertEqual(data["version"], "canonical-20260927-v14")
        self.assertEqual(data["evidenceDate"], "2026-09-27")
        self.assertEqual(data["provenance"]["verifiedInputHashes"], 1550)
        self.assertEqual(data["provenance"]["modelCalls"], 0)
        expected = {(p, task, size) for p, _ in website.build.PROFILES
                    for task in website.build.COUNTS for size in ("all", *website.SIZES)}
        self.assertEqual(len(data["metrics"]), 720)
        original_ids = {p for p, _ in website.build.PROFILES}
        original_rows = [r for r in data["metrics"] if r["model"] in original_ids]
        self.assertEqual(len(original_rows), 600)
        self.assertEqual({(r["model"], r["task"], r["nodeCount"]) for r in original_rows}, expected)
        aggregate = [r for r in data["metrics"] if r["nodeCount"] == "all"]
        for model, _ in website.build.PROFILES:
            rows = [r for r in aggregate if r["model"] == model]
            self.assertEqual(sum(r["scheduled"] for r in rows), 960)
            self.assertEqual(sum(r["unsupported"] for r in rows), 9 if model == "laya" else 0)
            self.assertEqual(sum(r["failures"] for r in rows),
                             9 if model == "gpt54_default_reasoning" else 0)
            self.assertEqual(sum(r["unattempted"] for r in rows), 0)
        for name, content in website.encoded_files(data).items():
            self.assertEqual((directory / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
