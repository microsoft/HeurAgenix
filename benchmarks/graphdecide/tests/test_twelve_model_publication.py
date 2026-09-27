"""Offline publication tests; fixtures and scratch files stay inside the project."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

from scripts import export_twelve_model_publication as publication
from scripts.paired_graph_ablation import Refusal
import test_paper_full_matrix


class TwelveModelPublicationTests(unittest.TestCase):
    def setUp(self):
        self.created_output = not publication.OUTPUT.exists()
        self.root = publication.OUTPUT / (".test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        shutil.rmtree(self.root)
        if self.created_output:
            publication.OUTPUT.rmdir()

    def fixture(self):
        summary = test_paper_full_matrix.FullMatrixPaperTests().fixture()
        original = publication.website.public_data(summary, {
            "source_summary_sha256": "a" * 64, "source_seal_sha256": "b" * 64,
            "source_completion_sha256": "c" * 64, "verified_input_hashes": 1550,
            "new_model_calls": 0,
        })
        episodes = []
        for task in publication.build.COUNTS:
            exact = task in publication.build.EXACT
            for size in publication.website.SIZES:
                count = 8 * size if task == "degree_exact" else 16 if exact else 8
                for i in range(count):
                    row = dict(instance_id=f"{task}-{size}-{i}", task=task, node_count=size,
                               feasible=True, failure_count=0, query_count=1,
                               private="/private/SECRET_PROMPT_RESPONSE")
                    if exact:
                        row["correct"] = size == 8
                    else:
                        row.update(optimal=i == 0, absolute_gap=float(i), objective=float(20 - i))
                    episodes.append(row)
        proof = {key: "d" * 64 for key in (
            "registrationSha256", "runSealSha256", "independentSealSha256",
            "independentReplaySha256", "independentReplayRecomputedSha256",
            "verificationSha256", "artifactManifestSha256", "bundleManifestSha256",
            "initialRequestsSha256")}
        proof.update(verifiedSourceHashes=37, recordedModelCalls=1920, modelCalls=0,
                     raw="/private/SECRET_PROMPT_RESPONSE")
        return original, {lane: deepcopy(episodes) for lane in publication.LANES}, {
            lane: deepcopy(proof) for lane in publication.LANES}

    def test_inventory_original_values_and_no_global_mutation(self):
        original, episodes, sources = self.fixture()
        before, profiles = deepcopy(original), publication.build.PROFILES
        data = publication.public_data(original, episodes, sources)
        self.assertEqual(original, before)
        self.assertIs(publication.build.PROFILES, profiles)
        self.assertEqual(data["metrics"][:600], original["metrics"])
        self.assertEqual(data["models"][:10], original["models"])
        self.assertEqual(len(data["metrics"]), 720)
        self.assertEqual(len(data["models"]), 12)
        self.assertEqual(data["schemaVersion"], 1)
        self.assertEqual(data["version"], "canonical-20260927-v14")
        self.assertEqual(data["evidenceDate"], "2026-09-27")
        self.assertEqual(data["provenance"]["modelCalls"], 0)
        self.assertEqual(data["provenance"]["recordedExtensionModelCalls"], 3840)
        for key, value in original["provenance"].items():
            self.assertEqual(data["provenance"][key], value)
        for missing in ("metric", "episode"):
            with self.subTest(missing=missing):
                broken = deepcopy(data)
                if missing == "metric":
                    broken["metrics"].pop()
                    with self.assertRaises(Refusal):
                        publication.validate_public(broken)
                else:
                    with self.assertRaises(Refusal):
                        publication.aggregate_lane(publication.LANES[0], episodes[publication.LANES[0]][:-1])

    def test_macro_not_pooled_degree(self):
        data = publication.public_data(*self.fixture())
        row = next(r for r in data["metrics"] if (r["model"], r["task"], r["nodeCount"]) ==
                   (publication.LANES[0], "degree_exact", "all"))
        self.assertEqual(row["accuracy"], .2)
        self.assertEqual(row["hits"] / row["scheduled"], .16)
        row["accuracy"] = .16
        with self.assertRaises(Refusal):
            publication.validate_public(data)

    def test_feasible_only_gap_null_coverage_and_csv(self):
        original, episodes, sources = self.fixture()
        lane = publication.LANES[0]
        for row in episodes[lane]:
            if row["task"] == "maxcut_construct" and row["node_count"] == 8:
                row.update(feasible=False, failure_count=1, absolute_gap=999999., objective=999999.)
        data = publication.public_data(original, episodes, sources)
        index = publication.validate_public(data)
        all_row = index[lane, "maxcut_construct", "all"]
        self.assertEqual(all_row["meanGap"], 3.5)
        self.assertEqual(all_row["feasible"], 32)
        self.assertEqual(all_row["failures"], 8)
        self.assertEqual(all_row["optimalRate"], 4 / 40)
        self.assertIsNone(index[lane, "maxcut_construct", 8]["meanGap"])
        self.assertIsNone(index[lane, "maxcut_construct", 8]["meanObjective"])
        files = publication.website.encoded_files(data)
        csv_rows = list(csv.DictReader(io.StringIO(files["benchmark.csv"].decode())))
        self.assertEqual(len(csv_rows), 720)
        for row, encoded in zip(data["metrics"], csv_rows, strict=True):
            self.assertEqual(encoded, {k: "" if v is None else str(v) for k, v in row.items()})
        tables = publication.render(data)
        self.assertEqual(set(tables), {"exact.tex", "optimal.tex", "gaps.tex", "coverage.tex"})
        self.assertIn("Qwen3.8-27B", tables["exact.tex"])
        self.assertIn("Qwen2.5-72B", tables["gaps.tex"])

    def test_allowlist_excludes_private_paths_and_raw_evidence(self):
        data = publication.public_data(*self.fixture())
        encoded = publication.website.encoded_files(data)
        for content in encoded.values():
            for forbidden in (b"/private/", b"SECRET_PROMPT_RESPONSE", b"source_sha256",
                              b"base_url", b"selected_option_id", b"instance_id"):
                self.assertNotIn(forbidden, content)
        for source in data["provenance"]["strongerSources"].values():
            self.assertNotIn("raw", source)
        original, episodes, sources = self.fixture()
        sources[publication.LANES[0]]["runSealSha256"] = "/private/leak"
        with self.assertRaises(Refusal):
            publication.public_data(original, episodes, sources)

    def test_all_direct_models_precede_gpt_only_lower_block_in_every_table(self):
        tables = publication.render(publication.public_data(*self.fixture()))
        for name, text in tables.items():
            with self.subTest(table=name):
                upper, lower = text.split("\\midrule\nGPT-5.4 &", maxsplit=1)
                direct_labels = [label for profile, label in publication.build.PROFILES
                                 if not profile.startswith("gpt")]
                direct_labels.extend(publication.DETAILS[lane][0] for lane in publication.LANES)
                self.assertEqual(len(direct_labels), 10)
                for label in direct_labels:
                    self.assertIn(f"\n{label} &", upper)
                    self.assertNotIn(f"\n{label} &", lower)
                self.assertIn("\nAstra &", lower)
                self.assertEqual(lower.count(" \\\\"), 2)
                self.assertLess(upper.index("\nQwen27 G &"), upper.index("\nQwen3.8-27B &"))
                self.assertLess(upper.index("\nQwen3.8-27B &"), upper.index("\nQwen2.5-72B &"))

    def test_source_tampering_missing_seals_and_path_escape_rejected(self):
        path = self.root / "source.json"
        path.write_text('{"sealed":true}\n')
        hashes = {"source.json": publication.suite.digest(path)}
        publication.checked_map(self.root, hashes, ("source.json",))
        path.write_text('{"sealed":false}\n')
        with self.assertRaises(Refusal):
            publication.checked_map(self.root, hashes)
        with self.assertRaises(Refusal):
            publication.checked_map(self.root, {}, ("report.json",))
        for name in ("../source.json", str(path.resolve())):
            with self.subTest(name=name), self.assertRaises(Refusal):
                publication.checked_map(self.root, {name: "a" * 64})
        link = self.root / "link.json"
        link.symlink_to(path)
        with self.assertRaises(Refusal):
            publication.checked_map(self.root, {"link.json": publication.suite.digest(path)})

    def test_lane_registration_tamper_stops_before_replay(self):
        root = self.root / "lane"
        root.mkdir()
        source = publication.STRONGER / publication.LANES[0]
        shutil.copyfile(source / "registration.json", root / "registration.json")
        shutil.copyfile(source / "registration.sha256", root / "registration.sha256")
        with (root / "registration.json").open("a") as stream:
            stream.write(" ")
        with patch.object(publication.subprocess, "run") as replay, self.assertRaises(Refusal):
            publication.verify_lane(root, publication.LANES[0], self.root)
        replay.assert_not_called()

    def test_replay_float_roundoff_is_bounded_not_success_fallback(self):
        publication.compare_replay({"gap": .3012386535704068, "hits": 2},
                                   {"gap": .30123865357040686, "hits": 2})
        for actual, recorded in ((.3, .30001), (float("nan"), float("nan")),
                                 (2, 2.), (True, 1), ({"calls": 0}, {"calls": 1})):
            with self.subTest(actual=actual), self.assertRaises(Refusal):
                publication.compare_replay(actual, recorded)

    def test_actual_recorded_degree_macros_and_construction_coverage(self):
        for lane, expected in zip(publication.LANES, (45.501893939393934, 41.238005050505055)):
            root = publication.STRONGER / lane
            rows = publication.suite.read_rows(root / "run/scores.jsonl")
            metrics = publication.aggregate_lane(lane, rows)
            degree = next(r for r in metrics if r["task"] == "degree_exact" and r["nodeCount"] == "all")
            self.assertAlmostEqual(100 * degree["accuracy"], expected, places=12)
            self.assertNotEqual(degree["accuracy"], degree["hits"] / degree["scheduled"])
            self.assertEqual(sum(r["feasible"] for r in metrics if r["nodeCount"] == "all"
                                 and r["task"] in publication.build.OPTIMIZATION), 160)
            report = publication.suite.read_json(root / "independent-v3/report.json")
            for row in metrics:
                if row["nodeCount"] == "all" and row["task"] in publication.build.OPTIMIZATION:
                    self.assertEqual(row["meanGap"],
                                     report["by_task"][row["task"]]["mean_absolute_gap_feasible"])

    def test_completed_publication_readback(self):
        seal = publication.OUTPUT / "export-seal.json"
        if not seal.exists():
            self.skipTest("Run exporter to enable persistent-artifact read-back.")
        publication.checked_map(publication.OUTPUT,
                                publication.suite.read_json(seal)["artifactSha256"])
        data = publication.suite.read_json(publication.OUTPUT / "benchmark.json")
        publication.validate_public(data)
        original = publication.suite.read_json(publication.build.SOURCE / "verified-summary.json")
        metrics = publication.build.validated_metrics(original)
        for row in data["metrics"][:600]:
            private = metrics[row["model"], row["task"], row["nodeCount"]]
            self.assertEqual(row["hits"], private["hits"])
            self.assertEqual(row["feasible"], private["feasible"])
        for name in ("benchmark.json", "benchmark.csv"):
            self.assertEqual((publication.OUTPUT / name).read_bytes(),
                             (publication.REPO / "web/data" / name).read_bytes())
        for name in ("exact.tex", "gaps.tex", "optimal.tex", "coverage.tex", "provenance.json"):
            self.assertEqual((publication.OUTPUT / name).read_bytes(),
                             (publication.PAPER / name).read_bytes())

    def test_custom_output_cli_does_not_require_or_modify_paper_destination(self):
        paper_before = ({p.name: p.read_bytes() for p in publication.PAPER.iterdir()}
                        if publication.PAPER.exists() else None)
        destination = self.root / "publication"
        result = subprocess.run(
            [sys.executable, str(publication.REPO / "scripts/export_twelve_model_publication.py"),
             "--output", str(destination)], check=True, text=True, capture_output=True)
        status = json.loads(result.stdout)
        self.assertEqual(status["metrics"], 720)
        self.assertEqual(status["models"], 12)
        self.assertEqual(status["modelCalls"], 0)
        self.assertTrue((destination / "gaps.tex").is_file())
        publication.checked_map(destination,
                                publication.suite.read_json(destination / "export-seal.json")["artifactSha256"])
        paper_after = ({p.name: p.read_bytes() for p in publication.PAPER.iterdir()}
                       if publication.PAPER.exists() else None)
        self.assertEqual(paper_before, paper_after)


if __name__ == "__main__":
    unittest.main()
