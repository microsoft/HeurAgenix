import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from scripts import finalize_expanded_pilots as finalizer


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FinalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior = finalizer.prior
        evidence = prior.Evidence()
        cls.panel, cls.instances, _, _ = prior.load_panel(finalizer.PANEL, evidence)
        cls.manifest, cls.rows, cls.attempts, _ = prior.load_candidate(
            finalizer.EXECUTION / "qwen27_grammar", cls.panel, finalizer.PANEL,
            cls.instances, evidence)

    def setUp(self):
        self.parent = finalizer.REPO / "output/expanded-pilot-finalization-20260924-v1"
        self.created_parent = not self.parent.exists()
        self.created_fixture_parent = not (self.parent / "test-fixtures").exists()
        self.workspace = self.parent / "test-fixtures" / uuid.uuid4().hex
        self.workspace.mkdir(parents=True)
        self.addCleanup(self.clean)

    def clean(self):
        shutil.rmtree(self.workspace)
        for path, owned in ((self.workspace.parent, self.created_fixture_parent),
                            (self.parent, self.created_parent)):
            if not owned:
                continue
            try:
                path.rmdir()
            except OSError:
                pass

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def write_rows(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare_fixture(self):
        root = self.workspace / "prepared"
        execution, trained = self.workspace / "execution", self.workspace / "trained"
        source = self.workspace / "source.py"
        source.write_text("offline_source = True\n")
        input_file = self.workspace / "identity.json"
        self.write(input_file, {"fixed": True})
        lanes = {name: str(execution / name) for name in finalizer.PROFILES[:2]}
        lanes.update({name: str(trained / f"run-{name}") for name in finalizer.PROFILES[2:]})
        for name, value in (("ROOT", root), ("EXECUTION", execution), ("TRAINED", trained)):
            patcher = patch.object(finalizer, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        with patch.object(finalizer, "fixed_plan", return_value=(
                lanes, {finalizer.expanded.relative(source): finalizer.suite.digest(source)},
                {finalizer.expanded.relative(input_file): finalizer.suite.digest(input_file)})), \
                patch.object(finalizer, "MAX_WAIT", 1):
            registration = finalizer.prepare(root)
        return root, registration, source

    def candidate_fixture(self, attempts=None, state="completed"):
        profile = "qwen27_thinking"
        run = self.workspace / profile
        attempts = copy.deepcopy(self.attempts if attempts is None else attempts)
        rows = finalizer.canonical.replay(
            self.instances, attempts, SimpleNamespace(model=self.manifest["model"]["model"]),
            native_action=False)
        observed = {"status": state, "profile": profile, "calls": len(attempts)}
        self.write(run / "status.json", observed)
        self.write_rows(run / "attempts.jsonl", attempts)
        self.write_rows(run / "scores.jsonl", rows)
        self.write(run / "summary.json", finalizer.suite.summarize(rows))
        self.write(run / "manifest.json", {"fixture": True})
        self.write(run.parent / "execution.json", {"fixture": True})
        self.write(run.parent / "execution.json.sha256",
                   finalizer.suite.digest(run.parent / "execution.json"))
        registration = {"panel": str(finalizer.PANEL), "source_sha256": {},
                        "historical": str(finalizer.prior.DEFAULT_HISTORY)}
        return profile, run, observed, registration

    def verify_fixture(self, fixture, reference_hook=None):
        reference = reference_hook or (
            lambda path, *args: ({"profile": path.name.split(".", 1)[1]}, [], []))
        with patch.object(finalizer, "verify_frozen"), \
                patch.object(finalizer.prior, "load_panel", return_value=(
                    self.panel, self.instances, [], {"fixture": True})), \
                patch.object(finalizer.expanded, "qwen_identity", return_value=(
                    self.manifest["model"]["model"], {})), \
                patch.object(finalizer.prior, "load_sealed_reference", side_effect=reference):
            return finalizer.verify_terminal(*fixture, self.workspace)

    def test_prepare_is_exclusive_and_source_snapshots_verified(self):
        root, registration, _ = self.prepare_fixture()
        finalizer.verify_frozen(root, registration)
        self.assertEqual(finalizer.suite.read_json(root / "status.json")["status"], "prepared_not_started")
        self.assertFalse((root / "finalizer-start.json").exists())
        with self.assertRaises(finalizer.suite.shared.Refusal):
            finalizer.prepare(root)

    def test_changed_frozen_source_refuses_before_wait_or_correctness(self):
        root, _, source = self.prepare_fixture()
        source.write_text("changed = True\n")
        with patch.object(finalizer, "observe") as observe:
            self.assertEqual(finalizer.run(root), 2)
            observe.assert_not_called()
        self.assertEqual(finalizer.suite.read_json(root / "status.json")["status"], "failed")

    def test_pending_guard_is_before_every_correctness_read(self):
        with patch.object(finalizer, "required_hashes") as hashes, \
                patch.object(finalizer.suite, "read_rows") as rows:
            with self.assertRaises(finalizer.suite.shared.Refusal):
                finalizer.verify_terminal("qwen27_thinking", self.workspace, {"status": "running"},
                                          {}, self.workspace)
            hashes.assert_not_called()
            rows.assert_not_called()

    def test_deadline_has_pending_diagnostics_without_reading_prefixes(self):
        root, registration, _ = self.prepare_fixture()
        for name, location in registration["lanes"].items():
            self.write(Path(location) / "status.json", {"status": "running", "calls": 17})
            (Path(location) / "scores.jsonl").write_text("DO NOT READ\n")
        clock = Clock()
        finalize = Mock()
        with patch.object(finalizer.suite, "read_rows") as rows:
            code = finalizer.run(root, clock=clock, sleep=clock.sleep, finalize=finalize)
            rows.assert_not_called()
        self.assertEqual(code, 2)
        finalize.assert_not_called()
        state = finalizer.suite.read_json(root / "status.json")
        self.assertEqual(state["status"], "deadline_exceeded")
        self.assertEqual(len(state["waiting"]), 4)
        self.assertTrue(all(v["calls"] == 17 for v in state["waiting"].values()))
        self.assertEqual(clock.value, 1)

    def test_missing_lanes_wait_without_creating_them(self):
        root, registration, _ = self.prepare_fixture()
        clock = Clock()
        self.assertEqual(finalizer.run(root, clock=clock, sleep=clock.sleep), 2)
        self.assertTrue(all(not Path(path).exists() for path in registration["lanes"].values()))

    def test_duplicate_launch_refused_without_overwriting_start_identity(self):
        root, _, _ = self.prepare_fixture()
        clock = Clock()
        finalizer.run(root, clock=clock, sleep=clock.sleep)
        before = (root / "finalizer-start.json").read_bytes()
        with self.assertRaises(FileExistsError):
            finalizer.run(root, clock=clock, sleep=clock.sleep)
        self.assertEqual((root / "finalizer-start.json").read_bytes(), before)
        identity = json.loads(before)
        self.assertIn("start_ticks", identity)
        self.assertIn("boot_id", identity)

    def test_abort_and_verification_failure_never_overall_success(self):
        root, registration, _ = self.prepare_fixture()
        for profile, location in registration["lanes"].items():
            self.write(Path(location) / "status.json", {
                "status": "aborted" if profile == "decider" else "completed", "calls": 0})

        def finalize(profile, path, observed, *args):
            if profile == "kev":
                raise ValueError("Cannot independently attest")
            return {"status": "verified", "source_status": observed["status"]}

        self.assertEqual(finalizer.run(root, finalize=finalize), 1)
        state = finalizer.suite.read_json(root / "status.json")
        self.assertEqual(state["status"], "completed_with_failures")
        self.assertEqual(state["outcomes"]["decider"]["source_status"], "aborted")
        self.assertEqual(state["outcomes"]["kev"]["status"], "verification_failed")

    def test_all_four_verified_completed_is_only_zero_exit(self):
        root, registration, _ = self.prepare_fixture()
        for location in registration["lanes"].values():
            self.write(Path(location) / "status.json", {"status": "completed", "calls": 160})
        self.assertEqual(finalizer.run(root, finalize=lambda *args: {
            "status": "verified", "source_status": "completed"}), 0)
        self.assertEqual(finalizer.suite.read_json(root / "status.json")["status"], "completed")

    def test_truthful_proof_requires_full_replay_and_independent_objectives(self):
        fixture = self.candidate_fixture()
        proof = self.verify_fixture(fixture)
        self.assertTrue(proof["independent_replay_verified"])
        self.assertEqual(proof["episodes"], 64)
        self.assertEqual(proof["calls"], 160)
        self.assertEqual(len(proof["construction_audit"]), 16)
        self.assertEqual(proof["input_sha256"], finalizer.required_hashes(fixture[0], fixture[1]))
        self.assertEqual(proof["counts"]["feasible"], 64)

    def test_changed_bytes_during_verification_refuse_even_semantically_identical(self):
        fixture = self.candidate_fixture()

        def mutate(path, *args):
            with (fixture[1] / "scores.jsonl").open("a") as stream:
                stream.write("\n")
            return {"profile": path.name.split(".", 1)[1]}, [], []

        with self.assertRaises(finalizer.suite.shared.Refusal):
            self.verify_fixture(fixture, reference_hook=mutate)

    def test_bad_score_or_summary_cannot_receive_verified_flag(self):
        fixture = self.candidate_fixture()
        self.write(fixture[1] / "summary.json", {"not": "canonical"})
        with self.assertRaises(finalizer.suite.shared.Refusal):
            self.verify_fixture(fixture)
        fixture = self.candidate_fixture()
        rows = copy.deepcopy(self.rows)
        rows[0]["correct"] = not rows[0]["correct"]
        self.write_rows(fixture[1] / "scores.jsonl", rows)
        with self.assertRaises(finalizer.suite.shared.Refusal):
            self.verify_fixture(fixture)

    def test_zero_attempt_abort_and_allfailed_counts_are_attested_honestly(self):
        proof = self.verify_fixture(self.candidate_fixture([], "aborted"))
        self.assertEqual(proof["source_status"], "aborted")
        self.assertEqual(proof["counts"], {"attempted": 0, "unattempted": 64, "failed": 0, "feasible": 0})
        attempts = [{
            "instance_id": i["id"], "step": 0, "status": "failure", "selected_option_id": None,
            "fatal": False, "diagnostic_code": "timeout", "resolved_model": None,
            "request": finalizer.suite.json_value(asdict(finalizer.canonical.public_request(i, []))),
        } for i in self.instances]
        proof = self.verify_fixture(self.candidate_fixture(attempts, "completed_with_failures"))
        self.assertEqual(proof["calls"], 64)
        self.assertEqual(proof["counts"], {"attempted": 64, "unattempted": 0, "failed": 64, "feasible": 0})
        self.assertEqual(proof["construction_audit"], [])

    def test_inconsistent_actual_count_refused(self):
        profile, run, observed, registration = self.candidate_fixture()
        observed["calls"] = 159
        self.write(run / "status.json", observed)
        with self.assertRaises(finalizer.suite.shared.Refusal):
            self.verify_fixture((profile, run, observed, registration))

    def test_no_proof_or_export_on_verification_failure(self):
        with patch.object(finalizer, "verify_terminal", side_effect=ValueError("replay differs")), \
                patch.object(finalizer.expanded, "export") as export:
            with self.assertRaises(ValueError):
                finalizer.finalize_lane("kev", self.workspace, {}, {}, self.workspace)
            export.assert_not_called()
        self.assertFalse((self.workspace / "proofs").exists())

    def test_exclusive_proof_written_before_export_and_never_reused(self):
        source = self.workspace / "sealed-input.json"
        self.write(source, {"terminal": True})
        proof = {"input_sha256": {finalizer.expanded.relative(source): finalizer.suite.digest(source)},
                 "independent_replay_verified": True}

        def exporter(profile, run, proof_path, destination, *args):
            self.assertEqual(finalizer.suite.read_json(proof_path), proof)
            destination.mkdir()
            self.write(destination / "verified-summary.json", {"source_status": "completed"})
            self.write(destination / "export-seal.json", {"artifact_sha256": {
                "verified-summary.json": finalizer.suite.digest(destination / "verified-summary.json")}})
            return {"source_status": "completed"}

        with patch.object(finalizer, "verify_terminal", return_value=proof), \
                patch.object(finalizer, "verify_frozen"), \
                patch.object(finalizer.expanded, "export", side_effect=exporter) as exporting:
            result = finalizer.finalize_lane(
                "kev", self.workspace, {"status": "completed"},
                {"panel": "fixture", "historical": "fixture"}, self.workspace)
            self.assertEqual(result["status"], "verified")
            with self.assertRaises(FileExistsError):
                finalizer.finalize_lane(
                    "kev", self.workspace, {"status": "completed"},
                    {"panel": "fixture", "historical": "fixture"}, self.workspace)
            self.assertEqual(exporting.call_count, 1)

    def test_illegal_own_history_cannot_receive_proof(self):
        fixture = self.candidate_fixture()
        attempts = copy.deepcopy(self.attempts)
        attempts[0]["selected_option_id"] = "not-an-option"
        self.write_rows(fixture[1] / "attempts.jsonl", attempts)
        with self.assertRaises(finalizer.suite.shared.Refusal):
            self.verify_fixture(fixture)

    def checkpoint_fixture(self):
        root = self.workspace / "checkpoint-fixture"
        artifacts = {"source_revisions": finalizer.expanded.SOURCES, "models": {}}
        for name, (repository, revision) in finalizer.expanded.PINS.items():
            directory = root / "models" / name
            self.write(directory / "weights.safetensors", {"fixture": name})
            if name == "decider":
                self.write(directory / "decider_config.json", {
                    "temperature_by_type": {"choice": 1.164}, "schema_first": False})
            if name == "kev":
                self.write(directory / "head.pt", {"fixture": "safe_loader_is_mocked"})
            artifacts["models"][name] = {
                "repository": repository, "revision": revision, "path": str(directory),
                "sha256": {p.name: finalizer.suite.digest(p) for p in directory.iterdir()}}
        self.write(root / "artifacts.json", artifacts)
        source = "scripts/trained_jevlike_adapter.py"
        self.write(root / "benchmark-registration-parent-locks.json", {
            "models": artifacts, "code_sha256": {source: finalizer.suite.digest(finalizer.REPO / source)},
            "packages": {"fixture": "no_model_loaded"}, "overlay": {}, "dependency_deviations": {}})
        for profile, temperature in finalizer.TEMPERATURES.items():
            self.write(root / f"cpu-smoke-{profile}/runtime.json", {
                "status": "ready", "runtime": {"candidate": profile, "device": "cpu",
                                              "cuda_initialized": False, "temperature": temperature}})
        head = {"base": finalizer.expanded.PINS["kev-base"][0],
                "base_revision": finalizer.expanded.PINS["kev-base"][1],
                "head": {"fixture_weight": [1]}, "temperature": finalizer.TEMPERATURES["kev"]}
        return root, head

    def test_checkpoint_inventory_and_independent_temperatures(self):
        root, head = self.checkpoint_fixture()
        for profile in finalizer.TEMPERATURES:
            proof = finalizer.checkpoint_evidence(
                profile, root, finalizer.prior.Evidence(), head_loader=lambda _: head)
            self.assertEqual(proof["effective_temperature"], finalizer.TEMPERATURES[profile])
            self.assertTrue(proof["checkpoint_inventory_independently_verified"])
            self.assertTrue(proof["temperature_independently_verified"])
            self.assertEqual(len(proof["checkpoint_payload_sha256"]), 5)

    def test_payload_checksum_mismatch_precedes_safe_head_load(self):
        root, _ = self.checkpoint_fixture()
        (root / "models/kev/weights.safetensors").write_bytes(b"changed")
        loader = Mock()
        with self.assertRaises(finalizer.suite.shared.Refusal):
            finalizer.checkpoint_evidence("kev", root, finalizer.prior.Evidence(), head_loader=loader)
        loader.assert_not_called()

    def test_head_temperature_and_base_mismatch_refuse(self):
        root, head = self.checkpoint_fixture()
        for key, value in (("temperature", 1.0), ("base_revision", "wrong")):
            bad = {**head, key: value}
            with self.subTest(key=key), self.assertRaises(finalizer.suite.shared.Refusal):
                finalizer.checkpoint_evidence(
                    "kev", root, finalizer.prior.Evidence(), head_loader=lambda _: bad)

    def test_preparation_temperature_disagreement_refuses(self):
        root, head = self.checkpoint_fixture()
        path = root / "cpu-smoke-decider/runtime.json"
        smoke = finalizer.suite.read_json(path)
        smoke["runtime"]["temperature"] = 1.145
        self.write(path, smoke)
        with self.assertRaises(finalizer.suite.shared.Refusal):
            finalizer.checkpoint_evidence(
                "decider", root, finalizer.prior.Evidence(), head_loader=lambda _: head)

    def test_decider_wrong_choice_temperature_refuses_even_with_matching_inventory(self):
        root, _ = self.checkpoint_fixture()
        path = root / "models/decider/decider_config.json"
        self.write(path, {"schema_first": False, "temperature_by_type": {"choice": 1.145}})
        artifacts = finalizer.suite.read_json(root / "artifacts.json")
        artifacts["models"]["decider"]["sha256"]["decider_config.json"] = finalizer.suite.digest(path)
        self.write(root / "artifacts.json", artifacts)
        active_path = root / "benchmark-registration-parent-locks.json"
        active = finalizer.suite.read_json(active_path)
        active["models"] = artifacts
        self.write(active_path, active)
        with self.assertRaises(finalizer.suite.shared.Refusal):
            finalizer.checkpoint_evidence("decider", root, finalizer.prior.Evidence())

    @unittest.skipUnless(sys.version_info[:2] == (3, 12), "Native Torch metadata check uses Python 3.12")
    def test_real_pinned_kev_head_cpu_metadata_without_model_instantiation(self):
        root = finalizer.REPO / "output/trained-jevlike-pilot-20260924-v1"
        artifacts = finalizer.suite.read_json(root / "artifacts.json")
        path = root / "models/kev/head.pt"
        finalizer.check_digest(path, artifacts["models"]["kev"]["sha256"]["head.pt"])
        head = finalizer.safe_head(path)
        self.assertEqual(head["temperature"], finalizer.TEMPERATURES["kev"])
        self.assertEqual(head["base_revision"], finalizer.expanded.PINS["kev-base"][1])
        import torch
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
