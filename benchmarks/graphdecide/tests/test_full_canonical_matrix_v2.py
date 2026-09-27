import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import Mock, patch
import uuid

from scripts import export_full_canonical_matrix as original
from scripts import export_full_canonical_matrix_v2 as exporter
from scripts import finalize_full_canonical_matrix_v2 as watcher


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.root = exporter.REPO / "output" / ("full-matrix-v2-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def records(self):
        parent = {"pid": 123, "start_ticks": 456,
                  "argv": ["python", str(exporter.REPO / exporter.COORDINATOR), "run"]}
        enriched_parent = {**parent, "start_ticks": "456", "ppid": 12, "cwd": str(exporter.REPO)}
        registration = {"python": "/registered/python"}
        expected = [registration["python"], str(exporter.REPO / exporter.COORDINATOR),
                    "worker-phase", "--root", str(self.root), "--handoff",
                    str(self.root / "standalone-coordinator/worker-handoff.json")]
        approval = {"approved": True, "coordinator_registration_sha256": exporter.COORDINATOR_SHA,
                    "base_registration_sha256": exporter.TRAINED_SHA}
        locks = {key: {"owner_pid": 123, "device_inode": "fake-recorded-inode", "path": "not-opened"}
                 for key in ("qwen08", "qwen2")}
        handoff = {"parent_pid": 123, "parent_start_ticks": 456,
                   "coordinator_registration_sha256": exporter.COORDINATOR_SHA,
                   "base_registration_sha256": exporter.TRAINED_SHA,
                   "cuda_visible_devices": "GPU-fixed", "locks": locks}
        services = {name: {"pid": i, "start_ticks": str(i * 2), "argv": [name],
                           "cuda_environment": {}, "cwd": str(exporter.REPO)}
                    for name, i in (("qwen08", 789), ("qwen2", 987))}
        return registration, [
            {"pid": 123, "approval": approval},
            {"argv": expected, "parent": enriched_parent, "retry_forbidden": True,
             "cuda_visible_devices": "GPU-fixed"},
            {"pid": 124, "ppid": 123, "start_ticks": "457", "argv": expected,
             "cuda_environment": {"CUDA_VISIBLE_DEVICES": "GPU-fixed"}, "cwd": str(exporter.REPO)},
            handoff,
            {"pid": 124, "parent": parent, "handoff": copy.deepcopy(handoff),
             "active_entry": exporter.COORDINATOR},
            {"parent": enriched_parent, "leases": locks, "target_gpu_uuids": ["GPU-fixed", "GPU-other"],
             "models_verified_idle": True, "services": services},
            {"models_verified_idle": True, "protected_services_unchanged": True,
             "services": copy.deepcopy(services)},
            {"status": "restored", "original_services_restored": True},
        ]

    def test_adapter_does_not_mutate_frozen_imported_modules(self):
        self.assertIsNot(exporter.base, original)
        self.assertIsNot(exporter.base.source_data, original.source_data)
        self.assertEqual(original.Evidence.__module__, original.__name__)
        self.assertEqual(Path(original.__file__).name, "export_full_canonical_matrix.py")

    def test_actual_production_subclass_route_needs_no_injected_boolean(self):
        registration, records = self.records()
        exporter.validate_route(self.root, registration, records)

    def test_empty_initial_process_capture_is_explicitly_unobserved_not_invented(self):
        registration, records = self.records()
        records[2]["argv"], records[2]["cuda_environment"] = [], {}
        observations = exporter.validate_route(self.root, registration, records)
        self.assertFalse(observations["captured_supervisor_argv_observed"])
        self.assertFalse(observations["captured_supervisor_cuda_environment_observed"])
        self.assertTrue(observations["launch_intent_argv_verified"])
        records[4]["pid"] = 999
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.validate_route(self.root, registration, records)

    def test_parent_start_ticks_normalize_string_to_integer_only(self):
        registration, records = self.records()
        exporter.validate_route(self.root, registration, records)
        records[1]["parent"]["start_ticks"] = "999"
        with self.assertRaises(exporter.suite.shared.Refusal):
            exporter.validate_route(self.root, registration, records)

    def test_wrong_production_entry_worker_phase_uuid_or_child_refused(self):
        for index, key, value in ((4, "active_entry", "scripts/full_matrix_trained_parent.py"),
                                  (1, "argv", ["python", "mock"]),
                                  (1, "cuda_visible_devices", "GPU-different"),
                                  (2, "pid", 999), (2, "ppid", 999),
                                  (1, "retry_forbidden", False)):
            with self.subTest(index=index, key=key):
                registration, records = self.records()
                records[index][key] = value
                with self.assertRaises(exporter.suite.shared.Refusal):
                    exporter.validate_route(self.root, registration, records)

    def test_unapproved_registration_or_unrestored_services_refused(self):
        for change in ("approval", "base", "lease", "restoration", "service"):
            registration, records = self.records()
            if change == "approval":
                records[0]["approval"]["approved"] = False
            elif change == "base":
                records[0]["approval"]["base_registration_sha256"] = "wrong"
            elif change == "lease":
                records[3]["locks"]["qwen08"]["owner_pid"] = 999
            elif change == "restoration":
                records[6]["protected_services_unchanged"] = False
            else:
                records[6]["services"]["qwen08"]["argv"] = ["changed"]
            with self.subTest(change=change), self.assertRaises(exporter.suite.shared.Refusal):
                exporter.validate_route(self.root, registration, records)

    def test_coordinator_gate_reads_metadata_only_and_refuses_missing_proofs(self):
        directory = self.root / "standalone-coordinator"
        directory.mkdir()
        status = directory / "status.json"
        with patch.object(exporter.suite, "read_rows", side_effect=AssertionError):
            status.write_text(json.dumps({"status": "running"}))
            self.assertFalse(exporter.coordinator_state(self.root)["ready"])
            status.write_text(json.dumps({"status": "failed"}))
            self.assertTrue(exporter.coordinator_state(self.root)["blocked"])
            status.write_text(json.dumps({"status": "terminal_verified_services_restored",
                                          "original_services_restored": True}))
            self.assertFalse(exporter.coordinator_state(self.root)["ready"])
            (directory / "run-verification.json").write_text("not read")
            (directory / "resources").mkdir()
            (directory / "resources/restoration.json").write_text("not read")
            self.assertTrue(exporter.coordinator_state(self.root)["ready"])

    def test_qwen_compatibility_never_ignores_score_or_seal_failures(self):
        producer = Mock()
        for error in ("Score replay differs.", "Output seal mismatch."):
            producer.verify.side_effect = exporter.suite.shared.Refusal(error)
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.verify_qwen(producer, self.root)
        producer.load.assert_not_called()

    def test_qwen_compatibility_refuses_material_summary_difference(self):
        producer = Mock()
        producer.verify.side_effect = exporter.suite.shared.Refusal("Summary replay differs.")
        producer.load.return_value = ({}, [])
        with patch.object(exporter.suite, "read_rows", return_value=[]), \
                patch.object(exporter.base.canonical, "replay", return_value=[]), \
                patch.object(exporter.suite, "summarize", return_value={"gap": 1.}), \
                patch.object(exporter.suite, "read_json", return_value={"gap": 1.0001}):
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.verify_qwen(producer, self.root)

    def test_v1_registration_cannot_be_launched_through_v2(self):
        with patch.object(watcher, "raw_load", return_value={"source_sha256": {}}), \
                patch.object(watcher.engine, "run") as run:
            with self.assertRaises(exporter.suite.shared.Refusal):
                watcher.run(self.root)
            run.assert_not_called()
        self.assertFalse((self.root / "runner.json").exists())


class CompletedRealAdapterTests(unittest.TestCase):
    def test_real_completed_new_sources_without_pending_correctness(self):
        instances = exporter.release.labeled_instances(
            exporter.BUNDLE, exporter.SPLIT, exporter.release.manifest_at(exporter.BUNDLE))
        for lane in exporter.lanes(watcher.engine.ASTRA):
            if lane.kind not in ("qwen27", "laya", "trained"):
                continue
            with self.subTest(profile=lane.name):
                if lane.kind in ("laya", "trained") and sys.version_info[:2] != (3, 12):
                    self.skipTest("Native registered source verification requires Python 3.12")
                if not exporter.readiness([lane])[lane.name]["ready"]:
                    self.skipTest("Source not terminal and restored; correctness not read")
                evidence = exporter.Evidence()
                manifest, terminal, attempts, saved, model, native = exporter.source_data(lane, evidence)
                rows, audits = exporter.replay_lane(
                    lane, instances, attempts, saved, model, native, terminal)
                self.assertEqual(len(rows), 960)
                self.assertEqual(len(exporter.task_metrics(rows)), 60)
                self.assertEqual(len(audits), sum(r["feasible"] for r in rows
                                                if r["task"] in exporter.CONSTRUCTION))
                evidence.recheck()
                if lane.kind == "trained":
                    self.assertTrue(manifest["source_verification"]["production_route_verified"])
                    self.assertTrue(manifest["source_verification"]["services_restoration_verified"])


if __name__ == "__main__":
    unittest.main()
