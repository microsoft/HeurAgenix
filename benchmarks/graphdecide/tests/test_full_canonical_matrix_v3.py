import copy
import unittest
from unittest.mock import patch

from scripts import export_full_canonical_matrix_v3 as exporter
from scripts import finalize_full_canonical_matrix_v3 as watcher


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = exporter.TRAINED
        cls.receipt = exporter.suite.read_json(exporter.RECEIPT)
        out = cls.root / "standalone-coordinator"
        cls.records = [exporter.suite.read_json(path) for path in (
            out / "resources/trained-supervisor-launch-intent.json",
            out / "resources/trained-supervisor-process.json",
            cls.root / "launch.json", out / "worker-handoff.json")]

    def test_existing_postexec_receipt_matches_frozen_chain(self):
        result = exporter.validate_receipt(self.root, self.receipt, *self.records)
        self.assertTrue(result["supervisor_argv_observed_and_verified"])
        self.assertTrue(result["supervisor_cuda_environment_observed_and_verified"])
        self.assertEqual(exporter.suite.digest(exporter.RECEIPT), exporter.RECEIPT_SHA)
        self.assertEqual(self.records[1]["argv"], [])
        self.assertEqual(self.records[1]["cuda_environment"], {})

    def test_modified_receipt_identity_or_postexec_fields_are_refused(self):
        for field, value in (("pid", 999), ("start_ticks", "999"), ("ppid", 999),
                             ("argv", []), ("cuda_environment", {}), ("cwd", str(self.root))):
            changed = copy.deepcopy(self.receipt)
            changed["supervisor"][field] = value
            with self.subTest(field=field), self.assertRaises(exporter.suite.shared.Refusal):
                exporter.validate_receipt(self.root, changed, *self.records)

    def test_receipt_is_not_used_as_terminal_admission(self):
        lane = next(lane for lane in exporter.lanes(watcher.prior.engine.ASTRA) if lane.kind == "trained")
        with patch.object(exporter.prior.base, "require_ready",
                          side_effect=exporter.suite.shared.Refusal("pending")), \
                patch.object(exporter.suite, "read_rows") as rows, \
                patch.object(exporter.suite, "read_json") as metadata:
            with self.assertRaises(exporter.suite.shared.Refusal):
                exporter.source_data(lane, exporter.Evidence())
            rows.assert_not_called()
            metadata.assert_not_called()

    def test_v2_registration_cannot_be_claimed_by_v3(self):
        with patch.object(watcher, "raw_load", return_value={"source_sha256": {}}), \
                patch.object(watcher.prior.engine, "run") as run:
            with self.assertRaises(exporter.suite.shared.Refusal):
                watcher.run(watcher.ROOT)
            run.assert_not_called()

    def test_native_real_trained_adapter_seals_postexec_receipt(self):
        import sys
        if sys.version_info[:2] != (3, 12):
            self.skipTest("Native registered environment requires Python 3.12")
        lane = next(lane for lane in exporter.lanes(watcher.prior.engine.ASTRA)
                    if lane.name == "decider")
        if not exporter.readiness([lane])[lane.name]["ready"]:
            self.skipTest("Trained source is not terminal/restored")
        evidence = exporter.Evidence()
        result = exporter.source_data(lane, evidence)
        proof = result[0]["source_verification"]
        self.assertTrue(proof["post_exec_parent_receipt"]["supervisor_argv_observed_and_verified"])
        self.assertFalse(proof["process_observations"]["captured_supervisor_argv_observed"])
        self.assertEqual(evidence.hashes[str(exporter.RECEIPT.relative_to(exporter.REPO))],
                         exporter.RECEIPT_SHA)
        self.assertEqual(len(result[3]), 960)
        evidence.recheck()


if __name__ == "__main__":
    unittest.main()
