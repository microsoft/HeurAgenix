import json
import shutil
import unittest
from unittest.mock import patch
import uuid

from scripts import trained_cpu_gpu_control as control


class CPUControlTests(unittest.TestCase):
    def setUp(self):
        self.root = control.REPO / "output" / ("cpu-control-tests-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def request(self, ids=("yes", "no")):
        return {"request_id": "authored:0", "state": {"nodes": [0, 1]},
                "question": "Choose one.", "options": [
                    {"id": key, "description": f"Meaning {key}"} for key in ids],
                "protocol_version": "choice-v1"}

    def native(self, p=.8, selected="yes"):
        return {"selected_option_id": selected, "probabilities": {"yes": p, "no": 1 - p},
                "input_tokens": 12, "logits": None}

    def test_selection_is_first_in_panel_order_not_prediction_order(self):
        cases = [{"task": "degree_exact", "node_count": 12, "index": 2},
                 {"task": "adjacency", "node_count": 8, "index": 3},
                 {"task": "degree_exact", "node_count": 12, "index": 4},
                 {"task": "adjacency", "node_count": 8, "index": 5}]
        self.assertEqual(control.choose_cases(cases), cases[:2])

    def test_restore_original_order_without_changing_saved_values(self):
        canonical = {**self.request(), "state": {"nodes": [0, 1], "vertex": 0,
                                                 "remaining_queries": 1}}
        persisted = json.loads(json.dumps(canonical, sort_keys=True))
        actual = control.wire_request(persisted, canonical)
        self.assertEqual(actual, persisted)
        self.assertEqual(list(actual["state"]), ["nodes", "vertex", "remaining_queries"])
        self.assertEqual(list(persisted["state"]), ["nodes", "remaining_queries", "vertex"])
        wrong = {**canonical, "question": "Changed"}
        with self.assertRaises(RuntimeError):
            control.wire_request(persisted, wrong)

    def test_descriptive_gate_no_renormalization(self):
        result = control.compare_vector(self.request(), self.native(.8), self.native(.795), "test")
        self.assertAlmostEqual(result["max_abs_difference"], .005)
        self.assertAlmostEqual(result["mean_abs_difference"], .005)
        self.assertTrue(result["gate_pass"])
        result = control.compare_vector(self.request(), self.native(.8), self.native(.78), "test")
        self.assertFalse(result["gate_pass"])

    def test_small_delta_with_changed_choice_fails_gate(self):
        result = control.compare_vector(self.request(), self.native(.501),
                                        self.native(.499, "no"), "test")
        self.assertLess(result["max_abs_difference"], .01)
        self.assertFalse(result["native_choice_agreement"])
        self.assertFalse(result["gate_pass"])

    def test_twelve_numeric_ids_coverage_order_and_finite_scores(self):
        ids = ["11", "0", "10", "1", "9", "2", "8", "3", "7", "4", "6", "5"]
        request = self.request(ids)
        native = {"selected_option_id": "11", "probabilities": {key: 1/12 for key in ids},
                  "input_tokens": 100}
        result = control.compare_vector(request, native, native, "test")
        self.assertEqual(result["option_ids"], ids)
        self.assertTrue(result["gate_pass"])
        for value in [float("nan"), -1, 2]:
            bad = {**native, "probabilities": {**native["probabilities"], "11": value}}
            with self.assertRaises(Exception):
                control.compare_vector(request, native, bad, "test")
        missing = {**native, "probabilities": {key: p for key, p in native["probabilities"].items()
                                              if key != "10"}}
        with self.assertRaises(Exception):
            control.compare_vector(request, native, missing, "test")

    def fixture(self):
        rows = []
        for candidate in control.MODELS:
            for index in range(2):
                request = self.request()
                rows.append({"candidate": candidate, "instance_id": f"id-{index}",
                             "request": request, "gpu_result": self.native(),
                             "payload_sha256": control.json_digest(control.adapter.public_payload(
                                 control.adapter.request_from_dict(request)))})
        control.save(self.root / "scheduled.json", rows)
        return {"models": {}, "packages": {}, "source_root": str(self.root),
                "frozen_code_sha256": {},
                "native_dtype": {"decider": "bfloat16", "kev": "float32"},
                "native_temperature": {"decider": 1.164, "kev": 2.406}}

    def run_fake(self, fail_first=False):
        manifest = self.fixture()
        owner = self
        seen, closed, loads, request_deadlines = [], [], [], []

        class Worker:
            def __init__(self, source, candidate, output):
                self.candidate = candidate

            def receive(self, deadline):
                loads.append(deadline)
                return {"status": "ready", "runtime": {
                    "cuda_initialized": False, "dtype": manifest["native_dtype"][self.candidate],
                    "temperature": manifest["native_temperature"][self.candidate]}}

            def predict(self, request, deadline):
                seen.append((self.candidate, request))
                request_deadlines.append(deadline)
                if fail_first and len(seen) == 1:
                    raise TimeoutError("authored timeout")
                return {"status": "success", "result": owner.native(), "cuda_initialized": False,
                        "payload_sha256": control.json_digest(control.adapter.public_payload(
                            control.adapter.request_from_dict(request)))}

            def close(self):
                closed.append(self.candidate)

        def comparison(output):
            rows = control.read(output / "results.json")
            value = {"successful": sum(r["status"] == "success" for r in rows),
                     "failed": sum(r["status"] == "failure" for r in rows),
                     "unattempted": sum(r["status"] == "unattempted" for r in rows),
                     "gate_pass_count": sum(r["status"] == "success" for r in rows)}
            control.save(output / "comparison.json", value)
            return value

        with patch.object(control, "verify", return_value=manifest), \
                patch.object(control.frozen, "verify_artifacts"), \
                patch.object(control.frozen, "verify_files"), \
                patch("importlib.metadata.distributions", return_value=[]), \
                patch.object(control, "offline_compare", side_effect=comparison):
            result = control.run(self.root, Worker)
        return result, seen, closed, loads, request_deadlines

    def test_exactly_four_requests_no_other_inference_and_deadlines(self):
        result, seen, closed, loads, deadlines = self.run_fake()
        self.assertEqual(result["successful"], 4)
        self.assertEqual([name for name, _ in seen], ["decider", "decider", "kev", "kev"])
        self.assertEqual(closed, ["decider", "kev"])
        self.assertTrue(all(0 < seconds <= 300 for seconds in loads))
        self.assertTrue(all(0 < seconds <= 180 for seconds in deadlines))
        self.assertTrue(all("gpu_result" not in request for _, request in seen))

    def test_failure_retained_no_retry_and_unattempted_row_preserved(self):
        result, seen, closed, _, _ = self.run_fake(fail_first=True)
        self.assertEqual([name for name, _ in seen], ["decider", "kev", "kev"])
        self.assertEqual(result, {"successful": 2, "failed": 1, "unattempted": 1,
                                  "gate_pass_count": 2})
        self.assertEqual(closed, ["decider", "kev"])
        rows = control.read(self.root / "results.json")
        self.assertEqual(rows[0]["error_type"], "TimeoutError")

    def test_refuse_reexecution_before_fresh_forwards(self):
        manifest = self.fixture()
        (self.root / "run-start.json").write_text("existing")
        with patch.object(control, "verify", return_value=manifest), \
                self.assertRaises(FileExistsError):
            control.run(self.root, lambda *args: self.fail("must not load"))

    def test_cuda_invariant_and_thread_guards(self):
        class Tensor:
            device = type("Device", (), {"type": "cpu"})()

        class Native:
            def parameters(self):
                return [Tensor()]

        class Torch:
            cuda = type("Cuda", (), {"is_initialized": lambda: False})()

            @staticmethod
            def get_num_threads():
                return 4

            @staticmethod
            def get_num_interop_threads():
                return 1

        with patch.dict(control.os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(Torch.cuda, "is_initialized", return_value=False):
            control.assert_cpu((None, Native()), "kev", Torch)
            with patch.object(Torch.cuda, "is_initialized", return_value=True), \
                    self.assertRaises(RuntimeError):
                control.assert_cpu((None, Native()), "kev", Torch)
