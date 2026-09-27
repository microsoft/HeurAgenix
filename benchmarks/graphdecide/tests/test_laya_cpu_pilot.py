import importlib.util
import os
import sys
from types import SimpleNamespace
import unittest

from scripts import laya_cpu_pilot as pilot


class LayaPilotTests(unittest.TestCase):
    def test_question_preserves_order_and_descriptions(self):
        request = pilot.smoke_requests()[1]
        question = pilot.question(request)
        self.assertEqual(list(question["criteria"]), [o["id"] for o in request["options"]])
        self.assertEqual(list(question["criteria"].values()),
                         [o["description"] for o in request["options"]])

    def test_option_limit_and_duplicate_rejected(self):
        request = pilot.smoke_requests()[1]
        request["options"] += [{"id": "extra", "description": "extra"}]
        with self.assertRaises(ValueError):
            pilot.question(request)
        request["options"] = [request["options"][0]] * 2
        with self.assertRaises(ValueError):
            pilot.question(request)

    def test_native_probability_rounding_not_recalibrated(self):
        request = {"options": [{"id": str(i)} for i in range(3)]}
        answer = {"choice": "0", "probabilities": {str(i): .3333 for i in range(3)}}
        pilot.validate_answer(request, answer, [0., 0., 0.])
        answer["choice"] = "illegal"
        with self.assertRaises(ValueError):
            pilot.validate_answer(request, answer, [0., 0., 0.])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            pilot.validate_answer({"options": [{"id": "a"}]},
                                  {"choice": "a", "probabilities": {"a": float("nan")}}, [0.])

    def test_cpu_environment(self):
        env = pilot.environment()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["LAYA_CPU_AMP"], "")
        self.assertEqual(pilot.LIMITS["max_forwards"], 160)
        self.assertEqual(pilot.LIMITS["context_tokens"], 1024)

    def test_deadline_includes_incomplete_response(self):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "rb", buffering=0)
        worker = pilot.Worker.__new__(pilot.Worker)
        worker.process = SimpleNamespace(stdout=stream)
        worker.pending = b""
        try:
            os.write(write_fd, b'{"partial":')
            with self.assertRaises(TimeoutError):
                worker.receive(.01)
        finally:
            stream.close()
            os.close(write_fd)

    def test_zero_attempt_replay_retains_64_denominator(self):
        from scripts import evaluate_graph_release as evaluation
        instances, _ = pilot.panel()
        rows = evaluation.replay(instances, [], SimpleNamespace(model=pilot.MODEL),
                                 native_action=False)
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(r["query_count"] == 0 and not r["feasible"] for r in rows))

    @unittest.skipUnless((pilot.ROOT / "upstream/laya/common.py").exists()
                         and importlib.util.find_spec("torch"), "Pinned artifact integration")
    def test_lossless_context_including_native_head_clipping(self):
        sys.path.insert(0, str(pilot.ROOT / "upstream"))
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(pilot.ROOT / "model/tokenizer",
                                           local_files_only=True, trust_remote_code=False)
        request = pilot.smoke_requests()[0]
        self.assertTrue(pilot.coverage(tok, request)["compatible"])
        request["state"] = " graph" * 1100
        check = pilot.coverage(tok, request)
        self.assertFalse(check["compatible"])
        self.assertIn("context_overlength", check["reasons"])
        request["state"] = "small"
        request["question"] = " question" * 300
        check = pilot.coverage(tok, request)
        self.assertLess(check["full_tokens"], 1024)
        self.assertIn("head_truncation", check["reasons"])
        request["question"] = "short"
        request["options"][0]["description"] = " option" * 60
        self.assertIn("option_truncation", pilot.coverage(tok, request)["reasons"])


if __name__ == "__main__":
    unittest.main()
