import unittest

from paper import build_deadline_table as build
from scripts.paired_graph_ablation import Refusal


class DeadlineTableTests(unittest.TestCase):
    def fixture(self):
        rows = [{"instance_id": f"{task}-{i}", "task": task, "feasible": False,
                 "failure_count": 1, "absolute_gap": None}
                for task in build.OPTIMIZATION for i in range(4)]
        attempts = [{"instance_id": r["instance_id"], "status": "failure",
                     "diagnostic_code": "finish_length", "usage": {"output_tokens": 8192},
                     "latency_seconds": 239} for r in rows]
        return {"qwen9_600": rows, "qwen9_180": [dict(r) for r in rows]}, attempts

    def test_explicit_missing_quality(self):
        self.assertEqual(build.render(*self.fixture()).count("0/4 & 0/4 & 4 & --"), 4)

    def test_no_timeout_relabeling(self):
        models, attempts = self.fixture()
        attempts[0]["diagnostic_code"] = "timeout"
        with self.assertRaises(Refusal):
            build.render(models, attempts)

    def test_no_invented_zero_gap(self):
        models, attempts = self.fixture()
        models["qwen9_600"][0]["absolute_gap"] = 0
        with self.assertRaises(Refusal):
            build.render(models, attempts)

    def test_no_duplicate_cases(self):
        models, attempts = self.fixture()
        attempts[-1] = attempts[0]
        with self.assertRaises(Refusal):
            build.render(models, attempts)


if __name__ == "__main__":
    unittest.main()
