import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "semif_cpu_pilot.py"
spec = importlib.util.spec_from_file_location("semif_cpu_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class OptionMappingTests(unittest.TestCase):
    def setUp(self):
        self.row = {"options": [{"id": "10"}, {"id": "2"}, {"id": "none"}]}

    def test_scores_map_to_ids_not_letters_or_integer_positions(self):
        self.assertEqual(pilot.choose(self.row, {
            "option_ids": ["10", "2", "none"], "probabilities": [0.1, 0.8, 0.1]}), "2")

    def test_reordered_return_ids_are_mapped_explicitly(self):
        self.assertEqual(pilot.choose(self.row, {
            "option_ids": ["none", "10", "2"], "probabilities": [0.8, 0.1, 0.1]}), "none")

    def test_ties_use_original_order(self):
        self.assertEqual(pilot.choose(self.row, {
            "option_ids": ["none", "2", "10"], "probabilities": [0, 0.5, 0.5]}), "10")

    def test_invalid_score_maps_fail(self):
        for ids, probs in [
            (["10", "2"], [0.5, 0.5]),
            (["10", "2", "2"], [0.3, 0.3, 0.4]),
            (["10", "2", "unknown"], [0.3, 0.3, 0.4]),
            (["10", "2", "none"], [float("nan"), 0.5, 0.5]),
            (["10", "2", "none"], [True, 0, 0]),
            (["10", "2", "none"], [0.1, 0.1, 0.1]),
        ]:
            with self.subTest(ids=ids, probabilities=probs), self.assertRaises(ValueError):
                pilot.choose(self.row, {"option_ids": ids, "probabilities": probs})

    def test_public_rows_and_reference_separation(self):
        rows, refs = pilot.cases()
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(r["expected"] is not None for r in refs.values()), 12)
        self.assertEqual(len({r["task"] for r in rows}), 10)
        for row in rows:
            self.assertNotIn("expected", row)
            self.assertNotIn("reference", row)
            self.assertGreaterEqual(len(row["options"]), 2)
            self.assertLessEqual(len(row["options"]), 16)
            for option in row["options"]:
                self.assertTrue(option["description"])
                if option["id"].isdigit():
                    self.assertIn(option["id"], option["description"])


if __name__ == "__main__":
    unittest.main()
