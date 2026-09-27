from collections import Counter
import unittest

from scripts import larger_local_pilot as pilot


class LargerPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = pilot.invariance.DEFAULT_BUNDLE
        manifest = pilot.release.manifest_at(cls.bundle)
        cls.public = pilot.release.load_public(cls.bundle, pilot.invariance.SPLIT, manifest)
        cls.metadata = pilot.suite.read_json(pilot.release.checked_file(
            cls.bundle, f"analysis/{pilot.invariance.SPLIT}.json", manifest))
        cls.selected = pilot.select_cases(cls.public, cls.metadata)
        labeled = {i["id"]: i for i in pilot.release.labeled_instances(
            cls.bundle, pilot.invariance.SPLIT, manifest)}
        cls.instances = [labeled[i["id"]] for i in cls.selected]

    def test_selection_is_fixed_and_balanced(self):
        self.assertEqual(self.selected, pilot.select_cases(list(reversed(self.public)), self.metadata))
        counts = Counter((i["task"], len(i["state"]["nodes"])) for i in self.selected)
        self.assertEqual(len(self.selected), 64)
        for (task, _), count in counts.items():
            self.assertEqual(count, 2 if task in pilot.CONSTRUCTION else 4)

    def test_missing_construction_fails_without_replacement(self):
        with self.assertRaises(pilot.suite.shared.Refusal):
            pilot.select_cases([i for i in self.public if i["task"] != "tsp_construct"],
                               self.metadata)

    def test_every_unattempted_episode_remains_in_denominator(self):
        rows = pilot.failure_denominators(self.instances)
        self.assertEqual(len(rows), 64)
        self.assertTrue(all(row["query_count"] == 0 and not row["feasible"] for row in rows))

    def test_native_option_mapping_and_configurations(self):
        for instance in self.instances:
            request = pilot.canonical.public_request(instance, [])
            self.assertEqual(len({o.id for o in request.options}), len(request.options))
            self.assertEqual(request.request_id, instance["id"] + ":0")
        configs = pilot.configurations()
        self.assertTrue(configs["qwen27_grammar"].constrain_choices)
        self.assertEqual(configs["qwen27_grammar"].max_tokens, 64)
        for name in ("qwen27_thinking", "qwen9_thinking"):
            self.assertFalse(configs[name].constrain_choices)
            self.assertEqual(configs[name].max_tokens, 8192)
            self.assertEqual(configs[name].timeout_seconds, 180)


if __name__ == "__main__":
    unittest.main()
