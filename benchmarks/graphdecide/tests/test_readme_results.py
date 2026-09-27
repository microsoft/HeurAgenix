from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from scripts import render_readme_results as render


class ReadmeResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = render.ROOT / "web/data/benchmark.json"
        cls.data = json.loads(cls.source.read_text())

    def test_assets_are_current_and_valid_svg(self):
        outputs = render.render(self.data, sha256(self.source.read_bytes()).hexdigest())
        for name, content in outputs.items():
            self.assertEqual((render.ROOT / name).read_text(), content)
            if name.endswith(".svg"):
                svg = ET.fromstring(content)
                self.assertEqual(svg.attrib["role"], "img")
                self.assertIn("GPT REFERENCES", content)
                self.assertNotIn("http://localhost", content)
        self.assertIn("41.24%", outputs["docs/RESULTS.md"])
        self.assertIn("45.50%", outputs["docs/RESULTS.md"])
        self.assertIn("152.4000 (40/40)", outputs["docs/RESULTS.md"])
        self.assertIn("(31/40)", outputs["docs/RESULTS.md"])

    def test_missing_duplicate_macro_and_gap_errors_fail(self):
        def mutate_macro(data):
            next(r for r in data["metrics"] if r["model"] == "qwen25_72b_bf16"
                 and r["task"] == "degree_exact" and r["nodeCount"] == "all")["accuracy"] = .405

        def mutate_gap(data):
            next(r for r in data["metrics"] if r["task"] == "tsp_construct"
                 and r["nodeCount"] == "all")["meanGap"] += 1

        for mutate in (lambda d: d["metrics"].pop(),
                       lambda d: d["metrics"].append(d["metrics"][0]),
                       mutate_macro, mutate_gap):
            data = deepcopy(self.data)
            mutate(data)
            with self.assertRaises(ValueError):
                render.validate(data)


if __name__ == "__main__":
    unittest.main()
