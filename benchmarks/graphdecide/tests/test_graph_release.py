from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from scripts import build_graph_release as release
from scripts import graph_release_api as api
from test_structural_integration import fixture


class GraphReleaseTests(TestCase):
    def source(self, directory):
        instances, metadata = fixture()
        metadata.update(requested_instances=len(instances), generated_instances=len(instances))
        source = directory / "source"
        with patch.dict("sys.modules", {"src.benchmark.structural_tasks": SimpleNamespace(
                build_structural_bank=lambda **_: (deepcopy(instances), deepcopy(metadata)))}):
            release.suite.plan(source, design="structural_dev", node_counts=(8,), models=("jev",))
        return source

    def test_portable_bundle_is_deterministic_and_has_no_provider_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.source(base)
            with patch.object(release.suite, "create_client", side_effect=AssertionError("Network")):
                a = release.build(base / "a", {"development": (source, None)}, base / "a.tgz")
                b = release.build(base / "b", {"development": (source, None)}, base / "b.tgz")
            self.assertEqual(a["archive_sha256"], b["archive_sha256"])
            self.assertEqual(a["verification"]["instances"], 10)
            with tarfile.open(base / "a.tgz") as archive:
                names = archive.getnames()
                self.assertIn("src/__init__.py", names)
                self.assertFalse(any("/runs/" in n or "config.py" in n or "typesafe.py" in n
                                     for n in names))
                self.assertTrue(all(not n.startswith("/") and ".." not in Path(n).parts for n in names))
            source.rename(base / "unavailable-source")
            checked = subprocess.run(
                [sys.executable, "-I", str(base / "a/release_api.py"), "verify",
                 "--root", str(base / "a")], cwd=base, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(checked.stdout)["model_calls"], 0)

    def test_request_never_reads_reference_and_protocol_changes_only_tsp(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.source(base)
            release.build(base / "bundle", {"development": (source, None)}, base / "bundle.tgz")
            root = base / "bundle"
            manifest = api.manifest_at(root)
            public = api.load_public(root, "development", manifest)
            for row in public:
                legacy = api.request_for(root, "development", row["id"], [], protocol="legacy_frozen")
                explicit = api.request_for(root, "development", row["id"], [],
                                           protocol="state_explicit_tsp")
                canonical = api.request_for(root, "development", row["id"], [])
                self.assertEqual(canonical["state"], legacy["state"])
                self.assertEqual([o["id"] for o in canonical["options"]],
                                 [o["id"] for o in legacy["options"]])
                self.assertTrue(all(o["description"] for o in canonical["options"]))
                self.assertNotIn("return only", canonical["question"].lower())
                self.assertEqual(legacy["state"], explicit["state"])
                self.assertNotIn("private", legacy["state"])
                if row["task"] == "tsp_construct":
                    self.assertIn("NOW at vertex", explicit["question"])
                else:
                    self.assertEqual(legacy, explicit)
            selected = public[0]
            before = api.request_for(root, "development", selected["id"], [])
            (root / "references/development.jsonl").write_text("not readable as references\n")
            self.assertEqual(before, api.request_for(root, "development", selected["id"], []))
            with self.assertRaisesRegex(ValueError, "integrity"):
                api.verify(root)

    def test_public_tamper_refused_and_archive_cannot_include_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self.source(base)
            with self.assertRaises(release.suite.Refusal):
                release.build(base / "bundle", {"development": (source, None)},
                              base / "bundle/archive.tgz")
            release.build(base / "bundle", {"development": (source, None)}, base / "bundle.tgz")
            root = base / "bundle"
            (root / "public/development.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                api.request_for(root, "development", "unknown", [])
