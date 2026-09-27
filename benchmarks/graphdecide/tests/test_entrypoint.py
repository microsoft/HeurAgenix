"""Offline checks for the direct script entry point; never call the provider."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "run_benchmark.py"


class ScriptEntryPointTests(unittest.TestCase):
    def test_script_delegates_to_shared_cli(self):
        with patch("src.benchmark.__main__.main") as main:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        main.assert_called_once_with()

    def test_import_does_not_start_evaluation(self):
        with patch("src.benchmark.__main__.main") as main:
            runpy.run_path(str(SCRIPT), run_name="imported_entrypoint")
        main.assert_not_called()

    def test_help_from_external_directory_without_key_or_pythonpath(self):
        env = {k: v for k, v in os.environ.items() if k not in {"TYPESAFE_API_KEY", "PYTHONPATH"}}
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--help"], cwd=cwd, env=env,
                capture_output=True, text=True, timeout=15, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_benchmark.py", result.stdout)
        self.assertIn("--config", result.stdout)
        self.assertIn("--no-progress", result.stdout)
        self.assertNotIn("--execute", result.stdout)


if __name__ == "__main__":
    unittest.main()