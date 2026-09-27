"""Recommended Qwen sampling settings must survive YAML and the actual HTTP wire."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import httpx
import yaml

from src.benchmark.config import ModelConfig, load_config
from src.clients.base import DecisionOption, DecisionRequest
from src.clients.vllm_client import VLLMClient

REPO = Path(__file__).resolve().parents[1]


class SamplingControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_config_yaml_and_wire_roundtrip(self):
        model = ModelConfig("vllm", "Qwen3.5-9B", 30, think=False,
                            temperature=1.0, top_p=1.0, top_k=40,
                            presence_penalty=2.0, max_tokens=64, output_format="answer_only")
        config = replace(load_config(REPO / "configs/adjacency.yaml"), model=model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config.snapshot()))
            rebuilt = load_config(path)
        self.assertEqual(rebuilt.model, model)
        self.assertEqual(rebuilt.sha256, config.sha256)
        observed = []

        def handler(request):
            body = json.loads(request.content)
            observed.append(body)
            return httpx.Response(200, json={
                "model": model.model,
                "choices": [{"finish_reason": "stop",
                             "message": {"role": "assistant", "content": "1"}}],
                "usage": {},
            })

        request = DecisionRequest("test", {"value": 1}, "Select one.",
                                  (DecisionOption("0"), DecisionOption("1")))
        async with VLLMClient(**model.client_kwargs(), api_key="",
                              transport=httpx.MockTransport(handler)) as client:
            self.assertEqual((await client.predict(request)).selected_option_id, "1")
        self.assertEqual(len(observed), 1)
        self.assertEqual({k: observed[0][k] for k in (
            "temperature", "top_p", "top_k", "presence_penalty", "max_tokens")},
            {"temperature": 1.0, "top_p": 1.0, "top_k": 40, "presence_penalty": 2.0, "max_tokens": 64})

    def test_optional_settings_preserve_legacy_defaults(self):
        config = ModelConfig("vllm", "test", 30)
        self.assertNotIn("top_k", config.client_kwargs())
        self.assertNotIn("presence_penalty", config.client_kwargs())
        explicit = replace(config, top_k=-1, presence_penalty=0)
        self.assertEqual(explicit.client_kwargs()["top_k"], -1)
        self.assertEqual(explicit.client_kwargs()["presence_penalty"], 0)

    def test_invalid_settings_and_provider_mismatch_rejected(self):
        for kwargs in ({"top_k": 0}, {"top_k": -2}, {"top_k": True}, {"top_k": 1.5},
                       {"presence_penalty": float("nan")}, {"presence_penalty": 2.1},
                       {"presence_penalty": -2.1}, {"presence_penalty": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ModelConfig("vllm", "test", 30, **kwargs)
                with self.assertRaises(ValueError):
                    VLLMClient(**kwargs)
        for provider in ("typesafe", "github_copilot"):
            for kwargs in ({"top_k": 20}, {"presence_penalty": 1.5}):
                with self.assertRaises(ValueError):
                    ModelConfig(provider, "test", 30, **kwargs)
