#!/usr/bin/env python3
"""SemIf-compatible explicit candidate scores via existing vLLM completions.

This bridge is not upstream SemIf execution and does not share its one-pass cost.
It never loads weights, starts a server, changes a server, or replaces lane locks.
The default uses one prefix and logprob_token_ids; teacher forcing is explicit fallback.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import AbstractContextManager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, ClientClosedError, ClientTimeoutError,
    ClientTransportError, DecisionOption, DecisionRequest, DecisionResponse,
    InvalidResponseError, ProviderHTTPError, TokenUsage, UnsupportedRequestError,
)

MODEL_ID = "Qwen3.5-4B"
BASE_URL = "http://127.0.0.1:8002/v1"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_REVISION = "23cf1f39fc9534fe81437200959b6dfc7106e45a"
DEFAULT_CPU = ROOT / "output/semif-cpu-pilot-20260924-v1"
DEFAULT_OUTPUT = ROOT / "output/semif-vllm-calibration-20260924-v1"
DEFAULT_MODEL = Path("/fastdata/xianya/models/Qwen3.5-4B")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def upstream_functions(cpu: Path, model: Path):
    manifest = json.loads((cpu / "upstream/source-manifest.json").read_text())
    if manifest["revision"] != SOURCE_REVISION:
        raise ValueError("Unexpected SemIf revision")
    for item in manifest["files"]:
        if digest(cpu / "upstream" / item["path"]) != item["sha256"]:
            raise ValueError("Upstream source changed")
    provenance = json.loads((cpu / "provenance.json").read_text())
    if provenance["declared_model_revision"] != MODEL_REVISION:
        raise ValueError("CPU reference revision differs")
    # Verify the tokenizer bytes without rereading/loading 9 GB of model weights.
    for item in provenance["model_files"]:
        if item["name"] in {"config.json", "tokenizer.json", "tokenizer_config.json",
                            "chat_template.jinja"}:
            if digest(model / item["name"]) != item["sha256"]:
                raise ValueError("Local tokenizer/config differs from CPU reference")
    for key in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
        os.environ[key] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(cpu / "upstream/src"))
    from semif_phase1.direct import encode_prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False)
    return tokenizer, encode_prompt


def row_for(request: DecisionRequest) -> dict:
    if len(request.options) > 16:
        raise UnsupportedRequestError("SemIf supports at most 16 options")
    return {"id": request.request_id, "state": request.state, "question": request.question,
            "options": [{"id": o.id, "description": o.description or o.id}
                        for o in request.options]}


def make_payload(prefix: list[int], slots: list[int], transport_id: str,
                 max_generated: int = 0) -> dict:
    if not 2 <= len(slots) <= 16 or len(set(slots)) != len(slots):
        raise UnsupportedRequestError("Need 2–16 distinct single-token candidates")
    if not prefix or any(type(i) is not int or i < 0 for i in prefix + slots):
        raise UnsupportedRequestError("Invalid prompt token IDs")
    if max_generated not in (0, 1):
        raise UnsupportedRequestError("Only explicit zero/one-token completion modes allowed")
    return {
        "model": MODEL_ID, "prompt": [prefix + [slot] for slot in slots],
        "request_id": transport_id, "n": 1, "stream": False,
        "echo": True, "max_tokens": max_generated, "logprobs": 0,
        "prompt_logprobs": 0, "return_token_ids": True,
        "add_special_tokens": False, "truncate_prompt_tokens": None,
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    }


def make_requested_payload(prefix, slots, transport_id):
    payload = make_payload(prefix, slots, transport_id, 1)
    payload.update({
        "prompt": prefix, "echo": False, "max_tokens": 1,
        "logprobs": len(slots), "logprob_token_ids": slots,
        "return_tokens_as_token_ids": True, "temperature": 1.0,
        "min_tokens": 0, "ignore_eos": True,
    })
    del payload["prompt_logprobs"]
    return payload


def parse_requested_response(body, request, payload):
    def require(ok, message):
        if not ok:
            raise InvalidResponseError(message)

    require(isinstance(body, dict), "Response must be an object")
    require(body.get("model") == MODEL_ID, "Unexpected served model")
    require(body.get("id") == "cmpl-" + payload["request_id"], "Response/case ID mismatch")
    choices = body.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "Expected one prefix response")
    choice = choices[0]
    require(isinstance(choice, dict) and type(choice.get("index")) is int
            and choice["index"] == 0, "Wrong prefix response index")
    actual = choice.get("prompt_token_ids")
    require(isinstance(actual, list) and all(type(t) is int for t in actual)
            and actual == payload["prompt"], "Prompt truncated, retokenized or mismatched")
    produced = choice.get("token_ids")
    require(isinstance(produced, list) and len(produced) == 1
            and type(produced[0]) is int and produced[0] >= 0, "Expected one generated token")
    require(choice.get("finish_reason") in {"stop", "length"}, "Incomplete response")
    logprobs = choice.get("logprobs")
    require(isinstance(logprobs, dict), "Requested scores missing")
    tables = logprobs.get("top_logprobs")
    require(isinstance(tables, list) and len(tables) == 1 and isinstance(tables[0], dict),
            "Expected one explicit-token score table")
    tokens = payload["logprob_token_ids"]
    require(len(tokens) == len(request.options) and len(set(tokens)) == len(tokens),
            "Candidate identity mismatch")
    keys = {f"token_id:{token}" for token in tokens}
    allowed = keys | {f"token_id:{produced[0]}"}
    require(keys <= set(tables[0]) <= allowed,
            "Explicit candidate coverage missing or unexpected natural top-K table")
    scores = [tables[0][f"token_id:{token}"] for token in tokens]
    # OpenAI formatting clamps -inf/very small values to -9999; reject that sentinel.
    require(all(type(v) in (int, float) and math.isfinite(v) and -9999 < v <= 1e-6
                for v in scores), "Nonfinite, clamped or non-logprob candidate scores")
    usage = body.get("usage")
    require(isinstance(usage, dict)
            and all(type(usage.get(k)) is int for k in
                    ["prompt_tokens", "completion_tokens", "total_tokens"]),
            "Usage missing or invalid")
    require(usage["prompt_tokens"] == len(actual) and usage["completion_tokens"] == 1
            and usage["total_tokens"] == len(actual) + 1, "Usage disagreement")
    maximum = max(scores)
    weights = [math.exp(v - maximum) for v in scores]
    probs = [v / sum(weights) for v in weights]
    ids = [o.id for o in request.options]
    selected = max(range(len(ids)), key=probs.__getitem__)
    return {
        "selected_option_id": ids[selected], "option_ids": ids,
        "candidate_logprobs": scores, "probabilities": dict(zip(ids, probs)),
        "candidate_count": len(ids), "candidate_prompts": 1,
        "shared_prefix_tokens": len(actual), "prefill_tokens": len(actual),
        "generated_tokens": 1, "requested_max_tokens": 1,
        "scoring_rule": "softmax of explicitly requested candidate token log probabilities",
        "generated_tokens_used_for_decision": False,
        "scoring_mode": "requested",
    }


def parse_response(body: dict, request: DecisionRequest, payload: dict) -> dict:
    def require(ok, message):
        if not ok:
            raise InvalidResponseError(message)

    require(isinstance(body, dict), "Response must be an object")
    require(body.get("model") == MODEL_ID, "Unexpected served model")
    require(body.get("id") == "cmpl-" + payload["request_id"], "Response/case ID mismatch")
    prompts = payload["prompt"]
    ids = [o.id for o in request.options]
    require(len(ids) == len(set(ids)) == len(prompts), "Invalid supplied option identity")
    choices = body.get("choices")
    require(isinstance(choices, list) and len(choices) == len(prompts),
            "Missing or extra candidate responses")
    values = {}
    generated = 0
    observed_prompt_tokens = 0
    for choice in choices:
        require(isinstance(choice, dict), "Candidate response must be an object")
        index = choice.get("index")
        require(type(index) is int and 0 <= index < len(prompts) and index not in values,
                "Duplicate, invalid or out-of-range batch index")
        expected = prompts[index]
        actual = choice.get("prompt_token_ids")
        require(isinstance(actual, list) and all(type(t) is int for t in actual)
                and actual == expected, "Prompt was truncated, retokenized or mismatched")
        probabilities = choice.get("prompt_logprobs")
        require(isinstance(probabilities, list) and len(probabilities) == len(expected),
                "Missing/misaligned prompt likelihoods")
        final = probabilities[-1]
        require(isinstance(final, dict), "Final prompt likelihood missing")
        entry = final.get(str(expected[-1]))
        require(isinstance(entry, dict), "Actual candidate token probability missing")
        value = entry.get("logprob")
        require(type(value) in (float, int) and math.isfinite(value) and value <= 1e-6,
                "Candidate log probability is invalid")
        require(choice.get("finish_reason") in {"length", "stop"}, "Incomplete candidate")
        produced = choice.get("token_ids")
        require(isinstance(produced, list) and len(produced) <= 1
                and all(type(t) is int and t >= 0 for t in produced),
                "Generated token accounting exceeds declared bound")
        generated += len(produced)
        observed_prompt_tokens += len(actual)
        values[index] = float(value)
    require(set(values) == set(range(len(prompts))), "Candidate coverage incomplete")
    usage = body.get("usage")
    require(isinstance(usage, dict), "Usage missing")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(usage.get(key)) is int and usage[key] >= 0, "Invalid usage")
    require(usage["prompt_tokens"] == observed_prompt_tokens
            and usage["completion_tokens"] == generated
            and usage["total_tokens"] == observed_prompt_tokens + generated,
            "Usage does not match actual prompt/generated token IDs")
    logs = [values[i] for i in range(len(ids))]
    maximum = max(logs)
    weights = [math.exp(v - maximum) for v in logs]
    probs = [w / sum(weights) for w in weights]
    selected = max(range(len(ids)), key=probs.__getitem__)
    return {
        "selected_option_id": ids[selected], "option_ids": ids,
        "candidate_logprobs": logs, "probabilities": dict(zip(ids, probs)),
        "candidate_count": len(ids), "shared_prefix_tokens": len(prompts[0]) - 1,
        "candidate_prompts": len(ids), "scoring_mode": "teacher_forced",
        "prefill_tokens": observed_prompt_tokens, "generated_tokens": generated,
        "requested_max_tokens": payload["max_tokens"],
        "scoring_rule": "softmax of exact final prompt-token log probabilities",
        "generated_tokens_used_for_decision": False,
    }


class SemIfVLLMBridge(BaseDecisionClient):
    """Each initialized instance owns the existing shared qwen4 lane lock.

    Do not wrap this instance in another acquisition of that same lane lock.
    The optional transport/guard are dependency injection for offline tests.
    """

    def __init__(self, tokenizer, encoder, *, max_prefix_tokens=2048, max_generated=0,
                 scoring_mode="requested", timeout=120.0, transport=None,
                 test_guard: AbstractContextManager | None = None):
        if max_generated not in (0, 1):
            raise ValueError("max_generated must be 0 or 1")
        if scoring_mode not in {"requested", "teacher_forced"}:
            raise ValueError("Unknown scoring mode")
        self.tokenizer, self.encoder = tokenizer, encoder
        self.max_prefix_tokens = max_prefix_tokens
        self.max_generated = max_generated
        self.scoring_mode = scoring_mode
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport,
                                      follow_redirects=False, trust_env=False)
        if test_guard is not None and transport is None:
            raise ValueError("A test guard requires injected HTTP transport")
        self._guard = test_guard
        self._test_transport = transport
        self._ready = False
        self._closed = False
        self._acquired = False

    @property
    def capabilities(self):
        return ClientCapabilities(supports_probabilities=True, max_options=16,
                                  max_context_tokens=self.max_prefix_tokens,
                                  supports_native_batch=False)

    async def initialize(self):
        if self._closed:
            raise ClientClosedError("Bridge is closed")
        if self._ready:
            return
        from scripts import paired_graph_ablation as shared
        if self._guard is None:
            # Exactly the existing coordination namespace; no replacement/tempdir override.
            self._guard = shared.lane_locks(["qwen4"])
        self._guard.__enter__()
        self._acquired = True
        try:
            config = shared.model_configs(models=("qwen4",))["qwen4"]
            if config.base_url != BASE_URL or config.model != MODEL_ID:
                raise ValueError("Shared qwen4 configuration differs")
            await shared.vllm_preflight([SimpleNamespace(model=config)],
                                        transport=self._test_transport)
            self._ready = True
        except BaseException:
            await self.aclose()
            raise

    async def predict(self, request: DecisionRequest) -> DecisionResponse:
        if self._closed:
            raise ClientClosedError("Bridge is closed")
        if not self._ready:
            raise ClientClosedError("Initialize and acquire the qwen4 lock first")
        self.validate_request(request)
        row = row_for(request)
        try:
            prefix, slots, prompt_hash = self.encoder(
                self.tokenizer, row, self.max_prefix_tokens)
        except ValueError as exc:
            raise UnsupportedRequestError(str(exc)) from exc
        # One extra teacher-forced token plus an engine-generated token must fit.
        if len(prefix) > self.max_prefix_tokens or len(prefix) + 2 > 40960:
            raise UnsupportedRequestError("Prompt exceeds pinned context budget")
        transport_id = "semif-" + uuid.uuid4().hex
        payload = (make_requested_payload(prefix, slots, transport_id)
                   if self.scoring_mode == "requested" else
                   make_payload(prefix, slots, transport_id, self.max_generated))
        started = time.perf_counter()
        try:
            response = await self._http.post(BASE_URL + "/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ClientTimeoutError("vLLM completion timed out; no automatic retry") from exc
        except httpx.TransportError as exc:
            raise ClientTransportError("vLLM completion transport failed") from exc
        if response.status_code != 200:
            raise ProviderHTTPError(response.status_code)
        try:
            body = response.json()
        except ValueError as exc:
            raise InvalidResponseError("vLLM returned non-JSON") from exc
        scores = (parse_requested_response(body, request, payload)
                  if self.scoring_mode == "requested" else
                  parse_response(body, request, payload))
        scores.update({"request_id": request.request_id, "transport_id": transport_id,
                       "prompt_sha256": prompt_hash,
                       "elapsed_seconds": time.perf_counter() - started})
        return DecisionResponse(
            request_id=request.request_id, selected_option_id=scores["selected_option_id"],
            resolved_model=MODEL_ID, probabilities=scores["probabilities"],
            probability_kind="option_logits",
            usage=TokenUsage(input_tokens=scores["prefill_tokens"],
                             output_tokens=scores["generated_tokens"]),
            raw_output={"bridge": scores, "provider": body})

    async def aclose(self):
        if not self._closed:
            await self._http.aclose()
            if self._acquired:
                self._guard.__exit__(None, None, None)
            self._acquired = False
            self._ready = False
            self._closed = True


def calibration_inputs(cpu: Path):
    rows = json.loads((cpu / "public-rows.json").read_text())
    # Scope is fixed by task names, not by answer labels or correctness.
    exact = {"adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
             "distance_threshold", "articulation_point"}
    selected = [row for row in rows if row["task"] in exact]
    if len(selected) != 12 or len({r["id"] for r in selected}) != 12:
        raise ValueError("Expected all twelve unique CPU exact probes")
    references = {r["id"]: r for r in
                  (json.loads(line) for line in (cpu / "decisions.jsonl").read_text().splitlines())}
    if any(row["id"] not in references or references[row["id"]]["status"] != "valid"
           for row in selected):
        raise ValueError("Missing valid CPU readout reference")
    return selected, references


def decision_request(row):
    return DecisionRequest(row["id"], row["state"], row["question"],
                           tuple(DecisionOption(o["id"], o["description"])
                                 for o in row["options"]))


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer, encoder = upstream_functions(args.cpu, args.model)
    rows, cpu = calibration_inputs(args.cpu)
    cases = []
    for row in rows:
        prefix, slots, prompt_hash = encoder(tokenizer, row, 2048)
        if prompt_hash != cpu[row["id"]]["upstream"]["prompt_sha256"]:
            raise ValueError("Canonical prompt differs from CPU reference")
        if len(prefix) != cpu[row["id"]]["upstream"]["input_tokens"]:
            raise ValueError("Canonical token count differs from CPU reference")
        cases.append({"id": row["id"], "task": row["task"], "prefix_tokens": len(prefix),
                      "slot_token_ids": slots, "option_ids": [o["id"] for o in row["options"]],
                      "prompt_sha256": prompt_hash,
                      "candidate_prefill_tokens": len(slots) * (len(prefix) + 1)})
    plan = {"scope": "twelve CPU exact probes; no ground truth used in requests",
            "source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
            "model_id": MODEL_ID, "base_url": BASE_URL, "cases": cases,
            "decision_calls": 12, "http_calls_per_decision": 1, "retry_policy": "none",
            "candidate_prompts": (len(cases) if args.scoring_mode == "requested" else
                                  sum(len(c["slot_token_ids"]) for c in cases)),
            "candidate_scores": sum(len(c["slot_token_ids"]) for c in cases),
            "logical_prefill_tokens": sum(c["prefix_tokens"] if args.scoring_mode == "requested"
                                          else c["candidate_prefill_tokens"] for c in cases),
            "scoring_mode": args.scoring_mode,
            "requested_max_tokens": 1 if args.scoring_mode == "requested" else args.max_generated,
            "installed_vllm_zero_echo_internal_generation": 1,
            "source_file_sha256": digest(Path(__file__)),
            "cpu_decisions_sha256": digest(args.cpu / "decisions.jsonl"),
            "probability_comparison": "all12 selected IDs and max absolute probability error",
            "predeclared_agreement_gate": {"all_choices_equal": True,
                                         "max_abs_probability_error_at_most": 0.01},
            "gate_caveat": "study engineering tolerance, not vendor numerical guarantee"}
    path = args.output / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Existing calibration plan differs; refusing overwrite")
    else:
        write_new(path, plan)
    return tokenizer, encoder, rows, cpu, plan


async def live(args, tokenizer, encoder, rows, cpu, plan):
    if (args.output / "results.jsonl").exists() or (args.output / "report.json").exists():
        raise ValueError("No automatic resume, repeats or overwrite")
    bridge = SemIfVLLMBridge(tokenizer, encoder, max_generated=args.max_generated,
                            scoring_mode=args.scoring_mode)
    comparisons, failures = [], []
    try:
        await bridge.initialize()
        with (args.output / "results.jsonl").open("x") as output:
            for row in rows:
                started = time.perf_counter()
                try:
                    answer = await bridge.predict(decision_request(row))
                    original = cpu[row["id"]]["upstream"]
                    cpu_probs = dict(zip(original["option_ids"], original["probabilities"]))
                    error = max(abs(answer.probabilities[k] - cpu_probs[k]) for k in cpu_probs)
                    record = {"id": row["id"], "status": "valid",
                              "choice": answer.selected_option_id,
                              "cpu_choice": cpu[row["id"]]["choice"],
                              "choices_equal": answer.selected_option_id == cpu[row["id"]]["choice"],
                              "max_abs_probability_error": error,
                              "cpu_option_logits": original["option_logits"],
                              "cpu_probabilities": cpu_probs, "response": answer.raw_output}
                    comparisons.append(record)
                except Exception as exc:
                    record = {"id": row["id"], "status": "failed",
                              "error_type": type(exc).__name__,
                              "http_status": getattr(exc, "status_code", None)}
                    failures.append(record)
                record["elapsed_seconds"] = time.perf_counter() - started
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                if record["status"] != "valid":
                    break
    except Exception as exc:
        failures.append({"phase": "initialization", "error_type": type(exc).__name__,
                         "http_status": getattr(exc, "status_code", None)})
    finally:
        await bridge.aclose()
    errors = [r["max_abs_probability_error"] for r in comparisons]
    accounting = [r["response"]["bridge"] for r in comparisons]
    attempted = len(comparisons) + sum("id" in failure for failure in failures)
    report = {"scheduled": 12, "attempted": attempted, "unattempted": 12 - attempted,
              "valid": len(comparisons), "failures": failures,
              "calibration_complete": len(comparisons) == 12,
              "matching_choices": sum(r["choices_equal"] for r in comparisons),
              "max_abs_probability_error": max(errors) if errors else None,
              "passes_predeclared_gate": len(comparisons) == 12
              and all(r["choices_equal"] for r in comparisons) and max(errors) <= 0.01,
              "valid_response_usage": {
                  "candidate_prompts": sum(r["candidate_prompts"] for r in accounting),
                  "candidate_scores": sum(r["candidate_count"] for r in accounting),
                  "prefill_tokens": sum(r["prefill_tokens"] for r in accounting),
                  "generated_tokens": sum(r["generated_tokens"] for r in accounting)},
              "failed_request_usage": "unknown; never inferred as zero",
              "no_generated_token_used_for_choice": True,
              "scoring_mode": args.scoring_mode,
              "same_scoring_rule_not_same_execution_cost": True}
    write_new(args.output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["calibration_complete"] else 2


async def readonly_preflight(output):
    from scripts import paired_graph_ablation as shared
    config = shared.model_configs(models=("qwen4",))["qwen4"]
    try:
        await shared.vllm_preflight([SimpleNamespace(model=config)])
        result = {"status": "passed", "inference_calls": 0,
                  "reservation": False, "note": "GET-only snapshot; not a lane reservation"}
    except Exception as exc:
        result = {"status": "blocked", "inference_calls": 0,
                  "error_type": type(exc).__name__,
                  "message": str(exc) if isinstance(exc, shared.Refusal)
                  else "GET-only preflight could not reach or validate the local endpoint"}
    write_new(output / "readonly-preflight.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, default=DEFAULT_CPU)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-generated", type=int, choices=(0, 1), default=0,
                        help="Teacher-forced fallback only; requested mode generates exactly one")
    parser.add_argument("--scoring-mode", choices=("requested", "teacher_forced"),
                        default="requested")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--readonly-preflight", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    tokenizer, encoder, rows, cpu, plan = prepare(args)
    if args.live:
        return asyncio.run(live(args, tokenizer, encoder, rows, cpu, plan))
    if args.readonly_preflight:
        asyncio.run(readonly_preflight(args.output))
    print(json.dumps({"status": "prepared_not_calibrated", "cases": len(rows),
                      "candidate_prompts": plan["candidate_prompts"],
                      "logical_prefill_tokens": plan["logical_prefill_tokens"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
