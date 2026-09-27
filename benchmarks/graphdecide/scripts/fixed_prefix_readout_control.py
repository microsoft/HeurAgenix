#!/usr/bin/env python3
"""Compare candidate argmax and masked greedy decoding on identical token prefixes."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import httpx

from scripts import evaluate_graph_release as evaluation
from scripts import extended_graph_suite as suite
from scripts import graph_release_api as release
from scripts import semif_vllm_bridge as bridge

ARMS = ("scores", "scores_repeat", "masked_greedy")


def payload_for(prefix, slots, request_id, arm):
    suite.require(arm in ARMS, "Unknown fixed-prefix arm.")
    payload = bridge.make_requested_payload(prefix, slots, request_id)
    payload["seed"] = 20261014
    if arm == "masked_greedy":
        payload.update(temperature=0.0, allowed_token_ids=list(slots))
    return payload


def parse_arm(body, request, payload, arm):
    scores = bridge.parse_requested_response(body, request, payload)
    generated = body["choices"][0]["token_ids"][0]
    slots = payload["logprob_token_ids"]
    decoded_action = None
    if arm == "masked_greedy":
        suite.require(generated in slots, "Greedy output escaped the candidate-token mask.")
        decoded_action = request.options[slots.index(generated)].id
        values = scores["candidate_logprobs"]
        suite.require(values[slots.index(generated)] == max(values),
                      "Greedy output is not a maximum of its own candidate scores.")
    return {key: scores[key] for key in (
        "candidate_logprobs", "probabilities", "option_ids", "selected_option_id",
        "prefill_tokens", "generated_tokens")} | {
            "generated_token_id": generated, "generated_action": decoded_action,
            "generated_token_used_for_decision": arm == "masked_greedy",
            "resolved_model": body["model"],
        }


def verify_result(case, row):
    result = row["result"]
    ids = [o["id"] for o in case["request"]["options"]]
    values = result["candidate_logprobs"]
    suite.require(result["resolved_model"] == bridge.MODEL_ID
                  and result["option_ids"] == ids and len(values) == len(ids)
                  and all(type(v) in (int, float) and math.isfinite(v) and -9999 < v <= 1e-6
                          for v in values), "Stored candidate scores or model differ.")
    weights = [math.exp(v - max(values)) for v in values]
    probabilities = {key: weight / sum(weights) for key, weight in zip(ids, weights)}
    suite.require(set(result["probabilities"]) == set(ids) and all(
        math.isclose(result["probabilities"][key], value, abs_tol=1e-12, rel_tol=0)
        for key, value in probabilities.items()), "Stored conditional probabilities differ.")
    suite.require(result["selected_option_id"] == max(ids, key=probabilities.__getitem__)
                  and result["prefill_tokens"] == len(case["prefix"])
                  and result["generated_tokens"] == 1, "Stored argmax or token accounting differs.")
    generated = result["generated_token_id"]
    suite.require(type(generated) is int and generated >= 0, "Invalid generated token.")
    masked = row["arm"] == "masked_greedy"
    suite.require(result["generated_token_used_for_decision"] is masked,
                  "Generated-token accounting differs.")
    if masked:
        suite.require(generated in case["slots"], "Stored greedy token escapes mask.")
        index = case["slots"].index(generated)
        suite.require(result["generated_action"] == ids[index] and values[index] == max(values),
                      "Stored greedy action is not the emitted maximum-score candidate.")
    else:
        suite.require(result["generated_action"] is None, "Unused token became an action.")


def pair_result(case, rows):
    suite.require(len(rows) == 3 and {r["arm"] for r in rows} == set(ARMS),
                  "A paired readout control needs all three arms.")
    by_arm = {r["arm"]: r for r in rows}
    if any(r["status"] != "success" for r in rows):
        return {"case_id": case["id"], "task": case["task"], "complete": False}
    a, repeat, greedy = (by_arm[k]["result"] for k in ARMS)
    ids = [o["id"] for o in case["request"]["options"]]
    suite.require(a["option_ids"] == repeat["option_ids"] == greedy["option_ids"] == ids,
                  "Option mapping changed between readouts.")
    index = ids.index(greedy["generated_action"])
    values = a["candidate_logprobs"]
    return {
        "case_id": case["id"], "task": case["task"], "complete": True,
        "choice_agreement": a["selected_option_id"] == greedy["generated_action"],
        "greedy_in_original_argmax_set": values[index] == max(values),
        "repeat_choice_agreement": a["selected_option_id"] == repeat["selected_option_id"],
        "max_probability_difference": max(abs(a["probabilities"][key] - greedy["probabilities"][key])
                                           for key in ids),
        "repeat_max_probability_difference": max(abs(a["probabilities"][key] - repeat["probabilities"][key])
                                                  for key in ids),
    }


def register(bundle, output):
    suite.require(not output.exists(), "Control output already exists.")
    release.verify(bundle)
    manifest = release.manifest_at(bundle)
    instances = release.labeled_instances(bundle, "structural_challenge", manifest)
    tokenizer, encoder = bridge.upstream_functions(bridge.DEFAULT_CPU, bridge.DEFAULT_MODEL)
    cases = []
    for task in sorted({i["task"] for i in instances}):
        for size in range(8, 13):
            selected = sorted((i for i in instances if i["task"] == task
                               and len(i["state"]["nodes"]) == size), key=lambda i: i["id"])[:2]
            suite.require(len(selected) == 2, "Fixed control quota is not available.")
            for instance in selected:
                request = evaluation.public_request(instance, [])
                prefix, slots, prompt_hash = encoder(tokenizer, bridge.row_for(request), 4096)
                suite.require(len(slots) == len(request.options), "Candidate count mismatch.")
                cases.append({"id": instance["id"], "task": task, "node_count": size,
                              "request": suite.json_value(asdict(request)), "prefix": prefix,
                              "slots": slots, "prompt_sha256": prompt_hash})
    suite.require(len(cases) == 100, "Expected ten tasks, five sizes, two initial states.")
    output.mkdir(parents=True)
    registration = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle.resolve()), "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "protocol": "fixed-prefix-readout-control-v1", "cases": cases,
        "max_calls": 300, "arms": ARMS, "seed": 20261014,
        "selection": "Two opaque-ID-first public initial states per task and size, no model-result selection.",
        "hypothesis": "Argmax and candidate-masked greedy should agree for identical logits; "
                      "ties and observed numerical/readout differences are retained.",
        "schedule": "Rotate the three arm orders by case index; no retries or repairs.",
        "scope": "Readout mechanism check at initial states, not an additional full-solution score "
                 "and not certified CPU SemIf reproduction.",
        "code_sha256": {**evaluation.code_hashes(),
                       "scripts/fixed_prefix_readout_control.py": suite.digest(__file__)},
    }
    suite.write_json(output / "manifest.json", registration)
    suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
    suite.write_json(output / "status.json", {"status": "registered", "calls": 0})
    return {"cases": len(cases), "max_calls": 300}


def load(output):
    manifest = suite.read_json(output / "manifest.json")
    suite.require(suite.digest(output / "manifest.json") == suite.read_json(output / "manifest.sha256"),
                  "Readout registration drift.")
    suite.require(all(suite.digest(REPO / name) == digest
                      for name, digest in manifest["code_sha256"].items()), "Readout source drift.")
    suite.require(suite.digest(Path(manifest["bundle"]) / "manifest.json")
                  == manifest["bundle_manifest_sha256"], "Readout bundle drift.")
    return manifest


def summarize(output):
    manifest = load(output)
    rows = suite.read_rows(output / "attempts.jsonl")
    suite.require(len(rows) == manifest["max_calls"], "Readout control is incomplete.")
    pairs = []
    for index, case in enumerate(manifest["cases"]):
        own = rows[3 * index:3 * index + 3]
        expected = list(ARMS[index % 3:] + ARMS[:index % 3])
        suite.require([row["arm"] for row in own] == expected
                      and all(row["case_id"] == case["id"] for row in own),
                      "Readout schedule differs from registration.")
        for row in own:
            suite.require(row["payload"] == payload_for(
                case["prefix"], case["slots"], f"paper-fixed-{index}-{row['arm']}", row["arm"]),
                "Readout token prefixes or settings differ.")
            suite.require(row["status"] in {"success", "failure"}, "Unknown attempt status.")
            if row["status"] == "success":
                verify_result(case, row)
        pairs.append(pair_result(case, own))
    complete = [p for p in pairs if p["complete"]]
    report = {
        "scheduled_cases": len(pairs), "calls": len(rows), "complete_cases": len(complete),
        "failed_calls": sum(r["status"] != "success" for r in rows),
        "choice_agreement": sum(p["choice_agreement"] for p in complete),
        "greedy_in_original_argmax_set": sum(p["greedy_in_original_argmax_set"] for p in complete),
        "repeat_choice_agreement": sum(p["repeat_choice_agreement"] for p in complete),
        "maximum_probability_difference": max((p["max_probability_difference"] for p in complete),
                                               default=None),
        "maximum_repeat_probability_difference": max((p["repeat_max_probability_difference"] for p in complete),
                                                      default=None),
        "pairs": pairs, "scope": manifest["scope"],
    }
    return report


async def run(output):
    manifest = load(output)
    suite.require(not (output / "attempts.jsonl").exists(), "No readout control resume.")
    config = suite.shared.model_configs(models=("qwen4",))["qwen4"]
    count = 0
    with suite.shared.lane_locks(["qwen4"]):
        await suite.shared.vllm_preflight([SimpleNamespace(model=config)])
        async with httpx.AsyncClient(timeout=120, trust_env=False, follow_redirects=False) as http:
            schema = await http.get(bridge.BASE_URL.removesuffix("/v1") + "/openapi.json")
            schema.raise_for_status()
            properties = schema.json()["components"]["schemas"]["CompletionRequest"]["properties"]
            suite.require("allowed_token_ids" in properties, "Server does not advertise candidate masking.")
            suite.write_json(output / "server-capability.json", {
                "model": config.model, "allowed_token_ids": properties["allowed_token_ids"],
                "openapi_sha256": hashlib.sha256(schema.content).hexdigest(),
            })
            suite.set_status(output, {"status": "running", "calls": count})
            with (output / "attempts.jsonl").open("x") as stream:
                for index, case in enumerate(manifest["cases"]):
                    request = bridge.decision_request({"id": case["request"]["request_id"],
                        **{key: case["request"][key] for key in ("state", "question", "options")}})
                    for arm in ARMS[index % 3:] + ARMS[:index % 3]:
                        payload = payload_for(case["prefix"], case["slots"],
                                              f"paper-fixed-{index}-{arm}", arm)
                        started = time.perf_counter()
                        try:
                            response = await http.post(bridge.BASE_URL + "/completions", json=payload)
                            response.raise_for_status()
                            result = parse_arm(response.json(), request, payload, arm)
                            record = {"status": "success", "result": result}
                        except (httpx.HTTPError, ValueError, suite.Refusal, bridge.InvalidResponseError) as exc:
                            record = {"status": "failure", "error_type": type(exc).__name__,
                                      "diagnostic": str(exc)}
                        suite.append_row(stream, {
                            "case_id": case["id"], "arm": arm, "payload": payload,
                            "latency_seconds": time.perf_counter() - started, **record,
                        })
                        count += 1
                        suite.set_status(output, {"status": "running", "calls": count})
    report = summarize(output)
    suite.write_json(output / "summary.json", report)
    suite.set_status(output, {"status": "completed_with_failures" if report["failed_calls"] else "completed",
                              "calls": count, "artifact_sha256": {
                                  name: suite.digest(output / name)
                                  for name in ("attempts.jsonl", "summary.json", "manifest.json")}})
    return {key: value for key, value in report.items() if key != "pairs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        suite.require(args.bundle is not None, "Registration needs a bundle.")
        result = register(args.bundle, args.output)
    elif args.command == "run":
        try:
            result = asyncio.run(run(args.output))
        except BaseException as exc:
            suite.set_status(args.output, {"status": "aborted", "error_type": type(exc).__name__,
                                          "diagnostic": str(exc)})
            raise
    else:
        status = suite.read_json(args.output / "status.json")
        suite.require(status["status"] in {"completed", "completed_with_failures"}, "Unsealed readout run.")
        suite.require(all(suite.digest(args.output / name) == digest
                          for name, digest in status["artifact_sha256"].items()), "Sealed artifact drift.")
        result = summarize(args.output)
        suite.require(result == suite.read_json(args.output / "summary.json"), "Readout summary differs.")
        result = {"status": "verified", "calls": result["calls"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
