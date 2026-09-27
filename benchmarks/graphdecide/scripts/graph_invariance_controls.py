#!/usr/bin/env python3
"""Preregistered exact-task relabeling, presentation and repeat controls."""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter, defaultdict
from contextlib import nullcontext
from copy import deepcopy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import networkx as nx
from scripts import evaluate_graph_release as canonical
from scripts import extended_graph_suite as suite
from scripts import graph_release_api as release
from scripts import native_choice_contract_audit as native
from scripts import summarize_release_matrix as matrix
from src.clients.base import DecisionOption, DecisionRequest, InvalidResponseError
from src.clients.registry import create_client

TASKS = ("adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
         "distance_threshold", "articulation_point")
SIZES = tuple(range(8, 13))
ARMS = ("original", "identical_repeat", "node_relabel", "edge_shuffle", "reversed_options")
PROFILES = tuple(canonical.PROFILES)
SEED = 20261013
SPLIT = "structural_challenge"
DEFAULT_BUNDLE = REPO / "output/paper-confirmatory-20260924-v1/bundle"
DEFAULT_ROOT = REPO / "output/graph-invariance-controls-20260924-v1"
PY311 = "/home/xianya/miniconda3/envs/jev/bin/python"
PY312 = "/fastdata/xianya/vllm-venv/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_hashes():
    """Static local import closure, with explicit roots for dynamically loaded clients."""
    pending = ["scripts.graph_invariance_controls", "src.clients.typesafe",
               "src.clients.vllm_client", "scripts.semif_vllm_bridge"]
    seen, paths = set(), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        base = REPO.joinpath(*name.split("."))
        path = base.with_suffix(".py")
        if not path.is_file():
            path = base / "__init__.py"
        if not path.is_file():
            if name == "graph_task_prompt":
                pending.append("scripts.graph_task_prompt")
            continue
        paths.add(path)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for parent in range(1, len(name.split("."))):
            pending.append(".".join(name.split(".")[:parent]))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name("." * node.level + module, package)
                pending.append(module)
                pending.extend(f"{module}.{alias.name}" for alias in node.names
                               if alias.name != "*")
    return {str(p.relative_to(REPO)): suite.digest(p) for p in sorted(paths)}


def truth(task, state):
    """Independent NetworkX oracle; no benchmark answer or score function."""
    require(task in TASKS and state["directed"] is False, "Expected exact undirected task.")
    nodes, edges = state["nodes"], state["edges"]
    require(set(nodes) == set(range(len(nodes))) and len(set(nodes)) == len(nodes),
            "Invalid node set.")
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    require(all(len(e) == 2 and e[0] in graph and e[1] in graph and e[0] != e[1]
                for e in edges), "Invalid simple edge.")
    graph.add_edges_from(edges)
    require(graph.number_of_edges() == len(edges), "Duplicate undirected edge.")
    if task == "degree_exact":
        return str(graph.degree[state["vertex"]])
    if task == "adjacency":
        answer = graph.has_edge(*state["pair"])
    elif task == "pair_connectivity":
        answer = nx.has_path(graph, *state["pair"])
    elif task == "cycle_detection":
        answer = bool(nx.cycle_basis(graph))
    elif task == "distance_threshold":
        u, v = state["pair"]
        answer = nx.has_path(graph, u, v) and nx.shortest_path_length(graph, u, v) <= state["threshold"]
    else:
        before = nx.number_connected_components(graph)
        graph.remove_node(state["vertex"])
        answer = nx.number_connected_components(graph) > before
    return "yes" if answer else "no"


def select_cases(public, metadata):
    require(len({i["id"] for i in public}) == len(public), "Duplicate public ID.")
    selected = []
    for task in TASKS:
        for n in SIZES:
            pool = [i for i in public if i["task"] == task and i["kind"] == "exact"
                    and len(i["state"]["nodes"]) == n]
            if task == "degree_exact":
                for degree in [k * (n - 1) // 3 for k in range(4)]:
                    candidates = sorted((i for i in pool if truth(task, i["state"]) == str(degree)),
                                        key=lambda i: i["id"])
                    require(bool(candidates), "Missing selected degree stratum.")
                    selected.append(candidates[0])
            else:
                pairs = defaultdict(list)
                for instance in pool:
                    pair = metadata["cases"][instance["id"]].get("pair_id")
                    if pair:
                        pairs[pair].append(instance)
                complete = sorted(pair for pair, cases in pairs.items()
                                  if len(cases) == 2 and
                                  {truth(task, i["state"]) for i in cases} == {"yes", "no"})
                require(len(complete) >= 2, "Missing two complete binary pairs.")
                for pair in complete[:2]:
                    selected.extend(sorted(pairs[pair], key=lambda i: i["id"]))
    require(len(selected) == len(TASKS) * len(SIZES) * 4
            and len({i["id"] for i in selected}) == len(selected), "Selection size differs.")
    return selected


def transform(instance, graph_id, arm):
    require(arm in ARMS, "Unknown intervention.")
    public = {key: deepcopy(instance[key]) for key in release.PUBLIC_KEYS}
    state = public["state"]
    details = {}
    if arm == "node_relabel":
        order = list(range(len(state["nodes"])))
        random.Random(f"{SEED}:{graph_id}:relabel").shuffle(order)
        mapping = dict(enumerate(order))
        state["nodes"] = sorted(mapping[v] for v in state["nodes"])
        state["edges"] = [[mapping[u], mapping[v]] for u, v in state["edges"]]
        if "pair" in state:
            state["pair"] = [mapping[v] for v in state["pair"]]
        if "vertex" in state:
            state["vertex"] = mapping[state["vertex"]]
        details["node_mapping"] = {str(k): v for k, v in mapping.items()}
    elif arm == "edge_shuffle":
        order = list(range(len(state["edges"])))
        random.Random(f"{SEED}:{graph_id}:edge-order").shuffle(order)
        state["edges"] = [state["edges"][index] for index in order]
        details["edge_index_order"] = order
    request = canonical.public_request(public, [])
    if arm == "reversed_options":
        request = replace(request, options=tuple(reversed(request.options)))
    return request, details


def plan_inputs(bundle):
    manifest = release.manifest_at(bundle)
    public = release.load_public(bundle, SPLIT, manifest)
    metadata = suite.read_json(release.checked_file(bundle, f"analysis/{SPLIT}.json", manifest))
    selected = select_cases(public, metadata)
    references = suite.read_rows(release.checked_file(bundle, f"references/{SPLIT}.jsonl", manifest))
    answers = {i["id"]: str(i["private"]["answer"]) for i in references
               if "answer" in i["private"]}
    cases, requests, interventions = [], [], []
    for instance in selected:
        ident, task = instance["id"], instance["task"]
        meta = metadata["cases"][ident]
        label = truth(task, instance["state"])
        require(label == answers[ident], "Independent truth differs from frozen reference.")
        graph_id = meta["cluster_id"]
        original = suite.json_value(asdict(transform(instance, graph_id, "original")[0]))
        cases.append({"instance_id": ident, "task": task, "node_count": len(instance["state"]["nodes"]),
                      "label": label, "cluster_id": graph_id, "pair_id": meta.get("pair_id")})
        for arm in ARMS:
            request, details = transform(instance, graph_id, arm)
            serialized = suite.json_value(asdict(request))
            require(truth(task, request.state) == label, "Transformation changed the label.")
            require(label in {o.id for o in request.options}, "Answer missing from supplied options.")
            require(request.request_id == original["request_id"], "Request ID changed.")
            if arm in ("original", "identical_repeat"):
                require(serialized == original, "Repeat is not byte-identical canonical request.")
            requests.append({"instance_id": ident, "arm": arm, "request": serialized,
                             "input_changed": serialized != original,
                             "state_changed": serialized["state"] != original["state"]})
            interventions.append({"instance_id": ident, "arm": arm, "cluster_id": graph_id,
                                  "input_changed": serialized != original, **details})
    return cases, requests, interventions


def backend_pins():
    paths = [
        "output/semif-vllm-calibration-20260924-v1/report.json",
        "output/semif-vllm-methodcheck-20260924-v1/independent-parser-check.json",
        "output/semif-cpu-pilot-20260924-v1/upstream/source-manifest.json",
        "output/semif-cpu-pilot-20260924-v1/provenance.json",
    ]
    return {p: suite.digest(REPO / p) for p in paths}


def write_rows(path, rows):
    with path.open("x") as stream:
        for row in rows:
            suite.append_row(stream, row)


def freeze_sources(root, hashes):
    snapshot = root / "source-snapshot"
    snapshot.mkdir()
    for name, expected in hashes.items():
        source, target = REPO / name, snapshot / name
        require(not source.is_symlink() and suite.digest(source) == expected, "Source drift while freezing.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        target.chmod(0o444)
        require(suite.digest(target) == expected, "Frozen source differs.")


def prepare(root, bundle):
    require(not (root / "registration.json").exists(), "Registration exists; no overwrite.")
    before = source_hashes()
    verified = release.verify(bundle)
    cases, requests, interventions = plan_inputs(bundle)
    require(len(cases) == 120 and len(requests) == 600, "Finite protocol cardinality differs.")
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "cases.jsonl", cases)
    write_rows(root / "requests.jsonl", requests)
    write_rows(root / "interventions.jsonl", interventions)
    source = source_hashes()
    require(source == before, "Relevant source changed during preparation.")
    freeze_sources(root, source)
    registration = {
        "schema_version": 1, "registered_at_utc": now(), "registered_before_inference": True,
        "bundle": str(bundle.resolve()), "split": SPLIT,
        "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "selection": "Only public cases and analysis metadata; first two COMPLETE binary pairs "
                     "by lexicographic opaque pair ID in each task/size, members by opaque case ID. "
                     "Degree: first opaque case ID for floor(k*(n-1)/3), k=0..3. Independent "
                     "NetworkX labels determine eligibility; frozen references cross-check only. "
                     "No completed model output file is read.",
        "tasks": list(TASKS), "sizes": list(SIZES), "cases": len(cases), "arms": list(ARMS),
        "calls_per_profile": len(requests), "total_calls": len(requests) * len(PROFILES),
        "profiles": {p: asdict(canonical.configuration(p)[0]) for p in PROFILES},
        "protocol": "explicit_actions", "transform_seed": SEED,
        "order": "Fixed task/size/selected-case order, then original, adjacent identical repeat, "
                 "node relabel, edge shuffle, reversed options. Same order for every profile.",
        "transformations": "Seed by shared graph cluster_id and intervention. Relabel nodes/queries "
                           "and endpoints consistently; preserve edge-list positions and endpoint "
                           "slots, retain canonical sorted node list. Degree IDs remain counts. "
                           "Shuffle edge positions only. Reverse whole option records without "
                           "changing IDs/descriptions. Never resample identity transforms or cases.",
        "input_changed": "Exact canonical DecisionRequest comparison against original, same request "
                         "ID for all arms; unchanged interventions are explicitly retained. "
                         "A graph automorphism may preserve topology even if serialization changes.",
        "hypotheses": {task: "Supplied semantic choice IDs and exact correctness are invariant "
                            "to each label-preserving transformation, conditional on both calls "
                            "valid; compare with identical-repeat variability." for task in TASKS},
        "analysis": "All-scheduled per-task/profile/arm accuracy and valid counts; original-versus-"
                    "each-arm choice flips with both-valid denominators, also stratified by actual "
                    "input_changed. Report robust-correct, invariant-wrong, correct-to-wrong, "
                    "wrong-to-correct, and wrong-to-other-wrong. Report five-arm joint patterns. "
                    "No pooled cross-task winner, hypothesis rejection or confidence intervals. "
                    "20 cases/task, four/size; paired roots and repeated graph queries are dependent.",
        "interpretation": "Finite descriptive robustness diagnostic on previously evaluated graphs, "
                          "not new task generalization. Fixed order, repeated IDs and shared serving "
                          "state/cache preclude attributing every flip causally to the intervention.",
        "native_policy": "Existing supplied-action eligibility; strict-native sum/argmax observations "
                         "separate, no normalization, choice substitution or probability repair.",
        "token_backend": "Existing requested-token bridge, exact candidate argmax, one generated "
                         "token ignored; distinct GPU reference, failed CPU numerical gate unchanged.",
        "resource_policy": "Existing physical locks, Jev/9B/4B grammar may parallel; 4B token follows "
                           "4B grammar, bridge alone acquires qwen4 lock. No namespace override, "
                           "endpoint changes, new weights or interaction with Astra/GPT work.",
        "failure_policy": "No retry/resume/repair. Nonfatal failure recorded then next request; "
                          "fatal failure stops profile. Unattempted cells remain in all denominators.",
        "source_sha256": source,
        "dependency_policy": "Explicit AST local import closure rooted at this script, Typesafe, "
                             "vLLM and SemIf bridge, including package initializers; no script glob.",
        "backend_prerequisite_sha256": backend_pins(),
        "input_sha256": {name: suite.digest(root / name) for name in
                         ("cases.jsonl", "requests.jsonl", "interventions.jsonl")},
        "validation": {**verified, "transformed_requests_independently_checked": len(requests),
                       "all_reference_labels_agree": True},
    }
    suite.write_json(root / "registration.json", registration)
    suite.write_json(root / "registration.sha256", suite.digest(root / "registration.json"))
    (root / "registration.json").chmod(0o444)
    return {"status": "prepared", "cases": len(cases), "requests_per_profile": len(requests),
            "total_calls": len(requests) * len(PROFILES), "model_calls": 0}


def verify(root):
    registration = suite.read_json(root / "registration.json")
    require(suite.digest(root / "registration.json") == suite.read_json(root / "registration.sha256"),
            "Registration hash drift.")
    require(registration["source_sha256"] == source_hashes(), "Relevant source hash drift.")
    for name, expected in registration["source_sha256"].items():
        frozen = root / "source-snapshot" / name
        require(frozen.is_file() and not frozen.is_symlink() and not frozen.stat().st_mode & 0o222
                and suite.digest(frozen) == expected, "Frozen source snapshot drift.")
    require(registration["backend_prerequisite_sha256"] == backend_pins(), "Backend prerequisite drift.")
    require(registration["profiles"] == {p: asdict(canonical.configuration(p)[0]) for p in PROFILES},
            "Canonical profile configuration drift.")
    require(all(suite.digest(root / name) == sha for name, sha in registration["input_sha256"].items()),
            "Prepared input hash drift.")
    bundle = Path(registration["bundle"])
    require(suite.digest(bundle / "manifest.json") == registration["bundle_manifest_sha256"],
            "Bundle manifest drift.")
    release.verify(bundle)
    rebuilt = plan_inputs(bundle)
    for name, expected in zip(("cases.jsonl", "requests.jsonl", "interventions.jsonl"), rebuilt):
        require(suite.read_rows(root / name) == expected, f"Prepared {name} replay differs.")
    return registration, rebuilt[0], rebuilt[1]


def request_from(row):
    data = row["request"]
    return DecisionRequest(**{**data, "options": tuple(DecisionOption(**o) for o in data["options"])})


def scores_for(cases, requests, attempts, profile):
    require(len(attempts) <= len(requests), "More attempts than scheduled.")
    config, _ = canonical.configuration(profile)
    cases_by_id = {c["instance_id"]: c for c in cases}
    scores = []
    for index, planned in enumerate(requests):
        case = cases_by_id[planned["instance_id"]]
        attempt = attempts[index] if index < len(attempts) else None
        if attempt is not None:
            require(attempt["instance_id"] == planned["instance_id"] and
                    attempt["arm"] == planned["arm"] and attempt["input_changed"] == planned["input_changed"],
                    "Attempt identity/order/intervention mismatch.")
            standard = {key: attempt[key] for key in (
                "instance_id", "step", "request", "status", "selected_option_id", "resolved_model",
                "usage", "diagnostic_code", "http_status", "fatal", "latency_seconds")}
            suite.validate_attempt(standard, request_from(planned), config)
            require(attempt["step"] == 0, "Exact task has only step zero.")
            require(not any(a["fatal"] for a in attempts[:index]), "Attempt continued after fatal failure.")
            if profile == "jev_action":
                observation = attempt["action_observation"]
                require(attempt["strict_status"] in ("success", "failure"), "Missing strict status.")
                require((attempt["status"] == "success") == observation["eligible"],
                        "Native action eligibility mismatch.")
                if observation["eligible"]:
                    require(not observation["ineligibility_reasons"] and
                            observation["selected_option_id"] == attempt["selected_option_id"],
                            "Native choice mismatch.")
            if profile == "qwen4_token_scores" and attempt["status"] == "success":
                matrix.verify_token_scores(attempt)
        valid = attempt is not None and attempt["status"] == "success"
        choice = attempt["selected_option_id"] if valid else None
        scores.append({**case, "arm": planned["arm"], "input_changed": planned["input_changed"],
                       "attempted": attempt is not None, "valid": valid,
                       "selected_option_id": choice, "correct": valid and choice == case["label"],
                       "failure_code": None if valid else (
                           attempt["diagnostic_code"] if attempt else "not_attempted")})
    return scores


def metrics(scores):
    cells, contrasts, joint = [], [], []
    for task in TASKS:
        own = [s for s in scores if s["task"] == task]
        if not own:
            continue
        lookup = {(s["instance_id"], s["arm"]): s for s in own}
        ids = sorted({s["instance_id"] for s in own})
        for size in ("all", *SIZES):
            for arm in ARMS:
                group = [s for s in own if s["arm"] == arm
                         and (size == "all" or s["node_count"] == size)]
                if not group:
                    continue
                cells.append({
                    "task": task, "node_count": size, "arm": arm, "scheduled": len(group),
                    "attempted": sum(s["attempted"] for s in group),
                    "valid": sum(s["valid"] for s in group),
                    "correct": sum(s["correct"] for s in group),
                    "accuracy_all_scheduled": sum(s["correct"] for s in group) / len(group),
                    "input_changed": sum(s["input_changed"] for s in group),
                })
        for arm in ARMS[1:]:
            paired = [(lookup[i, "original"], lookup[i, arm]) for i in ids]
            for changed in ("all", True, False):
                eligible = [(a, b) for a, b in paired if changed == "all" or b["input_changed"] == changed]
                both = [(a, b) for a, b in eligible if a["valid"] and b["valid"]]
                same = [(a, b) for a, b in both if a["selected_option_id"] == b["selected_option_id"]]
                flips = len(both) - len(same)
                contrasts.append({
                    "task": task, "arm": arm, "actual_input_changed": changed,
                    "scheduled_pairs": len(eligible), "both_valid": len(both),
                    "not_both_valid": len(eligible) - len(both),
                    "choice_flips": flips, "flip_rate_both_valid": flips / len(both) if both else None,
                    "robust_correct": sum(a["correct"] and b["correct"] for a, b in both),
                    "invariant_wrong": sum(not a["correct"] and not b["correct"] for a, b in same),
                    "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in both),
                    "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in both),
                    "wrong_to_other_wrong": sum(not a["correct"] and not b["correct"]
                                               and a["selected_option_id"] != b["selected_option_id"]
                                               for a, b in both),
                })
        groups = [[lookup[i, arm] for arm in ARMS] for i in ids]
        valid_groups = [g for g in groups if all(s["valid"] for s in g)]
        joint.append({
            "task": task, "scheduled_cases": len(groups), "all_five_valid": len(valid_groups),
            "all_five_correct": sum(all(s["correct"] for s in g) for g in valid_groups),
            "all_five_same_choice": sum(len({s["selected_option_id"] for s in g}) == 1 for g in valid_groups),
            "all_five_invariant_wrong": sum(not any(s["correct"] for s in g)
                                           and len({s["selected_option_id"] for s in g}) == 1
                                           for g in valid_groups),
        })
    return {"accuracy": cells, "paired_choice_changes": contrasts, "five_arm_patterns": joint}


def token_factory(requests):
    from scripts import semif_vllm_bridge as bridge
    calibration = suite.read_json(bridge.DEFAULT_OUTPUT / "report.json")
    checked = suite.read_json(REPO / "output/semif-vllm-methodcheck-20260924-v1/independent-parser-check.json")
    require(calibration["valid"] == calibration["matching_choices"] == 12
            and checked["independent_token_index_or_probability_mismatches"] == 0,
            "Token backend calibration/parser prerequisite failed.")
    tokenizer, encoder = bridge.upstream_functions(bridge.DEFAULT_CPU, bridge.DEFAULT_MODEL)
    maximum = max(len(encoder(tokenizer, bridge.row_for(request_from(r)), 4096)[0]) for r in requests)
    metadata = {"scoring_mode": "requested", "maximum_prefix_tokens": maximum,
                "max_prefix_tokens": 4096, "upstream_revision": bridge.SOURCE_REVISION,
                "declared_model_revision": bridge.MODEL_REVISION, "generated_token_used_for_decision": False,
                "cpu_numerical_gate": "failed; distinct GPU scoring reference"}
    return lambda: bridge.SemIfVLLMBridge(
        tokenizer, encoder, max_prefix_tokens=4096, scoring_mode="requested", timeout=120.0), metadata


async def run_profile(root, profile, *, client_factory=create_client,
                      preflight=suite.shared.vllm_preflight, lock_factory=suite.lane_locks):
    registration, cases, requests = verify(root)
    require(profile in PROFILES, "Unknown profile.")
    output = root / profile
    require(not output.exists(), "Profile output exists; no retry/resume/overwrite.")
    config, resource = canonical.configuration(profile)
    injected = client_factory is not create_client
    factory = None
    backend = ({"scoring_mode": "requested", "preprocessing_accounting": "backend-initialization.json"}
               if profile == "qwen4_token_scores" else {})
    output.mkdir()
    manifest = {
        "profile": profile, "model": asdict(config), "physical_resource": resource,
        "registration_sha256": suite.digest(root / "registration.json"),
        "source_sha256": registration["source_sha256"], "backend": backend,
        "environment": suite.environment_versions(), "injected_client": injected,
        "planned_calls": len(requests), "retry": False, "repair": False,
    }
    suite.write_json(output / "manifest.json", manifest)
    suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
    status = {"status": "running", "started_at_utc": now(), "diagnostic_code": None}
    suite.set_status(output, status)
    attempts, client, interrupted = [], None, None
    started = time.perf_counter()
    with (output / "attempts.jsonl").open("x") as stream:
        try:
            if profile == "qwen4_token_scores" and not injected:
                factory, initialized_backend = token_factory(requests)
                suite.write_json(output / "backend-initialization.json", initialized_backend)
            guard = nullcontext() if profile == "qwen4_token_scores" else lock_factory((resource,))
            with guard:
                await preflight([SimpleNamespace(model=config)])
                client = factory() if factory else client_factory(config.provider, **config.client_kwargs())
                try:
                    await client.initialize()
                    require(source_hashes() == registration["source_sha256"], "Source changed before inference.")
                    for planned in requests:
                        request, body = request_from(planned), None
                        tick = time.perf_counter()
                        try:
                            response = await client.predict(request)
                            body = response.raw_output
                            record = suite.response_record(response, request, config)
                            if profile == "qwen4_token_scores":
                                record["backend_accounting"] = canonical.bridge_metadata(response, request)
                        except BaseException as exc:
                            if isinstance(exc, InvalidResponseError):
                                body = exc.raw_output
                            record = {"status": "failure", "selected_option_id": None, "resolved_model": None,
                                      "usage": suite.safe_usage(getattr(exc, "usage", None)), **suite.failure(exc)}
                            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                                interrupted = exc
                        if profile == "jev_action":
                            record = native.action_execution_record(request, body, config.model, record)
                        row = {**planned, "step": 0, "latency_seconds": time.perf_counter() - tick, **record}
                        suite.append_row(stream, row)
                        attempts.append(suite.json_value(row))
                        if record["fatal"]:
                            status.update(status="aborted", diagnostic_code=record["diagnostic_code"])
                            break
                    if status["status"] == "running":
                        require(source_hashes() == registration["source_sha256"], "Source changed during inference.")
                        status["status"] = ("completed_with_failures" if any(a["status"] == "failure"
                                                                          for a in attempts) else "completed")
                finally:
                    if client is not None:
                        await client.aclose()
        except BaseException as exc:
            status.update(status="aborted", diagnostic_code=suite.failure(exc)["diagnostic_code"])
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                interrupted = exc
        finally:
            if interrupted is not None:
                status["status"] = "interrupted"
            scores = scores_for(cases, requests, attempts, profile)
            write_rows(output / "scores.jsonl", scores)
            suite.write_json(output / "summary.json", metrics(scores))
            artifacts = ["attempts.jsonl", "scores.jsonl", "summary.json"]
            if (output / "backend-initialization.json").exists():
                artifacts.append("backend-initialization.json")
            status.update(calls=len(attempts), planned_calls=len(requests), finished_at_utc=now(),
                          wall_seconds=time.perf_counter() - started,
                          strict_native_failures=sum(a.get("strict_status") == "failure" for a in attempts),
                          artifact_sha256={name: suite.digest(output / name) for name in artifacts})
            suite.set_status(output, status)
    if interrupted is not None:
        raise interrupted
    return status


def verify_run(root, profile, *, allow_injected=False):
    registration, cases, requests = verify(root)
    output = root / profile
    manifest = suite.read_json(output / "manifest.json")
    require(suite.digest(output / "manifest.json") == suite.read_json(output / "manifest.sha256"),
            "Profile manifest drift.")
    require(manifest["profile"] == profile and
            manifest["model"] == registration["profiles"][profile] and
            manifest["registration_sha256"] == suite.digest(root / "registration.json") and
            manifest["source_sha256"] == registration["source_sha256"], "Run provenance differs.")
    require(allow_injected or not manifest["injected_client"], "Injected clients are not paper results.")
    status = suite.read_json(output / "status.json")
    require(status["status"] in ("completed", "completed_with_failures", "aborted", "interrupted"),
            "Profile not sealed.")
    require(all(suite.digest(output / name) == sha for name, sha in status["artifact_sha256"].items()),
            "Profile artifact drift.")
    attempts = suite.read_rows(output / "attempts.jsonl")
    require(status["calls"] == len(attempts), "Call counter differs.")
    if status["status"].startswith("completed"):
        require(len(attempts) == len(requests), "Completed lane omits scheduled calls.")
    scores = scores_for(cases, requests, attempts, profile)
    require(scores == suite.read_rows(output / "scores.jsonl"), "Recomputed scores differ.")
    summary = metrics(scores)
    require(matrix.same_aggregates(summary, suite.read_json(output / "summary.json")),
            "Recomputed metrics differ.")
    return status, attempts, scores, summary


def summarize(root):
    require(not (root / "tables").exists(), "Tables exist; no overwrite.")
    registration, cases, requests = verify(root)
    exports = {key: [] for key in ("accuracy", "paired_choice_changes", "five_arm_patterns")}
    statuses, counts = {}, {}
    for profile in PROFILES:
        status, attempts, scores, summary = verify_run(root, profile)
        statuses[profile] = status
        for key in exports:
            exports[key].extend({"profile": profile, **row} for row in summary[key])
        counts[profile] = {
            "scheduled": len(requests), "actual_calls": len(attempts),
            "valid": sum(s["valid"] for s in scores), "unattempted": sum(not s["attempted"] for s in scores),
            "execution_failures": sum(a["status"] == "failure" for a in attempts),
            "strict_native_codes": dict(Counter(a["strict_diagnostic_code"] for a in attempts
                                               if a.get("strict_status") == "failure")),
            "candidate_scores": sum(a.get("backend_accounting", {}).get("candidate_count", 0) for a in attempts)
                                if profile == "qwen4_token_scores" else None,
        }
    output = root / "tables"
    output.mkdir()
    for key, rows in exports.items():
        with (output / f"{key}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    complete = all(s["status"].startswith("completed") for s in statuses.values())
    result = {"status": "completed" if complete else "incomplete_with_explicit_missingness",
              "verified_at_utc": now(), "registration_sha256": suite.digest(root / "registration.json"),
              "statuses": statuses, "counters": counts, "analysis": registration["analysis"],
              "profiles_receive_identical_scheduled_requests": True,
              "graph_clusters": len({c["cluster_id"] for c in cases}),
              "binary_pairs": len({c["pair_id"] for c in cases if c["pair_id"]}),
              "actual_input_changes": dict(Counter(r["arm"] for r in requests if r["input_changed"])),
              "table_sha256": {f"{key}.csv": suite.digest(output / f"{key}.csv") for key in exports}}
    suite.write_json(output / "verified-summary.json", result)
    return result


async def run_matrix(root):
    verify(root)
    async def launch(profile):
        python = PY312 if profile == "qwen4_token_scores" else PY311
        command = [python, str(Path(__file__).resolve()), "run-profile", "--root", str(root.resolve()),
                   "--profile", profile]
        with (root / f"{profile}.log").open("x") as log:
            child = await asyncio.create_subprocess_exec(*command, cwd=REPO, stdout=log,
                                                        stderr=asyncio.subprocess.STDOUT)
            suite.write_json(root / f"{profile}.launch.json",
                             {"pid": child.pid, "started_at_utc": now(), "command": command})
            code = await child.wait()
        suite.write_json(root / f"{profile}.exit.json", {"exit_code": code, "finished_at_utc": now()})
        return code
    async def qwen4():
        return [await launch("qwen4_grammar"), await launch("qwen4_token_scores")]
    exits = await asyncio.gather(launch("jev_action"), launch("qwen9_grammar"), qwen4())
    suite.write_json(root / "launcher-exits.json", {"exit_codes": exits})
    return summarize(root)


async def live_cli(args):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run_profile(args.root, args.profile)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run-profile", "run-matrix", "summarize"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--profile", choices=PROFILES)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, args.bundle)
    elif args.command == "verify":
        registration, cases, requests = verify(args.root)
        result = {"status": "verified", "cases": len(cases), "requests_per_profile": len(requests),
                  "registration_sha256": suite.digest(args.root / "registration.json"), "model_calls": 0}
    elif args.command == "run-profile":
        if args.profile is None:
            parser.error("--profile required")
        result = asyncio.run(live_cli(args))
    elif args.command == "run-matrix":
        result = asyncio.run(run_matrix(args.root))
    else:
        result = summarize(args.root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] in ("prepared", "verified", "completed", "completed_with_failures") else 1


if __name__ == "__main__":
    raise SystemExit(main())
