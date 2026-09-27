#!/usr/bin/env python3
"""Offline terminal-only, explicitly attested expanded-model pilot exporter."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import networkx as nx
from scripts import export_larger_local_pilot as prior
from scripts import trained_jevlike_adapter as adapter

suite, canonical = prior.suite, prior.canonical
PROFILES = ("qwen27_grammar", "qwen27_thinking", "qwen9_thinking", "decider", "kev")
TERMINAL = ("completed", "completed_with_failures", "aborted")
PINS = {
    "decider": ("Mapika/decider-2b", "533964dae8be954c5b5e19fa4948e48408094c1e"),
    "kev": ("jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"),
    "kev-base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
}
SOURCES = {"decider": "15ab28e8d8e4ec49728165b3426d81de5b77f864",
           "kev": "569ea449b1033c5aa8f149ec5e20d0c7a296d24a"}
SCOPE = (
    "One explicitly attested terminal candidate on the fixed 64-episode n8/n12 panel; "
    "four fully verified historical lanes are reused references, not fresh inference. "
    "Eight exact or four construction episodes per task. No pooled objective, p-values, "
    "universal ranking, equal-latency assumption, or causal timing comparison. "
    "Aborted terminal runs, if explicitly attested, are incomplete diagnostic outcomes "
    "with all scheduled denominators retained, not completed performance runs."
)


def relative(path):
    path = Path(path).resolve()
    suite.require(path.is_relative_to(REPO), "Inputs and output must remain in the repository.")
    return str(path.relative_to(REPO))


def required_inputs(profile, run):
    files = [run / name for name in ("status.json", *prior.ARTIFACTS)]
    if profile.startswith("qwen"):
        files += [run / "manifest.json", run.parent / "execution.json",
                  run.parent / "execution.json.sha256"]
    else:
        files += [run / "runtime.json"]
        files += [run.parent / name for name in (
            "benchmark-registration.json", "benchmark-registration.sha256",
            "benchmark-registration-parent-locks.json", "benchmark-registration-parent-locks.sha256",
            "artifacts.json", "panel.json", "initial-requests.json")]
    return {relative(p): p for p in files}


def admit(profile, run, attestation, evidence):
    """Never read candidate scores, attempts or runtime before terminal admission."""
    suite.require(profile in PROFILES, "Unsupported profile (Laya is not implemented here).")
    relative(run)
    relative(attestation)
    status = suite.read_json(run / "status.json")
    suite.require(status.get("status") in TERMINAL, "Pending/incomplete candidate: no correctness read.")
    suite.require(attestation.is_file(), "Explicit independent parent attestation is required.")
    proof = suite.read_json(attestation)
    evidence.track(attestation)
    if profile == "qwen27_grammar" and proof.get("status") == "verified_completed_bounded_pilot":
        suite.require(attestation.resolve() ==
                      (run.parent / "qwen27-grammar-parent-verification.json").resolve(),
                      "Legacy grammar proof path mismatch.")
        return status, proof, True
    suite.require(proof.get("schema_version") == 1
                  and proof.get("status") == "verified_terminal_expanded_pilot"
                  and proof.get("profile") == profile
                  and proof.get("run") == str(run.resolve())
                  and proof.get("source_status") == status["status"]
                  and proof.get("episodes") == 64
                  and type(proof.get("calls")) is int and 0 <= proof["calls"] <= 160
                  and proof["calls"] == status.get("calls")
                  and proof.get("independent_replay_verified") is True
                  and proof.get("new_model_calls") == 0,
                  "Invalid explicit terminal parent attestation.")
    files = required_inputs(profile, run)
    suite.require(set(proof.get("input_sha256", {})) == set(files),
                  "Attestation must seal exactly the required producer inputs.")
    for name, path in files.items():
        evidence.track(path, proof["input_sha256"][name])
    suite.require(suite.read_json(run / "status.json") == status
                  and suite.read_json(attestation) == proof, "Admission metadata changed while reading.")
    return status, proof, False


def source_hashes(hashes, evidence):
    suite.require(isinstance(hashes, dict) and bool(hashes), "Missing frozen source hashes.")
    for name, digest in hashes.items():
        evidence.track(REPO / name, digest)


def qwen_identity(profile, run, registration, panel_root, evidence):
    execution = suite.read_json(run.parent / "execution.json")
    execution_hash = evidence.track(run.parent / "execution.json",
                                   suite.read_json(run.parent / "execution.json.sha256"))
    suite.require(execution["panel"] == str(panel_root.resolve())
                  and execution["panel_registration_sha256"] ==
                  evidence.track(panel_root / "registration.json")
                  and execution["model"] == "Qwen/Qwen3.5-27B"
                  and execution["revision"] == "fc05daec18b0a78c049392ed2e771dde82bdf654",
                  "Thinking execution identity/pin mismatch.")
    source_hashes(execution["code_sha256"], evidence)
    expected = asdict(prior.panel.configurations()[profile])
    manifest = suite.read_json(run / "manifest.json")
    suite.require(manifest == {
        "execution_sha256": execution_hash, "fresh": True, "maximum_calls": 160,
        "model": expected, "planned_episodes": 64, "profile": profile}
        and execution["configurations"][profile] == registration["configurations"][profile] == expected,
        "Candidate profile/configuration drift.")
    return expected["model"], {"configuration": expected}


def trained_identity(profile, run, registration, panel_root, instances, proof, evidence):
    root = run.parent
    original = suite.read_json(root / "benchmark-registration.json")
    active = suite.read_json(root / "benchmark-registration-parent-locks.json")
    for stem in ("benchmark-registration", "benchmark-registration-parent-locks"):
        evidence.track(root / f"{stem}.json", suite.read_json(root / f"{stem}.sha256"))
    amendment = active["resource_amendment"]
    changed = "scripts/run_trained_jevlike_pilot.py"
    suite.require(amendment["prior_registration_sha256"] ==
                  suite.digest(root / "benchmark-registration.json")
                  and amendment["benchmark_calls_before_amendment"] == 0
                  and amendment["changed_source"] == changed
                  and amendment["selection_models_inference_scoring_unchanged"] is True
                  and amendment["prior_source_sha256"] == original["code_sha256"][changed]
                  and amendment["new_source_sha256"] == active["code_sha256"][changed],
                  "Invalid trained registration amendment.")
    restored = {**active, "code_sha256": {**active["code_sha256"],
                                         changed: original["code_sha256"][changed]}}
    restored.pop("resource_amendment")
    suite.require(restored == original, "Amendment changed more than permitted resource source.")
    suite.require(active["selection"] == str(panel_root.resolve())
                  and active["selection_sha256"] == registration["selection_sha256"]
                  and active["bundle_manifest_sha256"] == registration["bundle_manifest_sha256"]
                  and active["protocol"] == "explicit_actions"
                  and active["scheduled_episodes_per_model"] == 64
                  and active["maximum_calls_per_model"] == 160
                  and active["context_cap"] == 8192
                  and active["inference_timeout_seconds"] == 180
                  and active["policy"] == {
                      "retry": False, "repair": False, "truncate": False, "own_history": True,
                      "all_scheduled_in_denominator": True, "probability_calibration_claim": False,
                      "private_labels_sent": False}, "Trained panel/policy/configuration drift.")
    source_hashes(active["code_sha256"], evidence)
    artifacts = suite.read_json(root / "artifacts.json")
    suite.require(artifacts == active["models"]
                  and artifacts["source_revisions"] == SOURCES
                  and set(artifacts["models"]) == set(PINS),
                  "Trained artifact/source registration drift.")
    for name, (repository, revision) in PINS.items():
        model = artifacts["models"][name]
        suite.require(model["repository"] == repository and model["revision"] == revision
                      and bool(model["sha256"])
                      and all(len(value) == 64 for value in model["sha256"].values()),
                      "Trained checkpoint identity/inventory mismatch.")
    suite.require(proof.get("checkpoint_inventory_independently_verified") is True,
                  "Parent must independently verify checkpoint inventory, not merely copy metadata.")
    suite.require(suite.read_json(root / "panel.json") == [i["id"] for i in instances]
                  and suite.read_json(root / "initial-requests.json") ==
                  [suite.json_value(asdict(canonical.public_request(i, []))) for i in instances],
                  "Trained initial requests or panel order differ.")
    ready = suite.read_json(run / "runtime.json")
    # A loading failure may produce no ready runtime. That producer format has
    # no temperature evidence and is deliberately unsupported, not guessed.
    suite.require(ready.get("status") == "ready", "Missing ready trained runtime metadata.")
    runtime = ready["runtime"]
    temperature = runtime["temperature"]
    suite.require(type(temperature) in (int, float) and math.isfinite(temperature)
                  and temperature > 0 and temperature == proof.get("effective_temperature")
                  and proof.get("temperature_independently_verified") is True
                  and runtime["candidate"] == profile and runtime["device"] == "cuda"
                  and runtime["context_limit"] == 8192 and runtime["cpu_threads"] == 4
                  and runtime["dtype"] == ("float32" if profile == "kev" else "bfloat16")
                  and runtime["cuda_initialized"] is True and runtime["calibration_claim"] is False,
                  "Trained effective temperature/runtime mismatch.")
    expected_base = {"base": PINS["kev-base"][0],
                     "base_revision": runtime.get("base_metadata", {}).get("base_revision")} \
        if profile == "kev" else None
    if profile == "kev":
        suite.require(expected_base["base_revision"] in (None, PINS["kev-base"][1]),
                      "Unexpected Kev base revision.")
    suite.require(runtime["base_metadata"] == expected_base, "Trained base metadata mismatch.")
    return "@".join(PINS[profile]), {"runtime": runtime, "checkpoint_inventory": artifacts,
                                     "checkpoint_payloads_read_by_exporter": False}


def failure_category(attempt):
    if attempt["status"] == "success":
        return None
    diagnostic = str(attempt.get("diagnostic_code", "")).lower()
    if "unsupported" in diagnostic or "contextoverflow" in diagnostic:
        return "unsupported"
    if "timeout" in diagnostic:
        return "timeout"
    if diagnostic in adapter.InvalidResponseError._DIAGNOSTIC_CODES or any(
            s in diagnostic for s in ("invalid", "parse", "probability", "choice", "malformed")):
        return "parse_or_contract"
    return "backend_or_other"


def probability_audit(attempts, model_id):
    audit = []
    for attempt in attempts:
        observation = attempt.get("action_observation", {})
        ids = [o["id"] for o in attempt["request"]["options"]]
        eligible = observation.get("eligible")
        suite.require(type(eligible) is bool and
                      (observation.get("selected_option_id") in ids if eligible else
                       observation.get("selected_option_id") is None),
                      "Malformed native action eligibility observation.")
        shape = None
        if attempt["status"] == "success":
            backend = attempt["backend"]
            suite.require(backend["status"] == "success", "Native backend status mismatch.")
            response, diagnostic = adapter.validate_result(
                adapter.request_from_dict(attempt["request"]), backend["result"], model_id)
            suite.require(attempt["probabilities"] == response.probabilities
                          and attempt["probability_kind"] == "native"
                          and attempt["selected_option_id"] == response.selected_option_id
                          and attempt["strict_status"] == diagnostic["strict_status"]
                          and observation == diagnostic["action_observation"]
                          and attempt["usage"] == asdict(response.usage),
                          "Native probability/choice/usage replay mismatch.")
            shape = True
        else:
            suite.require(attempt.get("strict_status") == "failure",
                          "Failed native attempt has successful strict status.")
            if attempt.get("diagnostic_code") in ("invalid_probabilities", "invalid_probability_sum"):
                shape = False
        audit.append({"instance_id": attempt["instance_id"], "step": attempt["step"],
                      "status": attempt["status"], "failure_category": failure_category(attempt),
                      "choice_eligible": eligible, "strict_status": attempt["strict_status"],
                      "probability_shape_valid": shape,
                      "probability_shape_scope": "strict native contract; no calibration claim"})
    return audit


def replay_terminal(instances, attempts, recorded, status, model_id, trained=False):
    suite.require(type(status.get("calls")) is int and
                  status["calls"] == len(attempts) <= 160, "Actual call count/budget mismatch.")
    rows = canonical.replay(instances, attempts, SimpleNamespace(model=model_id), native_action=False)
    if trained:
        for row in rows:
            row.update(scheduled=True, unattempted=row["query_count"] == 0)
    suite.require(rows == recorded, "Own-history replayed scores differ.")
    if status["status"] in prior.TERMINAL:
        prior.complete_coverage(instances, attempts, rows, status)
    else:
        suite.require(status["status"] == "aborted", "Unrecognized terminal state.")
    first = [a for a in attempts if a["step"] == 0]
    by_id = {i["id"]: i for i in instances}
    suite.require(len(first) == len({a["instance_id"] for a in first}) and all(
        a["request"] == suite.json_value(asdict(canonical.public_request(by_id[a["instance_id"]], [])))
        for a in first), "Attempted initial requests differ.")
    if trained:
        suite.require(status["scheduled_episodes"] == 64
                      and status["unattempted_episodes"] == sum(r["unattempted"] for r in rows)
                      and status["failure_count"] == sum(a["status"] == "failure" for a in attempts)
                      and bool(status["fatal"]) == (status["status"] == "aborted"),
                      "Trained terminal accounting mismatch.")
    return rows


def independent_objectives(instances, rows):
    audits = []
    for instance, row in zip(instances, rows):
        if instance["kind"] != "optimization" or not row["feasible"]:
            continue
        state, task, solution = instance["state"], instance["task"], row["solution"]
        graph = nx.DiGraph() if state["directed"] else nx.Graph()
        graph.add_nodes_from(state["nodes"])
        if task == "tsp_construct":
            graph.add_weighted_edges_from(state["edges"])
            suite.require(solution[0] == solution[-1] == 0
                          and len(solution) == len(graph) + 1
                          and set(solution[:-1]) == set(graph), "Invalid closed tour.")
            objective = nx.path_weight(graph, solution, "weight")
        elif task == "lt_influence_construct":
            graph.add_edges_from((e["source"], e["target"],
                                  {"weight": Fraction(e["numerator"], e["denominator"])})
                                 for e in state["weights"])
            active = set(solution)
            threshold = Fraction(state["threshold"]["numerator"], state["threshold"]["denominator"])
            while True:
                updated = active | {v for v in graph if sum(
                    graph[u][v]["weight"] for u in graph.predecessors(v) if u in active) >= threshold}
                if updated == active:
                    break
                active = updated
            objective = len(active)
        else:
            graph.add_edges_from(state["edges"])
            communities = [{v for v, label in zip(state["nodes"], solution) if label == side}
                           for side in ("A", "B")]
            objective = nx.cut_size(graph, *communities) if task == "maxcut_construct" else \
                nx.community.modularity(graph, [c for c in communities if c])
        optimum = instance["private"]["objective"]
        suite.require(math.isclose(objective, row["objective"], rel_tol=0, abs_tol=1e-12)
                      and row["optimal"] == math.isclose(objective, optimum, rel_tol=0, abs_tol=1e-12),
                      "Independent NetworkX/Fraction objective or optimum-hit mismatch.")
        audits.append({"instance_id": instance["id"], "task": task,
                       "independent_objective": objective, "optimum": optimum, "optimal": row["optimal"]})
    return audits


def metrics(models):
    result = prior.task_metrics(models)
    for metric in result:
        rows = [r for r in models[metric["profile"]] if r["task"] == metric["task"]
                and (metric["node_count"] == "all" or r["node_count"] == metric["node_count"])]
        metric.update(attempted_episodes=sum(r["query_count"] > 0 for r in rows),
                      unattempted_episodes=sum(r["query_count"] == 0 for r in rows),
                      incomplete_attempted_episodes=sum(r["query_count"] > 0 and not r["feasible"]
                                                       and r["failure_count"] == 0 for r in rows),
                      objective_gap_denominator=metric["feasible"]
                      if rows[0]["kind"] == "optimization" else 0)
    return result


def export(profile, run, attestation, output, panel_root=prior.panel.DEFAULT_ROOT,
           historical=prior.DEFAULT_HISTORY):
    run, attestation, output = run.resolve(), attestation.resolve(), output.resolve()
    relative(output)
    suite.require(not output.exists(), "Output exists; no overwrite.")
    evidence = prior.Evidence()
    evidence.track(__file__)
    evidence.track(prior.__file__)
    status, proof, legacy = admit(profile, run, attestation, evidence)
    registration, instances, cases, validation = prior.load_panel(panel_root, evidence)
    if legacy:
        manifest, rows, attempts, _ = prior.load_candidate(
            run, registration, panel_root, instances, evidence)
        identity, probability = {"configuration": manifest["model"], "legacy_parent_proof": True}, []
    else:
        suite.require(proof.get("panel_registration_sha256") ==
                      suite.digest(panel_root / "registration.json"), "Attestation panel mismatch.")
        if profile.startswith("qwen"):
            model_id, identity = qwen_identity(profile, run, registration, panel_root, evidence)
            suite.require(status["profile"] == profile, "Thinking terminal profile mismatch.")
        else:
            model_id, identity = trained_identity(
                profile, run, registration, panel_root, instances, proof, evidence)
        attempts = suite.read_rows(run / "attempts.jsonl")
        rows = replay_terminal(instances, attempts, suite.read_rows(run / "scores.jsonl"),
                               status, model_id, trained=profile in PINS)
        suite.require(prior.matrix.same_aggregates(
            suite.summarize(rows), suite.read_json(run / "summary.json")),
            "Canonical aggregate replay mismatch.")
        probability = probability_audit(attempts, model_id) if profile in PINS else []
    audits = independent_objectives(instances, rows)
    models = {profile: rows}
    provenance = {profile: {"run": str(run), "status": status, "identity": identity,
                            "actual_calls": len(attempts), "fresh": True,
                            "failure_categories": dict(Counter(failure_category(a) for a in attempts
                                                              if a["status"] == "failure"))}}
    for reference in prior.HISTORICAL:
        root = historical / f'{registration["split"]}.{reference}'
        manifest, reference_rows, reference_attempts = prior.load_sealed_reference(
            root, Path(registration["bundle"]), registration["split"], instances, evidence)
        suite.require(manifest["profile"] == reference, "Historical profile mismatch.")
        models[reference] = reference_rows
        provenance[reference] = {"run": str(root), "fresh": False, "reused_reference": True,
                                 "selected_calls": len(reference_attempts)}
    task_metrics = metrics(models)
    pairs, paired = prior.paired_differences(models)
    indexed_cases = {c["instance_id"]: c for c in cases}
    episodes = []
    for name, own in models.items():
        for row in own:
            failures = [a for a in attempts if name == profile
                        and a["instance_id"] == row["instance_id"] and a["status"] == "failure"]
            outcome = ("unattempted" if row["query_count"] == 0 else
                       "failed" if row["failure_count"] else
                       "incomplete" if not row["feasible"] else
                       "correct_or_optimal" if prior.hit(row) else
                       "incorrect" if row["kind"] == "exact" else "suboptimal")
            episodes.append({"profile": name, "fresh": name == profile,
                             **indexed_cases[row["instance_id"]], **row, "outcome": outcome,
                             "failure_categories": [failure_category(a) for a in failures],
                             "failure_diagnostics": [a.get("diagnostic_code") for a in failures]})
    result = {"status": "verified_terminal_expanded_pilot_export", "scope": SCOPE,
              "profile": profile, "source_status": status["status"], "new_model_calls": 0,
              "episodes_per_profile": 64, "provenance": provenance,
              "independent_reference_validation": validation, "construction_audit": audits,
              "metrics": task_metrics, "paired": paired, "native_probability_audit": probability,
              "probability_scope": "Native strict probability contract and action eligibility are "
              "separate. Eligible actions from failed strict responses do not advance history or "
              "receive success credit. No graph-panel probability calibration claim.",
              "failure_policy": "All scheduled denominators retained, including unattempted. "
              "Objective/gap means conditional on feasibility; no feasible values means null, not 0. "
              "Errors are not model incorrectness; infeasible episodes still count as zero hits.",
              "source_sha256": evidence.hashes, "environment": suite.environment_versions()}
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("episodes", episodes), ("task-size-metrics", task_metrics),
                         ("paired-episodes", pairs), ("paired-task-size", paired)):
        prior.write_csv(output / f"{name}.csv", values)
    suite.write_json(output / "verified-summary.json", result)
    suite.write_json(output / "export-seal.json", {"status": "completed", "artifact_sha256": {
        p.name: suite.digest(p) for p in sorted(output.iterdir())}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=prior.panel.DEFAULT_ROOT)
    parser.add_argument("--historical", type=Path, default=prior.DEFAULT_HISTORY)
    args = parser.parse_args()
    result = export(args.profile, args.run, args.attestation, args.output, args.panel, args.historical)
    print(json.dumps({"status": result["status"], "profile": args.profile,
                      "source_status": result["source_status"], "output": str(args.output),
                      "new_model_calls": 0}))


if __name__ == "__main__":
    main()
