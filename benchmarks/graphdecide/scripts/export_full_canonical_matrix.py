#!/usr/bin/env python3
"""Offline, all-lanes-terminal export of the identical canonical 960-case matrix."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import evaluate_graph_release as canonical
from scripts import graph_release_api as release
from scripts import summarize_release_matrix as matrix

suite = canonical.suite
BUNDLE = REPO / "output/graphbench-portable-release-20260924-v2"
BUNDLE_SHA = "0735d1422c35b4593f9621706425aa13d72b1b9ffdcae959d67a1dbef072365d"
SPLIT = "structural_challenge"
REFERENCE = REPO / "output/explicit-actions-matrix-20260924-v1"
GPT54 = REPO / "output/paper-gpt54-canonical-anchor-20260924-v3"
QWEN27 = REPO / "output/full-matrix-qwen27-20260925-v1"
LAYA = REPO / "output/full-matrix-laya-20260925-v1"
TRAINED = REPO / "output/full-matrix-trained-20260925-v1"
COMPLETE = ("completed", "completed_with_failures")
EXACT = ("adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
         "distance_threshold", "articulation_point")
CONSTRUCTION = ("maxcut_construct", "tsp_construct", "lt_influence_construct",
                "community_bipartition_construct")
COUNTS = {**{task: 80 for task in EXACT}, "degree_exact": 400,
          **{task: 40 for task in CONSTRUCTION}}
ARTIFACTS = {"attempts.jsonl", "scores.jsonl", "summary.json"}


@dataclass(frozen=True)
class Lane:
    name: str
    family: str
    kind: str
    root: Path
    reused: bool = False
    group: str = "decision_matrix"

    @property
    def run(self):
        return self.root / ("run-" + self.name) if self.kind == "trained" else \
            self.root / "run" if self.kind in ("qwen27", "laya") else self.root

    @property
    def terminal(self):
        filename = {"qwen27": "terminal.json", "laya": "completion.json"}.get(self.kind, "status.json")
        return self.run / filename

    @property
    def seal(self):
        if self.kind == "reference":
            return self.run / "manifest.sha256"
        if self.kind == "trained":
            return self.root / "terminal-verification.json"
        if self.kind == "laya":
            return self.run / "offline-replay.json"
        return self.run / "seal.json"


def lanes(astra_root):
    """Explicit inventory: no glob discovery, pilots, thinking lanes or 952-case runs."""
    suite.require(astra_root.resolve().is_relative_to(REPO), "Astra evidence must be repository-local.")
    result = [Lane(name, "Jev" if name == "jev_action" else "Qwen", "reference",
                   REFERENCE / f"{SPLIT}.{name}", True)
              for name in ("jev_action", "qwen4_grammar", "qwen9_grammar", "qwen4_token_scores")]
    return result + [
        Lane("qwen27_grammar", "Qwen", "qwen27", QWEN27),
        Lane("laya", "Jev-like", "laya", LAYA),
        Lane("decider", "Jev-like", "trained", TRAINED),
        Lane("kev", "Jev-like", "trained", TRAINED),
        Lane("gpt54_default_reasoning", "GPT", "gpt54", GPT54, True, "strong_anchors"),
        Lane("gpt6astra_default_reasoning", "GPT", "astra", astra_root.resolve(), False, "strong_anchors"),
    ]


class Evidence:
    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = Path(path)
        suite.require(path.resolve().is_relative_to(REPO) and path.is_file() and not path.is_symlink(),
                      f"Missing, aliased or nonlocal evidence: {path}")
        name = str(path.resolve().relative_to(REPO))
        digest = suite.digest(path)
        suite.require(expected is None or digest == expected, f"Evidence hash mismatch: {name}")
        suite.require(name not in self.hashes or digest == self.hashes[name],
                      f"Evidence changed during export: {name}")
        self.hashes[name] = digest
        return digest

    def inventory(self, root, values, required=()):
        suite.require(set(required) <= set(values), "Incomplete terminal artifact inventory.")
        for name, digest in values.items():
            suite.require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                          "Invalid sealed relative path.")
            self.track(root / name, digest)

    def recheck(self):
        for name, expected in self.hashes.items():
            suite.require(suite.digest(REPO / name) == expected, f"Evidence changed after replay: {name}")


def readiness(source_lanes):
    """Status-only admission of every lane before any correctness stream is opened."""
    result = {}
    for lane in source_lanes:
        if not lane.terminal.is_file():
            result[lane.name] = {"ready": False, "reason": "terminal_marker_missing"}
            continue
        state = suite.read_json(lane.terminal)
        if state.get("status") not in COMPLETE:
            result[lane.name] = {"ready": False, "reason": "not_completed",
                                 "source_status": state.get("status")}
            continue
        result[lane.name] = {"ready": lane.seal.is_file(),
                             "reason": "terminal_unverified" if lane.seal.is_file() else "seal_missing",
                             "source_status": state["status"]}
    return result


def require_ready(source_lanes):
    states = readiness(source_lanes)
    blocked = {name: value for name, value in states.items() if not value["ready"]}
    suite.require(not blocked, "Full matrix not ready; no correctness read: " + json.dumps(blocked))
    return states


def load_panel(evidence):
    evidence.track(BUNDLE / "manifest.json", BUNDLE_SHA)
    release.verify(BUNDLE)
    manifest = release.manifest_at(BUNDLE)
    evidence.inventory(BUNDLE, manifest["sha256"])
    instances = release.labeled_instances(BUNDLE, SPLIT, manifest)
    check_panel(instances)
    return instances, canonical.tasks.validate_instances(instances)


def check_panel(instances):
    suite.require(len(instances) == len({i["id"] for i in instances}) == 960
                  and Counter(i["task"] for i in instances) == COUNTS
                  and sum(i["query_budget"] for i in instances) == 1920,
                  "Requires the complete unchanged canonical 960/1920 panel, not a pilot or 952 bank.")
    expected = {(task, size): (size * 8 if task == "degree_exact" else
                              16 if task in EXACT else 8)
                for task in (*EXACT, *CONSTRUCTION) for size in range(8, 13)}
    suite.require(Counter((i["task"], len(i["state"]["nodes"])) for i in instances) == expected,
                  "Canonical task/size strata differ.")


def source_inventory(evidence, registration, root):
    for field in ("source_sha256", "code_sha256", "files_sha256"):
        if field in registration:
            evidence.inventory(REPO, registration[field])
    if "bundle_manifest_sha256" in registration:
        evidence.track(BUNDLE / "manifest.json", registration["bundle_manifest_sha256"])
    for key, filename in (("panel_sha256", "panel.json"), ("initial_requests_sha256", "initial-requests.json")):
        if key in registration:
            path = root / filename
            if key == "initial_requests_sha256" and not path.exists():
                path = root / "initial-requests.jsonl"
            evidence.track(path, registration[key])
    if "artifacts_sha256" in registration:
        evidence.track(Path(registration["artifacts"]) / "artifacts.json", registration["artifacts_sha256"])
    if "historical_registration_sha256" in registration:
        evidence.track(Path(registration["assets"]) / "benchmark-registration-parent-locks.json",
                       registration["historical_registration_sha256"])
    for snapshot in ("source-snapshot", "frozen-source", registration.get("source_snapshot")):
        if snapshot and (root / snapshot).is_dir():
            for field in ("source_sha256", "code_sha256", "files_sha256"):
                for name, digest in registration.get(field, {}).items():
                    candidate = root / snapshot / name
                    if candidate.is_file():
                        evidence.track(candidate, digest)


def import_verifier(name, evidence):
    path = REPO / "scripts" / (name + ".py")
    evidence.track(path)
    return importlib.import_module("scripts." + name)


def no_test_results(manifest):
    for key in ("injected_client", "injected_worker", "test_only"):
        suite.require(manifest.get(key, False) is False, "Injected/test-only results are not real evidence.")


def verify_gpt(producer, root):
    """Retain frozen verifier checks; tolerate only cross-Python aggregate roundoff."""
    try:
        return producer.verify(root)
    except suite.shared.Refusal as exc:
        if str(exc) != "Summary replay mismatch.":
            raise
    # The producer has already checked registration, seals, requests and scores.
    # Python 3.12 changed float summation relative to the historical 3.11 run.
    helpers = getattr(producer, "core", producer)
    manifest, instances, config = helpers.load(root)
    attempts = suite.read_rows(root / "attempts.jsonl")
    rows = helpers.replay(instances, attempts, config)
    suite.require(matrix.same_aggregates(helpers.summary_for(rows, attempts),
                                        suite.read_json(root / "summary.json")),
                  "GPT summary differs beyond 1e-12 aggregate roundoff.")
    terminal = suite.read_json(root / "status.json")
    suite.require(terminal["status"] in COMPLETE and terminal["calls"] == len(attempts)
                  and all(r["feasible"] or r["failure_count"] for r in rows)
                  and (terminal["status"] == "completed") ==
                  all(a["status"] == "success" for a in attempts),
                  "GPT terminal accounting differs.")
    return {"status": "verified", "run_status": terminal["status"], "calls": len(attempts),
            "planned_instances": len(rows), "injected_client": manifest["injected_client"],
            "manifest_sha256": suite.digest(root / "manifest.json"),
            "seal_sha256": suite.digest(root / "seal.json"),
            "aggregate_roundoff_absolute_tolerance": 1e-12}


def source_data(lane, evidence):
    """Adapter calls are offline only; producer-specific seals precede their verifiers."""
    require_ready([lane])
    evidence.track(lane.terminal)
    evidence.track(lane.seal)
    terminal = suite.read_json(lane.terminal)
    if lane.kind == "reference":
        evidence.inventory(lane.run, terminal.get("artifact_sha256", {}), ARTIFACTS)
        manifest = suite.read_json(lane.root / "manifest.json")
        no_test_results(manifest)
        evidence.track(lane.root / "manifest.json", suite.read_json(lane.root / "manifest.sha256"))
        source_inventory(evidence, manifest, lane.root)
        manifest, terminal, attempts, rows = matrix.verified_run(lane.root, BUNDLE, SPLIT)
        suite.require(manifest["profile"] == lane.name and manifest["planned_instances"] == 960
                      and manifest["maximum_calls"] == 1920, "Reused canonical profile/budget mismatch.")
        model_id, native = manifest["model"]["model"], manifest["native_action_observation"]
    elif lane.kind == "qwen27":
        producer = import_verifier("full_matrix_qwen27", evidence)
        seal = suite.read_json(lane.seal)
        suite.require(set(seal) == ARTIFACTS | {"manifest.json", "terminal.json"},
                      "Incomplete Qwen27 seal.")
        evidence.inventory(lane.run, seal)
        manifest = suite.read_json(lane.run / "manifest.json")
        no_test_results(manifest)
        suite.require(manifest.get("injected_client") is False, "Missing Qwen27 real-client marker.")
        registration = suite.read_json(lane.root / "registration.json")
        evidence.track(lane.root / "registration.json", suite.read_json(lane.root / "registration.sha256"))
        evidence.track(lane.root / "registration.sha256")
        source_inventory(evidence, registration, lane.root)
        verification = producer.verify(lane.root)
        suite.require(verification["verified"] is True and verification["model_calls"] == 0,
                      "Qwen27 source verification failed.")
        model_id, native = manifest["model"]["model"], False
        attempts, rows = suite.read_rows(lane.run / "attempts.jsonl"), suite.read_rows(lane.run / "scores.jsonl")
        manifest = {**registration, "source_verification": verification}
    elif lane.kind in ("gpt54", "astra"):
        producer = import_verifier(
            "run_paper_gpt_anchor" if lane.kind == "gpt54" else "run_canonical_astra", evidence)
        seal = suite.read_json(lane.seal)
        evidence.inventory(lane.root, seal["sha256"], (
            "manifest.json", "manifest.sha256", "config.json", "status.json", "attempts.jsonl",
            "episodes.jsonl", "summary.json", "requests.jsonl"))
        manifest = suite.read_json(lane.root / "manifest.json")
        no_test_results(manifest)
        suite.require(manifest.get("test_only") is False and manifest.get("injected_client") is False
                      and manifest["profile"] == lane.name
                      and manifest["model"]["model"] == ("gpt-5.4" if lane.kind == "gpt54" else "gpt-6-astra"),
                      "Not a real canonical GPT strong anchor.")
        source_inventory(evidence, manifest, lane.root)
        verification = verify_gpt(producer, lane.root)
        suite.require(verification["status"] == "verified" and not verification["injected_client"],
                      "GPT source verifier refused real evidence.")
        model_id, native = manifest["model"]["model"], False
        attempts, rows = suite.read_rows(lane.root / "attempts.jsonl"), suite.read_rows(lane.root / "episodes.jsonl")
        manifest = {**manifest, "source_verification": verification}
    elif lane.kind == "laya":
        producer = import_verifier("full_matrix_laya", evidence)
        proof = suite.read_json(lane.seal)
        suite.require(terminal["sealed_replay"] == proof and proof["verified"] is True
                      and proof["model_calls"] == 0 and terminal["scheduled_episodes"] == 960
                      and terminal["fatal"] is None, "Laya completion/replay seal mismatch.")
        for filename in ("attempts.jsonl", "scores.json", "summary.json"):
            evidence.track(lane.run / filename, proof[filename.split(".")[0] + "_sha256"])
        registration = suite.read_json(lane.root / "registration.json")
        no_test_results(registration)
        evidence.track(lane.root / "registration.sha256")
        evidence.track(lane.root / "registration.json", terminal["registration_sha256"])
        source_inventory(evidence, registration, lane.root)
        for filename in ("runtime.json", "status.json", "supervisor-process.json", "worker-identity.json"):
            evidence.track(lane.run / filename)
        runtime = suite.read_json(lane.run / "runtime.json")
        supervisor = suite.read_json(lane.run / "supervisor-process.json")
        worker = suite.read_json(lane.run / "worker-identity.json")
        suite.require(runtime["status"] == "ready" and runtime["dtype"] == "float32"
                      and runtime["cuda_initialized"] is False and runtime["threads"] == 4
                      and runtime["interop_threads"] == 1
                      and supervisor["device"] == "cpu" and supervisor["cuda_visible_devices"] == "",
                      "Laya production CPU identity differs.")
        suite.require(all(type(value.get("pid")) is int and value["pid"] > 0
                          and type(value.get("startticks")) is int and value["startticks"] > 0
                          for value in (supervisor, worker)),
                      "Laya production process provenance missing.")
        # completion guarantees offline-replay.json already exists, so the
        # producer's optional proof-creation branch is never entered here.
        verification = producer.verify_results(lane.root)
        suite.require(verification == proof, "Laya replay proof differs.")
        model_id, native = registration["model"], False
        attempts, rows = suite.read_rows(lane.run / "attempts.jsonl"), suite.read_json(lane.run / "scores.json")
        manifest = {**registration, "source_verification": verification, "runtime": runtime}
    elif lane.kind == "trained":
        producer = import_verifier("full_matrix_trained", evidence)
        proofs = suite.read_json(lane.seal)
        proof = proofs[lane.name]
        evidence.inventory(lane.run, proof["sha256"], (
            "scheduled.json", "dispatch.jsonl", "attempts.jsonl", "scores.jsonl", "summary.json", "status.json"))
        evidence.track(lane.root / "registration.sha256")
        evidence.track(lane.root / "registration.json", suite.read_json(lane.root / "registration.sha256"))
        registration, instances = producer.verify_registration(lane.root)
        no_test_results(registration)
        source_inventory(evidence, registration, lane.root)
        for filename in ("runtime.json", "worker-process.json"):
            evidence.track(lane.run / filename)
        evidence.track(lane.root / "launch.json")
        launch = suite.read_json(lane.root / "launch.json")
        worker = suite.read_json(lane.run / "worker-process.json")
        suite.require(launch["handoff"]["approved"] is True
                      and launch["handoff"]["registration_sha256"] ==
                      suite.digest(lane.root / "registration.json")
                      and type(launch["pid"]) is int and launch["pid"] > 0
                      and type(worker["pid"]) is int and worker["pid"] > 0
                      and worker["device"] == "cuda",
                      "Missing registered production-launch/worker provenance.")
        if (lane.run / "manifest.json").exists():
            evidence.track(lane.run / "manifest.json")
            run_manifest = suite.read_json(lane.run / "manifest.json")
            no_test_results(run_manifest)
            suite.require(run_manifest.get("injected_worker") is False
                          and run_manifest["registration_sha256"] ==
                          suite.digest(lane.root / "registration.json")
                          and run_manifest["candidate"] == lane.name, "Trained run identity differs.")
        producer.runtime_gate(suite.read_json(lane.run / "runtime.json"), lane.name, registration)
        verification = producer.verify_terminal(lane.root, lane.name, instances)
        suite.require(verification == proof, "Trained terminal proof changed.")
        model_id, native = producer.model_id(lane.name), False
        attempts, rows = suite.read_rows(lane.run / "attempts.jsonl"), suite.read_rows(lane.run / "scores.jsonl")
        manifest = {**registration, "source_verification": verification,
                    "runtime": suite.read_json(lane.run / "runtime.json")}
    else:
        raise ValueError("Unsupported producer layout")
    suite.require(manifest["protocol"] == "explicit_actions" and manifest["split"] == SPLIT
                  and Path(manifest["bundle"]).resolve() == BUNDLE.resolve(),
                  "Different bank/protocol; pilot and confirmation sources cannot be blended.")
    evidence.recheck()
    return manifest, terminal, attempts, rows, model_id, native


def execution_category(row, attempts):
    if row["query_count"] == 0:
        return "unattempted"
    if row.get("initial_context_compatible") is False:
        return "initial_unsupported"
    failures = [a for a in attempts if a["status"] == "failure"]
    unsupported = [a for a in failures if str(a.get("diagnostic_code", "")).lower() in
                   ("unsupported", "unsupportedrequesterror", "contextoverflow")]
    if unsupported:
        return "initial_unsupported" if unsupported[0]["step"] == 0 else "execution_unsupported"
    if failures:
        return "execution_failure"
    return "feasible" if row["feasible"] else "incomplete_execution"


def replay_lane(lane, instances, attempts, saved, model_id, native, terminal):
    replayed = canonical.replay(instances, attempts, SimpleNamespace(model=model_id), native_action=native)
    suite.require(len(saved) == len(instances) == 960 and len(attempts) <= 1920,
                  "Missing full scheduled rows or exceeded call budget.")
    suite.require(all({key: old.get(key) for key in new} == new
                      for new, old in zip(replayed, saved)), "Canonical own-history score replay differs.")
    suite.require(not any(a["fatal"] for a in attempts)
                  and all(r["query_count"] > 0 and (r["feasible"] or r["failure_count"] == 1)
                          for r in replayed), "Completed lane contains fatal/unattempted/incomplete episodes.")
    if lane.kind != "laya":
        suite.require(terminal["calls"] == len(attempts)
                      and (terminal["status"] == "completed") ==
                      all(a["status"] == "success" for a in attempts),
                      "Terminal completion/count accounting differs.")
    initial = {a["instance_id"]: a["request"] for a in attempts if a["step"] == 0}
    expected = {i["id"]: suite.json_value(asdict(canonical.public_request(i, []))) for i in instances}
    suite.require(initial == expected, "Canonical initial public requests differ.")
    own = {}
    for attempt in attempts:
        own.setdefault(attempt["instance_id"], []).append(attempt)
    rows = []
    audits = []
    for instance, row in zip(instances, saved):
        category = execution_category(row, own.get(instance["id"], []))
        suite.require((category == "feasible") == row["feasible"], "Coverage/feasibility category mismatch.")
        if instance["kind"] == "optimization" and row["feasible"]:
            evaluate, _ = canonical.tasks._independent_objectives(instance["task"], instance["state"])
            objective = evaluate(row["solution"])
            suite.require(math.isclose(objective, row["objective"], rel_tol=0, abs_tol=1e-12)
                          and row["optimal"] == math.isclose(
                              objective, instance["private"]["objective"], rel_tol=0, abs_tol=1e-12),
                          "Independent NetworkX/Fraction/coordinate objective mismatch.")
            audits.append({"instance_id": instance["id"], "task": instance["task"],
                           "objective": objective, "optimum": instance["private"]["objective"],
                           "optimal": row["optimal"]})
        rows.append({**row, "profile": lane.name, "family": lane.family, "group": lane.group,
                     "reused_reference": lane.reused, "coverage_outcome": category,
                     "initial_coverage_recorded": row.get("initial_context_compatible"),
                     "strict_contract_observed_calls": sum("strict_status" in a
                                                          for a in own.get(instance["id"], [])),
                     "strict_contract_failures": sum(a.get("strict_status") == "failure"
                                                    for a in own.get(instance["id"], []))})
    return rows, audits


def mean(values):
    return sum(values) / len(values) if values else None


def task_metrics(rows):
    metrics = []
    for task in (*EXACT, *CONSTRUCTION):
        all_rows = [r for r in rows if r["task"] == task]
        for size in ("all", *range(8, 13)):
            own = [r for r in all_rows if size == "all" or r["node_count"] == size]
            suite.require(bool(own), "Missing scheduled task/size denominator.")
            feasible = [r for r in own if r["feasible"]]
            exact = task in EXACT
            hits = sum(bool(r.get("correct" if exact else "optimal", False)) and r["feasible"] for r in own)
            size_rates = [sum(bool(r.get("correct", False)) and r["feasible"]
                              for r in own if r["node_count"] == n) /
                          sum(r["node_count"] == n for r in own)
                          for n in sorted({r["node_count"] for r in own})] if exact else []
            metrics.append({
                "profile": own[0]["profile"], "family": own[0]["family"], "group": own[0]["group"],
                "task": task, "node_count": size, "scheduled": len(own), "feasible": len(feasible),
                "initial_unsupported": sum(r["coverage_outcome"] == "initial_unsupported" for r in own),
                "execution_failures": sum(r["coverage_outcome"] in
                                          ("execution_failure", "execution_unsupported") for r in own),
                "unattempted": sum(r["coverage_outcome"] == "unattempted" for r in own),
                "failure_calls": sum(r["failure_count"] for r in own),
                "strict_contract_failure_calls": sum(r["strict_contract_failures"] for r in own),
                "strict_contract_observed_calls": sum(r["strict_contract_observed_calls"] for r in own),
                "hits": hits, "hit_definition": "correct" if exact else "optimal",
                "all_scheduled_hit_rate": hits / len(own),
                "macro_size_accuracy": mean(size_rates) if exact else None,
                "conditional_objective_denominator": len(feasible) if not exact else 0,
                "mean_objective_feasible": mean([r["objective"] for r in feasible]) if not exact else None,
                "mean_absolute_gap_feasible": mean([r["absolute_gap"] for r in feasible]) if not exact else None,
                "objective_direction": own[0].get("direction") if not exact else None,
            })
    return metrics


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def latex_escape(value):
    return "".join({"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                    "#": r"\#", "{": r"\{", "}": r"\}", "$": r"\$"}.get(c, c)
                   for c in str(value))


def write_tex(path, metrics):
    lines = [
        "% Generated from the verified complete canonical matrix; do not edit.",
        "% Ten taskwise tables; no pooled score or equal-compute GPT claim.",
    ]
    for task in (*EXACT, *CONSTRUCTION):
        exact = task in EXACT
        rows = [r for r in metrics if r["task"] == task and r["node_count"] == "all"]
        lines.extend([
            r"\begin{table}[htbp]\centering\small",
            r"\caption{" + latex_escape(task) +
            (r": full scheduled exact outcomes; macro equally weights sizes 8--12.}"
             if exact else r": full scheduled construction outcomes; means condition on feasibility.}"),
            r"\begin{tabular}{lrrrrr}\hline",
            (r"Profile & Scheduled & Feasible & Correct & Micro (\%) & Macro (\%) \\"
             if exact else r"Profile & Scheduled & Feasible & Optima & Mean objective & Mean gap \\"),
            r"\hline",
        ])
        group = None
        for row in rows:
            if row["group"] != group:
                group = row["group"]
                label = ("Decision configurations" if group == "decision_matrix" else
                         "GPT strong anchors (not equal compute)")
                lines.append(r"\multicolumn{6}{l}{\textit{" + label + r"}} \\")
            if exact:
                values = [f'{100 * row["all_scheduled_hit_rate"]:.2f}',
                          f'{100 * row["macro_size_accuracy"]:.2f}']
            else:
                values = ["n/a" if row[key] is None else f"{row[key]:.6g}" for key in
                          ("mean_objective_feasible", "mean_absolute_gap_feasible")]
            lines.append(" & ".join([latex_escape(row["profile"]), str(row["scheduled"]),
                                     str(row["feasible"]), str(row["hits"]), *values]) + r" \\")
        lines.extend([r"\hline\end{tabular}", r"\end{table}", ""])
    with path.open("x") as stream:
        stream.write("\n".join(lines))


def export(astra_root, output):
    source_lanes = lanes(astra_root)
    suite.require(not output.exists() and output.resolve().is_relative_to(REPO), "Output exists or is not local.")
    require_ready(source_lanes)
    evidence = Evidence()
    for module in (__file__, canonical.__file__, release.__file__, matrix.__file__):
        evidence.track(module)
    instances, validation = load_panel(evidence)
    rows, metrics, provenance, objective_audits = [], [], {}, {}
    for lane in source_lanes:
        manifest, terminal, attempts, saved, model_id, native = source_data(lane, evidence)
        own, audit = replay_lane(lane, instances, attempts, saved, model_id, native, terminal)
        rows.extend(own)
        metrics.extend(task_metrics(own))
        objective_audits[lane.name] = audit
        provenance[lane.name] = {
            "root": str(lane.root), "family": lane.family, "group": lane.group,
            "reused_reference": lane.reused, "model_id": model_id, "source_status": terminal["status"],
            "recorded_attempts": len(attempts),
            "observed_model_forwards": sum(a["forward_count"] for a in attempts)
            if all("forward_count" in a for a in attempts) else None,
            "configuration": manifest.get("model", manifest.get("model_settings")),
            "source_models": manifest.get("models"),
            "runtime": manifest.get("runtime"),
            "environment": manifest.get("packages", manifest.get("environment")),
            "dependency_deviations": manifest.get("dependency_deviations"),
            "source_verification": manifest.get("source_verification", "canonical verified_run"),
            "probability_scope": "Source-native conformance checked separately from eligible choices; "
                                 "no probability calibration guarantee.",
        }
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for group in ("decision_matrix", "strong_anchors"):
        write_csv(output / f"{group}-episodes.csv", [r for r in rows if r["group"] == group])
        write_csv(output / f"{group}-task-size.csv", [r for r in metrics if r["group"] == group])
    write_tex(output / "full-matrix-tables.tex", metrics)
    suite.write_json(output / "verified-summary.json", {
        "status": "verified_full_canonical_matrix", "new_model_calls": 0,
        "protocol": "explicit_actions", "split": SPLIT, "bundle_manifest_sha256": BUNDLE_SHA,
        "scheduled_per_profile": 960, "maximum_decisions_per_profile": 1920,
        "task_counts": COUNTS, "profiles": provenance, "metrics": metrics,
        "independent_reference_validation": validation, "independent_objectives": objective_audits,
        "initial_requests_identical": True, "source_sha256": evidence.hashes,
        "scope": "Identical full canonical 960-case bank. No 64-case pilot, 952-case confirmation, "
                 "new thinking configuration, selective task subsets or partial-run publication. "
                 "Reused Jev/Qwen/GPT54 identities are retained. GPT strong anchors are exported "
                 "separately, not treated as equal-compute or equal-latency methods. All scheduled "
                 "denominators retain initial unsupported and execution failures. Degree uses "
                 "equal-weight size macro accuracy (64/72/80/88/96 cases), not pooled accuracy. "
                 "Conditional construction gaps use feasible denominators and are null when empty. "
                 "No cross-objective pooled score, universal ranking or significance test.",
    })
    suite.write_json(output / "export-seal.json", {
        "status": "completed", "artifact_sha256": {
            path.name: suite.digest(path) for path in sorted(output.iterdir())}})
    return {"status": "verified_full_canonical_matrix", "profiles": len(source_lanes),
            "scheduled_episodes": len(rows), "output": str(output), "new_model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export"))
    parser.add_argument("--astra-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        result = readiness(lanes(args.astra_root))
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if all(r["ready"] for r in result.values()) else 2)
    if args.output is None:
        parser.error("--output is required for export")
    print(json.dumps(export(args.astra_root, args.output)))


if __name__ == "__main__":
    main()
