#!/usr/bin/env python3
"""Offline, replay-verified export of the completed bounded 27B grammar pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import larger_local_pilot as panel
from scripts import summarize_release_matrix as matrix

suite, canonical, release = panel.suite, panel.canonical, panel.release
HISTORICAL = ("jev_action", "qwen9_grammar", "qwen4_grammar", "qwen4_token_scores")
CANDIDATE = "qwen27_grammar"
TERMINAL = ("completed", "completed_with_failures")
ARTIFACTS = ("attempts.jsonl", "scores.jsonl", "summary.json")
DEFAULT_EXECUTION = REPO / "output/larger-local-pilot-execution-20260924-v1"
DEFAULT_HISTORY = REPO / "output/paper-confirmatory-20260924-v1"
SCOPE = (
    "Descriptive, score-blind selected n=8/12 panel only. Historical lanes are reused "
    "references, not fresh runs or latency controls. No causal latency comparison, "
    "formal significance claim, pooled cross-objective score, or universal size claim. "
    "Only completed 27B grammar and four sealed historical lanes are read; other "
    "thinking/GPT/trained lanes are outside this export."
)


class Evidence:
    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = Path(path).resolve()
        suite.require(path.is_relative_to(REPO), "Evidence must be inside the repository.")
        digest = suite.digest(path)
        suite.require(expected is None or digest == expected, f"Evidence hash drift: {path}")
        self.hashes[str(path.relative_to(REPO))] = digest
        return digest

    def recheck(self):
        for name, digest in self.hashes.items():
            suite.require(suite.digest(REPO / name) == digest, f"Evidence changed during export: {name}")


def terminal_status(root):
    # Read status before attempts/scores: an incomplete lane is never analyzed.
    status = suite.read_json(root / "status.json")
    suite.require(status.get("status") in TERMINAL, "Refusing partial or nonterminal lane.")
    return status


def complete_coverage(instances, attempts, rows, status):
    suite.require(len(rows) == len(instances)
                  and [r["instance_id"] for r in rows] == [i["id"] for i in instances],
                  "Incomplete or duplicate score coverage.")
    suite.require(status["calls"] == len(attempts)
                  and not any(a["fatal"] for a in attempts),
                  "Completed lane call count/fatal failure mismatch.")
    suite.require(all(r["query_count"] > 0 and
                      (r["feasible"] or r["failure_count"] == 1) for r in rows),
                  "Completed lane contains unattempted or unfinished episodes.")
    suite.require((status["status"] == "completed") ==
                  all(a["status"] == "success" for a in attempts), "Completion status mismatch.")


def load_panel(root, evidence):
    registration = suite.read_json(root / "registration.json")
    evidence.track(root / "registration.sha256")
    evidence.track(root / "registration.json", suite.read_json(root / "registration.sha256"))
    evidence.track(root / "pilot-cases.jsonl", registration["selection_sha256"])
    evidence.track(root / "initial-requests.jsonl", registration["initial_requests_sha256"])
    for name, digest in registration["source_sha256"].items():
        evidence.track(REPO / name, digest)
        evidence.track(root / "source-snapshot" / name, digest)
    bundle = Path(registration["bundle"])
    evidence.track(bundle / "manifest.json", registration["bundle_manifest_sha256"])
    release.verify(bundle)
    manifest = release.manifest_at(bundle)
    public = release.load_public(bundle, registration["split"], manifest)
    metadata = suite.read_json(release.checked_file(
        bundle, f"analysis/{registration['split']}.json", manifest))
    selected = panel.select_cases(public, metadata)
    cases = suite.read_rows(root / "pilot-cases.jsonl")
    expected_cases = [
        {"index": index, "instance_id": i["id"], "task": i["task"], "kind": i["kind"],
         "node_count": len(i["state"]["nodes"]), "query_budget": i["query_budget"],
         "cluster_id": metadata["cases"][i["id"]]["cluster_id"],
         "pair_id": metadata["cases"][i["id"]].get("pair_id")}
        for index, i in enumerate(selected)]
    suite.require(cases == expected_cases, "Score-blind selection or panel metadata drift.")
    labeled = {i["id"]: i for i in release.labeled_instances(
        bundle, registration["split"], manifest)}
    instances = [labeled[c["instance_id"]] for c in cases]
    counts = Counter((i["task"], len(i["state"]["nodes"])) for i in instances)
    suite.require(len(instances) == len(labeled.keys() & {i["id"] for i in instances}) == 64
                  and sum(i["query_budget"] for i in instances) == 160
                  and len(counts) == 20
                  and all(n in (8, 12) and count == (2 if task in panel.CONSTRUCTION else 4)
                          for (task, n), count in counts.items()), "Panel denominator drift.")
    initial = suite.read_rows(root / "initial-requests.jsonl")
    suite.require(initial == [{"instance_id": i["id"], "request": suite.json_value(
        asdict(canonical.public_request(i, [])))} for i in instances],
        "Canonical initial request mismatch.")
    validation = canonical.tasks.validate_instances(instances)
    for instance in instances:
        if instance["kind"] == "exact":
            suite.require(panel.invariance.truth(instance["task"], instance["state"]) ==
                          str(instance["private"]["answer"]), "Independent exact truth mismatch.")
    return registration, instances, cases, validation


def check_initial(instances, attempts):
    first = [a for a in attempts if a["step"] == 0]
    suite.require(len(first) == len(instances) and
                  {a["instance_id"]: a["request"] for a in first} ==
                  {i["id"]: suite.json_value(asdict(canonical.public_request(i, [])))
                   for i in instances}, "Selected canonical initial requests differ.")


def load_sealed_reference(root, bundle, split, instances, evidence):
    """Canonical source verifier first, subset second; usable only for sealed terminal sources."""
    status = terminal_status(root)
    suite.require(set(status.get("artifact_sha256", {})) == set(ARTIFACTS),
                  "Terminal source lacks complete artifact seal.")
    evidence.track(root / "status.json")
    evidence.track(root / "manifest.sha256")
    evidence.track(root / "manifest.json", suite.read_json(root / "manifest.sha256"))
    for name in ARTIFACTS:
        evidence.track(root / name, status["artifact_sha256"][name])
    manifest = suite.read_json(root / "manifest.json")
    required = {"profile", "model", "protocol", "split", "bundle_manifest_sha256",
                "injected_client", "native_action_observation", "code_sha256",
                "planned_instances", "maximum_calls"}
    suite.require(required <= manifest.keys(), "Incomplete canonical source manifest.")
    full_instances = release.labeled_instances(bundle, split, release.manifest_at(bundle))
    suite.require(manifest["planned_instances"] == len(full_instances)
                  and manifest["maximum_calls"] == sum(i["query_budget"] for i in full_instances),
                  "Canonical source manifest has incomplete planned coverage.")
    manifest, status, attempts, rows = matrix.verified_run(root, bundle, split)
    for name, digest in manifest["code_sha256"].items():
        evidence.track(REPO / name, digest)
    complete_coverage(full_instances, attempts, rows, status)
    ids = {i["id"] for i in instances}
    selected_attempts = [a for a in attempts if a["instance_id"] in ids]
    by_id = {r["instance_id"]: r for r in rows}
    selected_rows = [by_id[i["id"]] for i in instances]
    check_initial(instances, selected_attempts)
    return manifest, selected_rows, selected_attempts


def load_candidate(root, registration, panel_root, instances, evidence):
    status = terminal_status(root)
    evidence.track(root / "status.json")
    execution_root = root.parent
    execution = suite.read_json(execution_root / "execution.json")
    evidence.track(execution_root / "execution.json.sha256")
    execution_hash = evidence.track(execution_root / "execution.json",
                                   suite.read_json(execution_root / "execution.json.sha256"))
    suite.require(execution["panel"] == str(panel_root.resolve())
                  and execution["panel_registration_sha256"] ==
                  evidence.track(panel_root / "registration.json"), "Execution/panel identity mismatch.")
    for name, digest in execution["code_sha256"].items():
        evidence.track(REPO / name, digest)
    manifest = suite.read_json(root / "manifest.json")
    evidence.track(root / "manifest.json")
    suite.require(manifest == {
        "execution_sha256": execution_hash, "fresh": True, "maximum_calls": 160,
        "model": registration["configurations"][CANDIDATE], "planned_episodes": 64,
        "profile": CANDIDATE}, "Unexpected candidate manifest or configuration.")
    suite.require(execution["configurations"][CANDIDATE] == manifest["model"]
                  and status["profile"] == CANDIDATE, "Candidate identity mismatch.")
    # This legacy runtime has no status artifact seal. Its completed, hash-bound
    # parent attestation is required instead; no generic unsealed lane is accepted.
    proof_path = execution_root / "qwen27-grammar-parent-verification.json"
    evidence.track(proof_path)
    proof = suite.read_json(proof_path)
    suite.require(proof["status"] == "verified_completed_bounded_pilot"
                  and proof["exact_replay_equal"] is True and proof["episodes"] == 64
                  and proof["calls"] == status["calls"] and proof["new_model_calls"] == 0,
                  "Missing completed parent attestation.")
    evidence.track(root / "attempts.jsonl", proof["attempts_sha256"])
    evidence.track(root / "scores.jsonl", proof["scores_sha256"])
    evidence.track(root / "summary.json")
    attempts = suite.read_rows(root / "attempts.jsonl")
    rows = canonical.replay(instances, attempts, matrix.ModelConfig(**manifest["model"]),
                            native_action=False)
    suite.require(rows == suite.read_rows(root / "scores.jsonl"), "Candidate score replay mismatch.")
    suite.require(matrix.same_aggregates(suite.summarize(rows),
                                        suite.read_json(root / "summary.json")),
                  "Candidate aggregate replay mismatch.")
    complete_coverage(instances, attempts, rows, status)
    check_initial(instances, attempts)
    return manifest, rows, attempts, proof


def anomaly_audit(instances, rows, proof):
    """Reuse parent's independent NX cut/modularity checks; audit all zero-hit tasks."""
    import networkx as nx
    from fractions import Fraction

    checked = {r["instance_id"]: r for r in proof["independent_networkx_checks"]}
    expected = {i["id"] for i in instances if i["task"] in
                ("maxcut_construct", "community_bipartition_construct")}
    suite.require(set(checked) == expected and len(proof["independent_networkx_checks"]) == 8,
                  "Independent parent objective coverage mismatch.")
    audits = []
    for instance, row in zip(instances, rows):
        if instance["kind"] != "optimization":
            continue
        if not row["feasible"]:
            audits.append({"instance_id": instance["id"], "task": instance["task"],
                           "feasible": False, "independent_objective": None})
            continue
        state, task, solution = instance["state"], instance["task"], row["solution"]
        if task in ("maxcut_construct", "community_bipartition_construct"):
            check = checked[instance["id"]]
            suite.require(check["task"] == task, "Parent proof task mismatch.")
            objective, method = check["independent_objective"], "parent independent NetworkX proof"
        elif task == "tsp_construct":
            graph = nx.Graph()
            graph.add_weighted_edges_from(state["edges"])
            objective = nx.path_weight(graph, solution, "weight")
            method = "NetworkX path_weight on the complete closed tour"
        else:
            graph = nx.DiGraph()
            graph.add_nodes_from(state["nodes"])
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
            objective, method = len(active), "NetworkX predecessors / exact Fraction LT closure"
        suite.require(math.isclose(objective, row["objective"], rel_tol=0, abs_tol=1e-12),
                      "Independent construction objective mismatch.")
        suite.require(row["optimal"] == math.isclose(
            objective, instance["private"]["objective"], rel_tol=0, abs_tol=1e-12),
            "Independent optimum-hit mismatch.")
        audits.append({"instance_id": instance["id"], "task": task, "feasible": True,
                       "independent_objective": objective, "optimum": row["optimum"],
                       "optimal": row["optimal"], "method": method})
    return audits


def hit(row):
    return int(row["feasible"] and row.get("correct", row.get("optimal", False)))


def mean(values):
    return sum(values) / len(values) if values else None


def task_metrics(models):
    metrics = []
    for profile, rows in models.items():
        for task in sorted({r["task"] for r in rows}):
            for size in ("all", 8, 12):
                own = [r for r in rows if r["task"] == task and
                       (size == "all" or r["node_count"] == size)]
                feasible = [r for r in own if r["feasible"]]
                construction = own[0]["kind"] == "optimization"
                metrics.append({
                    "profile": profile, "task": task, "node_count": size,
                    "episodes": len(own), "feasible": len(feasible),
                    "failed_episodes": sum(r["failure_count"] > 0 for r in own),
                    "failure_calls": sum(r["failure_count"] for r in own),
                    "hits": sum(hit(r) for r in own), "hit_rate": mean([hit(r) for r in own]),
                    "hit_definition": "optimum" if construction else "exact",
                    "objective_direction": own[0].get("direction"),
                    "mean_objective_feasible": mean([r["objective"] for r in feasible])
                    if construction else None,
                    "mean_optimum": mean([r["optimum"] for r in own]) if construction else None,
                    "mean_absolute_gap_feasible": mean([r["absolute_gap"] for r in feasible])
                    if construction else None,
                })
    return metrics


def paired_differences(models):
    pairs, aggregates = [], []
    left = {r["instance_id"]: r for r in models["jev_action"]}
    suite.require(len(left) == len(models["jev_action"]), "Duplicate paired instance IDs.")
    for profile, rows in models.items():
        if profile == "jev_action":
            continue
        suite.require(len(rows) == len(left) and {r["instance_id"] for r in rows} == set(left),
                      "Incomplete or duplicate selected-instance pairing.")
        for right in rows:
            a = left[right["instance_id"]]
            suite.require(all(a[key] == right[key] for key in ("task", "kind", "node_count")),
                          "Paired instance metadata mismatch.")
            both = a["feasible"] and right["feasible"]
            construction = a["kind"] == "optimization"
            pairs.append({
                "left": "jev_action", "right": profile, "instance_id": a["instance_id"],
                "task": a["task"], "node_count": a["node_count"],
                "left_feasible": a["feasible"], "right_feasible": right["feasible"],
                "left_hit": hit(a), "right_hit": hit(right), "hit_difference": hit(a) - hit(right),
                "objective_direction": a.get("direction"),
                "objective_difference": a["objective"] - right["objective"]
                if construction and both else None,
                "absolute_gap_difference": a["absolute_gap"] - right["absolute_gap"]
                if construction and both else None,
            })
        for task, size in itertools.product(sorted({r["task"] for r in rows}), ("all", 8, 12)):
            own = [r for r in pairs if r["right"] == profile and r["task"] == task
                   and (size == "all" or r["node_count"] == size)]
            aggregates.append({
                "left": "jev_action", "right": profile, "task": task, "node_count": size,
                "objective_direction": own[0]["objective_direction"] if own else None,
                "episodes": len(own), "left_only_hits": sum(r["hit_difference"] == 1 for r in own),
                "right_only_hits": sum(r["hit_difference"] == -1 for r in own),
                "equal_hits": sum(r["hit_difference"] == 0 for r in own),
                "mean_hit_difference": mean([r["hit_difference"] for r in own]),
                "objective_pairs": sum(r["objective_difference"] is not None for r in own),
                "mean_objective_difference": mean([r["objective_difference"] for r in own
                                                  if r["objective_difference"] is not None]),
                "mean_absolute_gap_difference": mean([r["absolute_gap_difference"] for r in own
                                                     if r["absolute_gap_difference"] is not None]),
            })
    return pairs, aggregates


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows)


def report_table(metrics):
    profiles = (*HISTORICAL, CANDIDATE)
    indexed = {(r["task"], r["profile"]): r for r in metrics if r["node_count"] == "all"}
    lines = ["| Task | Episodes/lane | Jev | 9B grammar | 4B grammar | 4B token | 27B grammar |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in sorted({r["task"] for r in metrics}):
        own = [indexed[task, p] for p in profiles]
        values = [f'{r["hits"]}/{r["episodes"]}' + (
            f'; {r["mean_objective_feasible"]:.6g}' if r["mean_objective_feasible"] is not None else "")
                  for r in own]
        lines.append(f'| {task} | {own[0]["episodes"]} | ' + " | ".join(values) + " |")
    return "\n".join(lines)


def export(output, panel_root=panel.DEFAULT_ROOT, execution=DEFAULT_EXECUTION,
           historical=DEFAULT_HISTORY):
    output = output.resolve()
    suite.require(output.is_relative_to(REPO) and not output.exists(), "Output exists or is not local.")
    evidence = Evidence()
    evidence.track(__file__)
    registration, instances, cases, validation = load_panel(panel_root, evidence)
    models, provenance, all_attempts = {}, {}, {}
    manifest, rows, attempts, proof = load_candidate(
        execution / CANDIDATE, registration, panel_root, instances, evidence)
    models[CANDIDATE], all_attempts[CANDIDATE] = rows, attempts
    provenance[CANDIDATE] = {"source": str(execution / CANDIDATE), "fresh": True,
                             "selected_calls": len(attempts), "model": manifest["model"]}
    audits = anomaly_audit(instances, rows, proof)
    for profile in HISTORICAL:
        root = historical / f'{registration["split"]}.{profile}'
        manifest, rows, attempts = load_sealed_reference(
            root, Path(registration["bundle"]), registration["split"], instances, evidence)
        suite.require(manifest["profile"] == profile, "Historical profile mismatch.")
        models[profile], all_attempts[profile] = rows, attempts
        provenance[profile] = {"source": str(root), "fresh": False, "reused_reference": True,
                               "selected_calls": len(attempts), "model": manifest["model"]}
    metrics = task_metrics(models)
    pairs, paired = paired_differences(models)
    by_id = {c["instance_id"]: c for c in cases}
    episodes = []
    for profile, rows in models.items():
        for row in rows:
            failures = [a for a in all_attempts[profile] if a["instance_id"] == row["instance_id"]
                        and a["status"] == "failure"]
            episodes.append({"profile": profile, "fresh": profile == CANDIDATE,
                             **by_id[row["instance_id"]], **row, "hit": hit(row),
                             "failure_diagnostics": failures})
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "task-size-metrics.csv", metrics)
    write_csv(output / "paired-episodes.csv", pairs)
    write_csv(output / "paired-task-size.csv", paired)
    summary = {"status": "verified_completed_bounded_pilot_export", "scope": SCOPE,
               "new_model_calls": 0, "episodes_per_profile": len(instances),
               "canonical_initial_requests_identical": True, "provenance": provenance,
               "independent_reference_validation": validation, "construction_audit": audits,
               "metrics": metrics, "paired": paired,
               "failure_policy": "All planned episodes retained; infeasible episodes are zero hits. "
               "Objective means use feasible episodes only with explicit denominators; no imputation.",
               "paired_policy": "Jev minus reference on identical instance IDs, per task/size. "
               "Hit differences retain failures; objective/gap differences require both feasible. "
               "No confidence intervals or hypothesis tests on this tiny dependent panel.",
               "environment": suite.environment_versions(), "source_sha256": evidence.hashes}
    suite.write_json(output / "verified-summary.json", summary)
    (output / "summary.md").write_text(
        "# Completed bounded 27B grammar pilot\n\n" + SCOPE + "\n\n" + report_table(metrics) +
        "\n\nExact cells are correct/planned. Construction cells are optimum hits/planned; "
        "mean feasible objective (cut edges, tour cost, active vertices, modularity respectively). "
        "Never pool these objective units. Per-size denominators are 4 exact or 2 construction. "
        "Optima, gaps, failures and paired differences are in the CSV/JSON audit files.\n")
    suite.write_json(output / "export-seal.json", {
        "status": "completed", "artifact_sha256": {
            p.name: suite.digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=panel.DEFAULT_ROOT)
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    result = export(args.output, args.panel, args.execution, args.historical)
    print(report_table(result["metrics"]))


if __name__ == "__main__":
    main()
