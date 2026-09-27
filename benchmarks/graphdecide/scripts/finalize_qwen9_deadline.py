#!/usr/bin/env python3
"""Terminal-only, immutable offline finalization of the Qwen9 deadline control."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
from dataclasses import asdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import networkx as nx
from scripts import qwen9_deadline_control as control
from scripts import export_larger_local_pilot as reference

shared, suite, canonical = control.shared, control.suite, control.canonical
ROOT = REPO / "output/qwen9-deadline-finalization-20260924-v1"
RUN = control.ROOT
SEALED = REPO / "output/expanded-pilot-finalization-20260924-v1/exports/qwen9_thinking"
MAX_WAIT, MAX_ANALYSIS = 86400, 300
TERMINAL = frozenset(("completed", "completed_with_failures", "wait_deadline_exceeded",
                     "run_deadline_exceeded", "interrupted", "blocked_or_aborted"))
PENDING = frozenset(("waiting_predecessor", "admission", "running"))
COMPLETED = frozenset(("completed", "completed_with_failures"))
SCOPE = ("Post-hoc 600-versus-180-second wall-budget contrast on the same sixteen construction "
         "episodes under the existing service; not repaired original results, confirmatory accuracy, "
         "or a causal reasoning/planning-quality claim. Jev and 27B grammar are bounded reused "
         "historical references, not matched latency controls. No bootstrap, significance testing, "
         "pooled cross-objective quality score, or universal model ranking.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def status(root, **fields):
    path = root / "status.next.json"
    save(path, {"at_utc": shared.now(), "model_calls": 0, "resource_actions": 0, **fields})
    os.replace(path, root / "status.json")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def checked(path, expected=None):
    require(path.is_file() and not path.is_symlink(), f"Missing or aliased input: {path}")
    digest = suite.digest(path)
    require(expected is None or digest == expected, f"Input hash mismatch: {path}")
    return digest


class Evidence:
    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = Path(path)
        digest = checked(path, expected)
        key = str(path.resolve())
        require(key not in self.hashes or self.hashes[key] == digest, "Input changed between reads.")
        self.hashes[key] = digest
        return digest

    def recheck(self):
        for path, digest in self.hashes.items():
            checked(Path(path), digest)


def sealed_files(root, evidence):
    evidence.track(root / "export-seal.json")
    seal = suite.read_json(root / "export-seal.json")
    require(seal["status"] == "completed" and isinstance(seal["artifact_sha256"], dict)
            and bool(seal["artifact_sha256"]), "Incomplete reference export seal.")
    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "export-seal.json"}
    require(actual == set(seal["artifact_sha256"]), "Reference seal inventory differs.")
    for name, digest in seal["artifact_sha256"].items():
        require(Path(name).name == name, "Unsafe sealed reference filename.")
        evidence.track(root / name, digest)


def prepare(root=ROOT):
    require(root == ROOT and not root.exists(), "Fresh fixed finalizer root required.")
    registration, instances = control.verified_inputs(RUN)
    require(len(instances) == 16 and sum(i["query_budget"] for i in instances) == 112,
            "Expected fixed construction panel.")
    evidence = Evidence()
    for name in ("registration.json", "registration.sha256", "cases.jsonl",
                 "initial-requests.jsonl", "checkpoint.json", "controller-start.json"):
        evidence.track(RUN / name)
    sealed_files(SEALED, evidence)
    history = reference.DEFAULT_HISTORY / "structural_challenge.jev_action"
    for name in ("manifest.json", "manifest.sha256", "status.json"):
        evidence.track(history / name)
    for name in ("execution.json", "execution.json.sha256", "qwen27-grammar-parent-verification.json"):
        evidence.track(control.PREVIOUS / name)
    for name in ("status.json", "manifest.json"):
        evidence.track(control.PREVIOUS / "qwen27_grammar" / name)
    sources = dict(registration["source_sha256"])
    sources.update({str(Path(module).relative_to(REPO)): checked(Path(module))
                    for module in (__file__, reference.__file__)})
    for name, digest in sources.items():
        evidence.track(REPO / name, digest)
    root.mkdir(parents=True)
    control.panel.invariance.freeze_sources(root, sources)
    value = {"schema_version": 1, "registered_at_utc": shared.now(), "model_calls": 0,
             "run": str(RUN), "sealed_original": str(SEALED), "scope": SCOPE,
             "maximum_wait_seconds": MAX_WAIT, "maximum_analysis_seconds": MAX_ANALYSIS,
             "child_termination_grace_seconds": 10,
             "source_sha256": sources, "fixed_input_sha256": evidence.hashes,
             "environment": control.environment(),
             "policy": {"status_only_before_terminal": True, "resume": False, "retry": False,
                        "server_actions": False, "gpu_actions": False, "leases": False,
                        "model_loads": False, "checkpoint_payload_sha256_recheck": True,
                        "paper_edits": False, "all_scheduled_denominator": 16}}
    evidence.recheck()
    save(root / "registration.json", value)
    save(root / "registration.sha256", checked(root / "registration.json"))
    for name in ("registration.json", "registration.sha256"):
        (root / name).chmod(0o444)
    status(root, status="prepared_not_started")
    return value


def frozen(root):
    registration = suite.read_json(root / "registration.json")
    checked(root / "registration.json", suite.read_json(root / "registration.sha256"))
    require(control.environment() == registration["environment"], "Offline environment drift.")
    for name, digest in registration["source_sha256"].items():
        checked(REPO / name, digest)
        checked(root / "source-snapshot" / name, digest)
    for path, digest in registration["fixed_input_sha256"].items():
        checked(Path(path), digest)
    return registration


async def wait_terminal(path, *, deadline, clock=time.monotonic,
                        sleeper=asyncio.sleep, reader=suite.read_json):
    while True:
        if clock() >= deadline:
            raise TimeoutError("Finalizer 24-hour terminal wait expired.")
        observed = reader(path)
        state = observed.get("status")
        if state in TERMINAL:
            return observed
        require(state in PENDING, "Unknown controller state; no correctness read permitted.")
        await sleeper(min(30, max(0, deadline - clock())))


def finite(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def redacted_attempts(attempts):
    fields = {"instance_id", "step", "request", "response_diagnostics", "latency_seconds",
              "status", "selected_option_id", "resolved_model", "usage", "diagnostic_code",
              "http_status", "fatal", "exception_type"}
    diagnostics = {"response_is_object", "choice_count", "finish_reason", "final_content_characters",
                   "final_content_sha256", "reasoning_present"}
    for row in attempts:
        require(set(row) == fields, "Unexpected attempt fields or full response body.")
        require(type(row["step"]) is int and row["step"] >= 0
                and finite(row["latency_seconds"]) and type(row["fatal"]) is bool,
                "Malformed attempt metadata.")
        require(row["status"] in ("success", "failure"), "Unknown attempt status.")
        require(set(row["usage"]) == {"input_tokens", "output_tokens", "reasoning_tokens"}
                and all(v is None or type(v) is int and v >= 0 for v in row["usage"].values()),
                "Malformed token accounting.")
        diagnostic = row["response_diagnostics"]
        require(diagnostic is None or isinstance(diagnostic, dict) and set(diagnostic) <= diagnostics,
                "Non-allowlisted response diagnostics.")
        if diagnostic:
            for key, value in diagnostic.items():
                if key in ("response_is_object", "reasoning_present"):
                    require(type(value) is bool, "Invalid diagnostic boolean.")
                elif key in ("choice_count", "final_content_characters"):
                    require(type(value) is int and value >= 0, "Invalid diagnostic count.")
                elif key == "final_content_sha256":
                    require(isinstance(value, str) and len(value) == 64
                            and set(value) <= set("0123456789abcdef"), "Invalid content digest.")
                else:
                    require(value in ("stop", "length", "tool_calls", "function_call", "content_filter",
                                      "error", "abort", "unrecognized"), "Unrecognized finish reason.")
        require(row["exception_type"] is None or isinstance(row["exception_type"], str)
                and row["exception_type"].isidentifier() and len(row["exception_type"]) <= 80,
                "Invalid exception class.")
        require(row["diagnostic_code"] is None or row["diagnostic_code"] in
                set(suite.DIAGNOSTICS) | {"run_deadline_exceeded"}, "Unknown failure diagnostic.")
        require(row["http_status"] is None or type(row["http_status"]) is int
                and 100 <= row["http_status"] <= 599, "Invalid HTTP status.")
        if row["status"] == "success":
            require(row["diagnostic_code"] is None and row["exception_type"] is None
                    and row["fatal"] is False, "Invalid success metadata.")
        else:
            require(row["selected_option_id"] is None and row["resolved_model"] is None
                    and row["diagnostic_code"] is not None, "Invalid failure metadata.")


def replay_control(instances, attempts, recorded, observed, summary, diagnostics, config):
    require(observed["status"] in TERMINAL, "Candidate not terminal.")
    require(len(instances) == 16 and len({i["id"] for i in instances}) == 16
            and sum(i["query_budget"] for i in instances) == 112, "Invalid fixed panel.")
    require(len(attempts) <= 112 and observed.get("calls") == len(attempts)
            and observed.get("planned_episodes") == 16, "Terminal call/denominator mismatch.")
    redacted_attempts(attempts)
    counts = Counter(a["instance_id"] for a in attempts)
    require(set(counts) <= {i["id"] for i in instances}
            and all(counts[i["id"]] <= i["query_budget"] for i in instances), "Episode budget exceeded.")
    rows = canonical.replay(instances, attempts, config, native_action=False)
    require(rows == recorded and len(rows) == 16, "Saved score coverage or own-history replay differs.")
    require(suite.summarize(rows) == summary, "Saved summary differs from complete canonical replay.")
    expected = {"planned_episodes": 16, "calls": len(attempts),
                "failures": dict(Counter(a["diagnostic_code"] for a in attempts
                                         if a["status"] == "failure")),
                "unattempted_episodes": sum(r["query_count"] == 0 for r in rows),
                "infeasible_episodes": sum(not r["feasible"] for r in rows),
                "length_cap_note": "8192-token cap unchanged; finish_length remains a failure, never retried."}
    require(diagnostics == expected, "Saved diagnostics differ from ledger.")
    if observed["status"] in COMPLETED:
        require(not any(a["fatal"] for a in attempts)
                and all(r["query_count"] > 0 and (r["feasible"] or r["failure_count"] == 1)
                        for r in rows), "Completed status has partial, unattempted or fatal episodes.")
        require((observed["status"] == "completed") ==
                all(a["status"] == "success" for a in attempts), "Completion/failure status mismatch.")
        require(observed.get("source_and_service_verified_after") is True,
                "Completed controller lacks final source/service verification.")
    return rows


def independent_objectives(instances, rows):
    audits = []
    for instance, row in zip(instances, rows):
        if not row["feasible"]:
            continue
        state, task, solution = instance["state"], instance["task"], row["solution"]
        graph = nx.DiGraph() if state["directed"] else nx.Graph()
        graph.add_nodes_from(state["nodes"])
        if task == "tsp_construct":
            graph.add_weighted_edges_from(state["edges"])
            require(solution[0] == solution[-1] == 0 and len(solution) == len(graph) + 1
                    and set(solution[:-1]) == set(graph), "Invalid independent tour.")
            value = Fraction(nx.path_weight(graph, solution, "weight"))
        elif task == "lt_influence_construct":
            graph.add_edges_from((e["source"], e["target"], {
                "weight": Fraction(e["numerator"], e["denominator"])}) for e in state["weights"])
            active = set(solution)
            threshold = Fraction(state["threshold"]["numerator"], state["threshold"]["denominator"])
            while True:
                updated = active | {v for v in graph if sum(
                    graph[u][v]["weight"] for u in graph.predecessors(v) if u in active) >= threshold}
                if updated == active:
                    break
                active = updated
            value = Fraction(len(active))
        else:
            graph.add_edges_from(state["edges"])
            parts = [{v for v, label in zip(state["nodes"], solution) if label == side}
                     for side in ("A", "B")]
            if task == "maxcut_construct":
                value = Fraction(nx.cut_size(graph, *parts))
            else:
                edges = graph.number_of_edges()
                value = sum(Fraction(graph.subgraph(part).number_of_edges(), edges) -
                            Fraction(sum(dict(graph.degree(part)).values()), 2 * edges) ** 2
                            for part in parts) if edges else Fraction(0)
        require(math.isclose(float(value), row["objective"], rel_tol=0, abs_tol=1e-12)
                and row["optimal"] == math.isclose(float(value), instance["private"]["objective"],
                                                  rel_tol=0, abs_tol=1e-12),
                "Independent NX/Fraction objective mismatch.")
        audits.append({"instance_id": instance["id"], "task": task,
                       "objective_fraction": str(value), "objective": float(value)})
    return audits


def metrics(rows):
    output = []
    for task in control.panel.CONSTRUCTION:
        for size in (8, 12, "all"):
            subset = [r for r in rows if r["task"] == task
                      and (size == "all" or r["node_count"] == size)]
            gaps = [r["absolute_gap"] for r in subset if r["feasible"]]
            output.append({"task": task, "node_count": size, "planned": len(subset),
                           "attempted": sum(r["query_count"] > 0 for r in subset),
                           "feasible": len(gaps), "optimal_all_planned": sum(r["optimal"] for r in subset),
                           "coverage": len(gaps) / len(subset),
                           "optimal_rate_all_planned": sum(r["optimal"] for r in subset) / len(subset),
                           "conditional_gap_denominator": len(gaps),
                           "conditional_mean_absolute_gap": sum(gaps) / len(gaps) if gaps else None})
    return output


def paired(candidate, original):
    require([r["instance_id"] for r in candidate] == [r["instance_id"] for r in original],
            "Paired episode IDs/order differ.")
    out = []
    for task in control.panel.CONSTRUCTION:
        pairs = [(a, b) for a, b in zip(candidate, original) if a["task"] == task]
        both = [(a, b) for a, b in pairs if a["feasible"] and b["feasible"]]
        out.append({"task": task, "planned_pairs": len(pairs), "both_feasible_denominator": len(both),
                    "coverage_600": sum(a["feasible"] for a, _ in pairs),
                    "coverage_180": sum(b["feasible"] for _, b in pairs),
                    "optimal_600_all_planned": sum(a["optimal"] for a, _ in pairs),
                    "optimal_180_all_planned": sum(b["optimal"] for _, b in pairs),
                    "paired_mean_gap_delta_600_minus_180":
                        sum(a["absolute_gap"] - b["absolute_gap"] for a, b in both) / len(both)
                        if both else None})
    return out


def usage(attempts):
    successful = [a for a in attempts if a["status"] == "success"]
    values = {}
    for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
        known = [a["usage"][key] for a in successful if a["usage"][key] is not None]
        values[key] = {"reported_successful_calls": len(known),
                       "sum": sum(known) if known else None,
                       "minimum": min(known) if known else None, "maximum": max(known) if known else None}
    return {"attempted_calls": len(attempts), "successful_calls": len(successful),
            "failed_calls": len(attempts) - len(successful),
            "successful_call_token_lengths": values,
            "successful_call_latency_seconds": [a["latency_seconds"] for a in successful],
            "observed_attempt_latency_seconds_sum": sum(a["latency_seconds"] for a in attempts),
            "failure_diagnostics": dict(Counter(a["diagnostic_code"] for a in attempts
                                               if a["status"] == "failure")),
            "scope": "Only completed client-call observations on this fixed subset. Failed-call "
                     "latency may be censored and provider work may continue after timeout. "
                     "Not throughput, isolated reasoning time, or matched latency across models."}


def subset(rows, instances):
    require(len(rows) == len({r["instance_id"] for r in rows}), "Duplicate reference score ID.")
    indexed = {r["instance_id"]: r for r in rows}
    require({i["id"] for i in instances} <= set(indexed), "Reference lacks selected IDs.")
    return [indexed[i["id"]] for i in instances]


def verify_configurations(registration):
    original, current = registration["original_configuration"], registration["configuration"]
    require(set(current) == set(original)
            and {k for k in current if current[k] != original[k]} == {"timeout_seconds"}
            and original["timeout_seconds"] == 180 and current["timeout_seconds"] == 600
            and asdict(control.configuration()) == current
            and asdict(control.panel.configurations()["qwen9_thinking"]) == original,
            "Control configuration changed beyond timeout.")


def references(instances, evidence, registration):
    sealed_files(SEALED, evidence)
    sealed = suite.read_json(SEALED / "verified-summary.json")
    require(sealed["profile"] == "qwen9_thinking" and sealed["source_status"] in COMPLETED,
            "Original 9B export is not sealed terminal completion.")
    require(sealed["provenance"]["qwen9_thinking"]["identity"]["configuration"] ==
            registration["original_configuration"], "Sealed original/control configuration differs.")
    for name, digest in sealed["source_sha256"].items():
        evidence.track(REPO / name, digest)
    original_root = control.PREVIOUS / "qwen9_thinking"
    for name in ("status.json", "attempts.jsonl", "scores.jsonl", "summary.json", "manifest.json"):
        require(str((original_root / name).relative_to(REPO)) in sealed["source_sha256"],
                "Original 9B payload is not hash-bound by the sealed export.")
    _, all_instances = shared.load_panel(Path(registration["panel"]))
    config = control.panel.configurations()["qwen9_thinking"]
    attempts = suite.read_rows(original_root / "attempts.jsonl")
    rows = canonical.replay(all_instances, attempts, config, native_action=False)
    require(rows == suite.read_rows(original_root / "scores.jsonl"), "Original 9B replay differs.")
    require(suite.summarize(rows) == suite.read_json(original_root / "summary.json"),
            "Original 9B summary replay differs.")
    reference.complete_coverage(all_instances, attempts, rows, suite.read_json(original_root / "status.json"))
    selected = subset(rows, instances)
    with (SEALED / "episodes.csv").open() as stream:
        csv_rows = [r for r in csv.DictReader(stream) if r["profile"] == "qwen9_thinking"]
    require(len(csv_rows) == 64 and len({r["instance_id"] for r in csv_rows}) == 64,
            "Original sealed CSV IDs duplicate or incomplete.")
    exported = {r["instance_id"]: r for r in csv_rows}
    require(all(exported[r["instance_id"]]["feasible"] == str(r["feasible"])
                and int(exported[r["instance_id"]]["query_count"]) == r["query_count"]
                and exported[r["instance_id"]]["optimal"] == str(r["optimal"])
                and exported[r["instance_id"]]["task"] == r["task"]
                and int(exported[r["instance_id"]]["node_count"]) == r["node_count"]
                for r in selected), "Original sealed export differs from canonical subset.")
    models, calls_by_model = {"qwen9_180": selected}, {
        "qwen9_180": [a for a in attempts if a["instance_id"] in {i["id"] for i in instances}]}
    panel_reg = suite.read_json(Path(registration["panel"]) / "registration.json")
    history = reference.DEFAULT_HISTORY / f'{panel_reg["split"]}.jev_action'
    _, models["jev_reused"], calls_by_model["jev_reused"] = reference.load_sealed_reference(
        history, Path(panel_reg["bundle"]), panel_reg["split"], instances, evidence)
    _, grammar, grammar_attempts, _ = reference.load_candidate(
        control.PREVIOUS / "qwen27_grammar", panel_reg, Path(registration["panel"]), all_instances, evidence)
    models["qwen27_grammar_reused"] = subset(grammar, instances)
    calls_by_model["qwen27_grammar_reused"] = [
        a for a in grammar_attempts if a["instance_id"] in {i["id"] for i in instances}]
    return models, calls_by_model


def table(models):
    lines = [r"\begin{tabular}{llrrr}", r"\hline",
             r"Configuration & Task & Feasible / planned & Optimal / planned & Gap (feasible $n$) \\",
             r"\hline"]
    for name, rows in models.items():
        for metric in metrics(rows):
            if metric["node_count"] != "all":
                continue
            gap = metric["conditional_mean_absolute_gap"]
            value = "--" if gap is None else f"{gap:.4g}"
            label = name.replace("_", r"\_")
            task = metric["task"].removesuffix("_construct").replace("_", r"\_")
            lines.append(f"{label} & {task} & {metric['feasible']}/{metric['planned']} & "
                         f"{metric['optimal_all_planned']}/{metric['planned']} & "
                         f"{value} ({metric['conditional_gap_denominator']}) " + r"\\")
    return "\n".join(lines + [r"\hline", r"\end{tabular}", ""])


def analyze(root):
    finalizer = frozen(root)
    observed = suite.read_json(RUN / "status.json")
    require(observed.get("status") in TERMINAL, "Candidate pending; no correctness reads.")
    evidence = Evidence()
    for path, digest in finalizer["fixed_input_sha256"].items():
        evidence.track(Path(path), digest)
    for name, digest in finalizer["source_sha256"].items():
        evidence.track(REPO / name, digest)
        evidence.track(root / "source-snapshot" / name, digest)
    for name in ("status.json", "attempts.jsonl", "scores.jsonl", "summary.json", "diagnostics.json"):
        evidence.track(RUN / name)
    require(suite.read_json(RUN / "status.json") == observed, "Terminal status changed.")
    registration, instances = control.verified_inputs(RUN)
    verify_configurations(registration)
    checkpoint = suite.read_json(RUN / "checkpoint.json")
    for name, entry in checkpoint["files"].items():
        path = Path(checkpoint["model_directory"]) / name
        require(path.stat().st_size == entry["bytes"], "Checkpoint payload size drift.")
        evidence.track(path, entry["sha256"])
    require(set(checkpoint["files"]) == {p.name for p in Path(checkpoint["model_directory"]).iterdir()
                                       if p.name != ".cache"}, "Checkpoint payload inventory drift.")
    bundle = Path(suite.read_json(Path(registration["panel"]) / "registration.json")["bundle"])
    manifest = control.panel.release.manifest_at(bundle)
    evidence.track(bundle / "manifest.json")
    for name, digest in manifest["sha256"].items():
        evidence.track(bundle / name, digest)
    reference_validation = canonical.tasks.validate_instances(instances)
    attempts = suite.read_rows(RUN / "attempts.jsonl")
    rows = replay_control(instances, attempts, suite.read_rows(RUN / "scores.jsonl"), observed,
                          suite.read_json(RUN / "summary.json"), suite.read_json(RUN / "diagnostics.json"),
                          control.configuration())
    models, prior_calls = references(instances, evidence, registration)
    models = {"qwen9_600": rows, **models}
    audits = {name: independent_objectives(instances, own) for name, own in models.items()}
    calls_by_model = {"qwen9_600": attempts, **prior_calls}
    result = {"status": "verified_completed_analysis" if observed["status"] in COMPLETED
              else "verified_abort_analysis", "source_status": observed["status"],
              "scope": SCOPE, "new_model_calls": 0, "episodes_per_configuration": 16,
              "maximum_followup_calls": 112, "configuration_delta": {"timeout_seconds": [180, 600]},
              "reference_validation": reference_validation, "independent_objectives": audits,
              "metrics": {name: metrics(own) for name, own in models.items()},
              "paired_600_vs_180": paired(rows, models["qwen9_180"]),
              "usage_and_time": {name: usage(own) for name, own in calls_by_model.items()},
              "quality_policy": "Conditional gap denominators explicit; empty feasible sets produce "
                                "null, never zero. No pooled quality mean. Original 9B both-feasible "
                                "paired gap delta is undefined when its feasibility is zero.",
              "input_sha256_before_and_after": evidence.hashes,
              "source_status_preserved": True, "paper_modified": False}
    evidence.recheck()
    frozen(root)
    output = root / "export"
    require(not output.exists(), "Immutable output already exists; no overwrite.")
    output.mkdir()
    save(output / "analysis.json", result)
    save(output / "episodes.json", models)
    with (output / "deadline-table.tex").open("x") as stream:
        stream.write(table(models))
        stream.flush()
        os.fsync(stream.fileno())
    evidence.recheck()
    save(output / "export-seal.json", {"status": result["status"],
         "registration_sha256": checked(root / "registration.json"),
         "artifact_sha256": {p.name: checked(p) for p in sorted(output.iterdir())}})
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return {"status": result["status"], "export": str(output),
            "seal_sha256": checked(output / "export-seal.json")}


async def bounded_child(root, seconds=MAX_ANALYSIS):
    child = await asyncio.create_subprocess_exec(
        sys.executable, str(Path(__file__).resolve()), "_analyze", "--root", str(root),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    save(root / "analysis-child.json", {**identity(child.pid), "parent_pid": os.getpid(),
                                      "analysis_deadline_seconds": seconds})
    try:
        out, err = await asyncio.wait_for(child.communicate(), seconds)
    except BaseException:
        if child.returncode is None:
            child.terminate()
            try:
                await asyncio.wait_for(child.wait(), 10)
            except TimeoutError:
                child.kill()
                await child.wait()
        raise
    require(child.returncode == 0, f"Offline analysis failed: {err.decode(errors='replace')[-1500:]}")
    return json.loads(out)


async def run(root):
    frozen(root)
    save(root / "finalizer-start.json", {**identity(), "started_at_utc": shared.now(),
         "registration_sha256": checked(root / "registration.json"),
         "maximum_wait_seconds": MAX_WAIT, "maximum_analysis_seconds": MAX_ANALYSIS})
    status(root, status="waiting_terminal")
    try:
        async with asyncio.timeout(MAX_WAIT):
            terminal = await wait_terminal(RUN / "status.json", deadline=time.monotonic() + MAX_WAIT)
        save(root / "terminal-observation.json", terminal)
        frozen(root)
        status(root, status="analyzing", source_status=terminal["status"])
        result = await bounded_child(root)
        status(root, **result)
        return result
    except BaseException as exc:
        state = "deadline_exceeded" if isinstance(exc, TimeoutError) else "failed"
        status(root, status=state, exception_type=type(exc).__name__, message=str(exc)[:1500])
        return {"status": state, "exception_type": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "dry-run", "run", "_analyze"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = shared.local(args.root)
    require(root == ROOT, "Fixed finalizer output required.")
    if args.command == "prepare":
        result = prepare(root)
    elif args.command == "dry-run":
        reg = frozen(root)
        result = {"status": "verified_zero_call_preparation", "model_calls": 0,
                  "source_files": len(reg["source_sha256"]), "candidate_results_read": False}
    elif args.command == "_analyze":
        starter = suite.read_json(root / "finalizer-start.json")
        require(starter["pid"] == os.getppid() and identity(os.getppid()) ==
                {k: starter[k] for k in ("pid", "start_ticks", "boot_id")},
                "Offline child requires its exclusively registered live parent.")
        result = analyze(root)
    else:
        async def execute():
            import signal
            loop, task = asyncio.get_running_loop(), asyncio.current_task()
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            try:
                return await run(root)
            finally:
                loop.remove_signal_handler(signal.SIGTERM)
        result = asyncio.run(execute())
    print(json.dumps(result, sort_keys=True))
    return int(args.command == "run" and result["status"] not in
               ("verified_completed_analysis", "verified_abort_analysis"))


if __name__ == "__main__":
    raise SystemExit(main())
