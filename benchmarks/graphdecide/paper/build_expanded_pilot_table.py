#!/usr/bin/env python3
"""Generate coverage-aware paper tables from the five sealed expanded-pilot exports."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import export_expanded_model_pilots as expanded
from scripts.build_paper_evidence import EXACT, OPTIMIZATION, TASK_LABELS, latex_table

prior, suite = expanded.prior, expanded.suite
FINALIZATION = REPO / "output/expanded-pilot-finalization-20260924-v1"
BOUNDED = REPO / "output/larger-local-verified-pilot-summary-20260924-v2"
NEW = ("qwen27_thinking", "qwen9_thinking", "decider", "kev")
PROFILES = ("jev_action", "qwen27_grammar", *NEW)
LABELS = ("Jev", "27B G", "27B T", "9B T", "Decider", "Kev")
TASKS = EXACT + OPTIMIZATION
CORE = ("instance_id", "task", "kind", "node_count", "query_budget", "query_count",
        "successful_queries", "failure_count", "feasible", "decisions",
        "correct", "prediction", "answer", "optimal", "objective", "optimum", "absolute_gap", "direction")


class Evidence(prior.Evidence):
    def track(self, path, expected=None):
        name = expanded.relative(Path(path))
        if name in self.hashes:
            suite.require(expected is None or self.hashes[name] == expected,
                          "Conflicting source hash claims across exports.")
            return self.hashes[name]
        return super().track(path, expected)


def boolean(value):
    suite.require(value in ("True", "False"), "Malformed CSV boolean.")
    return value == "True"


def finite(value):
    if value == "":
        return None
    result = float(value)
    suite.require(math.isfinite(result), "Nonfinite CSV objective.")
    return result


def read_episodes(path):
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    result = {}
    for source in raw:
        row = {key: source[key] for key in ("instance_id", "task", "kind")}
        row.update({key: int(source[key]) for key in (
            "node_count", "query_budget", "query_count", "successful_queries", "failure_count")})
        row["feasible"] = boolean(source["feasible"])
        row["decisions"] = json.loads(source["decisions"])
        if row["kind"] == "exact":
            row.update(correct=boolean(source["correct"]), prediction=source["prediction"] or None,
                       answer=source["answer"])
        else:
            row.update(optimal=boolean(source["optimal"]), direction=source["direction"])
            row.update({key: finite(source[key]) for key in ("objective", "optimum", "absolute_gap")})
            suite.require(row["optimum"] is not None and all(
                (row[key] is not None) == row["feasible"] for key in ("objective", "absolute_gap")),
                "Missing feasible objective or invented infeasible objective/gap.")
        suite.require(row["feasible"] or not row.get("correct", row.get("optimal")),
                      "Infeasible episode credited as a hit.")
        result.setdefault(source["profile"], []).append(row)
    return result


def verify_panel_rows(models, cases):
    expected = {row["instance_id"]: row for row in cases}
    suite.require(len(expected) == len(cases) == 64, "Panel must contain 64 unique IDs.")
    for profile, rows in models.items():
        suite.require(len(rows) == len({r["instance_id"] for r in rows}) == 64
                      and {r["instance_id"] for r in rows} == set(expected), "Missing/duplicate panel rows.")
        for row in rows:
            case = expected[row["instance_id"]]
            suite.require(all(row[k] == case[k] for k in ("task", "kind", "node_count", "query_budget")),
                          "Selected instance metadata mismatch.")
            suite.require(0 < row["query_count"] <= row["query_budget"]
                          and row["successful_queries"] + row["failure_count"] == row["query_count"]
                          and len(row["decisions"]) == row["successful_queries"]
                          and row["failure_count"] in (0, 1)
                          and (row["feasible"] or row["failure_count"] == 1),
                          "Incomplete terminal episode or inconsistent call accounting.")
        counts = Counter((r["task"], r["node_count"]) for r in rows)
        suite.require(set(counts) == {(task, size) for task in TASKS for size in (8, 12)}
                      and all(count == (4 if task in EXACT else 2)
                              for (task, _), count in counts.items()), "Task/size denominators differ.")


def check_metrics(report, models):
    actual = expanded.metrics(models)
    indexed = {(r["profile"], r["task"], r["node_count"]): r for r in actual}
    saved = report["metrics"]
    suite.require(len(saved) == len(indexed)
                  and len({(r["profile"], r["task"], r["node_count"]) for r in saved}) == len(saved),
                  "Missing or duplicate metric cells.")
    for metric in saved:
        key = metric["profile"], metric["task"], metric["node_count"]
        suite.require(key in indexed and metric.keys() <= indexed[key].keys()
                      and prior.matrix.same_aggregates(
                          metric, {k: indexed[key][k] for k in metric}),
                      "Metrics disagree with all scheduled audit rows.")
    for profile, rows in models.items():
        source = report["provenance"][profile]
        calls = source.get("actual_calls", source.get("selected_calls"))
        suite.require(calls == sum(r["query_count"] for r in rows), "Selected actual call count differs.")
    return actual


def verified(source, profile, cases, evidence):
    seal = suite.read_json(source / "export-seal.json")
    evidence.track(source / "export-seal.json")
    suite.require(seal["status"] == "completed"
                  and {"verified-summary.json", "episodes.csv"} <= seal["artifact_sha256"].keys(),
                  "Missing terminal export seal or audit artifacts.")
    for name, digest in seal["artifact_sha256"].items():
        path = source / name
        suite.require(not Path(name).is_absolute() and ".." not in Path(name).parts
                      and path.resolve().is_relative_to(source.resolve()) and not path.is_symlink(),
                      "Invalid sealed export path.")
        evidence.track(path, digest)
    report = suite.read_json(source / "verified-summary.json")
    status = "verified_completed_bounded_pilot_export" if profile == "bounded" else \
        "verified_terminal_expanded_pilot_export"
    suite.require(report["status"] == status and report["episodes_per_profile"] == 64,
                  "Unexpected completed report.")
    if profile == "bounded":
        suite.require(report["canonical_initial_requests_identical"] is True,
                      "Bounded canonical initial requests not verified.")
    else:
        suite.require(report["profile"] == profile and report["source_status"] in prior.TERMINAL,
                      "Only fully completed candidate lanes enter these paper tables.")
    for name, digest in report["source_sha256"].items():
        path = REPO / name
        suite.require(path.resolve().is_relative_to(REPO) and not path.is_symlink(),
                      "Invalid original evidence path.")
        evidence.track(path, digest)
    models = read_episodes(source / "episodes.csv")
    expected = {*prior.HISTORICAL, "qwen27_grammar" if profile == "bounded" else profile}
    suite.require(set(models) == expected, "Unexpected source profiles.")
    verify_panel_rows(models, cases)
    metrics = check_metrics(report, models)
    return report, models, metrics


def combine(sources):
    """Require identical reused histories, not merely matching aggregate accuracy."""
    merged = {}
    for models in sources:
        for profile, rows in models.items():
            ordered = sorted(({k: row[k] for k in CORE if k in row} for row in rows),
                             key=lambda row: row["instance_id"])
            if profile in merged:
                suite.require(prior.matrix.same_aggregates(merged[profile], ordered),
                              "Same-selected historical reference mismatch.")
            else:
                merged[profile] = ordered
    suite.require(set(PROFILES) <= merged.keys(), "Missing required expanded-paper profiles.")
    return {profile: merged[profile] for profile in PROFILES}


def tables(metrics):
    all_rows = [row for row in metrics if row["node_count"] == "all"]
    cells = {(row["task"], row["profile"]): row for row in all_rows}
    suite.require(len(cells) == len(all_rows) == 60, "Missing or duplicated task/profile table cells.")
    exact, feasible, optima, gaps = [], [], [], []
    for task in TASKS:
        own = [cells[task, profile] for profile in PROFILES]
        denominator = 8 if task in EXACT else 4
        suite.require(all(row["episodes"] == denominator for row in own), "Wrong scheduled denominator.")
        if task in EXACT:
            exact.append([TASK_LABELS[task], *(f'{r["hits"]}/8' for r in own)])
        else:
            feasible.append([TASK_LABELS[task], *(f'{r["feasible"]}/4' for r in own)])
            optima.append([TASK_LABELS[task], *(f'{r["hits"]}/4' for r in own)])
            values = []
            for row in own:
                gap = row["mean_absolute_gap_feasible"]
                suite.require(row["objective_gap_denominator"] == row["feasible"]
                              and (gap is None) == (row["feasible"] == 0),
                              "Conditional gap denominator or null contract differs.")
                values.append(r"\textemdash" if gap is None else
                              f"{gap:.6f}" if task == "community_bipartition_construct" else f"{gap:.2f}")
            gaps.append([TASK_LABELS[task], *values])
    exact.append(["Exact total", *(f'{sum(cells[t, p]["hits"] for t in EXACT)}/48' for p in PROFILES)])
    execution = []
    for profile, label in zip(PROFILES, LABELS):
        own = [cells[task, profile] for task in TASKS]
        execution.append([label, str(sum(r["episodes"] for r in own)),
                          str(sum(r["failed_episodes"] for r in own)),
                          f'{sum(cells[t, profile]["feasible"] for t in EXACT)}/48',
                          f'{sum(cells[t, profile]["feasible"] for t in OPTIMIZATION)}/16'])
    return {
        "exact.tex": latex_table(("Task", *LABELS), exact),
        "construction-feasible.tex": latex_table(("Task", *LABELS), feasible),
        "construction-optima.tex": latex_table(("Task", *LABELS), optima),
        "construction-gaps.tex": latex_table(("Task", *LABELS), gaps),
        "coverage.tex": latex_table(
            ("Configuration", "Scheduled", "Failed", "Exact returned", "Constructed"), execution),
    }


def diagnostics(models, reports, evidence):
    result = {}
    for profile, rows in models.items():
        exact = [r for r in rows if r["kind"] == "exact"]
        construction = [r for r in rows if r["kind"] == "optimization"]
        result[profile] = {
            "actual_calls": sum(r["query_count"] for r in rows),
            "failed_calls": sum(r["failure_count"] for r in rows),
            "exact_correct": sum(prior.hit(r) for r in exact),
            "exact_returned": sum(r["feasible"] for r in exact),
            "exact_failed": sum(r["failure_count"] for r in exact),
            "construction_feasible": sum(r["feasible"] for r in construction),
            "construction_optima": sum(prior.hit(r) for r in construction),
            "construction_failed": sum(r["failure_count"] for r in construction),
        }
        if profile in NEW:
            categories = reports[profile]["provenance"][profile]["failure_categories"]
            suite.require(all(type(v) is int and v >= 0 for v in categories.values())
                          and sum(categories.values()) == result[profile]["failed_calls"],
                          "Failure classification/count mismatch.")
            result[profile]["failure_categories"] = categories
    for profile in ("decider", "kev"):
        report = reports[profile]
        path = Path(report["provenance"][profile]["run"]) / "attempts.jsonl"
        name = expanded.relative(path)
        suite.require(name in report["source_sha256"], "Native diagnostics need sealed original attempts.")
        evidence.track(path, report["source_sha256"][name])
        attempts = {a["instance_id"]: a for a in suite.read_rows(path) if a["step"] == 0}
        degree = [r for r in models[profile] if r["task"] == "degree_exact"]
        for row in degree:
            attempt = attempts[row["instance_id"]]
            ids = [o["id"] for o in attempt["request"]["options"]]
            suite.require(len(ids) == row["node_count"] and set(ids) ==
                          {str(v) for v in range(row["node_count"])}
                          and attempt["selected_option_id"] == row["prediction"]
                          and row["prediction"] in ids
                          and attempt["action_observation"]["eligible"] is True,
                          "Native degree vocabulary or semantic mapping differs.")
        errors = [int(r["prediction"]) - int(r["answer"]) for r in degree]
        result[profile]["degree"] = {
            "full_8_and_12_class_coverage": True,
            "predictions": [{"size": r["node_count"], "answer": r["answer"],
                             "prediction": r["prediction"]} for r in degree],
            "varied_predictions": len({r["prediction"] for r in degree}) > 1,
            "global_plus_one": all(v == 1 for v in errors),
            "global_minus_one": all(v == -1 for v in errors),
        }
        result[profile]["all_B_cut_decisions"] = all(
            set(r["decisions"]) == {"B"} for r in models[profile] if r["task"] == "maxcut_construct")
        result[profile]["all_A_community_decisions"] = all(
            set(r["decisions"]) == {"A"} for r in models[profile]
            if r["task"] == "community_bipartition_construct")
    return result


def numbers(metrics, audit):
    names = dict(zip(PROFILES, ("Jev", "Grammar", "LargeThinking", "SmallThinking", "Decider", "Kev")))
    commands = []
    for profile, prefix in names.items():
        for suffix, field in (("Exact", "exact_correct"), ("Returned", "exact_returned"),
                              ("Calls", "actual_calls"), ("Failures", "failed_calls"),
                              ("ExactFailures", "exact_failed"), ("Optima", "construction_optima")):
            value = audit[profile][field]
            denominator = "/48" if suffix == "Exact" else "/16" if suffix == "Optima" else ""
            commands.append(r"\newcommand{\EP" + prefix + suffix + "}{" + str(value) + denominator + "}")
    for profile, prefix in (("jev_action", "Jev"), ("kev", "Kev")):
        metric = next(r for r in metrics if r["profile"] == profile
                      and r["task"] == "community_bipartition_construct" and r["node_count"] == "all")
        commands.append(r"\newcommand{\EP" + prefix + "CommunityGap}{" +
                        f'{metric["mean_absolute_gap_feasible"]:.6f}' + "}")
    return "\n".join(commands) + "\n"


def export(output, expanded_root=FINALIZATION / "exports", bounded=BOUNDED):
    output = output.resolve()
    suite.require(output.is_relative_to(REPO) and not output.exists(), "Output exists or is not local.")
    evidence = Evidence()
    evidence.track(__file__)
    evidence.track(expanded.__file__)
    evidence.track(prior.__file__)
    evidence.track(REPO / "scripts/build_paper_evidence.py")
    cases_path = prior.panel.DEFAULT_ROOT / "pilot-cases.jsonl"
    evidence.track(cases_path)
    cases = suite.read_rows(cases_path)
    reports, sources = {}, []
    for profile, path in (("bounded", bounded), *((name, expanded_root / name) for name in NEW)):
        report, models, _ = verified(path, profile, cases, evidence)
        reports[profile] = report
        sources.append(models)
    models = combine(sources)
    metrics = expanded.metrics(models)
    audit = diagnostics(models, reports, evidence)
    rendered = tables(metrics)
    rendered["numbers.tex"] = numbers(metrics, audit)
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, text in rendered.items():
        (output / name).write_text(text)
    suite.write_json(output / "audit.json", {"metrics": metrics, "diagnostics": audit})
    suite.write_json(output / "provenance.json", {
        "status": "verified_complete_sources", "source_sha256": evidence.hashes,
        "new_model_calls": 0, "profiles": list(PROFILES),
        "generated_sha256": {p.name: suite.digest(p) for p in sorted(output.iterdir())},
        "scope": "Same 64 selected episodes; whole historical lanes verified upstream, "
        "sealed source bytes rechecked here, every CSV denominator/metric recomputed and "
        "all shared reference trajectories required identical. Feasibility and all-scheduled "
        "optimum hits are separate; conditional gaps with no feasible episodes are null. "
        "No pooled construction objective, significance claim, universal ranking or latency control.",
    })
    return {"status": "verified_and_exported", "output": str(output), "new_model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expanded-root", type=Path, default=FINALIZATION / "exports")
    parser.add_argument("--bounded", type=Path, default=BOUNDED)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.expanded_root, args.bounded)))
