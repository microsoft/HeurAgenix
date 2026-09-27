#!/usr/bin/env python3
"""Offline twelve-model publication; frozen experiments and ten-model values stay intact.

Run with the existing project Python:
  python scripts/export_twelve_model_publication.py --publish
  python scripts/export_twelve_model_publication.py --output output/new-publication
Tests: PYTHONPATH=tests python -m unittest test_twelve_model_publication

Custom --output directories include all tables without changing the default paper
directory; --paper optionally copies tables to another new project-local directory.
New output/paper directories are mandatory. Publication replaces only the public
JSON/CSV, after both runner and independent bundled-scorer verification succeeds.
Checkpoint manifests and metadata are checked, never weight payloads. No inference,
GPU initialization, network calls, or provider invocation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import export_website_data as website

build, suite = website.build, website.build.suite
LANES = ("qwen38_27b_bf16", "qwen25_72b_bf16")
STRONGER = REPO / "output/stronger-local-execution-20260926-v3"
OUTPUT = REPO / "output/twelve-model-publication-20260927-v2"
PAPER = REPO / "paper/generated-twelve-model"
VERSION = "canonical-20260927-v14"
FLOAT_TOLERANCE = 1e-12
DETAILS = {
    LANES[0]: ("Qwen3.8-27B", "27B",
               "Newer Qwen3.8 generation; separate inspected-bank extension, not fresh confirmation."),
    LANES[1]: ("Qwen2.5-72B", "72B",
               "Older-generation Qwen2.5 (2024); not a scale-only ablation against Qwen3.5 or Qwen3.8."),
}


def checked_hash(path, expected):
    suite.require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                  "Invalid source digest.")
    suite.require(path.is_file() and not path.is_symlink()
                  and suite.digest(path) == expected, "Source/seal hash differs.")


def checked_map(root, hashes, required=()):
    suite.require(isinstance(hashes, dict) and set(required) <= hashes.keys(),
                  "Required sealed artifacts missing.")
    for name, expected in hashes.items():
        path = root / name
        suite.require(not Path(name).is_absolute() and ".." not in Path(name).parts
                      and path.resolve().is_relative_to(root.resolve()), "Unsafe sealed path.")
        checked_hash(path, expected)


def compare_replay(actual, recorded):
    """Allow only finite float roundoff (Python sum changed across interpreter versions)."""
    if isinstance(actual, dict):
        suite.require(isinstance(recorded, dict) and actual.keys() == recorded.keys(),
                      "Replay fields differ.")
        for key in actual:
            compare_replay(actual[key], recorded[key])
    elif isinstance(actual, list):
        suite.require(isinstance(recorded, list) and len(actual) == len(recorded),
                      "Replay list length differs.")
        for left, right in zip(actual, recorded, strict=True):
            compare_replay(left, right)
    elif type(actual) is float:
        suite.require(type(recorded) is float and math.isfinite(actual)
                      and math.isfinite(recorded)
                      and math.isclose(actual, recorded, rel_tol=0, abs_tol=FLOAT_TOLERANCE),
                      "Replay float differs beyond roundoff tolerance.")
    else:
        suite.require(type(actual) is type(recorded) and actual == recorded,
                      "Replay value differs.")


def verify_lane(root, lane, workspace):
    registration = suite.read_json(root / "registration.json")
    checked_hash(root / "registration.json", suite.read_json(root / "registration.sha256"))
    suite.require(registration["lane"] == lane and registration["episodes"] == 960
                  and registration["maximum_calls"] == 1920
                  and registration["task_counts"] == build.COUNTS, "Lane identity differs.")
    checked_map(REPO, registration["source_sha256"])
    checked_map(root / "source-snapshot", registration["source_sha256"])
    run, independent = root / "run", root / "independent-v3"
    checked_map(run, suite.read_json(run / "seal.json"),
                ("manifest.json", "attempts.jsonl", "scores.jsonl", "summary.json", "terminal.json"))
    checked_map(independent, suite.read_json(independent / "seal.json"),
                ("report.json", "construction-gaps.csv"))
    verification = suite.read_json(root / "verification.json")
    checked_hash(independent / "report.json", verification["independent_replay_sha256"])
    checked_hash(independent / "construction-gaps.csv", verification["construction_gaps_sha256"])
    old = suite.read_json(independent / "report.json")
    checked_map(REPO, old["replay_source_sha256"],
                ("scripts/stronger_local_replay_v1.py", "scripts/stronger_local_replay_v3.py"))
    checked_hash(root / "registration.json", old["input_registration_sha256"])
    checked_hash(run / "seal.json", old["input_run_seal_sha256"])
    recovered = workspace / lane
    subprocess.run(
        [sys.executable, str(REPO / "scripts/stronger_local_replay_v3.py"),
         "--root", str(root), "--bundle", registration["bundle"], "--output", str(recovered)],
        cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    fresh = suite.read_json(recovered / "report.json")
    compare_replay(fresh, old)
    checked_hash(recovered / "report.json", suite.digest(independent / "report.json"))
    checked_hash(recovered / "construction-gaps.csv",
                 suite.digest(independent / "construction-gaps.csv"))
    suite.require(fresh["verified"] is True and fresh["episodes"] == 960
                  and fresh["calls"] == 1920 and fresh["model_calls"] == 0
                  and fresh["run_status"] == "completed"
                  and fresh["provider_import_blocker_active"] is True
                  and fresh["provider_modules_imported"] == [], "Independent replay incomplete.")

    from scripts import stronger_local_runtime_v1 as runtime

    registered, instances = runtime.load(root)
    artifact = runtime.verify_artifacts(lane, Path(registered["materials"]), payload=False)
    suite.require(artifact["revision"] == registered["checkpoint_revision"],
                  "Recorded checkpoint revision differs.")
    checked_hash(runtime.base.REFERENCE / "status.json", registered["reference_status_sha256"])
    reference = {r["instance_id"]: r["request"] for r in
                 suite.read_rows(runtime.base.REFERENCE / "attempts.jsonl") if r["step"] == 0}
    suite.require(all(reference[r["instance_id"]] == r["request"]
                      for r in suite.read_rows(root / "initial-requests.jsonl")),
                  "Initial requests differ from canonical reference.")
    manifest = suite.read_json(run / "manifest.json")
    suite.require(manifest["model"] == registered["model"]
                  and manifest["registration_sha256"] == suite.digest(root / "registration.json")
                  and manifest["injected_client"] is False
                  and manifest["injected_preflight"] is False, "Invalid runner identity.")
    attempts = suite.read_rows(run / "attempts.jsonl")
    rows = runtime.canonical.replay(instances, attempts, runtime.configuration(lane),
                                    native_action=False)
    suite.require(rows == suite.read_rows(run / "scores.jsonl"), "Runner score replay differs.")
    compare_replay(suite.json_value(suite.summarize(rows)), suite.read_json(run / "summary.json"))
    terminal = suite.read_json(run / "terminal.json")
    suite.require(len(rows) == 960 and len(attempts) == terminal["calls"] == 1920
                  and terminal["status"] == "completed" and terminal["failure_counts"] == {}
                  and all(r["feasible"] and r["failure_count"] == 0 for r in rows),
                  "Extension coverage/calls/failures differ.")
    compare_replay(dict(verified=True, status="completed", episodes=960, calls=1920, model_calls=0),
                   verification["runner"])
    source = dict(
        registrationSha256=suite.digest(root / "registration.json"),
        runSealSha256=suite.digest(run / "seal.json"),
        independentSealSha256=suite.digest(independent / "seal.json"),
        independentReplaySha256=suite.digest(independent / "report.json"),
        independentReplayRecomputedSha256=suite.digest(recovered / "report.json"),
        verificationSha256=suite.digest(root / "verification.json"),
        artifactManifestSha256=registered["artifact_manifest_sha256"],
        bundleManifestSha256=registered["bundle_manifest_sha256"],
        initialRequestsSha256=registered["initial_requests_sha256"],
        verifiedSourceHashes=len(registered["source_sha256"]),
        recordedModelCalls=1920, modelCalls=0,
    )
    return rows, fresh, source


def aggregate_lane(lane, rows):
    suite.require(len(rows) == 960 and len({r["instance_id"] for r in rows}) == 960
                  and Counter(r["task"] for r in rows) == build.COUNTS,
                  "Extension episode inventory differs.")
    metrics = []
    for task in build.COUNTS:
        exact = task in build.EXACT
        own = [r for r in rows if r["task"] == task]
        cells = {}
        for size in ("all", *website.SIZES):
            group = own if size == "all" else [r for r in own if r["node_count"] == size]
            expected = build.COUNTS[task] if size == "all" else (
                8 * size if task == "degree_exact" else 16 if exact else 8)
            suite.require(len(group) == expected, "Extension size inventory differs.")
            feasible = [r for r in group if r["feasible"]]
            failures = sum(not r["feasible"] and r["failure_count"] > 0 for r in group)
            unattempted = sum(not r["feasible"] and r["query_count"] == 0 for r in group)
            hits = sum(bool(r["correct" if exact else "optimal"]) for r in feasible)
            cell = dict(
                model=lane, task=task, nodeCount=size, scheduled=len(group),
                feasible=len(feasible), hits=hits, unsupported=0,
                failures=failures, unattempted=unattempted,
                accuracy=hits / len(group) if exact else None,
                optimalRate=None if exact else hits / len(group),
                meanGap=None if exact or not feasible else
                math.fsum(r["absolute_gap"] for r in feasible) / len(feasible),
                meanObjective=None if exact or not feasible else
                math.fsum(r["objective"] for r in feasible) / len(feasible),
            )
            cells[size] = cell
            metrics.append(cell)
        if exact:
            cells["all"]["accuracy"] = math.fsum(cells[s]["accuracy"] for s in website.SIZES) / 5
    return metrics


def validate_public(data):
    expected = {(p, task, size) for p in [p for p, _ in build.PROFILES] + list(LANES)
                for task in build.COUNTS for size in ("all", *website.SIZES)}
    index = {(r["model"], r["task"], r["nodeCount"]): r for r in data["metrics"]}
    suite.require(len(index) == len(data["metrics"]) == 720 and set(index) == expected
                  and len(data["models"]) == 12
                  and {m["id"] for m in data["models"]} == {k[0] for k in expected},
                  "Public model/metric inventory differs.")
    for key, row in index.items():
        suite.require(set(row) == set(website.METRIC_FIELDS), "Public metric fields differ.")
        exact = row["task"] in build.EXACT
        for field in ("scheduled", "feasible", "hits", "unsupported", "failures", "unattempted"):
            suite.require(type(row[field]) is int and row[field] >= 0, "Invalid coverage count.")
        suite.require(row["scheduled"] > 0 and 0 <= row["hits"] <= row["feasible"]
                      and row["feasible"] + row["unsupported"] + row["failures"]
                      + row["unattempted"] == row["scheduled"], "Coverage does not partition.")
        suite.require((row["optimalRate"] is None and row["meanGap"] is None
                       and row["meanObjective"] is None) if exact else row["accuracy"] is None,
                      "Exact/construction null contract differs.")
        if exact:
            accuracy = (math.fsum(index[key[0], key[1], s]["hits"] /
                                 index[key[0], key[1], s]["scheduled"] for s in website.SIZES) / 5
                        if key[2] == "all" else row["hits"] / row["scheduled"])
            suite.require(math.isclose(row["accuracy"], accuracy, abs_tol=1e-12, rel_tol=0),
                          "Exact accuracy is not macro-size.")
        else:
            suite.require(row["optimalRate"] == row["hits"] / row["scheduled"],
                          "Optimum denominator differs.")
            for field in ("meanGap", "meanObjective"):
                value = row[field]
                suite.require(value is None if row["feasible"] == 0 else
                              type(value) in (int, float) and math.isfinite(value),
                              "Invalid feasible-only objective/gap.")
            suite.require(row["meanGap"] is None or row["meanGap"] >= 0, "Negative additive gap.")
        if key[2] == "all":
            for field in ("scheduled", "feasible", "hits", "unsupported", "failures", "unattempted"):
                suite.require(row[field] == sum(index[key[0], key[1], s][field] for s in website.SIZES),
                              "All-size coverage differs.")
    return index


def public_data(original, extensions, sources):
    data = {**original, "models": list(original["models"]), "metrics": list(original["metrics"]),
            "provenance": dict(original["provenance"])}
    data.update(version=VERSION, evidenceDate="2026-09-27")
    data["scope"] += (
        " Two completed local BF16 extensions use the identical bank and initial requests: "
        "Qwen3.8-27B and older-generation Qwen2.5-72B. Distinct from Qwen3.5; "
        "not a scale-only ablation or fresh confirmation. Construction additive gaps are primary; "
        "optimum hits are supplemental. The extensions recorded 3,840 execution calls; "
        "this offline export makes zero new model calls.")
    safe_sources = {}
    digest_fields = (
        "registrationSha256", "runSealSha256", "independentSealSha256",
        "independentReplaySha256", "independentReplayRecomputedSha256", "verificationSha256",
        "artifactManifestSha256", "bundleManifestSha256", "initialRequestsSha256")
    for lane in LANES:
        name, parameters, note = DETAILS[lane]
        data["models"].append(dict(id=lane, name=name, family="Qwen", group="decision_matrix",
                                   parameters=parameters,
                                   protocol="BF16; no thinking; greedy choice grammar; 64-token ceiling",
                                   note=note))
        data["metrics"].extend(aggregate_lane(lane, extensions[lane]))
        source = sources[lane]
        for field in digest_fields:
            suite.require(isinstance(source[field], str)
                          and re.fullmatch(r"[0-9a-f]{64}", source[field]), "Unsafe public digest.")
        suite.require(type(source["verifiedSourceHashes"]) is int and source["verifiedSourceHashes"] > 0
                      and source["recordedModelCalls"] == 1920 and source["modelCalls"] == 0,
                      "Invalid extension provenance.")
        safe_sources[lane] = {field: source[field] for field in
                             (*digest_fields, "verifiedSourceHashes", "recordedModelCalls", "modelCalls")}
    data["provenance"].update(strongerSources=safe_sources, recordedExtensionModelCalls=3840,
                              runnerSummaryFloatTolerance=FLOAT_TOLERANCE,
                              weightPayloadsRehashed=False)
    validate_public(data)
    return data


def render(data):
    index = validate_public(data)
    labels = []
    for profile, label in build.PROFILES:
        labels.append((profile, label))
        if profile == "qwen27_grammar":
            labels.extend((lane, DETAILS[lane][0]) for lane in LANES)
    exact, optimal, gaps, coverage = [], [], [], []
    for profile, label in labels:
        own = {task: index[profile, task, "all"] for task in build.COUNTS}
        exact.append([label, *(f'{100 * own[t]["accuracy"]:.2f}' for t in build.EXACT)])
        optimal.append([label, *(f'{own[t]["hits"]}/40' for t in build.OPTIMIZATION)])
        gaps.append([label, *("--" if own[t]["meanGap"] is None else f'{own[t]["meanGap"]:.4f}'
                              for t in build.OPTIMIZATION)])
        coverage.append([label, f'{sum(own[t]["feasible"] for t in build.EXACT)}/800',
                         f'{sum(own[t]["feasible"] for t in build.OPTIMIZATION)}/160',
                         *(str(sum(r[field] for r in own.values()))
                           for field in ("unsupported", "failures", "unattempted"))])
    return {
        "exact.tex": build.table(("Model", *(build.TASK_LABELS[t] for t in build.EXACT)), exact),
        "optimal.tex": build.table(("Model", *(build.TASK_LABELS[t] for t in build.OPTIMIZATION)), optimal),
        "gaps.tex": build.table(("Model", *(build.TASK_LABELS[t] for t in build.OPTIMIZATION)), gaps),
        "coverage.tex": build.table(("Model", "Exact feasible", "Construct feasible",
                                     "Input rejects", "Exec. failures", "Unattempted"), coverage),
    }


def publish(files):
    target = REPO / "web/data"
    previous = {name: (target / name).read_bytes() for name in files}
    try:
        for name, content in files.items():
            (target / name).write_bytes(content)
        suite.require(all((target / name).read_bytes() == content for name, content in files.items()),
                      "Public publication read-back differs.")
    except BaseException:
        for name, content in previous.items():
            (target / name).write_bytes(content)
        raise


def export(source=build.SOURCE, stronger=STRONGER, output=OUTPUT, paper=None, *, publish_web=False):
    source, stronger, output = map(Path, (source, stronger, output))
    paper = Path(paper) if paper is not None else (
        PAPER if output.resolve() == OUTPUT.resolve() else None)
    for destination in (output, *([paper] if paper is not None else [])):
        suite.require(not destination.exists() and not destination.is_symlink()
                      and destination.resolve().is_relative_to(REPO)
                      and not destination.resolve().is_relative_to(source.resolve())
                      and not destination.resolve().is_relative_to(stronger.resolve()),
                      "Use new project-local publication directories outside source evidence.")
    if paper is not None:
        suite.require(not output.resolve().is_relative_to(paper.resolve())
                      and not paper.resolve().is_relative_to(output.resolve()), "Overlapping outputs.")
    output.mkdir(parents=True)
    workspace = output / ".verification"
    workspace.mkdir()
    paper_created = False
    try:
        build.export(source, workspace / "original")
        original_summary = suite.read_json(source / "verified-summary.json")
        original_proof = suite.read_json(workspace / "original/provenance.json")
        suite.require(original_proof["verified_input_hashes"] == 1550,
                      "Original canonical source inventory differs.")
        original = website.public_data(original_summary, original_proof)
        checked_hash(stronger / "registration.json", suite.read_json(stronger / "registration.sha256"))
        parent = suite.read_json(stronger / "registration.json")
        checked_map(REPO, parent["source_sha256"])
        checked_map(stronger / "source-snapshot", parent["source_sha256"])
        extensions, sources = {}, {}
        for lane in LANES:
            rows, replay, proof = verify_lane(stronger / lane, lane, workspace)
            extensions[lane], sources[lane] = rows, proof
            computed = {r["task"]: r for r in aggregate_lane(lane, rows) if r["nodeCount"] == "all"}
            for task in build.OPTIMIZATION:
                compare_replay(computed[task]["meanGap"],
                               replay["by_task"][task]["mean_absolute_gap_feasible"])
        data = public_data(original, extensions, sources)
        suite.require(data["metrics"][:600] == original["metrics"], "Original values changed.")
        for profile, field in (("laya", "unsupported"), ("gpt54_default_reasoning", "failures")):
            suite.require(sum(r[field] for r in data["metrics"]
                              if r["model"] == profile and r["nodeCount"] == "all") == 9,
                          "Historical missing outcomes lost.")
        for lane, expected in zip(LANES, (45.501893939393934, 41.238005050505055)):
            degree = next(r for r in data["metrics"]
                          if (r["model"], r["task"], r["nodeCount"]) == (lane, "degree_exact", "all"))
            suite.require(math.isclose(100 * degree["accuracy"], expected, abs_tol=1e-12, rel_tol=0),
                          "Extension degree macro differs.")
        files = website.encoded_files(data)
        tables = render(data)
        for name, text in build.render(original_summary).items():
            original_lines = text.splitlines()
            suite.require(all(line in tables[name].splitlines() for line in original_lines),
                          "Original table rows changed.")
        for name, content in files.items():
            (output / name).write_bytes(content)
        for name, content in tables.items():
            (output / name).write_text(content)
        provenance = dict(status="verified_twelve_model_publication", version=VERSION,
                          models=12, metrics=720, scheduledPerModel=960,
                          maximumCallsPerModel=1920, **data["provenance"],
                          exporterSha256=suite.digest(Path(__file__)),
                          testSha256=suite.digest(REPO / "tests/test_twelve_model_publication.py"))
        suite.write_json(output / "provenance.json", provenance)
        checked_hash(source / "verified-summary.json", original_proof["source_summary_sha256"])
        checked_hash(source / "export-seal.json", original_proof["source_seal_sha256"])
        checked_hash(source.parent / "completion.json", original_proof["source_completion_sha256"])
        for lane in LANES:
            checked_map(stronger / lane / "run", suite.read_json(stronger / lane / "run/seal.json"))
        shutil.rmtree(workspace)
        names = (*files, *tables, "provenance.json")
        suite.write_json(output / "export-seal.json",
                         dict(status="completed", modelCalls=0,
                              artifactSha256={name: suite.digest(output / name) for name in names}))
        checked_map(output, suite.read_json(output / "export-seal.json")["artifactSha256"], names)
        if paper is not None:
            paper.mkdir(parents=True)
            paper_created = True
            for name in (*tables, "provenance.json"):
                shutil.copyfile(output / name, paper / name)
                checked_hash(paper / name, suite.digest(output / name))
        if publish_web:
            publish(files)
        return provenance
    except BaseException:
        shutil.rmtree(output)
        if paper_created:
            shutil.rmtree(paper)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=build.SOURCE)
    parser.add_argument("--stronger", type=Path, default=STRONGER)
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="New project-local directory containing public data and all four tables.")
    parser.add_argument("--paper", type=Path,
                        help="Optional additional new table directory; default only for canonical output.")
    parser.add_argument("--publish", action="store_true", help="Replace public JSON/CSV after verification.")
    args = parser.parse_args()
    proof = export(args.source, args.stronger, args.output, args.paper, publish_web=args.publish)
    print(json.dumps({k: proof[k] for k in
                      ("status", "version", "models", "metrics", "modelCalls", "verifiedInputHashes")}))


if __name__ == "__main__":
    main()
