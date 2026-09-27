#!/usr/bin/env python3
"""Independent bundled-scorer replay, including failures and unattempted cases."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay(instances, attempts, recorded_scores, api, model):
    require(len(instances) == len(recorded_scores) == 960, "Expected all 960 scheduled cases.")
    require(len({i["id"] for i in instances}) == 960 and
            sum(i["query_budget"] for i in instances) == 1920, "Canonical identity/budget differs.")
    require(sum(i["kind"] == "exact" for i in instances) == 800, "Exact denominator differs.")
    require(Counter(i["task"] for i in instances) == {
        "adjacency": 80, "degree_exact": 400, "cycle_detection": 80,
        "pair_connectivity": 80, "distance_threshold": 80, "articulation_point": 80,
        "maxcut_construct": 40, "tsp_construct": 40,
        "lt_influence_construct": 40, "community_bipartition_construct": 40},
        "Canonical task counts differ.")
    require(len(attempts) <= 1920, "Call budget exceeded.")
    position, aborted, groups = 0, False, defaultdict(list)
    for instance, recorded in zip(instances, recorded_scores, strict=True):
        history, own = [], []
        while position < len(attempts) and attempts[position]["instance_id"] == instance["id"]:
            row = attempts[position]
            require(not aborted and not (own and own[-1]["status"] == "failure"),
                    "Attempt after terminal failure.")
            request = api.tasks.next_request(instance, history)
            require(request is not None, "Attempt after completed episode.")
            request = api.canonical_graph_request(instance["task"], request)
            require(row["step"] == len(history)
                    and row["request"] == json.loads(json.dumps(asdict(request))),
                    "Independent own-history request mismatch.")
            if row["status"] == "success":
                require(row["resolved_model"] == model and not row["fatal"]
                        and row["selected_option_id"] in {o.id for o in request.options},
                        "Invalid successful action.")
                history.append(row["selected_option_id"])
            else:
                require(row["status"] == "failure" and row["selected_option_id"] is None,
                        "Invalid failure.")
                aborted = row["fatal"]
            own.append(row)
            position += 1
        score = api.tasks.score(instance, history)
        if position < len(attempts):
            require(score["feasible"] or own and own[-1]["status"] == "failure",
                    "Unfinished episode skipped.")
        require(recorded["instance_id"] == instance["id"] and recorded["decisions"] == history
                and recorded["query_count"] == len(own)
                and recorded["failure_count"] == sum(r["status"] == "failure" for r in own),
                "Recorded episode accounting differs.")
        for key, value in score.items():
            actual = recorded[key]
            require(math.isclose(value, actual, rel_tol=0, abs_tol=1e-12)
                    if type(value) is float else value == actual,
                    f"Independent score mismatch: {instance['id']} {key}")
        groups[instance["task"]].append({**score, "attempted": bool(own),
                                        "failures": sum(r["status"] == "failure" for r in own)})
    require(position == len(attempts), "Unknown/out-of-order attempts.")
    metrics = {}
    for task, group in sorted(groups.items()):
        feasible = [r for r in group if r["feasible"]]
        cell = {"scheduled": len(group), "feasible": len(feasible),
                "feasible_coverage": len(feasible) / len(group),
                "unattempted": sum(not r["attempted"] for r in group),
                "failed_episodes": sum(bool(r["failures"]) for r in group)}
        if "correct" in group[0]:
            cell.update(correct=sum(r["correct"] for r in group),
                        accuracy_all_scheduled=sum(r["correct"] for r in group) / len(group))
        else:
            cell.update(
                mean_absolute_gap_feasible=math.fsum(r["absolute_gap"] for r in feasible)
                / len(feasible) if feasible else None,
                gap_denominator=len(feasible),
                optimum_count_supplemental=sum(r["optimal"] for r in group),
                mean_objective_feasible=math.fsum(r["objective"] for r in feasible)
                / len(feasible) if feasible else None)
        metrics[task] = cell
    return {"verified": True, "episodes": 960, "calls": len(attempts),
            "task_counts": dict(Counter(i["task"] for i in instances)),
            "by_task": metrics, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle, root = args.bundle.resolve(), args.root.resolve()
    registration = read(root / "registration.json")
    require(digest(root / "registration.json") == read(root / "registration.sha256"),
            "Registration hash differs.")
    require(str(bundle) == registration["bundle"]
            and digest(bundle / "manifest.json") == registration["bundle_manifest_sha256"],
            "Bundle identity differs.")
    sys.path.insert(0, str(bundle))
    api = importlib.import_module("release_api")
    require(Path(api.__file__).resolve() == bundle / "release_api.py"
            and Path(api.tasks.__file__).resolve() ==
            bundle / "src/benchmark/extended_tasks.py", "Ambient scorer imports forbidden.")
    api.verify(bundle)
    instances = api.labeled_instances(bundle, "structural_challenge", api.manifest_at(bundle))
    out = root / "run"
    for name, expected in read(out / "seal.json").items():
        require(digest(out / name) == expected, "Run seal differs.")
    manifest = read(out / "manifest.json")
    require(manifest["model"] == registration["model"]
            and manifest["registration_sha256"] == digest(root / "registration.json"),
            "Run/registration model identity differs.")
    require(not manifest["injected_client"] and not manifest["injected_preflight"],
            "Injected execution cannot be real model evidence.")
    result = replay(instances, rows(out / "attempts.jsonl"), rows(out / "scores.jsonl"),
                    api, registration["model"]["model"])
    terminal = read(out / "terminal.json")
    require(terminal["calls"] == result["calls"], "Terminal call count differs.")
    if terminal["status"].startswith("completed"):
        require(all(cell["feasible"] + cell["failed_episodes"] == cell["scheduled"]
                    for cell in result["by_task"].values()),
                "Completed status hides incomplete/unattempted episodes.")
    result["run_status"] = terminal["status"]
    result["provider_modules_imported"] = sorted(
        name for name in sys.modules if name.startswith(("src.clients", "vllm", "transformers")))
    require(not result["provider_modules_imported"], "Independent replay imported a provider.")
    (root / "independent-replay.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    fields = ("task", "scheduled", "feasible", "feasible_coverage", "unattempted",
              "failed_episodes", "mean_absolute_gap_feasible", "gap_denominator",
              "optimum_count_supplemental")
    with (root / "construction-gaps.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for task, metric in result["by_task"].items():
            if "mean_absolute_gap_feasible" in metric:
                writer.writerow({"task": task, **metric})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
