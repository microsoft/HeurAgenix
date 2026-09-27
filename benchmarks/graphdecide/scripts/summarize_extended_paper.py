#!/usr/bin/env python3
"""Verified task/size tables and paired generation-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
import random
import subprocess
import sys

import networkx as nx

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite


def size_plot(summaries):
    colors = ("#1768ac", "#dd6b20", "#27823b", "#8846a3", "#bd3441", "#0e8b8b", "#595959")
    tasks = sorted(next(iter(summaries.values()))["by_task"])
    height = 80 + 240 * ((len(tasks) + 1) // 2)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="{height}" '
             f'viewBox="0 0 1040 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="11" fill="#222">',
             '<text x="20" y="20">Exact accuracy / optimal-solution rate '
             '(failures included); interpret using the frozen split definition</text>']
    for index, model in enumerate(summaries):
        x, y = 20 + (index % 7) * 145, 42 + 15 * (index // 7)
        color = colors[index % len(colors)]
        parts.append(f'<text x="{x}" y="{y}" fill="{color}">{escape(model)}</text>')
    for index, task in enumerate(tasks):
        x, y = 55 + (index % 2) * 510, 100 + (index // 2) * 240
        parts.append(f'<text x="{x}" y="{y - 18}">{escape(task)}</text>')
        for tick in (0, 0.5, 1):
            top = y + 150 * (1 - tick)
            parts.extend((f'<path d="M{x} {top}h410" stroke="#ddd"/>',
                          f'<text x="{x - 32}" y="{top + 4}">{int(tick * 100)}%</text>'))
        for n in range(5, 13):
            left = x + (n - 5) * 410 / 7
            parts.append(f'<text x="{left - 3}" y="{y + 170}">{n}</text>')
        parts.append(f'<text x="{x + 165}" y="{y + 188}">Number of nodes</text>')
        for model_index, (model, summary) in enumerate(summaries.items()):
            points = []
            for size, cell in sorted(summary["by_task_and_size"][task].items(), key=lambda p: int(p[0])):
                value = cell.get("accuracy", cell.get("optimality_rate"))
                points.append((x + (int(size) - 5) * 410 / 7, y + 150 * (1 - value)))
            color = colors[model_index % len(colors)]
            coordinates = " ".join(f"{a:.2f},{b:.2f}" for a, b in points)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            parts.extend(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="2.5" fill="{color}"/>'
                         for a, b in points)
    parts.append("</g></svg>")
    return "\n".join(parts)


def paired_intervals(instances, left, right, *, repetitions=2000, seed=20260928, clusters=None):
    """Macro-size correctness/optimality differences; resample whole seed clusters."""
    by_model = [{row["instance_id"]: row for row in rows} for rows in (left, right)]
    kinds = {instance["task"]: instance["kind"] for instance in instances}
    if clusters is not None:
        suite.require(set(clusters) == {i["id"] for i in instances},
                      "Every instance needs an explicit generation cluster.")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for instance in instances:
        a, b = (rows[instance["id"]] for rows in by_model)
        key = "correct" if instance["kind"] == "exact" else "optimal"
        cluster = clusters[instance["id"]] if clusters is not None else instance["id"].rsplit("-", 1)[0]
        grouped[instance["task"]][len(instance["state"]["nodes"])][cluster].append(
            int(a[key]) - int(b[key]))
    rng, result = random.Random(seed), {}
    for task, sizes in sorted(grouped.items()):
        strata = [[sum(values) / len(values) for values in clusters.values()]
                  for _, clusters in sorted(sizes.items())]
        observed = sum(sum(s) / len(s) for s in strata) / len(strata)
        sampled = sorted(
            sum(sum(rng.choice(s) for _ in s) / len(s) for s in strata) / len(strata)
            for _ in range(repetitions))
        interval = [sampled[int(0.025 * repetitions)],
                    sampled[min(repetitions - 1, int(0.975 * repetitions))]]
        result[task] = {
            "metric": "macro-size accuracy difference" if kinds[task] == "exact" else
                      "macro-size optimal-solution-rate difference",
            "difference_left_minus_right": observed,
            "percentile_95": interval,
            "degenerate_empirical_interval": interval[0] == interval[1],
            "generation_clusters": sum(len(s) for s in strata),
            "sizes": len(strata), "replicates": repetitions,
            "warning": "Exploratory pointwise interval conditional on the sampled graph families; "
                       "few clusters per size. Zero-width empirical intervals do not establish "
                       "population certainty or equivalence. No pooled cross-task inference.",
        }
    return result


def detail_tables(output, rows, metadata=None):
    degree, quality, families = [], [], []
    for model, scores in rows.items():
        by_degree, by_family = defaultdict(list), defaultdict(list)
        for row in scores:
            if row["task"] == "degree_exact":
                by_degree[row["node_count"], row["answer"]].append(row)
            if metadata is not None:
                family = metadata["cases"][row["instance_id"]]["family"]
                by_family[row["task"], row["node_count"], family].append(row)
            if row["kind"] == "optimization":
                ratio_defined = (row["feasible"] and row["optimum"] > 0
                                 and row["task"] != "community_bipartition_construct")
                quality.append({
                    "model": model, **{key: row[key] for key in (
                        "instance_id", "task", "node_count", "feasible", "objective", "optimum",
                        "absolute_gap")},
                    "objective_ratio": row["objective"] / row["optimum"] if ratio_defined else None,
                    "relative_gap": row["absolute_gap"] / row["optimum"] if ratio_defined else None,
                    "direction": row["direction"],
                })
        for (size, answer), cell in sorted(by_degree.items(), key=lambda item: (
                item[0][0], int(item[0][1]))):
            degree.append({"model": model, "node_count": size, "answer": answer,
                           "instances": len(cell), "feasible": sum(r["feasible"] for r in cell),
                           "correct": sum(r["correct"] for r in cell),
                           "recall": sum(r["correct"] for r in cell) / len(cell)})
        for (task, size, family), cell in sorted(by_family.items()):
            metrics = suite.metrics(cell)
            families.append({"model": model, "task": task, "node_count": size, "family": family,
                             **{key: metrics.get(key) for key in (
                                 "instances", "feasible", "failure_count", "accuracy",
                                 "optimality_rate", "mean_objective_feasible",
                                 "mean_absolute_gap_feasible")}})
    for filename, values in (("degree-class-recall.csv", degree),
                             ("optimization-quality.csv", quality),
                             ("family-metrics.csv", families)):
        if values:
            with (output / filename).open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)


def structural_clusters(metadata):
    """Keep both shared graph identities and non-isomorphic construction twins together."""
    graph = nx.Graph()
    for instance_id, case in metadata["cases"].items():
        node = ("instance", instance_id)
        graph.add_edge(node, ("graph", case["cluster_id"]))
        if case.get("pair_id") is not None:
            graph.add_edge(node, ("pair", case["pair_id"]))
    result = {}
    for component in nx.connected_components(graph):
        instances = sorted(value for kind, value in component if kind == "instance")
        result.update((instance_id, instances[0]) for instance_id in instances)
    return result


def verified_inputs(root, models):
    root = Path(root).resolve()
    frozen = root / "frozen-source/scripts/extended_graph_suite.py"
    if frozen.exists():
        completed = subprocess.run(
            [sys.executable, str(frozen), "report", "--root", str(root), "--models", *models],
            capture_output=True, text=True, check=True,
        )
        report = json.loads(completed.stdout)
        manifest = suite.read_json(root / "manifest.json")
        suite.require(suite.digest(root / "manifest.json") ==
                      suite.read_json(root / "manifest.sha256"), "Manifest changed during verification.")
        suite.require(all(suite.digest(root / name) == expected
                          for name, expected in manifest["input_sha256"].items()),
                      "Inputs changed during verification.")
        instances = suite.read_rows(root / "instances.jsonl")
    else:
        root, manifest, instances, _ = suite.load_plan(root)
        report = suite.report(root, models)
    return root, manifest, instances, report


def export(root, output, models):
    root, manifest, instances, report = verified_inputs(root, models)
    suite.require(manifest["design"] in ("curriculum", "structural_dev", "structural_holdout"),
                  "This design has no declared clustering protocol.")
    clusters, metadata = None, None
    if manifest["design"].startswith("structural_"):
        metadata = suite.read_json(root / "design_metadata.json")
        clusters = structural_clusters(metadata)
    suite.require(all(cell["status"] in ("completed", "completed_with_failures")
                      for cell in report["models"].values()),
                  "Paper table requires completed selected lanes, not observed prefixes.")
    output = Path(output).resolve()
    suite.require(not output.exists(), "Table output already exists.")
    rows = {model: suite.read_rows(root / "runs" / model / "scores.jsonl") for model in models}
    summaries = {model: suite.summarize(values) for model, values in rows.items()}
    timing = {}
    for model in models:
        attempts = suite.read_rows(root / "runs" / model / "attempts.jsonl")
        status = suite.read_json(root / "runs" / model / "status.json")
        latency = sorted(row["latency_seconds"] for row in attempts)
        timing[model] = {
            "attempts": len(attempts), "request_seconds_sum": sum(latency),
            "lane_wall_seconds": status.get("wall_seconds"),
            "started_at_utc": status.get("started_at_utc"),
            "finished_at_utc": status.get("finished_at_utc"),
            "mean_seconds": sum(latency) / len(latency),
            "median_seconds": latency[len(latency) // 2],
            "p95_seconds": latency[min(len(latency) - 1, int(0.95 * len(latency)))],
            "token_totals_available": {
                key: sum(row["usage"][key] for row in attempts if row["usage"][key] is not None)
                if any(row["usage"][key] is not None for row in attempts) else None
                for key in suite.USAGE_KEYS},
            "token_observation_counts": {
                key: sum(row["usage"][key] is not None for row in attempts) for key in suite.USAGE_KEYS},
        }
    comparisons = {
        f"{left}-minus-{right}": paired_intervals(instances, rows[left], rows[right], clusters=clusters)
        for left, right in (("jev", model) for model in models if model != "jev")
        if left in models
    }
    output.mkdir(parents=True, exist_ok=False)
    suite.write_json(output / "verified-report.json", report)
    suite.write_json(output / "paired-intervals.json", comparisons)
    if clusters is not None:
        suite.write_json(output / "analysis-clusters.json", clusters)
    suite.write_json(output / "timing.json", timing)
    with (output / "size-rates.svg").open("x", encoding="utf-8") as stream:
        stream.write(size_plot(summaries))
    suite.write_json(output / "provenance.json", {
        "manifest_sha256": suite.digest(root / "manifest.json"),
        "analysis_code_sha256": suite.digest(Path(__file__)),
        "verification_source": "frozen-source" if (root / "frozen-source").exists() else "current-source",
        "selected_models": list(models),
        "design": manifest["design"],
        "notice": ("Prior-identity-excluded structural challenge, not universal generator-OOD, "
                   "unseen pretraining data or equal-compute evaluation."
                   if manifest["design"] == "structural_holdout" else
                   "Development results, not structure-held-out or equal-compute evaluation."),
    })
    fields = ("model", "task", "node_count", "instances", "feasible", "failure_count",
              "accuracy", "macro_size_accuracy", "optimality_rate",
              "mean_objective_feasible", "mean_absolute_gap_feasible")
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for model, summary in summaries.items():
            for task, metrics in summary["by_task"].items():
                for size, values in [("all", metrics), *summary["by_task_and_size"][task].items()]:
                    writer.writerow({"model": model, "task": task, "node_count": size,
                                     **{key: values.get(key) for key in fields[3:]}})
    detail_tables(output, rows, metadata)
    return {"output": str(output), "models": list(models), "paired_comparisons": len(comparisons)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=suite.MODELS, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output, args.models), sort_keys=True))


if __name__ == "__main__":
    main()
