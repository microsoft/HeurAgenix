#!/usr/bin/env python3
"""Generate manuscript tables and exact on-policy regret from sealed real runs."""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts import extended_reference_analysis as references
from scripts import graph_release_api as release
from scripts import summarize_release_matrix as matrix_tools
from scripts.construction_prefix_controls import continuation_values
from scripts.summarize_extended_paper import paired_intervals, structural_clusters
from src.benchmark import extended_tasks as tasks

PROFILES = ("jev_action", "qwen4_grammar", "qwen9_grammar", "qwen4_token_scores")
LABELS = ("Jev action", "4B grammar", "9B grammar", "4B scores")
EXACT = ("adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
         "distance_threshold", "articulation_point")
OPTIMIZATION = ("maxcut_construct", "tsp_construct", "lt_influence_construct",
                "community_bipartition_construct")
TASK_LABELS = dict(zip(EXACT + OPTIMIZATION, (
    "Adjacency", "Degree", "Cycle", "Connectivity", "Distance", "Articulation",
    "MaxCut", "TSP", "Deterministic LT", "Modularity")))


def trajectory_regrets(instance, decisions, value_fn=continuation_values):
    """Best-continuation loss telescopes to the observed final objective gap."""
    score = tasks.score(instance, decisions)
    suite.require(instance["kind"] == "optimization" and score["feasible"],
                  "Regret decomposition requires a complete construction.")
    minimize = instance["task"] == "tsp_construct"
    previous = float(score["optimum"])
    steps = []
    for step, action in enumerate(decisions):
        values = value_fn(instance, decisions[:step])
        best = float(min(values.values()) if minimize else max(values.values()))
        suite.require(math.isclose(best, previous, rel_tol=0, abs_tol=1e-12),
                      "Consecutive continuation values disagree.")
        chosen = float(values[action])
        regret = chosen - best if minimize else best - chosen
        suite.require(regret >= -1e-12, "Negative continuation regret.")
        steps.append({
            "step": step + 1, "action": action, "best_before": best,
            "best_after": chosen, "incremental_regret": regret,
            "informative": len(set(values.values())) > 1,
            "locally_optimal": abs(regret) <= 1e-12,
        })
        previous = chosen
    total = math.fsum(s["incremental_regret"] for s in steps)
    suite.require(math.isclose(previous, float(score["objective"]), rel_tol=0, abs_tol=1e-12)
                  and math.isclose(total, float(score["absolute_gap"]), rel_tol=0, abs_tol=1e-12),
                  "Regret does not telescope to independently replayed episode gap.")
    return steps


def latex_table(headers, rows):
    suite.require(all(len(row) == len(headers) for row in rows), "Ragged manuscript table.")
    return "\n".join([
        r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        r"\toprule", " & ".join(headers) + r" \\",
        r"\midrule", *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule", r"\end{tabular}", "",
    ])


def size_figure(summaries):
    parts = [r"\begin{tikzpicture}",
             r"\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.3cm,vertical sep=1.4cm},",
             r"width=.46\textwidth,height=4.8cm,xlabel={Number of vertices},",
             r"xmin=7.8,xmax=12.2,xtick={8,9,10,11,12},ymin=0,ymax=100,grid=major,",
             r"legend style={font=\scriptsize},legend columns=2]"]
    styles = ("blue,mark=*", "orange,mark=square*", "teal,mark=triangle*", "violet,mark=diamond*")
    for index, task in enumerate(("degree_exact", "pair_connectivity", "tsp_construct",
                                  "lt_influence_construct")):
        metric = "accuracy" if task in EXACT else "optimality_rate"
        ylabel = r"Accuracy (\%)" if task in EXACT else r"Optimum hits (\%)"
        legend = ",legend to name=modellegend" if index == 0 else ""
        parts.append(r"\nextgroupplot[title={" + TASK_LABELS[task] + "},ylabel={" + ylabel + "}" + legend + "]")
        for profile, label, style in zip(PROFILES, LABELS, styles):
            data = summaries[profile]["by_task_and_size"][task]
            coordinates = " ".join(f"({n},{100 * cell[metric]:.8f})"
                                   for n, cell in sorted(data.items(), key=lambda pair: int(pair[0])))
            parts.append(r"\addplot+[" + style + "] coordinates {" + coordinates + "};")
            if index == 0:
                parts.append(r"\addlegendentry{" + label + "}")
    parts.extend([r"\end{groupplot}", r"\end{tikzpicture}",
                  r"\par\ref{modellegend}", ""])
    return "\n".join(parts)


def export(matrix, bundle, split, output):
    suite.require(not output.exists(), "Paper evidence exists; use a fresh output path.")
    release.verify(bundle)
    manifest = release.manifest_at(bundle)
    instances = release.labeled_instances(bundle, split, manifest)
    metadata = suite.read_json(bundle / f"analysis/{split}.json")
    clusters = structural_clusters(metadata)
    scores, summaries, statuses, timing, initial, sources = {}, {}, {}, {}, None, {}
    for profile in PROFILES:
        root = matrix / f"{split}.{profile}"
        run_manifest, status, attempts, rows = matrix_tools.verified_run(root, bundle, split)
        suite.require(run_manifest["profile"] == profile, "Profile path/manifest mismatch.")
        first = {row["instance_id"]: row["request"] for row in attempts if row["step"] == 0}
        suite.require(len(first) == len(instances) and (initial is None or first == initial),
                      "Initial requests or execution coverage differ.")
        initial = first
        scores[profile] = {row["instance_id"]: row for row in rows}
        summaries[profile], statuses[profile] = suite.summarize(rows), status
        latencies = sorted(row["latency_seconds"] for row in attempts)
        timing[profile] = (1000 * latencies[len(latencies)//2],
                           1000 * latencies[int(.95 * len(latencies))])
        for filename in ("manifest.json", "status.json", "attempts.jsonl", "scores.jsonl"):
            path = root / filename
            sources[str(path)] = suite.digest(path)
    classical = references.analyze(instances)
    paired = paired_intervals(instances, list(scores[PROFILES[0]].values()),
                              list(scores[PROFILES[1]].values()), clusters=clusters)
    step_rows, episode_rows = [], []
    for instance in instances:
        if instance["kind"] != "optimization":
            continue

        @lru_cache(maxsize=None)
        def cached_values(history):
            return continuation_values(instance, list(history))

        for profile in PROFILES:
            scored = scores[profile][instance["id"]]
            if not scored["feasible"]:
                episode_rows.append({
                    "profile": profile, "instance_id": instance["id"], "task": instance["task"],
                    "feasible": False, "first_loss_step": None, "final_gap": None,
                    "informative_steps": None, "optimal_informative_steps": None,
                })
                continue
            steps = trajectory_regrets(instance, scored["decisions"],
                                       lambda inst, history: cached_values(tuple(history)))
            first_loss = next((s["step"] for s in steps if not s["locally_optimal"]), None)
            episode_rows.append({
                "profile": profile, "instance_id": instance["id"], "task": instance["task"],
                "feasible": True, "first_loss_step": first_loss,
                "final_gap": scored["absolute_gap"],
                "informative_steps": sum(s["informative"] for s in steps),
                "optimal_informative_steps": sum(s["informative"] and s["locally_optimal"] for s in steps),
            })
            step_rows.extend({"profile": profile, "instance_id": instance["id"],
                              "task": instance["task"], "node_count": len(instance["state"]["nodes"]),
                              "budget": instance["query_budget"], **s} for s in steps)
    output.mkdir(parents=True, exist_ok=False)
    exact_rows, opt_rows, gap_rows, interval_rows, regret_rows = [], [], [], [], []
    for task in EXACT:
        exact_rows.append([TASK_LABELS[task], *(
            f'{100 * summaries[p]["by_task"][task]["macro_size_accuracy"]:.2f}' for p in PROFILES)])
    for task in OPTIMIZATION:
        cells = [summaries[p]["by_task"][task] for p in PROFILES]
        ref = classical["summary"][task]
        opt_rows.append([TASK_LABELS[task], *(f'{c["optimal_count"]}/{c["instances"]}' for c in cells),
                         f'{ref["improved_optimal"]}/{ref["instances"]}'])
        gap_rows.append([TASK_LABELS[task], *(
            "--" if c["mean_absolute_gap_feasible"] is None else f'{c["mean_absolute_gap_feasible"]:.3f}'
            for c in cells), f'{ref["improved_mean_gap"]:.3f}', f'{ref["uniform_mean_expected_gap"]:.3f}'])
        regrets = []
        for profile in PROFILES:
            selected = [r for r in episode_rows if r["profile"] == profile and r["task"] == task]
            feasible = [r for r in selected if r["feasible"]]
            regrets.append(f'{sum(r["first_loss_step"] == 1 for r in feasible)}/{len(feasible)}')
        regret_rows.append([TASK_LABELS[task], *regrets])
    for task in EXACT + OPTIMIZATION:
        cell = paired[task]
        low, high = cell["percentile_95"]
        interval_rows.append([TASK_LABELS[task], f'{100 * cell["difference_left_minus_right"]:.2f}',
                              f'[{100 * low:.2f}, {100 * high:.2f}]',
                              str(cell["generation_clusters"])])
    tables = {
        "exact.tex": latex_table(("Exact task (macro \\%)", *LABELS), exact_rows),
        "optimal.tex": latex_table(("Task", *LABELS, "Classical"), opt_rows),
        "gaps.tex": latex_table(("Task", *LABELS, "Classical", "Uniform"), gap_rows),
        "intervals.tex": latex_table(("Task", r"$\Delta$ (pp)", "Pointwise 95\\% interval", "Clusters"), interval_rows),
        "first-loss.tex": latex_table(("Task", *LABELS), regret_rows),
        "timing.tex": latex_table(("Method", "Median (ms)", "p95 (ms)"),
                                  [[label, f"{timing[p][0]:.2f}", f"{timing[p][1]:.2f}"]
                                   for p, label in zip(PROFILES, LABELS)]),
        "size-figure.tex": size_figure(summaries),
    }
    for filename, content in tables.items():
        (output / filename).write_text(content)
    for filename, rows in (("onpolicy-steps.csv", step_rows), ("onpolicy-episodes.csv", episode_rows)):
        with (output / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    suite.write_json(output / "classical.json", classical)
    suite.write_json(output / "paired-intervals.json", paired)
    suite.write_json(output / "provenance.json", {
        "schema_version": 1, "split": split, "profiles": PROFILES,
        "matrix": str(matrix), "bundle": str(bundle),
        "source_artifact_sha256": sources,
        "bundle_manifest_sha256": suite.digest(bundle / "manifest.json"),
        "analysis_source_sha256": {
            **matrix_tools.evaluator.code_hashes(),
            "scripts/build_paper_evidence.py": suite.digest(__file__),
            "scripts/construction_prefix_controls.py": suite.digest(REPO / "scripts/construction_prefix_controls.py"),
            "scripts/extended_reference_analysis.py": suite.digest(references.__file__),
            "scripts/summarize_extended_paper.py": suite.digest(REPO / "scripts/summarize_extended_paper.py"),
            "scripts/summarize_release_matrix.py": suite.digest(matrix_tools.__file__),
        },
        "generated_sha256": {path.name: suite.digest(path) for path in sorted(output.iterdir())},
        "instance_evaluations": len(instances) * len(PROFILES),
        "calls": sum(s["calls"] for s in statuses.values()),
        "regret_episodes": len(episode_rows), "regret_steps": len(step_rows),
        "regret_telescope_mismatches": 0, "initial_requests_identical": True,
        "inference_calls": 0, "environment": suite.environment_versions(),
        "scope": "Retrospective exact on-policy diagnostic; no oracle information given to models.",
    })
    return {"output": str(output), "instances": len(instances), "regret_episodes": len(episode_rows),
            "regret_steps": len(step_rows), "telescope_mismatches": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--split", default="structural_challenge")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.matrix, args.bundle, args.split, args.output), indent=2))


if __name__ == "__main__":
    main()
