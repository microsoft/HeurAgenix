#!/usr/bin/env python3
"""Export public aggregates offline; never publish raw evidence or overwrite output.

Usage: python3 scripts/export_website_data.py --output web/data
The output directory must not exist. The default source is the sealed canonical
2026-09-25 export. Verification reuses the frozen paper builder and checks all
original input hashes. Its private intermediate files are removed after use.
CSV columns match the public JSON metric fields; blank numeric cells mean null,
not zero. Exact all-size accuracy is macro-size accuracy, while construction
optimum rates retain all scheduled episodes in their denominator.
Run using the existing project Python environment (no dependencies are added).
Tests: PYTHONPATH=tests python3 -m unittest test_website_data
"""

import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from paper import build_full_matrix_table as build


SIZES = (8, 9, 10, 11, 12)
METRIC_FIELDS = (
    "model", "task", "nodeCount", "scheduled", "feasible", "hits", "unsupported",
    "failures", "unattempted", "accuracy", "optimalRate", "meanGap", "meanObjective",
)
MODEL_DETAILS = {
    "jev_action": ("Jev", "Jev", "Undisclosed", "Native action eligibility",
                   "Native selections; not a probability-argmax surrogate."),
    "qwen4_grammar": ("Qwen3.5-4B grammar", "Qwen", "4B",
                      "No thinking; greedy choice grammar; 64-token ceiling",
                      "Bounded direct-decision configuration."),
    "qwen9_grammar": ("Qwen3.5-9B grammar", "Qwen", "9B",
                      "No thinking; greedy choice grammar; 64-token ceiling",
                      "Bounded direct-decision configuration."),
    "qwen27_grammar": ("Qwen3.5-27B grammar", "Qwen", "27B",
                       "No thinking; greedy choice grammar; 64-token ceiling",
                       "Bounded direct-decision configuration."),
    "qwen4_token_scores": ("Qwen3.5-4B candidate scores", "Jev-like", "4B",
                           "No thinking; requested candidate-token log probabilities",
                           "Interface reference, not a trained selector; SemIf numerical gate failed."),
    "decider": ("Decider-2B v11", "Jev-like", "2B", "Native trained selector; BF16",
                "Trained Jev-like baseline; not parameter-matched to Jev."),
    "kev": ("Kev-4B", "Jev-like", "4B", "Native trained selector; FP32",
            "Trained Jev-like baseline; not parameter-matched to Jev."),
    "laya": ("Laya", "Jev-like", "~421M", "Native encoder and trained head; CPU FP32",
             "Nine unsupported LT inputs retained; no truncation."),
    "gpt54_default_reasoning": ("GPT-5.4", "GPT", "Undisclosed",
                                "Tool-free default reasoning; 4,096 output-token ceiling",
                                "Separate reasoning anchor; nine historical construction timeouts retained."),
    "gpt6astra_default_reasoning": ("GPT-6-Astra", "GPT", "Undisclosed",
                                    "Tool-free default reasoning; 4,096 output-token ceiling",
                                    "Separate reasoning anchor; not compute-matched or certified no-thinking."),
}
TASK_DETAILS = {
    "adjacency": ("Adjacency", "Does the specified pair share an undirected edge?",
                  "accuracy", "maximize"),
    "degree_exact": ("Exact degree", "Count the specified vertex's distinct neighbors.",
                     "accuracy", "maximize"),
    "cycle_detection": ("Cycle detection",
                        "Does any component contain a simple undirected cycle of length at least three?",
                        "accuracy", "maximize"),
    "pair_connectivity": ("Pair connectivity",
                          "Does an undirected path connect the specified pair?",
                          "accuracy", "maximize"),
    "distance_threshold": ("Distance threshold",
                           "Is the specified pair's shortest-path distance at most the stated threshold? "
                           "Disconnected pairs have infinite distance.",
                           "accuracy", "maximize"),
    "articulation_point": ("Articulation point",
                           "Does deleting the specified vertex and its edges increase "
                           "the graph's number of connected components?",
                           "accuracy", "maximize"),
    "maxcut_construct": ("MaxCut",
                         "Maximize crossing edges in an unweighted undirected graph; fix vertex 0 "
                         "to A and assign the remaining vertices to A or B in order.",
                         "edges", "maximize"),
    "tsp_construct": ("Manhattan TSP",
                      "Minimize a closed Hamiltonian tour's Manhattan distance, including the return "
                      "edge; start at city 0 and append the last unvisited city implicitly.",
                      "distance", "minimize"),
    "lt_influence_construct": ("Deterministic normalized LT influence",
                               "Choose two distinct seeds in a directed graph; maximize final "
                               "activated nodes, including seeds, under synchronous irreversible "
                               "diffusion with equal normalized incoming weights and threshold 1/2.",
                               "activated nodes", "maximize"),
    "community_bipartition_construct": ("At-most-two-community modularity",
                                        "Maximize standard resolution-1 modularity in an undirected "
                                        "unweighted graph using at most two communities; fix vertex "
                                        "0 to A. A single community is legal; balance is not required.",
                                        "modularity", "maximize"),
}


def public_data(summary, verification):
    """Construct the public contract by allowlist, never by copying source records."""
    index = build.validated_metrics(summary)
    provenance = {}
    for public, private in (
        ("summarySha256", "source_summary_sha256"),
        ("sealSha256", "source_seal_sha256"),
        ("completionSha256", "source_completion_sha256"),
    ):
        value = verification[private]
        build.suite.require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
                            "Invalid provenance digest.")
        provenance[public] = value
    count = verification["verified_input_hashes"]
    build.suite.require(type(count) is int and count > 0
                        and verification["new_model_calls"] == 0,
                        "Invalid offline verification provenance.")
    provenance.update(verifiedInputHashes=count, modelCalls=0)
    models = []
    for profile, _ in build.PROFILES:
        name, family, parameters, protocol, note = MODEL_DETAILS[profile]
        models.append(dict(id=profile, name=name, family=family,
                           group="strong_anchors" if profile.startswith("gpt") else "decision_matrix",
                           parameters=parameters, protocol=protocol, note=note))
    tasks = []
    for task, count in build.COUNTS.items():
        name, description, unit, direction = TASK_DETAILS[task]
        tasks.append(dict(id=task, name=name, kind="exact" if task in build.EXACT else "construction",
                          description=description, unit=unit, direction=direction, scheduled=count))
    metrics = []
    for profile, _ in build.PROFILES:
        for task in build.COUNTS:
            exact = task in build.EXACT
            for size in ("all", *SIZES):
                row = index[profile, task, size]
                rate = row["hits"] / row["scheduled"]
                metrics.append(dict(
                    model=profile, task=task, nodeCount=size,
                    scheduled=row["scheduled"], feasible=row["feasible"], hits=row["hits"],
                    unsupported=row["initial_unsupported"], failures=row["execution_failures"],
                    unattempted=row["unattempted"],
                    accuracy=(row["macro_size_accuracy"] if size == "all" else rate) if exact else None,
                    optimalRate=None if exact else rate,
                    meanGap=None if exact else row["mean_absolute_gap_feasible"],
                    meanObjective=None if exact else row["mean_objective_feasible"],
                ))
    return dict(
        schemaVersion=1, title="JevGraphBench", version="canonical-20260925",
        evidenceDate="2026-09-25",
        scope="Full inspected canonical structural-challenge bank; 960 scheduled episodes per model. "
              "GPT default-reasoning anchors are separate, not compute-matched. All failures and "
              "unsupported inputs remain in scheduled denominators. Exact aggregate accuracy is "
              "macro-averaged over sizes, not pooled; construction optimum rates use scheduled "
              "denominators and absolute gaps/objectives are conditional on feasible episodes. "
              "No cross-task quality pooling or new model calls.",
        provenance=provenance, models=models, tasks=tasks, sizes=list(SIZES), metrics=metrics,
    )


def encoded_files(data):
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(data["metrics"])
    return {"benchmark.json": text.encode("utf-8"), "benchmark.csv": stream.getvalue().encode("utf-8")}


def export(source, output):
    """Verify through the frozen builder, then publish only two sanitized files."""
    source, output = Path(source), Path(output)
    build.suite.require(not output.exists() and not output.is_symlink(), "Output exists; no overwrite.")
    build.suite.require(not output.resolve().is_relative_to(source.resolve()),
                        "Output must not be inside sealed source evidence.")
    workspace = REPO / "output" / ("website-verification-" + uuid.uuid4().hex)
    workspace.mkdir(parents=True, exist_ok=False)
    created_output = False
    try:
        private = workspace / "paper"
        build.export(source, private)
        verification = build.suite.read_json(private / "provenance.json")
        summary_path = source / "verified-summary.json"
        summary = build.suite.read_json(summary_path)
        for path, field in (
            (summary_path, "source_summary_sha256"),
            (source / "export-seal.json", "source_seal_sha256"),
            (source.parent / "completion.json", "source_completion_sha256"),
        ):
            build.suite.require(build.suite.digest(path) == verification[field],
                                "Evidence changed after verification.")
        data = public_data(summary, verification)
        files = encoded_files(data)
        output.mkdir(parents=True, exist_ok=False)
        created_output = True
        for name, content in files.items():
            with (output / name).open("xb") as stream:
                stream.write(content)
        return data["provenance"]
    except BaseException:
        if created_output:
            shutil.rmtree(output)
        raise
    finally:
        shutil.rmtree(workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=build.SOURCE)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for benchmark.json and benchmark.csv; no overwrite.")
    args = parser.parse_args()
    provenance = export(args.source, args.output)
    print(json.dumps(dict(status="verified_public_export", metrics=600, **provenance), sort_keys=True))


if __name__ == "__main__":
    main()
