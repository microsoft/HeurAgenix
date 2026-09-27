#!/usr/bin/env python3
"""Compact taskwise manuscript tables from the sealed canonical ten-model export."""
import argparse
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts.build_paper_evidence import EXACT, OPTIMIZATION, TASK_LABELS, latex_table

SOURCE = REPO / "output/full-matrix-offline-finalization-20260925-v3/export"
PROFILES = (
    ("jev_action", "Jev"),
    ("qwen4_grammar", "Qwen4 G"),
    ("qwen9_grammar", "Qwen9 G"),
    ("qwen27_grammar", "Qwen27 G"),
    ("qwen4_token_scores", "Qwen4 scores"),
    ("decider", "Decider"),
    ("kev", "Kev"),
    ("laya", "Laya"),
    ("gpt54_default_reasoning", "GPT-5.4"),
    ("gpt6astra_default_reasoning", "Astra"),
)
COUNTS = {**{task: 80 for task in EXACT}, "degree_exact": 400,
          **{task: 40 for task in OPTIMIZATION}}


def table(headers, rows):
    text = latex_table(headers, rows)
    return text.replace("GPT-5.4 &", "\\midrule\nGPT-5.4 &", 1)


def validated_metrics(summary):
    suite.require(summary["status"] == "verified_full_canonical_matrix"
                  and summary["scheduled_per_profile"] == 960
                  and summary["maximum_decisions_per_profile"] == 1920
                  and summary["initial_requests_identical"] is True
                  and summary["protocol"] == "explicit_actions"
                  and summary["split"] == "structural_challenge",
                  "Not the complete canonical matrix.")
    suite.require(set(summary["profiles"]) == {p for p, _ in PROFILES}
                  and summary["task_counts"] == COUNTS, "Profile/task inventory differs.")
    index = {}
    for row in summary["metrics"]:
        key = row["profile"], row["task"], row["node_count"]
        suite.require(key not in index, "Duplicate metric row.")
        index[key] = row
    expected = {(p, task, size) for p, _ in PROFILES for task in COUNTS
                for size in ("all", *range(8, 13))}
    suite.require(set(index) == expected, "Incomplete task/size/model matrix.")
    for profile, _ in PROFILES:
        for task, count in COUNTS.items():
            exact = task in EXACT
            all_row = index[profile, task, "all"]
            cells = [index[profile, task, size] for size in range(8, 13)]
            for row in [all_row, *cells]:
                fields = ("scheduled", "feasible", "hits", "initial_unsupported",
                          "execution_failures", "unattempted")
                suite.require(all(type(row[k]) is int and row[k] >= 0 for k in fields),
                              "Invalid integer coverage counts.")
                suite.require(row["scheduled"] > 0
                              and 0 <= row["hits"] <= row["feasible"] <= row["scheduled"]
                              and row["feasible"] + row["initial_unsupported"]
                              + row["execution_failures"] + row["unattempted"] == row["scheduled"],
                              "Coverage outcomes do not partition scheduled cases.")
                suite.require(row["hit_definition"] == ("correct" if exact else "optimal"),
                              "Wrong hit definition.")
                suite.require(row["group"] == ("strong_anchors" if profile.startswith("gpt")
                                              else "decision_matrix"),
                              "GPT reasoning anchor mixed into the direct-decision group.")
                suite.require(math.isclose(row["all_scheduled_hit_rate"],
                                           row["hits"] / row["scheduled"], abs_tol=1e-12,
                                           rel_tol=0), "Wrong scheduled hit rate.")
                if not exact:
                    suite.require(row["conditional_objective_denominator"] == row["feasible"],
                                  "Wrong conditional objective denominator.")
                    values = [row[k] for k in ("mean_objective_feasible", "mean_absolute_gap_feasible")]
                    suite.require(all(v is None for v in values) if row["feasible"] == 0 else
                                  all(isinstance(v, (float, int)) and not isinstance(v, bool)
                                      and math.isfinite(v) for v in values),
                                  "Missing or invented conditional objective.")
                    suite.require(row["feasible"] == 0 or row["mean_absolute_gap_feasible"] >= 0,
                                  "Negative objective gap.")
            suite.require(all_row["scheduled"] == count, "Wrong task denominator.")
            for size, cell in zip(range(8, 13), cells):
                expected_n = 8 * size if task == "degree_exact" else 16 if exact else 8
                suite.require(cell["scheduled"] == expected_n, "Wrong size denominator.")
            for field in ("scheduled", "feasible", "hits", "initial_unsupported",
                          "execution_failures", "unattempted"):
                suite.require(all_row[field] == sum(c[field] for c in cells),
                              "Aggregate coverage differs from size cells.")
            if exact:
                macro = math.fsum(c["hits"] / c["scheduled"] for c in cells) / 5
                suite.require(math.isclose(all_row["macro_size_accuracy"], macro,
                                           abs_tol=1e-12, rel_tol=0),
                              "Macro-size accuracy differs from size cells.")
    return index


def render(summary):
    metrics = validated_metrics(summary)
    exact, optimal, gaps, coverage = [], [], [], []
    for profile, label in PROFILES:
        exact.append([label, *(f'{100 * metrics[profile, task, "all"]["macro_size_accuracy"]:.2f}'
                                for task in EXACT)])
        optimal.append([label, *(f'{metrics[profile, task, "all"]["hits"]}/40'
                                  for task in OPTIMIZATION)])
        gaps.append([label, *("--" if metrics[profile, task, "all"]["mean_absolute_gap_feasible"] is None
                              else f'{metrics[profile, task, "all"]["mean_absolute_gap_feasible"]:.4f}'
                              for task in OPTIMIZATION)])
        own = [metrics[profile, task, "all"] for task in COUNTS]
        coverage.append([
            label, f'{sum(metrics[profile, task, "all"]["feasible"] for task in EXACT)}/800',
            f'{sum(metrics[profile, task, "all"]["feasible"] for task in OPTIMIZATION)}/160',
            str(sum(r["initial_unsupported"] for r in own)),
            str(sum(r["execution_failures"] for r in own)),
            str(sum(r["unattempted"] for r in own)),
        ])
    return {
        "exact.tex": table(("Model", *(TASK_LABELS[t] for t in EXACT)), exact),
        "optimal.tex": table(("Model", *(TASK_LABELS[t] for t in OPTIMIZATION)), optimal),
        "gaps.tex": table(("Model", *(TASK_LABELS[t] for t in OPTIMIZATION)), gaps),
        "coverage.tex": table(("Model", "Exact feasible", "Construct feasible",
                               "Input rejects", "Exec. failures", "Unattempted"), coverage),
    }


def export(source, output):
    suite.require(not output.exists(), "Output exists; no overwrite.")
    seal = suite.read_json(source / "export-seal.json")
    completion = suite.read_json(source.parent / "completion.json")
    suite.require(seal["status"] == "completed"
                  and completion["status"] in ("completed", "completed_with_source_failures")
                  and completion["export_seal_sha256"] == suite.digest(source / "export-seal.json"),
                  "Final matrix verification has not completed.")
    for name, digest in seal["artifact_sha256"].items():
        path = source / name
        suite.require(path.resolve().is_relative_to(source.resolve()) and not path.is_symlink()
                      and suite.digest(path) == digest, "Export artifact drift.")
    suite.require("verified-summary.json" in seal["artifact_sha256"], "Summary not sealed.")
    summary = suite.read_json(source / "verified-summary.json")
    for name, digest in summary["source_sha256"].items():
        path = REPO / name
        suite.require(path.resolve().is_relative_to(REPO) and not path.is_symlink()
                      and suite.digest(path) == digest, "Original source evidence drift.")
    texts = render(summary)
    output.mkdir(parents=True, exist_ok=False)
    for name, text in texts.items():
        with (output / name).open("x") as stream:
            stream.write(text)
    suite.write_json(output / "provenance.json", {
        "status": "verified_and_rendered", "new_model_calls": 0,
        "source": str(source), "source_seal_sha256": suite.digest(source / "export-seal.json"),
        "source_completion_sha256": suite.digest(source.parent / "completion.json"),
        "source_summary_sha256": suite.digest(source / "verified-summary.json"),
        "builder_sha256": suite.digest(__file__), "verified_input_hashes": len(summary["source_sha256"]),
        "scheduled_per_profile": 960, "profiles": [p for p, _ in PROFILES],
        "generated_sha256": {name: suite.digest(output / name) for name in texts},
        "scope": "Full inspected canonical bank; GPT reasoning anchors separate; "
                 "failures retained, feasible-only gaps, no new thinking runs or quality pooling.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
