#!/usr/bin/env python3
"""Offline, model-independent label/cheap-feature audit of frozen task instances."""

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.paired_graph_ablation import digest, read_rows, write_json


def audit(instances):
    tasks = defaultdict(list)
    for instance in instances:
        if instance["kind"] == "exact":
            tasks[instance["task"]].append(instance)
    result = {}
    for task, rows in sorted(tasks.items()):
        labels = Counter(str(i["private"]["answer"]) for i in rows)
        sizes = {}
        for n in sorted({len(i["state"]["nodes"]) for i in rows}):
            subset = [i for i in rows if len(i["state"]["nodes"]) == n]
            counts = Counter(str(i["private"]["answer"]) for i in subset)
            sizes[str(n)] = {"instances": len(subset), "labels": dict(counts),
                             "majority_rate": max(counts.values()) / len(subset)}
        cell = {"instances": len(rows), "labels": dict(labels), "by_size": sizes}
        if task != "degree_exact":
            groups = defaultdict(Counter)
            for instance in rows:
                state = instance["state"]
                degrees = Counter(v for edge in state["edges"] for v in edge)
                features = [len(state["nodes"]), len(state["edges"]),
                            sorted(degrees[v] for v in state["nodes"])]
                if "pair" in state:
                    features.append(sorted(degrees[v] for v in state["pair"]))
                if "vertex" in state:
                    features.append(degrees[state["vertex"]])
                if "threshold" in state:
                    features.append(state["threshold"])
                key = json.dumps(features, separators=(",", ":"))
                groups[key][instance["private"]["answer"]] += 1
            cell.update(
                cheap_feature_cells=len(groups),
                empirical_lookup_ceiling=sum(max(g.values()) for g in groups.values()) / len(rows),
                mixed_label_instance_fraction=sum(sum(g.values()) for g in groups.values()
                                                   if len(g) > 1) / len(rows),
                features="n, m, degree multiset; endpoint/queried-vertex degree and threshold when present",
            )
        result[task] = cell
    return {
        "notice": "Feature-cell resubstitution ceiling is a diagnostic, not a trained/test accuracy. "
                  "High values may reflect sparse unique cells, not proof of leakage. "
                  "Degree is deliberately a counting control, so its defining feature is not audited as leakage.",
        "by_task": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    inputs = args.root / "instances.jsonl"
    result = audit(read_rows(inputs))
    result["input_sha256"] = digest(inputs)
    result["analysis_sha256"] = digest(__file__)
    write_json(args.root / "shortcut-audit.json", result)
    print(json.dumps({task: {"instances": cell["instances"],
                            "lookup_ceiling": cell.get("empirical_lookup_ceiling")}
                      for task, cell in result["by_task"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
