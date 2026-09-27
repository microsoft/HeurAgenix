#!/usr/bin/env python3
"""Replay completed action traces using only the bundled graph scorer."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def value_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_values(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_values(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_values(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=0, abs_tol=1e-12)
    return left == right


def replay_lane(instances, traces, api):
    require(bool(instances) and len(instances) == len(traces), "Scheduled episode coverage differs.")
    groups = defaultdict(list)
    decisions = 0
    for instance, trace in zip(instances, traces, strict=True):
        require(set(trace) == {"id", "actions", "request_sha256", "score"}
                and trace["id"] == instance["id"], "Trace schema, order or identity differs.")
        actions, hashes = trace["actions"], trace["request_sha256"]
        require(type(actions) is list and all(type(a) is str for a in actions)
                and type(hashes) is list and len(actions) == len(hashes),
                "Invalid action/request lists.")
        history = []
        for action, expected_hash in zip(actions, hashes, strict=True):
            request = api.tasks.next_request(instance, history)
            require(request is not None, "Trace continues after completion.")
            request = api.canonical_graph_request(instance["task"], request)
            require(value_hash(asdict(request)) == expected_hash, "Own-history request mismatch.")
            require(action in {option.id for option in request.options}, "Illegal recorded action.")
            history.append(action)
        score = api.tasks.score(instance, history)
        require(score["feasible"] and score["status"] == "complete",
                "This completed-trace release cannot omit failed or incomplete episodes.")
        require(same_values(score, trace["score"]), "Recomputed mathematical score differs.")
        require(api.tasks.next_request(instance, history) is None, "Unfinished trace.")
        groups[instance["task"]].append((len(instance["state"]["nodes"]), score))
        decisions += len(history)
    metrics = {}
    for task, rows in sorted(groups.items()):
        cell = {"instances": len(rows), "feasible": len(rows)}
        if "correct" in rows[0][1]:
            sizes = sorted({n for n, _ in rows})
            by_size = {str(n): sum(score["correct"] for size, score in rows if size == n)
                       / sum(size == n for size, _ in rows) for n in sizes}
            cell.update(correct=sum(score["correct"] for _, score in rows),
                        accuracy=sum(score["correct"] for _, score in rows) / len(rows),
                        macro_size_accuracy=math.fsum(by_size.values()) / len(by_size),
                        accuracy_by_size=by_size)
        else:
            cell.update(optimal_count=sum(score["optimal"] for _, score in rows),
                        optimality_rate=sum(score["optimal"] for _, score in rows) / len(rows),
                        mean_objective_feasible=math.fsum(score["objective"] for _, score in rows)
                        / len(rows),
                        mean_absolute_gap_feasible=math.fsum(
                            score["absolute_gap"] for _, score in rows) / len(rows))
        metrics[task] = cell
    return {"episodes": len(traces), "decisions": decisions, "by_task": metrics}


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    require(digest(root / "manifest.json") == json.loads((root / "manifest.sha256").read_text()),
            "Result manifest integrity failure.")
    require(manifest["schema_version"] == 1 and manifest["protocol"] == "explicit_actions"
            and manifest["evidence_kind"] == "verified_completed_real_actions",
            "Unsupported protocol or evidence kind.")
    for relative, expected_hash in manifest["sha256"].items():
        path = root / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
                and path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root) and digest(path) == expected_hash,
                f"Result artifact integrity failure: {relative}")
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual_files == set(manifest["sha256"]) | {"manifest.json", "manifest.sha256"},
            "Unexpected or missing result artifact.")
    sys.path.insert(0, str(root / "bundles/original"))
    api = importlib.import_module("release_api")
    require(Path(api.__file__).resolve() == root / "bundles/original/release_api.py"
            and Path(api.tasks.__file__).resolve()
            == root / "bundles/original/src/benchmark/extended_tasks.py",
            "Run in a fresh isolated Python process; ambient scorer imports are not permitted.")
    bundles = {}
    for name in ("original", "confirmation"):
        bundle = root / "bundles" / name
        api.verify(bundle)
        bundles[name] = api.manifest_at(bundle)
    software = {name: h for name, h in bundles["original"]["sha256"].items()
                if name.endswith(".py")}
    require(software == {name: h for name, h in bundles["confirmation"]["sha256"].items()
                         if name.endswith(".py")}, "Bundled scorers differ across banks.")
    require(len(manifest["lanes"]) == len({lane["name"] for lane in manifest["lanes"]}),
            "Duplicate result lanes.")
    expected_lanes = {f"{bank}.{split}.{profile}"
                      for bank, split in (("original", "development"),
                                          ("original", "structural_challenge"),
                                          ("confirmation", "structural_challenge"))
                      for profile in ("jev_action", "qwen4_grammar", "qwen9_grammar",
                                      "qwen4_token_scores")}
    require({lane["name"] for lane in manifest["lanes"]} == expected_lanes,
            "Primary matrix lane coverage differs.")
    results = {}
    for lane in manifest["lanes"]:
        require(lane["bank"] in bundles and lane["trace_file"] in manifest["sha256"],
                "Unknown bank or unsealed trace.")
        require(lane["name"] == f'{lane["bank"]}.{lane["split"]}.{lane["profile"]}',
                "Lane identity differs from its bank/split/profile.")
        bundle = root / "bundles" / lane["bank"]
        instances = api.labeled_instances(bundle, lane["split"], bundles[lane["bank"]])
        rows = api.read_rows(root / lane["trace_file"])
        result = replay_lane(instances, rows, api)
        require(same_values(result, lane["expected"]), "Replayed lane aggregate differs.")
        results[lane["name"]] = result
    return {"status": "verified", "episodes": sum(r["episodes"] for r in results.values()),
            "decisions": sum(r["decisions"] for r in results.values()), "lanes": results,
            "model_calls": 0,
            "scope": "Completed mathematical action replay, not provider conformance, timing, "
                     "calibration, classical heuristics or bootstrap re-analysis.",
            "provider_modules_imported": sorted(
                name for name in ("httpx", "torch", "transformers", "copilot") if name in sys.modules)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(canonical(verify(args.root)))


if __name__ == "__main__":
    main()
