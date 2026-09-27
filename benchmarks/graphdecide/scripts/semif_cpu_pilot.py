#!/usr/bin/env python3
"""Bounded CPU-only feasibility pilot of pinned, unmodified upstream SemIf."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


SOURCE_REVISION = "23cf1f39fc9534fe81437200959b6dfc7106e45a"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict | list) -> None:
    pending = path.with_suffix(path.suffix + ".next")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def choose(row: dict, result: dict) -> str:
    """Map upstream option scores to original action IDs; ties use input order."""
    expected = [o["id"] for o in row["options"]]
    ids, probabilities = result["option_ids"], result["probabilities"]
    if len(expected) != len(set(expected)) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate action IDs")
    if set(ids) != set(expected) or len(probabilities) != len(ids):
        raise ValueError("Action coverage differs")
    if any(isinstance(p, bool) or not isinstance(p, (int, float))
           or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid conditional probabilities")
    if abs(sum(probabilities) - 1) > 1e-6:
        raise ValueError("Direct softmax probabilities are not normalized")
    by_id = dict(zip(ids, probabilities))
    return max(expected, key=by_id.__getitem__)


def cases() -> tuple[list[dict], dict]:
    """Authored public graphs; references are never included in model rows."""
    rows, references = [], {}
    nodes = [str(i) for i in range(6)]
    path = [[str(i), str(i + 1)] for i in range(5)]
    ring = path + [["0", "5"]]
    disconnected = [e for e in path if e != ["2", "3"]]

    def graph(edges):
        return {"graph_type": "simple undirected unweighted",
                "vertices": nodes, "edges": edges}

    def add(task, state, question, options, reference=None):
        ident = f"{task}-{sum(r['task'] == task for r in rows)}"
        row = {"id": ident, "task": task, "state": state, "question": question,
               "options": [{"id": i, "description": d} for i, d in options]}
        rows.append(row)
        references[ident] = {"task": task, "expected": reference,
                            "scope": "exact_authored" if reference is not None
                            else "fixed_state_action_diagnostic"}

    yesno = [("yes", "Yes: the stated graph property holds."),
             ("no", "No: the stated graph property does not hold.")]
    for v, expected in [(1, "yes"), (5, "no")]:
        add("adjacency", graph(path), f"Is there a direct edge between 0 and {v}?",
            yesno, expected)
    for v, expected in [(0, "1"), (2, "2")]:
        add("degree_exact", graph(path), f"What is the degree of vertex {v}?",
            [(str(i), f"Degree {i}: exactly {i} distinct neighbors.") for i in range(6)],
            expected)
    for edges, expected in [(ring, "yes"), (path, "no")]:
        add("cycle_detection", graph(edges),
            "Does this undirected graph contain any simple cycle of length at least 3?",
            yesno, expected)
    for edges, expected in [(path, "yes"), (disconnected, "no")]:
        add("pair_connectivity", graph(edges),
            "Does any path connect vertex 0 to vertex 5?", yesno, expected)
    for k, expected in [(5, "yes"), (4, "no")]:
        add("distance_threshold", graph(path),
            f"Is the shortest-path distance from 0 to 5 at most {k} edges?",
            yesno, expected)
    for edges, expected in [(path, "yes"), (ring, "no")]:
        add("articulation_point", graph(edges),
            "Does removing vertex 2 and its incident edges increase the number of connected components?",
            yesno, expected)

    blocks = [["0", "1"], ["1", "2"], ["2", "0"],
              ["3", "4"], ["4", "5"], ["5", "3"], ["2", "3"]]
    for task in ["maxcut_construct", "community_bipartition_construct"]:
        for v in range(1, 4):
            state = graph(blocks)
            state.update({"assigned": {str(i): "A" for i in range(v)},
                          "next_vertex": str(v)})
            goal = ("maximize the final number of edges crossing between sides"
                    if task == "maxcut_construct" else
                    "maximize standard final modularity Q=sum_C[internal_edges(C)/m"
                    " -(degree_sum(C)/(2*m))^2], allowing one or two communities")
            add(task, state,
                f"Assign vertex {v} to A or B to {goal}. Remaining vertices will be assigned later.",
                [(label, f"Assign vertex {v} to group {label}.") for label in ["A", "B"]])
    coordinates = {"0": [0, 0], "1": [1, 0], "2": [2, 1],
                   "3": [4, 4], "4": [1, 4], "5": [0, 2]}
    for count in [1, 2, 3]:
        visited = nodes[:count]
        add("tsp_construct",
            {"coordinates": coordinates, "visited_in_order": visited,
             "distance": "abs(x_i-x_j)+abs(y_i-y_j)", "start": "0"},
            "Choose the next unvisited city to minimize the final closed Manhattan tour length.",
            [(v, f"Visit city {v} next, at coordinates {coordinates[v]}.")
             for v in nodes if v not in visited])
    arcs = [("0", "2"), ("1", "2"), ("2", "3"),
            ("3", "4"), ("1", "4"), ("4", "5")]
    weighted = [{"from": u, "to": v, "weight_numerator": 1,
                 "weight_denominator": sum(b == v for _, b in arcs)} for u, v in arcs]
    for seeds in [[], ["0"], ["1"]]:
        add("lt_influence_construct",
            {"vertices": nodes, "directed_edges": weighted, "selected_seeds": seeds,
             "seed_budget": 2, "threshold": {"numerator": 1, "denominator": 2}},
            "Choose a new seed to maximize final active vertices. Both seeds start simultaneously; "
            "synchronous irreversible rounds activate a node when incoming active weight is at least "
            "1/2. Unseeded nodes with no incoming edges stay inactive.",
            [(v, f"Select vertex {v} as a new seed.") for v in nodes if v not in seeds])
    assert len(rows) == 24
    return rows, references


def worker(args) -> int:
    root = args.output
    sys.path[:0] = [str(root / "deps"), str(root / "upstream" / "src")]
    status = root / "status.json"
    try:
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        from semif_phase1.core import load_causal_model
        from semif_phase1.direct import score
        save(status, {"phase": "loading", "time": time.time(), "pid": os.getpid()})
        started = time.monotonic()
        model, tokenizer, metadata = load_causal_model(
            str(args.model), MODEL_REVISION, device="cpu", dtype="bfloat16")
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise RuntimeError("CPU-only invariant failed")
        save(root / "runtime.json",
             {"model": metadata, "loading_seconds": time.monotonic() - started,
              "torch_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(),
              "upstream_unmodified": True, "dependency_deviation": "Torch 2.13.0 vs pinned 2.10.0",
              "model_parameter_devices": ["cpu"]})
        rows = json.loads((root / "public-rows.json").read_text())
        with (root / "decisions.jsonl").open("x") as out:
            for row in rows:
                save(status, {"phase": "forward", "time": time.time(),
                              "id": row["id"], "pid": os.getpid()})
                started = time.monotonic()
                try:
                    result = score(model, tokenizer, row, metadata, max_tokens=2048)
                    record = {"id": row["id"], "task": row["task"], "status": "valid",
                              "choice": choose(row, result), "upstream": result}
                except Exception as exc:
                    record = {"id": row["id"], "task": row["task"], "status": "failed",
                              "error_type": type(exc).__name__, "error": str(exc)}
                record["elapsed_seconds"] = time.monotonic() - started
                out.write(json.dumps(record, allow_nan=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
                if record["status"] != "valid":
                    save(status, {"phase": "blocked", "time": time.time(),
                                  "id": row["id"], "reason": record["error_type"]})
                    return 2
        save(status, {"phase": "complete", "time": time.time(), "calls": len(rows)})
        return 0
    except Exception as exc:
        save(status, {"phase": "blocked", "time": time.time(),
                      "error_type": type(exc).__name__, "error": str(exc)})
        traceback.print_exc()
        return 2


def supervise(args) -> int:
    root = args.output
    if (root / "report.json").exists() or (root / "decisions.jsonl").exists():
        raise ValueError("Refusing to overwrite an existing pilot")
    if not args.model.is_dir():
        raise ValueError("Existing local model directory is required")
    manifest = json.loads((root / "upstream" / "source-manifest.json").read_text())
    if manifest["revision"] != SOURCE_REVISION:
        raise ValueError("Wrong upstream revision")
    for entry in manifest["files"]:
        if digest_file(root / "upstream" / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Modified upstream file: {entry['path']}")
    rows, references = cases()
    save(root / "public-rows.json", rows)
    save(root / "references-not-model-input.json", references)
    model_files = sorted(args.model.glob("*.safetensors"))
    model_files += [p for p in [args.model / "config.json",
                                args.model / "tokenizer_config.json",
                                args.model / "tokenizer.json",
                                args.model / "chat_template.jinja"] if p.is_file()]
    provenance = {"source": manifest, "model_path": str(args.model),
                  "declared_model_revision": MODEL_REVISION,
                  "model_files": [{"name": p.name, "bytes": p.stat().st_size,
                                   "sha256": digest_file(p)} for p in model_files],
                  "script_sha256": digest_file(Path(__file__)),
                  "public_rows_sha256": digest_file(root / "public-rows.json"),
                  "limits": {"max_calls": 24, "forward_seconds": 120,
                             "load_seconds": 300, "whole_worker_seconds": 900,
                             "torch_blas_threads": 4, "max_input_tokens": 2048},
                  "scope": "CPU feasibility; 12 exact authored probes and 12 fixed-state "
                           "optimization-action diagnostics, not benchmark episodes",
                  "no_new_weights": True, "no_jev_training": True}
    save(root / "provenance.json", provenance)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
                "VECLIB_MAXIMUM_THREADS": "4", "BLIS_NUM_THREADS": "4",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
                "TMPDIR": str(root / "pip-work"), "XDG_CACHE_HOME": str(root / "runtime-cache")})
    started = time.monotonic()
    timeout_reason = None
    last_status = {}
    with (root / "worker.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
                                 "--output", str(root), "--model", str(args.model)],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:
                last_status = json.loads((root / "status.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            elapsed = time.monotonic() - started
            phase = last_status.get("phase")
            if elapsed > 900:
                timeout_reason = "whole_worker_900s_limit"
            elif phase == "forward" and time.time() - last_status["time"] > 120:
                timeout_reason = "forward_120s_limit"
            elif phase in (None, "loading") and elapsed > 300:
                timeout_reason = "load_300s_limit"
            if timeout_reason:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(0.1)
    try:
        last_status = json.loads((root / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    records = []
    if (root / "decisions.jsonl").exists():
        records = [json.loads(line) for line in (root / "decisions.jsonl").read_text().splitlines()]
    valid = [r for r in records if r["status"] == "valid"]
    exact = [r for r in records if references[r["id"]]["expected"] is not None]
    correct = sum(r.get("choice") == references[r["id"]]["expected"] for r in exact)
    inflight = last_status.get("id") if timeout_reason and last_status.get("phase") == "forward" else None
    report = {"status": "complete" if proc.returncode == 0 and not timeout_reason else "blocked",
              "returncode": proc.returncode, "timeout_reason": timeout_reason,
              "worker_seconds": time.monotonic() - started, "last_status": last_status,
              "scheduled": 24, "completed_records": len(records), "valid": len(valid),
              "attempted": len(records) + int(inflight is not None),
              "unattempted": 24 - len(records) - int(inflight is not None),
              "inflight_failed_id": inflight, "exact_attempted": len(exact),
              "exact_correct": correct, "optimization_scope": "fixed-state action validity only",
              "optimization_valid": sum(r["task"].endswith("_construct") for r in valid),
              "first_decision_seconds": records[0]["elapsed_seconds"] if records else None,
              "provenance_sha256": digest_file(root / "provenance.json")}
    save(root / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output, args.model = args.output.resolve(), args.model.resolve()
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
