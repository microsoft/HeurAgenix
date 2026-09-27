#!/usr/bin/env python3
"""Create a portable, model-free, deterministic graph-benchmark data bundle."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from importlib.metadata import version
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import extended_graph_suite as suite
from scripts.graph_release_api import PUBLIC_KEYS
from src.benchmark import extended_tasks as tasks

SOFTWARE = {
    "src/__init__.py": "src/__init__.py",
    "src/benchmark/__init__.py": "src/benchmark/__init__.py",
    "src/benchmark/extended_tasks.py": "src/benchmark/extended_tasks.py",
    "src/clients/__init__.py": "src/clients/__init__.py",
    "src/clients/base.py": "src/clients/base.py",
    "scripts/graph_release_api.py": "release_api.py",
    "scripts/graph_task_prompt.py": "graph_task_prompt.py",
}

DATA_CARD = """# JevGraphBench portable structural data

This local release candidate contains generated graphs, separated reference
labels, analysis metadata and a model-free request builder/scorer. It contains
no provider responses, credentials, model weights, endpoint configurations or
SDK clients. It is prepared locally, not automatically published.

Public JSONL rows have exactly: id, task, kind, state, query_budget.
References are separate rows with id and private. Analysis metadata, including
families, labels and paired-generation IDs, is NOT model input.

Only the output of `release_api.py request` may be passed to a model.
It reads public data only. `score` reads references; never expose those files to
a model. Invalid/incomplete construction histories receive no objective.
TSP forces the final city and return edge by contract, not by answer repair.
The legacy normalized_quality field uses task-specific objective units and
must not be averaged across tasks.

Three prompt protocols are available: legacy_frozen, state_explicit_tsp, and
explicit_actions. The last is recommended for new cross-backend comparisons:
mathematical questions are wire-neutral, action meanings are explicit, and TSP
names its actual current state. The surrounding provider interface, not the
mathematical question, specifies IDs versus internal letter codes.
Native probability conformance and supplied-action validity are separate
experimental protocols; this model-free dataset does not choose a provider
validation policy. Previously frozen experiment scores are not rewritten.

Identity exclusion is relative to the included pinned graph pools: unweighted
topology, directed LT topology and weighted TSP metric identity, respectively.
It is not generator-family OOD, size-OOD or unseen pretraining data. Source
splits were inspected during this project; this public release is not a secret
future test. Reused graph identities and construction twins form dependent
analysis groups; union cluster_id and pair_id transitively.

Development covers n5..12 with explicit small-space omissions. The structural
challenge covers n8..12. TSP confirmation includes only its forty selected
metric instances, not unexecuted companion tasks from its source generator.
The manifest gives the exact realized counts and source hashes.

Python >=3.11 is required; the only external verifier dependency is NetworkX
at the version pinned in requirements.txt. No network/model call is made by:

    python release_api.py verify --root .
    python release_api.py request --root . --split development --id INSTANCE_ID
    python release_api.py score --root . --split development --id INSTANCE_ID --history '["yes"]'

The archive is deterministic for identical source data/software. Inspect the
repository's task cards, provenance and applicable licensing before public
distribution. No new license grant for repository code is implied.
"""


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(suite.canonical(row) + "\n")


def source_data(root, selected_task):
    manifest = suite.read_json(root / "manifest.json")
    suite.require(suite.digest(root / "manifest.json") == suite.read_json(root / "manifest.sha256"),
                  "Source manifest drift.")
    suite.verify_source_snapshot(root, manifest["code_sha256"])
    suite.require(all(not (root / name).is_symlink() and suite.digest(root / name) == sha
                      for name, sha in manifest["input_sha256"].items()), "Source input drift.")
    instances = suite.read_rows(root / "instances.jsonl")
    if selected_task is not None:
        instances = [i for i in instances if i["task"] == selected_task]
    suite.require(bool(instances), "Empty release split.")
    validation = tasks.validate_instances(instances)
    metadata = suite.read_json(root / "design_metadata.json")
    ids = {i["id"] for i in instances}
    selected_metadata = {
        "not_model_input": True,
        "cases": {i: metadata["cases"][i] for i in sorted(ids)},
        "coverage": [c for c in metadata["coverage"]
                     if selected_task is None or c["task"] == selected_task],
        "source_seed": manifest["seed"],
        "source_design": manifest["design"],
        "source_requested_instances": metadata["requested_instances"],
        "source_generated_instances": metadata["generated_instances"],
        "selected_task": selected_task,
    }
    return manifest, instances, validation, selected_metadata


def build(destination, sources, archive):
    suite.require(not destination.exists() and not archive.exists(),
                  "Release destination/archive already exists.")
    suite.require(not archive.resolve().is_relative_to(destination.resolve()),
                  "Archive must be outside the release directory.")
    prepared = {}
    for split, (root, selected_task) in sources.items():
        suite.require(split in ("development", "structural_challenge", "tsp_confirmation"),
                      "Unknown release split.")
        prepared[split] = source_data(root, selected_task)
    destination.mkdir(parents=True, exist_ok=False)
    specifications = {}
    for split, (manifest, instances, validation, metadata) in prepared.items():
        source_root, _ = sources[split]
        write_rows(destination / f"public/{split}.jsonl",
                   [{key: i[key] for key in PUBLIC_KEYS} for i in instances])
        write_rows(destination / f"references/{split}.jsonl",
                   [{"id": i["id"], "private": i["private"]} for i in instances])
        (destination / "analysis").mkdir(exist_ok=True)
        suite.write_json(destination / f"analysis/{split}.json", metadata)
        specifications[split] = {
            "source_bank": source_root.name,
            "source_manifest_sha256": suite.digest(source_root / "manifest.json"),
            "source_input_sha256": manifest["input_sha256"], "validation": validation,
            "node_counts": sorted({len(i["state"]["nodes"]) for i in instances}),
            "tasks": dict(Counter(i["task"] for i in instances)),
        }
        if "prior_graph_pool.jsonl" in manifest["input_sha256"]:
            (destination / "prior_pools").mkdir(exist_ok=True)
            shutil.copyfile(source_root / "prior_graph_pool.jsonl",
                            destination / f"prior_pools/{split}.jsonl")
    for source, target in SOFTWARE.items():
        target_path = destination / target
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / source, target_path)
    (destination / "DATA_CARD.md").write_text(DATA_CARD)
    (destination / "requirements.txt").write_text(f"networkx=={version('networkx')}\n")
    suite.write_json(destination / "task-contracts.json", {
        "questions": tasks._QUESTIONS,
        "state_fields": {task: sorted(keys) for task, keys in tasks._STATE_KEYS.items()},
        "tasks": sorted((*tasks._ALL_EXACT, *tasks._OPTIMIZATION)),
        "no_private_labels_in_model_requests": True,
        "construction": {"maxcut_construct": "n-1 decisions; vertex0 fixed A",
                         "community_bipartition_construct": "n-1 decisions; at most two groups",
                         "tsp_construct": "n-2 decisions; forced last city and return0; Manhattan",
                         "lt_influence_construct": "two distinct seeds; normalized deterministic LT"}},
    )
    hashes = {p.relative_to(destination).as_posix(): suite.digest(p)
              for p in sorted(destination.rglob("*")) if p.is_file()}
    suite.write_json(destination / "manifest.json", {
        "schema_version": 1, "release": "structural-ten-task-explicit-actions-v3",
        "protocols": ["legacy_frozen", "state_explicit_tsp", "explicit_actions"],
        "recommended_protocol": "explicit_actions",
        "splits": specifications, "sha256": hashes,
        "exporter_sha256": suite.digest(__file__), "model_calls": 0,
    })
    suite.write_json(destination / "manifest.sha256", suite.digest(destination / "manifest.json"))
    verified = subprocess.run(
        [sys.executable, "-I", str(destination.resolve() / "release_api.py"), "verify",
         "--root", str(destination.resolve())],
        cwd=destination.parent, capture_output=True, text=True, check=True)
    report = json.loads(verified.stdout)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for path in sorted(destination.rglob("*")):
                    if path.is_file():
                        data = path.read_bytes()
                        info = tarfile.TarInfo(path.relative_to(destination).as_posix())
                        info.size, info.mode, info.mtime = len(data), 0o444, 0
                        bundle.addfile(info, io.BytesIO(data))
    return {"verification": report, "archive_sha256": suite.digest(archive),
            "archive_bytes": archive.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--tsp-confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output, {
        "development": (args.development, None),
        "structural_challenge": (args.challenge, None),
        "tsp_confirmation": (args.tsp_confirmation, "tsp_construct"),
    }, args.archive)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
