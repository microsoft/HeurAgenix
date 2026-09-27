#!/usr/bin/env python3
"""Offline reader, oracle verifier and scorer for the portable graph release."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE if (HERE / "src").is_dir() else HERE.parent
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))

from src.benchmark import extended_tasks as tasks
from graph_task_prompt import canonical_graph_request, state_explicit_tsp_request

PUBLIC_KEYS = {"id", "task", "kind", "state", "query_budget"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_file(root, relative, manifest):
    path = root / relative
    if (Path(relative).is_absolute() or ".." in Path(relative).parts
            or not path.is_file() or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or digest(path) != manifest["sha256"][relative]):
        raise ValueError(f"Release file integrity failure: {relative}")
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def manifest_at(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(root / "manifest.json") != json.loads((root / "manifest.sha256").read_text()):
        raise ValueError("Release manifest integrity failure")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported release schema")
    return manifest


def load_public(root, split, manifest):
    if split not in manifest["splits"]:
        raise ValueError("Unknown release split")
    rows = read_rows(checked_file(root, f"public/{split}.jsonl", manifest))
    if any(set(row) != PUBLIC_KEYS for row in rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Invalid public record schema or duplicate IDs")
    return rows


def labeled_instances(root, split, manifest):
    public = load_public(root, split, manifest)
    references = read_rows(checked_file(root, f"references/{split}.jsonl", manifest))
    by_id = {r["id"]: r["private"] for r in references}
    if (len(by_id) != len(references) or set(by_id) != {r["id"] for r in public}
            or any(set(r) != {"id", "private"} for r in references)):
        raise ValueError("Reference IDs or schema differ")
    return [{**row, "private": by_id[row["id"]]} for row in public]


def request_for(root, split, ident, history, *, protocol=None):
    manifest = manifest_at(root)
    protocol = protocol or manifest.get("recommended_protocol", "state_explicit_tsp")
    if protocol not in manifest["protocols"]:
        raise ValueError("Unknown prompt protocol")
    matches = [r for r in load_public(root, split, manifest) if r["id"] == ident]
    if len(matches) != 1:
        raise ValueError("Unknown instance")
    instance = matches[0]
    request = tasks.next_request(instance, history)
    if request is not None and protocol == "explicit_actions":
        request = canonical_graph_request(instance["task"], request)
    if request is not None and protocol == "state_explicit_tsp" and instance["task"] == "tsp_construct":
        request = state_explicit_tsp_request(request)
    return None if request is None else asdict(request)


def verify(root):
    manifest = manifest_at(root)
    for relative in manifest["sha256"]:
        checked_file(root, relative, manifest)
    validations, seen = {}, set()
    for split, specification in manifest["splits"].items():
        instances = labeled_instances(root, split, manifest)
        ids = {i["id"] for i in instances}
        if seen & ids:
            raise ValueError("Instance ID reused across release splits")
        seen |= ids
        actual = tasks.validate_instances(instances)
        if actual != specification["validation"]:
            raise ValueError("Split validation differs")
        metadata = json.loads(checked_file(root, f"analysis/{split}.json", manifest).read_text())
        if set(metadata["cases"]) != ids:
            raise ValueError("Analysis metadata coverage differs")
        validations[split] = actual
    return {"status": "verified", "instances": len(seen), "splits": validations,
            "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "request", "score"))
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--split")
    parser.add_argument("--id")
    parser.add_argument("--history", default="[]", help="JSON array of exact action IDs")
    parser.add_argument("--protocol", choices=("legacy_frozen", "state_explicit_tsp", "explicit_actions"))
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.root)
    elif args.command == "request":
        result = request_for(args.root, args.split, args.id, json.loads(args.history),
                             protocol=args.protocol)
    else:
        manifest = manifest_at(args.root)
        matches = [i for i in labeled_instances(args.root, args.split, manifest) if i["id"] == args.id]
        if len(matches) != 1:
            raise ValueError("Unknown instance")
        result = tasks.score(matches[0], json.loads(args.history))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
