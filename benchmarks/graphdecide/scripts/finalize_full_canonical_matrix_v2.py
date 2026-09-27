#!/usr/bin/env python3
"""Fresh registration for the compatible, standalone-coordinator-aware offline watcher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scripts import export_full_canonical_matrix_v2 as exporter

engine = exporter.isolated("_full_canonical_watcher_v2_base", "finalize_full_canonical_matrix.py")
engine.exporter = exporter
ROOT = REPO / "output/full-matrix-offline-finalization-20260925-v2"
raw_freeze, raw_statuses, raw_load = engine.source_freeze, engine.statuses, engine.load


def source_freeze():
    hashes = raw_freeze()
    for path in (Path(__file__), Path(exporter.base.__file__),
                 REPO / "tests/test_full_canonical_matrix_v2.py",
                 REPO / exporter.COORDINATOR):
        hashes[str(path.relative_to(REPO))] = exporter.suite.digest(path)
    root = exporter.TRAINED / "standalone-coordinator"
    for name in ("registration.json", "registration.sha256"):
        hashes[str((root / name).relative_to(REPO))] = exporter.suite.digest(root / name)
    exporter.suite.require(exporter.suite.digest(root / "registration.json") == exporter.COORDINATOR_SHA,
                           "Different standalone coordinator registration.")
    exporter.suite.require(exporter.suite.digest(exporter.TRAINED / "registration.json") ==
                           exporter.TRAINED_SHA
                           and exporter.suite.digest(engine.ASTRA / "manifest.json") == exporter.ASTRA_SHA,
                           "Fixed trained/Astra source identity differs.")
    for name, digest in exporter.suite.read_json(root / "registration.json")["source_sha256"].items():
        exporter.suite.require(name not in hashes or hashes[name] == digest, "Source registration conflict.")
        hashes[name] = digest
    engine.verify_freeze(hashes)
    return hashes


def statuses():
    result = raw_statuses()
    for name in ("decider", "kev"):
        state = exporter.coordinator_state(exporter.TRAINED)
        result[name]["coordinator"] = state
        result[name]["ready"] = result[name]["ready"] and state["ready"]
        result[name]["blocked"] = result[name]["blocked"] or state["blocked"]
    return result


def load(root):
    registration = raw_load(root)
    required = (Path(__file__), Path(exporter.__file__), REPO / exporter.COORDINATOR)
    exporter.suite.require(all(
        registration["source_sha256"].get(str(path.relative_to(REPO))) == exporter.suite.digest(path)
        for path in required), "Not a registered v2 watcher; v1 must remain unlaunched.")
    return registration


def run(root, **kwargs):
    root = engine.local(root)
    load(root)
    return engine.run(root, read_statuses=statuses, export=exporter.export, **kwargs)


engine.source_freeze, engine.statuses, engine.load = source_freeze, statuses, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--max-wait-seconds", type=int, default=engine.MAX_WAIT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = engine.prepare(args.root, args.max_wait_seconds)
        print(json.dumps({"status": "prepared", "root": str(args.root), "deadline_unix": result["deadline_unix"]}))
    elif args.command == "run":
        raise SystemExit(run(args.root))
    else:
        print(json.dumps(statuses(), indent=2))


if __name__ == "__main__":
    main()
