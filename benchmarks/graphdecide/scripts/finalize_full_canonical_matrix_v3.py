#!/usr/bin/env python3
"""Final prepared watcher revision including the existing post-exec parent receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scripts import export_full_canonical_matrix_v3 as exporter

prior = exporter.prior.isolated(
    "_full_canonical_watcher_v3_prior", "finalize_full_canonical_matrix_v2.py")
prior.exporter = exporter
prior.engine.exporter = exporter
ROOT = REPO / "output/full-matrix-offline-finalization-20260925-v3"
raw_freeze, raw_load = prior.source_freeze, prior.load


def source_freeze():
    hashes = raw_freeze()
    for path in (Path(__file__), REPO / "scripts/export_full_canonical_matrix_v2.py",
                 REPO / "tests/test_full_canonical_matrix_v3.py", exporter.RECEIPT):
        hashes[str(path.relative_to(REPO))] = exporter.suite.digest(path)
    exporter.suite.require(hashes[str(exporter.RECEIPT.relative_to(REPO))] == exporter.RECEIPT_SHA,
                           "Existing parent post-exec receipt changed.")
    prior.engine.verify_freeze(hashes)
    return hashes


def load(root):
    registration = raw_load(root)
    for path in (Path(__file__), Path(exporter.__file__), exporter.RECEIPT):
        exporter.suite.require(registration["source_sha256"].get(str(path.relative_to(REPO))) ==
                               exporter.suite.digest(path),
                               "Not a registered receipt-aware v3 watcher.")
    return registration


prior.engine.source_freeze, prior.engine.load = source_freeze, load


def run(root, **kwargs):
    root = prior.engine.local(root)
    load(root)
    return prior.engine.run(root, read_statuses=prior.statuses, export=exporter.export, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--max-wait-seconds", type=int, default=prior.engine.MAX_WAIT)
    args = parser.parse_args()
    if args.command == "prepare":
        registration = prior.engine.prepare(args.root, args.max_wait_seconds)
        print(json.dumps({"status": "prepared", "root": str(args.root),
                          "deadline_unix": registration["deadline_unix"]}))
    elif args.command == "run":
        raise SystemExit(run(args.root))
    else:
        print(json.dumps(prior.statuses(), indent=2))


if __name__ == "__main__":
    main()
