#!/usr/bin/env python3
"""Download only pinned public, non-executable model artifacts; no CUDA imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

PINS = {
    "decider": ("Mapika/decider-2b", "533964dae8be954c5b5e19fa4948e48408094c1e"),
    "kev": ("jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"),
    "kev-base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
}
SOURCES = {"decider": "15ab28e8d8e4ec49728165b3426d81de5b77f864",
           "kev": "569ea449b1033c5aa8f149ec5e20d0c7a296d24a"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".next")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (repo, revision) in PINS.items():
        save(root / "download-status.json", {"phase": "downloading", "model": name,
                                           "time": time.time(), "completed": list(records)})
        target = root / "models" / name
        snapshot_download(repo, revision=revision, local_dir=target,
                          cache_dir=root / "hub-cache", max_workers=2,
                          allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                          "*.txt", "*.pt", "LICENSE", "README.md"],
                          ignore_patterns=["decider/*", "train.log", "training_metrics.json"])
        files = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob("*"))
                 if p.is_file() and ".cache" not in p.parts}
        records[name] = {"repository": repo, "revision": revision, "path": str(target),
                         "sha256": files}
        save(root / "artifacts.json", {"models": records, "source_revisions": SOURCES})
    save(root / "download-status.json", {"phase": "complete", "time": time.time(),
                                       "completed": list(records)})


if __name__ == "__main__":
    main()
