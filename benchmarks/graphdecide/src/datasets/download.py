"""Explicit, checksum-pinned public data downloads and provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .real_graphs import load_graph, verify_file
from .sources import DEFAULT_DATA_DIR, DatasetSource, available_datasets, get_source


MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024


def download_file(source: DatasetSource, raw_dir: Path, client: httpx.Client) -> dict:
    """Verify cached files or atomically install a pinned archive, without retries.

    Corrupt existing files are never silently replaced. The caller must explicitly
    remove or move one before redownloading. No API credentials are needed.
    """
    path = raw_dir / source.filename
    if path.exists():
        verify_file(path, source.sha256)
        return {"status": "verified-cache", "bytes": path.stat().st_size}
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        # Manual redirects prevent accidental HTTPS -> HTTP downgrade.
        url = httpx.URL(source.download_url)
        for _ in range(6):
            if url.scheme != "https":
                raise ValueError("Refusing a non-HTTPS dataset download or redirect")
            with client.stream("GET", url, follow_redirects=False) as response:
                if response.is_redirect:
                    if "location" not in response.headers:
                        raise ValueError("Dataset redirect has no Location")
                    url = response.url.join(response.headers["location"])
                    continue
                response.raise_for_status()
                digest = hashlib.sha256()
                count = 0
                with tempfile.NamedTemporaryFile(dir=raw_dir, suffix=".part", delete=False) as stream:
                    temp_path = Path(stream.name)
                    for chunk in response.iter_bytes():
                        count += len(chunk)
                        if count > MAX_DOWNLOAD_BYTES:
                            raise ValueError("Dataset exceeds download size limit")
                        digest.update(chunk)
                        stream.write(chunk)
                if digest.hexdigest() != source.sha256:
                    raise ValueError(f"SHA-256 mismatch for downloaded {source.name}")
                os.replace(temp_path, path)
                return {
                    "status": "downloaded", "bytes": count,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "resolved_download_url": str(response.url),
                }
        raise ValueError("Too many dataset redirects")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_manifest(path: Path, manifest: dict) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=available_datasets(), default=list(available_datasets()))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--verify-only", action="store_true", help="Offline verification and summary, without changing files")
    args = parser.parse_args(argv)
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1,
        "normalization": "simple unweighted topology; retain numeric IDs; merge duplicate/reverse edges; remove self-loops but retain their nodes; deterministic node/edge order",
        "datasets": {},
    }
    with httpx.Client(timeout=60.0, headers={"User-Agent": "JevGraphBench-data/0.1"}) as client:
        for name in dict.fromkeys(args.datasets):
            source = get_source(name)
            receipt = None if args.verify_only else download_file(source, args.data_dir / "raw", client)
            dataset = load_graph(name, data_dir=args.data_dir)
            print(f"{name}: {json.dumps(asdict(dataset.stats), sort_keys=True)}")
            if receipt is not None:
                previous = manifest["datasets"].get(name, {})
                record = {
                    "source": asdict(source), "raw_file": f"raw/{source.filename}",
                    "bytes": receipt["bytes"], "sha256": dataset.raw_sha256,
                    "downloaded_at_utc": previous.get("downloaded_at_utc"),
                    "resolved_download_url": previous.get("resolved_download_url"),
                    "stats": asdict(dataset.stats),
                }
                if receipt["status"] == "downloaded":
                    record.update({k: receipt[k] for k in ("downloaded_at_utc", "resolved_download_url")})
                manifest["datasets"][name] = record
                _write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()