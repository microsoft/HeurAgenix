#!/usr/bin/env python3
"""Package verified primary actions for provider-free mathematical replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import graph_release_api as api
from scripts import portable_result_replay as replay
from scripts import summarize_release_matrix as matrix
from scripts import extended_graph_suite as suite
from scripts.build_graph_release import write_rows

PROFILES = ("jev_action", "qwen4_grammar", "qwen9_grammar", "qwen4_token_scores")
CARD = """# Portable primary action replay

This locally prepared artifact contains both existing data/scorer bundles and
the four completed primary methods on development, original challenge and
prospective confirmation. It contains no provider clients, endpoints, tokens,
weights, raw provider bodies or hidden reasoning. It is not public publication
or a new license grant. Consult repository/provider terms before distribution.

Python >=3.11 and NetworkX (version in bundles/original/requirements.txt) suffice:

    python -I replay_results.py --root .

Run in a fresh process. No original workspace, model service or network access
is needed. Each action's public own-history request is reconstructed and hashed;
mathematical correctness, objective, optimum hits and macro-size accuracy are
recomputed. No retry, repair or oracle prefix is inserted. Both supplied data
bundles retain their public/reference separation and independent oracle checks.
These completed traces must not be treated as new independent graph samples.
The forty TSP wording cases are included in the original data bundle but are
not counted as primary model evaluations.

The exporter first verifies the original sealed real runs, including native
action eligibility and candidate-score argmax checks. Source artifact digests
provide provenance, not cryptographic proof that a provider produced a response.
The portable replayer verifies mathematical actions, not the original transport,
probability eligibility, timing, CPU SemIf agreement or bootstrap intervals.
Those analyses remain separate evidence. This version explicitly refuses mock,
unsealed or incomplete source runs rather than silently removing failed cases.

Scores retain the existing exact per-episode contract. Cross-runtime derived
floating-point comparisons use absolute tolerance 1e-12 and zero relative
tolerance; integer, Boolean, ID, action and request comparisons remain exact.
No cross-task objective aggregate or universal model ranking is produced.
"""


def collect_lane(matrix_root, bundle, bank, split, profile):
    source = matrix_root / f"{split}.{profile}"
    manifest, status, attempts, scores = matrix.verified_run(source, bundle, split)
    suite.require(manifest["profile"] == profile and not manifest["injected_client"],
                  "Profile mismatch or mock evidence.")
    suite.require(all(row["status"] == "success" for row in attempts)
                  and all(row["feasible"] and row["failure_count"] == 0 for row in scores),
                  "Portable completed-trace release refuses incomplete or failed evidence.")
    instances = api.labeled_instances(bundle, split, api.manifest_at(bundle))
    by_id = defaultdict(list)
    for row in attempts:
        by_id[row["instance_id"]].append(row)
    traces = []
    for instance, scored in zip(instances, scores, strict=True):
        own = by_id[instance["id"]]
        mathematical = api.tasks.score(instance, scored["decisions"])
        suite.require(all(matrix.same_aggregates(v, scored[k]) for k, v in mathematical.items()),
                      "Source mathematical score mismatch.")
        traces.append({"id": instance["id"], "actions": scored["decisions"],
                       "request_sha256": [replay.value_hash(row["request"]) for row in own],
                       "score": mathematical})
    expected = replay.replay_lane(instances, traces, api)
    original = suite.summarize(scores)
    for task, cell in expected["by_task"].items():
        for key, value in cell.items():
            if key == "accuracy_by_size":
                source_value = {n: original["by_task_and_size"][task][n]["accuracy"] for n in value}
            else:
                source_value = original["by_task"][task][key]
            suite.require(matrix.same_aggregates(value, source_value),
                          "Independent portable aggregate differs from primary analysis.")
    name = f"{bank}.{split}.{profile}"
    return {"name": name, "bank": bank, "split": split, "profile": profile,
            "model_id": manifest["model"]["model"], "provider": manifest["model"]["provider"],
            "trace_file": f"traces/{name}.jsonl", "expected": expected,
            "source_manifest_sha256": suite.digest(source / "manifest.json"),
            "source_status_sha256": suite.digest(source / "status.json"),
            "source_artifact_sha256": status["artifact_sha256"]}, traces


def archive_bundle(root, archive):
    suite.require(not archive.exists() and not archive.resolve().is_relative_to(root.resolve()),
                  "Archive must be a new path outside the result bundle.")
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as stream:
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        suite.require(not path.is_symlink(), "Symlink in result package.")
                        data = path.read_bytes()
                        info = tarfile.TarInfo(path.relative_to(root).as_posix())
                        info.size, info.mode, info.mtime = len(data), 0o444, 0
                        stream.addfile(info, io.BytesIO(data))


def build(original_matrix, original_bundle, confirmation, output, archive):
    suite.require(not output.exists() and not archive.exists()
                  and not archive.resolve().is_relative_to(output.resolve()),
                  "Result output/archive must be new and non-nested.")
    bundles = {"original": original_bundle, "confirmation": confirmation / "bundle"}
    for bundle in bundles.values():
        api.verify(bundle)
    prepared = []
    for bank, split, source in (
            ("original", "development", original_matrix),
            ("original", "structural_challenge", original_matrix),
            ("confirmation", "structural_challenge", confirmation)):
        for profile in PROFILES:
            prepared.append(collect_lane(source, bundles[bank], bank, split, profile))
    output.mkdir(parents=True, exist_ok=False)
    for bank, bundle in bundles.items():
        manifest = api.manifest_at(bundle)
        for relative in (*manifest["sha256"], "manifest.json", "manifest.sha256"):
            source = bundle / relative
            target = output / "bundles" / bank / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copyfile(Path(replay.__file__), output / "replay_results.py")
    (output / "RESULTS_CARD.md").write_text(CARD)
    for lane, traces in prepared:
        write_rows(output / lane["trace_file"], traces)
    hashes = {p.relative_to(output).as_posix(): suite.digest(p)
              for p in sorted(output.rglob("*")) if p.is_file()}
    suite.write_json(output / "manifest.json", {
        "schema_version": 1, "protocol": "explicit_actions",
        "evidence_kind": "verified_completed_real_actions",
        "lanes": [lane for lane, _ in prepared], "sha256": hashes,
        "exporter_sha256": suite.digest(__file__), "model_calls": 0})
    suite.write_json(output / "manifest.sha256", suite.digest(output / "manifest.json"))
    checked = subprocess.run([sys.executable, "-I", str(output.resolve() / "replay_results.py"),
                              "--root", str(output.resolve())],
                             capture_output=True, text=True)
    suite.require(checked.returncode == 0,
                  f"Standalone result replay failed:\n{checked.stdout}\n{checked.stderr}")
    report = json.loads(checked.stdout)
    suite.require(not report["provider_modules_imported"], "Unexpected provider dependency.")
    archive_bundle(output, archive)
    return {"verification": report, "archive_sha256": suite.digest(archive),
            "archive_bytes": archive.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(suite.canonical(build(args.matrix, args.bundle, args.confirmation, args.output, args.archive)))


if __name__ == "__main__":
    main()
