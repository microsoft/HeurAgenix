#!/usr/bin/env python3
"""Verify the completed CPU pilot and render coverage-aware exploratory tables."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import laya_cpu_pilot as pilot
from scripts import export_expanded_model_pilots as expanded
from scripts.build_paper_evidence import EXACT, TASK_LABELS, latex_table
from paper.build_pilot_table import CONSTRUCTIONS

prior, suite, canonical = expanded.prior, expanded.suite, expanded.canonical


def tables(rows, references):
    suite.require(len(rows) == len(references) == 64, "Expected the whole 64-case panel.")
    reference = {r["instance_id"]: r for r in references}
    suite.require(len(reference) == 64 and
                  {r["instance_id"] for r in rows} == set(reference), "Reference identity mismatch.")
    exact, construction = [], []
    for task in (*EXACT, *CONSTRUCTIONS):
        own = [r for r in rows if r["task"] == task]
        expected = 8 if task in EXACT else 4
        suite.require(len(own) == expected, "Wrong task denominator.")
        supported = [r for r in own if r["initial_context_compatible"]]
        suite.require(all(r["feasible"] == r["initial_context_compatible"] for r in own),
                      "This completed-pilot table cannot hide later failures.")
        matched = [reference[r["instance_id"]] for r in supported]
        suite.require(all(r["task"] == task and r["feasible"] for r in matched),
                      "Incompatible historical reference.")
        field = "correct" if task in EXACT else "optimal"
        hits = [f'{sum(bool(r[field]) for r in group)}/{len(group)}'
                for group in (supported, matched)]
        if task in EXACT:
            suite.require(len(supported) == expected, "Exact table requires complete coverage.")
            exact.append([TASK_LABELS[task], *hits])
        else:
            gaps = [f'{sum(r["absolute_gap"] for r in group) / len(group):.3f}'
                    if group else "--" for group in (supported, matched)]
            construction.append([TASK_LABELS[task], f"{len(supported)}/{expected}", *hits, *gaps])
    return {"exact.tex": latex_table(("Task", "Laya CPU", "Reused Jev"), exact),
            "construction.tex": latex_table(
                ("Task", "Covered", "Laya hits", "Jev hits", "Laya gap", "Jev gap"), construction)}


def export(root, output):
    suite.require(not output.exists(), "Output exists; no overwrite.")
    evidence = prior.Evidence()
    evidence.track(__file__)
    manifest = suite.read_json(root / "final-artifact-manifest.json")
    suite.require(manifest["status"] == "complete_offline_verified", "Pilot is not finalized.")
    evidence.track(root / "final-artifact-manifest.json")
    for name, digest in manifest["files_sha256"].items():
        path = REPO / name
        suite.require(path.resolve().is_relative_to(REPO) and not path.is_symlink(),
                      "Invalid sealed artifact path.")
        evidence.track(path, digest)
    pilot.verify_registration(root)
    completion = suite.read_json(root / "run/completion.json")
    suite.require(completion["status"] == "completed_with_initial_unsupported"
                  and completion["scheduled_episodes"] == 64
                  and completion["inference_retried"] is False
                  and completion["source_or_registration_modified"] is False,
                  "Unexpected completion/recovery contract.")
    for name, digest in completion["artifacts_sha256"].items():
        path = root / "run" / name
        suite.require(path.resolve().is_relative_to((root / "run").resolve()),
                      "Invalid completion artifact path.")
        evidence.track(path, digest)
    registration, instances, _, _ = prior.load_panel(pilot.SELECTION, evidence)
    attempts = suite.read_rows(root / "run/attempts.jsonl")
    replayed = canonical.replay(instances, attempts, SimpleNamespace(model=pilot.MODEL),
                                native_action=False)
    saved = suite.read_json(root / "run/scores.json")
    suite.require(len(saved) == len(replayed) == 64, "Missing scheduled rows.")
    coverage = {r["instance_id"]: r for r in suite.read_json(root / "coverage.json")["rows"]}
    grouped = {i["id"]: [a for a in attempts if a["instance_id"] == i["id"]] for i in instances}
    for new, old in zip(replayed, saved):
        own = grouped[new["instance_id"]]
        compatible = coverage[new["instance_id"]]["compatible"]
        new.update(initial_context_compatible=compatible, scheduled=True,
                   forward_count=sum(a["forward_count"] for a in own),
                   execution_category="completed" if compatible else "unsupported_initial")
        suite.require(new == old, "Canonical own-history replay or coverage differs.")
    success = [a for a in attempts if a["status"] == "success"]
    for attempt in success:
        backend = attempt["backend"]
        pilot.validate_answer(attempt["request"], backend["answer"], backend["raw_logits"])
        suite.require(backend["answer"]["choice"] == attempt["selected_option_id"]
                      and backend["coverage"]["compatible"]
                      and backend["cuda_initialized"] is False, "Native action/CPU contract differs.")
    suite.require(len(success) == completion["official_forwards"]
                  == sum(a["forward_count"] for a in attempts) == 156,
                  "Successful forward count differs.")
    suite.require(dict(Counter(r["execution_category"] for r in saved)) ==
                  completion["execution_categories"], "Completion accounting differs.")
    reference_root = prior.DEFAULT_HISTORY / f'{registration["split"]}.jev_action'
    _, reference, _ = prior.load_sealed_reference(
        reference_root, Path(registration["bundle"]), registration["split"], instances, evidence)
    suite.require(reference == suite.read_json(root / "reused-jev.json")["rows"],
                  "Reused Jev subset differs from fully verified original lane.")
    objectives = expanded.independent_objectives(instances, saved)
    rendered = tables(saved, reference)
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, text in rendered.items():
        (output / name).write_text(text)
    suite.write_json(output / "provenance.json", {
        "status": "independently_replayed", "source": str(root), "new_model_calls": 0,
        "source_sha256": evidence.hashes,
        "generated_sha256": {p.name: suite.digest(p) for p in output.iterdir()},
        "independent_objectives": objectives,
        "scope": "Post-hoc CPU FP32 exploratory pilot. Full 64-case denominator retained; "
                 "construction quality uses initial-context-compatible cases, not truncated inputs. "
                 "Reused Jev is not a fresh latency comparison. Offline recovery disclosed."})
    return {"status": "verified_and_exported", "model_calls": 0, "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=pilot.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(suite.canonical(export(args.source, args.output)))
