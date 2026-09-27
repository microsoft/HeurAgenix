#!/usr/bin/env python3
"""One-shot offline finalization of four fixed expanded pilot lanes; never inference."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import export_expanded_model_pilots as expanded

prior, suite, canonical = expanded.prior, expanded.suite, expanded.canonical
ROOT = REPO / "output/expanded-pilot-finalization-20260924-v1"
QUEUE = REPO / "output/trained-pilot-queue-20260924-v1"
EXECUTION = prior.DEFAULT_EXECUTION
TRAINED = REPO / "output/trained-jevlike-pilot-20260924-v1"
PANEL = prior.panel.DEFAULT_ROOT
PROFILES = ("qwen27_thinking", "qwen9_thinking", "decider", "kev")
TEMPERATURES = {"decider": 1.164, "kev": 2.406050072164233}
MAX_WAIT = 24 * 60 * 60


def durable_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    durable_directory(path.parent)


def status(root, **values):
    pending = root / "status.next.json"
    exclusive_json(pending, {"at": time.time(), "new_model_calls": 0,
                             "resource_actions": 0, **values})
    os.replace(pending, root / "status.json")
    durable_directory(root)


def event(root, phase, **values):
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": time.time(), "phase": phase, **values}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    durable_directory(root)


def environment():
    packages = {}
    for name in ("networkx", "httpx", "PyYAML", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "packages": packages, "purpose": "offline replay and CPU-only weights metadata",
            "training_environment_reproduction_claim": False}


def process_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": os.getpid(), "start_ticks": fields[19], "ppid": os.getppid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "argv": sys.argv}


def check_digest(path, expected):
    suite.require(path.is_file() and not path.is_symlink() and suite.digest(path) == expected,
                  f"Frozen byte hash mismatch: {path}")


def fixed_plan():
    queue = suite.read_json(QUEUE / "registration.json")
    check_digest(QUEUE / "registration.json", suite.read_json(QUEUE / "registration.sha256"))
    suite.require(queue["prior_root"] == str(EXECUTION)
                  and queue["trained_root"] == str(TRAINED), "Queue roots differ from fixed four lanes.")
    check_digest(EXECUTION / "execution.json", queue["prior_execution_sha256"])
    check_digest(TRAINED / "benchmark-registration-parent-locks.json",
                 queue["trained_registration_sha256"])
    check_digest(TRAINED / "parent-entry-registration.json",
                 queue["parent_entry_registration_sha256"])
    lanes = {name: str(EXECUTION / name) for name in PROFILES[:2]}
    lanes.update({name: str(TRAINED / f"run-{name}") for name in PROFILES[2:]})
    panel = suite.read_json(PANEL / "registration.json")
    trained = suite.read_json(TRAINED / "benchmark-registration-parent-locks.json")
    execution = suite.read_json(EXECUTION / "execution.json")
    sources = {}
    for inventory in (panel["source_sha256"], trained["code_sha256"], execution["code_sha256"]):
        for name, digest in inventory.items():
            suite.require(name not in sources or sources[name] == digest,
                          f"Incompatible registered source pins: {name}")
            sources[name] = digest
    for module in (__file__, expanded.__file__, prior.__file__):
        sources[expanded.relative(Path(module))] = suite.digest(module)
    paths = [QUEUE / "registration.json", QUEUE / "registration.sha256",
             PANEL / "registration.json", PANEL / "registration.sha256",
             PANEL / "pilot-cases.jsonl", PANEL / "initial-requests.jsonl",
             EXECUTION / "execution.json", EXECUTION / "execution.json.sha256"]
    paths += [TRAINED / name for name in (
        "benchmark-registration.json", "benchmark-registration.sha256",
        "benchmark-registration-parent-locks.json", "benchmark-registration-parent-locks.sha256",
        "parent-entry-registration.json", "parent-entry-registration.sha256", "artifacts.json",
        "panel.json", "initial-requests.json", "cpu-smoke-decider/runtime.json",
        "cpu-smoke-kev/runtime.json")]
    inputs = {expanded.relative(path): suite.digest(path) for path in paths}
    return lanes, sources, inputs


def prepare(root=ROOT):
    suite.require(root == ROOT and not root.exists(), "Fixed finalization root exists; no overwrite.")
    lanes, sources, inputs = fixed_plan()
    for name, digest in {**sources, **inputs}.items():
        check_digest(REPO / name, digest)
    root.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = root / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((REPO / name).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        check_digest(target, digest)
        target.chmod(0o444)
    registration = {
        "schema_version": 1, "prepared_at": time.time(), "lanes": lanes,
        "panel": str(PANEL), "historical": str(prior.DEFAULT_HISTORY),
        "maximum_wait_seconds": MAX_WAIT, "poll_seconds": 30,
        "source_sha256": sources, "fixed_input_sha256": inputs,
        "environment": environment(), "expected_native_temperatures": TEMPERATURES,
        "policy": {"model_calls": 0, "resource_actions": 0, "retry": False,
                   "resume": False, "paper_edits": False, "pending_correctness_reads": False},
    }
    exclusive_json(root / "registration.json", registration)
    exclusive_json(root / "registration.sha256", suite.digest(root / "registration.json"))
    (root / "registration.json").chmod(0o444)
    (root / "registration.sha256").chmod(0o444)
    verify_frozen(root, registration)
    status(root, status="prepared_not_started", profiles=list(lanes))
    event(root, "prepared", profiles=list(lanes))
    return registration


def verify_frozen(root, registration):
    check_digest(root / "registration.json", suite.read_json(root / "registration.sha256"))
    suite.require(suite.read_json(root / "registration.json") == registration,
                  "Registration changed after admission.")
    suite.require(registration["environment"] == environment(), "Offline environment changed.")
    for name, digest in registration["source_sha256"].items():
        check_digest(REPO / name, digest)
        check_digest(root / "source-snapshot" / name, digest)
    for name, digest in registration["fixed_input_sha256"].items():
        check_digest(REPO / name, digest)


def required_hashes(profile, run):
    result = {}
    for name, path in expanded.required_inputs(profile, run).items():
        suite.require(path.is_file() and not path.is_symlink(), "Missing or aliased terminal input.")
        result[name] = suite.digest(path)
    return result


def safe_head(path):
    suite.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024,
                  "Kev head is missing, aliased, or too large for bounded metadata inspection.")
    import torch
    suite.require(not torch.cuda.is_initialized(), "Offline metadata loader must not initialize CUDA.")
    value = torch.load(path, map_location="cpu", weights_only=True)
    suite.require(not torch.cuda.is_initialized() and isinstance(value, dict),
                  "Unsafe or malformed Kev metadata load.")
    return value


def checkpoint_evidence(profile, root, evidence, *, head_loader=safe_head):
    """Hash every pinned payload; derive temperature from config/head, never GPU runtime."""
    active = suite.read_json(root / "benchmark-registration-parent-locks.json")
    artifacts = suite.read_json(root / "artifacts.json")
    suite.require(artifacts == active["models"] and artifacts["source_revisions"] == expanded.SOURCES
                  and set(artifacts["models"]) == set(expanded.PINS), "Checkpoint registration mismatch.")
    expanded.source_hashes(active["code_sha256"], evidence)
    payloads = {}
    for name, (repository, revision) in expanded.PINS.items():
        model = artifacts["models"][name]
        directory = root / "models" / name
        suite.require(model["repository"] == repository and model["revision"] == revision
                      and Path(model["path"]) == directory and bool(model["sha256"]),
                      "Pinned checkpoint repository, revision or directory differs.")
        for filename, digest in model["sha256"].items():
            path = directory / filename
            suite.require(path.resolve().is_relative_to(directory.resolve())
                          and path.is_file() and not path.is_symlink(),
                          "Invalid checkpoint inventory path.")
            payloads[expanded.relative(path)] = evidence.track(path, digest)
    if profile == "decider":
        config = suite.read_json(root / "models/decider/decider_config.json")
        temperature = config["temperature_by_type"]["choice"]
        suite.require(config.get("schema_first") is False, "Unexpected Decider schema-first layout.")
        method = "pinned decider_config.json temperature_by_type.choice; plain choice/state-first"
    else:
        head = head_loader(root / "models/kev/head.pt")
        suite.require(head.get("base") == expanded.PINS["kev-base"][0]
                      and head.get("base_revision") == expanded.PINS["kev-base"][1]
                      and isinstance(head.get("head"), dict) and bool(head["head"]),
                      "Kev head/base metadata differs from the registered release.")
        temperature = head["temperature"]
        method = "pinned head.pt weights_only=True, map_location=cpu; explicit checkpoint temperature"
    suite.require(type(temperature) in (int, float) and math.isfinite(temperature)
                  and temperature == TEMPERATURES[profile], "Independently derived temperature mismatch.")
    preparation = root / f"cpu-smoke-{profile}/runtime.json"
    evidence.track(preparation)
    smoke = suite.read_json(preparation)
    suite.require(smoke["status"] == "ready" and smoke["runtime"]["candidate"] == profile
                  and smoke["runtime"]["device"] == "cpu"
                  and smoke["runtime"]["cuda_initialized"] is False
                  and smoke["runtime"]["temperature"] == temperature,
                  "Independent checkpoint temperature disagrees with earlier preparation evidence.")
    return {"checkpoint_inventory_independently_verified": True,
            "effective_temperature": temperature, "temperature_independently_verified": True,
            "temperature_derivation": method, "checkpoint_payload_sha256": payloads,
            "producer_environment": {"packages": active["packages"], "overlay": active["overlay"],
                                     "dependency_deviations": active["dependency_deviations"]}}


def verify_terminal(profile, run, observed, registration, root):
    """Independent offline analysis relative to the producer, before setting attestation flags."""
    suite.require(observed.get("status") in expanded.TERMINAL, "Pending lane: no correctness access.")
    verify_frozen(root, registration)
    before = required_hashes(profile, run)
    suite.require(suite.read_json(run / "status.json") == observed, "Terminal status changed before replay.")
    evidence = prior.Evidence()
    panel_root = Path(registration["panel"])
    panel, instances, _, validation = prior.load_panel(panel_root, evidence)
    identity = {}
    if profile in TEMPERATURES:
        identity = checkpoint_evidence(profile, run.parent, evidence)
        model_id, _ = expanded.trained_identity(
            profile, run, panel, panel_root, instances, identity, evidence)
    else:
        model_id, _ = expanded.qwen_identity(profile, run, panel, panel_root, evidence)
        suite.require(observed.get("profile") == profile, "Thinking status profile differs.")
    attempts = suite.read_rows(run / "attempts.jsonl")
    recorded = suite.read_rows(run / "scores.jsonl")
    rows = canonical.replay(instances, attempts, SimpleNamespace(model=model_id), native_action=False)
    if profile in TEMPERATURES:
        for row in rows:
            row.update(scheduled=True, unattempted=row["query_count"] == 0)
    suite.require(len(rows) == 64 and rows == recorded, "Independent full-row replay differs.")
    expanded.replay_terminal(instances, attempts, recorded, observed, model_id,
                             trained=profile in TEMPERATURES)
    suite.require(prior.matrix.same_aggregates(suite.summarize(rows),
                                               suite.read_json(run / "summary.json")),
                  "Independent summary replay differs.")
    probabilities = expanded.probability_audit(attempts, model_id) if profile in TEMPERATURES else []
    objectives = expanded.independent_objectives(instances, rows)
    for reference in prior.HISTORICAL:
        manifest, _, _ = prior.load_sealed_reference(
            Path(registration["historical"]) / f'{panel["split"]}.{reference}',
            Path(panel["bundle"]), panel["split"], instances, evidence)
        suite.require(manifest["profile"] == reference, "Historical reference identity differs.")
    after = required_hashes(profile, run)
    suite.require(before == after, "Candidate bytes changed during independent verification.")
    evidence.recheck()
    verify_frozen(root, registration)
    return {
        "schema_version": 1, "status": "verified_terminal_expanded_pilot",
        "profile": profile, "run": str(run.resolve()), "source_status": observed["status"],
        "episodes": len(rows), "calls": len(attempts), "independent_replay_verified": True,
        "new_model_calls": 0, "panel_registration_sha256": suite.digest(panel_root / "registration.json"),
        "input_sha256": before, **identity,
        "independence_scope": "Offline canonical request/own-history replay and independent "
        "NetworkX/Fraction references relative to producer; local attestation, not a signature.",
        "verification_source_sha256": registration["source_sha256"],
        "verified_evidence_sha256": evidence.hashes, "offline_environment": environment(),
        "environment_compatibility": "Offline verifier environment is fingerprinted separately; "
        "trained producer packages/overlay/deviations are preserved when available. Thinking "
        "execution has no producer package inventory. No inference-environment reproduction claim. "
        "Canonical aggregate float tolerance is 1e-12; per-episode scores are exact.",
        "independent_reference_validation": validation, "construction_audit": objectives,
        "native_probability_audit": probabilities,
        "counts": {"attempted": sum(r["query_count"] > 0 for r in rows),
                   "unattempted": sum(r["query_count"] == 0 for r in rows),
                   "failed": sum(r["failure_count"] > 0 for r in rows),
                   "feasible": sum(r["feasible"] for r in rows)},
    }


def finalize_lane(profile, run, observed, registration, root):
    proof = verify_terminal(profile, run, observed, registration, root)
    for name, digest in proof["input_sha256"].items():
        check_digest(REPO / name, digest)
    proof_path = root / "proofs" / f"{profile}.json"
    proof_path.parent.mkdir(exist_ok=True)
    exclusive_json(proof_path, proof)
    (root / "exports").mkdir(exist_ok=True)
    destination = root / "exports" / profile
    result = expanded.export(profile, run, proof_path, destination,
                             Path(registration["panel"]), Path(registration["historical"]))
    seal = suite.read_json(destination / "export-seal.json")
    for name, digest in seal["artifact_sha256"].items():
        check_digest(destination / name, digest)
    verify_frozen(root, registration)
    suite.require(result["source_status"] == observed["status"], "Export terminal state changed.")
    return {"status": "verified", "source_status": observed["status"], "proof": str(proof_path),
            "proof_sha256": suite.digest(proof_path), "export": str(destination),
            "export_seal_sha256": suite.digest(destination / "export-seal.json")}


def overall(outcomes):
    return ("completed" if len(outcomes) == 4 and all(
        item["status"] == "verified" and item["source_status"] == "completed"
        for item in outcomes.values()) else "completed_with_failures")


def observe(run):
    path = run / "status.json"
    if not path.exists():
        return {"status": "waiting_for_lane_creation"}
    return suite.read_json(path)


def run(root=ROOT, *, clock=time.monotonic, sleep=time.sleep, finalize=finalize_lane):
    suite.require(root == ROOT, "Only the fixed finalization root is supported.")
    registration = suite.read_json(root / "registration.json")
    check_digest(root / "registration.json", suite.read_json(root / "registration.sha256"))
    expected = {name: str(EXECUTION / name) for name in PROFILES[:2]}
    expected.update({name: str(TRAINED / f"run-{name}") for name in PROFILES[2:]})
    suite.require(registration["lanes"] == expected, "Unexpected fixed lane map.")
    suite.require(0 < registration["maximum_wait_seconds"] <= MAX_WAIT, "Wait exceeds finite bound.")
    started = clock()
    exclusive_json(root / "finalizer-start.json", {
        **process_identity(), "started_at": time.time(),
        "deadline_at": time.time() + registration["maximum_wait_seconds"],
        "registration_sha256": suite.digest(root / "registration.json")})
    outcomes, waiting = {}, {}
    try:
        verify_frozen(root, registration)
        event(root, "started")
        deadline = started + registration["maximum_wait_seconds"]
        while len(outcomes) < len(PROFILES):
            if clock() >= deadline:
                status(root, status="deadline_exceeded", outcomes=outcomes, waiting=waiting,
                       diagnostic="Finite wait exhausted; no pending correctness was read.")
                event(root, "deadline_exceeded", waiting=waiting)
                return 2
            for profile in PROFILES:
                if profile in outcomes:
                    continue
                observed = observe(Path(registration["lanes"][profile]))
                if observed.get("status") not in expanded.TERMINAL:
                    waiting[profile] = {key: observed[key] for key in ("status", "calls", "benchmark_calls")
                                        if key in observed}
                    continue
                event(root, "terminal_observed", profile=profile, source_status=observed["status"])
                try:
                    outcomes[profile] = finalize(
                        profile, Path(registration["lanes"][profile]), observed, registration, root)
                except Exception as exc:
                    outcomes[profile] = {"status": "verification_failed",
                                         "source_status": observed["status"],
                                         "error_type": type(exc).__name__, "error": str(exc)[:1600]}
                waiting.pop(profile, None)
                exclusive_json(root / f"{profile}-outcome.json", outcomes[profile])
                event(root, "lane_finalized", profile=profile, **outcomes[profile])
            if len(outcomes) == len(PROFILES):
                result = overall(outcomes)
                status(root, status=result, outcomes=outcomes, waiting={})
                event(root, "finished", status=result)
                return 0 if result == "completed" else 1
            remaining = deadline - clock()
            if remaining <= 0:
                status(root, status="deadline_exceeded", outcomes=outcomes, waiting=waiting,
                       diagnostic="24-hour finite wait exhausted; pending correctness was not read.")
                event(root, "deadline_exceeded", waiting=waiting)
                return 2
            status(root, status="waiting_for_terminal_lanes", outcomes=outcomes, waiting=waiting,
                   remaining_seconds=remaining)
            sleep(min(registration["poll_seconds"], remaining))
    except BaseException as exc:
        status(root, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               outcomes=outcomes, waiting=waiting, error_type=type(exc).__name__, error=str(exc)[:1600])
        event(root, "failed", error_type=type(exc).__name__)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "inspect", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "inspect":
        print(json.dumps(suite.read_json(ROOT / "status.json"), indent=2))
    else:
        raise SystemExit(run())


if __name__ == "__main__":
    main()
