#!/usr/bin/env python3
"""Versioned terminal-only adapter for the registered standalone trained producer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True


def isolated(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = isolated("_full_canonical_export_v2_base", "export_full_canonical_matrix.py")
suite = base.suite
COORDINATOR = "scripts/full_matrix_trained_coordinator.py"
COORDINATOR_SHA = "ce2ed02fcf5ccda06e27cc75de93140aad21c3e26ac0e36c90579f10ad9f77ed"
TRAINED_SHA = "368ac82ea076696f4cddad5b5b3133e8e3aad7ba1c533b96f3c1015f801bfc39"
ASTRA_SHA = "2c81fb3b6c7c656500462bd7edcb41ec6ea208c8be70c54b8caf6d1224c90a48"
raw_source_data, raw_readiness = base.source_data, base.readiness


class Evidence(base.Evidence):
    def __init__(self):
        super().__init__()
        self.track(__file__)


def coordinator_state(root):
    out = root / "standalone-coordinator"
    state = suite.read_json(out / "status.json") if (out / "status.json").is_file() else {}
    status = state.get("status", "not_started")
    return {
        "status": status,
        "ready": status == "terminal_verified_services_restored"
        and state.get("original_services_restored") is True
        and all((out / name).is_file() for name in ("run-verification.json", "resources/restoration.json")),
        "blocked": status in ("failed", "aborted", "interrupted"),
    }


def readiness(lanes):
    states = raw_readiness(lanes)
    for lane in lanes:
        if lane.kind == "trained":
            coordinator = coordinator_state(lane.root)
            states[lane.name]["coordinator"] = coordinator
            if not coordinator["ready"]:
                states[lane.name].update(ready=False, reason="coordinator_not_verified_restored")
    return states


def verify_qwen(producer, root):
    try:
        return producer.verify(root)
    except suite.shared.Refusal as exc:
        if str(exc) != "Summary replay differs.":
            raise
    # The frozen verifier has already checked registration, seals and every score.
    registration, instances = producer.load(root)
    attempts = suite.read_rows(root / "run/attempts.jsonl")
    rows = base.canonical.replay(instances, attempts, producer.configuration(), native_action=False)
    suite.require(base.matrix.same_aggregates(
        suite.json_value(suite.summarize(rows)), suite.read_json(root / "run/summary.json")),
        "Qwen summary differs beyond 1e-12 aggregate roundoff.")
    terminal = suite.read_json(root / "run/terminal.json")
    suite.require(terminal["status"] in base.COMPLETE and terminal["calls"] == len(attempts)
                  and all(r["feasible"] or r["failure_count"] for r in rows)
                  and (terminal["status"] == "completed") ==
                  all(a["status"] == "success" for a in attempts),
                  "Qwen terminal accounting differs.")
    return {"verified": True, "status": terminal["status"], "episodes": len(rows),
            "calls": len(attempts), "model_calls": 0,
            "aggregate_roundoff_absolute_tolerance": 1e-12}


def identity(value):
    pid, ticks = value["pid"], value["start_ticks"]
    suite.require(type(pid) is int and pid > 1
                  and (type(ticks) is int or isinstance(ticks, str) and ticks.isdecimal())
                  and int(ticks) > 0, "Invalid recorded process identity.")
    return pid, int(ticks)


def same_process(left, right):
    return identity(left) == identity(right) and left["argv"] == right["argv"]


def validate_route(root, registration, records):
    """Static production-chain proof only: never inspect live processes or locks."""
    start, intent, child, handoff, launch, admission, restoration, resource_status = records
    approval = start["approval"]
    suite.require(approval.get("approved") is True
                  and approval["coordinator_registration_sha256"] == COORDINATOR_SHA
                  and approval["base_registration_sha256"] == TRAINED_SHA,
                  "Production approval does not bind both fixed registrations.")
    parent = launch["parent"]
    pid, ticks = identity(parent)
    expected = [registration["python"], str(REPO / COORDINATOR), "worker-phase",
                "--root", str(root), "--handoff", str(root / "standalone-coordinator/worker-handoff.json")]
    uuid = handoff["cuda_visible_devices"]
    suite.require(launch["active_entry"] == COORDINATOR and launch["handoff"] == handoff
                  and start["pid"] == pid == handoff["parent_pid"]
                  and ticks == handoff["parent_start_ticks"]
                  and type(handoff["parent_start_ticks"]) is int
                  and same_process(parent, intent["parent"])
                  and same_process(parent, admission["parent"])
                  and any(Path(arg).resolve() == REPO / COORDINATOR for arg in parent["argv"][1:3])
                  and "run" in parent["argv"],
                  "Trained coordinator parent/handoff identity differs.")
    suite.require(handoff["coordinator_registration_sha256"] == COORDINATOR_SHA
                  and handoff["base_registration_sha256"] == TRAINED_SHA
                  and isinstance(uuid, str) and uuid.startswith("GPU-") and "," not in uuid
                  and intent["argv"] == expected and child["argv"] in ([], expected)
                  and intent["retry_forbidden"] is True
                  and intent["cuda_visible_devices"] == uuid
                  and (child["cuda_environment"] == {} or
                       child["cuda_environment"].get("CUDA_VISIBLE_DEVICES") == uuid)
                  and child["cwd"] == str(REPO) and child["ppid"] == pid
                  and launch["pid"] == identity(child)[0],
                  "Trained supervisor command/process/UUID differs.")
    suite.require(handoff["locks"] == admission["leases"]
                  and set(handoff["locks"]) == {"qwen08", "qwen2"}
                  and all(row["owner_pid"] == pid and bool(row["device_inode"])
                          for row in handoff["locks"].values())
                  and admission["target_gpu_uuids"][0] == uuid
                  and admission["models_verified_idle"] is True,
                  "Registered production lease evidence differs.")
    suite.require(resource_status["status"] == "restored"
                  and resource_status["original_services_restored"] is True
                  and restoration["models_verified_idle"] is True
                  and restoration["protected_services_unchanged"] is True
                  and set(restoration["services"]) == set(admission["services"]) == {"qwen08", "qwen2"},
                  "Original services are not verifiably restored.")
    for name, restored in restoration["services"].items():
        original = admission["services"][name]
        identity(restored)
        suite.require(all(restored[key] == original[key] for key in
                          ("argv", "cwd", "cuda_environment")),
                      "Restored service configuration differs.")
    return {"launch_intent_argv_verified": True,
            "captured_supervisor_argv_observed": bool(child["argv"]),
            "captured_supervisor_cuda_environment_observed": bool(child["cuda_environment"]),
            "worker_written_entry_and_handoff_verified": True}


def trained_source(lane, evidence):
    base.require_ready([lane])
    root, out = lane.root, lane.root / "standalone-coordinator"
    producer = base.import_verifier("full_matrix_trained", evidence)
    coordinator = base.import_verifier("full_matrix_trained_coordinator", evidence)
    for path, expected in ((root / "registration.json", TRAINED_SHA),
                           (out / "registration.json", COORDINATOR_SHA)):
        evidence.track(path, expected)
        evidence.track(path.with_suffix(".sha256"))
    amendment = coordinator.verify(root)
    registration, instances = producer.verify_registration(root)
    base.source_inventory(evidence, registration, root)
    base.source_inventory(evidence, amendment, out)
    base.no_test_results(registration)
    paths = ("start.json", "resources/trained-supervisor-launch-intent.json",
             "resources/trained-supervisor-process.json", "worker-handoff.json",
             "../launch.json", "resources/admission.json", "resources/restoration.json",
             "resources/status.json")
    records = []
    for name in paths:
        path = out / name
        evidence.track(path)
        records.append(suite.read_json(path))
    route_observations = validate_route(root, registration, records)
    evidence.track(root / "standalone-parent-approval.json")
    suite.require(suite.read_json(root / "standalone-parent-approval.json") == records[0]["approval"],
                  "Recorded production approval differs from the parent-approved input.")
    evidence.track(out / "resources/trained-supervisor.log")
    restoration = records[6]
    for name, service in restoration["services"].items():
        path = out / "resources" / f"restore-{name}-process.json"
        evidence.track(path)
        recorded = suite.read_json(path)
        suite.require(same_process(service, recorded)
                      and all(service[key] == recorded[key] for key in ("cwd", "cuda_environment")),
                      "Restoration process evidence differs.")
        path = out / "resources" / f"restore-{name}-intent.json"
        evidence.track(path)
        intent = suite.read_json(path)
        suite.require(intent["retry_forbidden"] is True
                      and all(intent[key] == service[key] for key in ("argv", "cwd", "cuda_environment")),
                      "Restoration launch intent differs.")
    for path in (lane.terminal, lane.seal, out / "status.json", out / "run-verification.json",
                 lane.run / "worker.log", lane.run / "worker-process.json", lane.run / "runtime.json"):
        evidence.track(path)
    proofs = suite.read_json(lane.seal)
    suite.require(proofs == suite.read_json(out / "run-verification.json")
                  and set(proofs) == {"decider", "kev"}, "Coordinator terminal proofs differ.")
    proof = proofs[lane.name]
    evidence.inventory(lane.run, proof["sha256"], (
        "scheduled.json", "dispatch.jsonl", "attempts.jsonl", "scores.jsonl", "summary.json", "status.json"))
    worker = suite.read_json(lane.run / "worker-process.json")
    suite.require(type(worker["pid"]) is int and worker["pid"] > 1 and worker["device"] == "cuda",
                  "Missing real trained worker process.")
    runtime = suite.read_json(lane.run / "runtime.json")
    producer.runtime_gate(runtime, lane.name, registration)
    verification = producer.verify_terminal(root, lane.name, instances)
    suite.require(verification == proof, "Trained terminal replay differs.")
    terminal = suite.read_json(lane.terminal)
    attempts, rows = suite.read_rows(lane.run / "attempts.jsonl"), suite.read_rows(lane.run / "scores.jsonl")
    evidence.recheck()
    verification = {**verification, "production_entry": COORDINATOR,
                    "coordinator_registration_sha256": COORDINATOR_SHA,
                    "production_route_verified": True, "services_restoration_verified": True,
                    "process_observations": route_observations,
                    "runtime_and_process_hashes": "source_sha256 (not the producer six-file proof)"}
    return ({**registration, "runtime": runtime, "source_verification": verification,
             "production_entry": COORDINATOR}, terminal, attempts, rows, producer.model_id(lane.name), False)


def source_data(lane, evidence):
    if lane.kind == "trained":
        return trained_source(lane, evidence)
    if lane.kind != "qwen27":
        if lane.kind == "astra":
            base.require_ready([lane])
            evidence.track(lane.root / "manifest.json", ASTRA_SHA)
        result = raw_source_data(lane, evidence)
        if lane.kind == "laya":
            path = lane.run / "worker-process.json"
            evidence.track(path)
            worker = suite.read_json(path)
            supervisor = suite.read_json(lane.run / "supervisor-process.json")
            identity_record = suite.read_json(lane.run / "worker-identity.json")
            suite.require(worker["pid"] == identity_record["pid"]
                          and worker["supervisor_pid"] == supervisor["pid"] and worker["device"] == "cpu",
                          "Laya worker-parent identity differs.")
            evidence.recheck()
        return result
    base.require_ready([lane])
    producer = base.import_verifier("full_matrix_qwen27", evidence)
    for path in (lane.terminal, lane.seal, lane.root / "registration.sha256"):
        evidence.track(path)
    seal = suite.read_json(lane.seal)
    suite.require(set(seal) == base.ARTIFACTS | {"manifest.json", "terminal.json"}, "Qwen seal differs.")
    evidence.inventory(lane.run, seal)
    manifest = suite.read_json(lane.run / "manifest.json")
    base.no_test_results(manifest)
    suite.require(manifest.get("injected_client") is False, "Qwen real-client marker missing.")
    evidence.track(lane.root / "registration.json", suite.read_json(lane.root / "registration.sha256"))
    registration = suite.read_json(lane.root / "registration.json")
    base.source_inventory(evidence, registration, lane.root)
    verification = verify_qwen(producer, lane.root)
    attempts, rows = suite.read_rows(lane.run / "attempts.jsonl"), suite.read_rows(lane.run / "scores.jsonl")
    evidence.recheck()
    return ({**registration, "source_verification": verification},
            suite.read_json(lane.terminal), attempts, rows, manifest["model"]["model"], False)


# Only this isolated copy is adapted; the frozen v1 module and all producers stay untouched.
base.Evidence, base.source_data, base.readiness = Evidence, source_data, readiness
export = base.export


def __getattr__(name):
    return getattr(base, name)


if __name__ == "__main__":
    base.main()
