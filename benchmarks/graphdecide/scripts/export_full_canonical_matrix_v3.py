#!/usr/bin/env python3
"""Receipt-aware revision; retain the initial capture and verify later post-exec evidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "_full_canonical_export_v3_prior", REPO / "scripts/export_full_canonical_matrix_v2.py")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
suite = prior.suite
RECEIPT = prior.TRAINED / "parent-launch-verification.json"
RECEIPT_SHA = "fa85aa3645f5229c14edbb6d3ad484c49a86fdebbc17395dab0a0f9601731fc0"


class Evidence(prior.Evidence):
    def __init__(self):
        super().__init__()
        self.track(__file__)


def validate_receipt(root, receipt, intent, captured, launch, handoff):
    supervisor, parent = receipt["supervisor"], receipt["coordinator"]
    expected = [intent["argv"][0], str(REPO / prior.COORDINATOR), "worker-phase",
                "--root", str(root), "--handoff", str(root / "standalone-coordinator/worker-handoff.json")]
    suite.require(receipt["status"] == "running_not_completed"
                  and receipt["producer_source_unchanged"] is True
                  and type(receipt["new_probe_inference_calls"]) is int
                  and receipt["new_probe_inference_calls"] == 0
                  and receipt["standard_resources"] == ["qwen08", "qwen2"],
                  "Parent receipt scope differs.")
    suite.require(prior.identity(supervisor) == prior.identity(captured)
                  and supervisor["pid"] == launch["pid"]
                  and supervisor["ppid"] == parent["pid"] == handoff["parent_pid"]
                  and prior.same_process(parent, launch["parent"])
                  and prior.same_process(parent, intent["parent"])
                  and prior.identity(parent)[1] == handoff["parent_start_ticks"],
                  "Post-exec parent/supervisor identity differs.")
    expected_environment = {**intent["parent"]["cuda_environment"],
                            "CUDA_VISIBLE_DEVICES": handoff["cuda_visible_devices"]}
    suite.require(supervisor["argv"] == expected == intent["argv"]
                  and supervisor["cuda_environment"] == expected_environment
                  and supervisor["cwd"] == parent["cwd"] == str(REPO),
                  "Observed post-exec command/environment differs.")
    return {"receipt_sha256": RECEIPT_SHA, "observed_at_utc": receipt["at_utc"],
            "supervisor_pid": supervisor["pid"], "supervisor_start_ticks": prior.identity(supervisor)[1],
            "supervisor_argv_observed_and_verified": True,
            "supervisor_cuda_environment_observed_and_verified": True,
            "historical_running_observation_not_terminal_proof": True,
            "initial_empty_capture_retained": True}


def source_data(lane, evidence):
    postexec = None
    if lane.kind == "trained":
        prior.base.require_ready([lane])
        evidence.track(RECEIPT, RECEIPT_SHA)
        paths = (
            lane.root / "standalone-coordinator/resources/trained-supervisor-launch-intent.json",
            lane.root / "standalone-coordinator/resources/trained-supervisor-process.json",
            lane.root / "launch.json", lane.root / "standalone-coordinator/worker-handoff.json",
        )
        for path in paths:
            evidence.track(path)
        postexec = validate_receipt(lane.root, suite.read_json(RECEIPT),
                                    *(suite.read_json(path) for path in paths))
    result = prior.source_data(lane, evidence)
    if postexec is not None:
        manifest = result[0]
        manifest["source_verification"] = {
            **manifest["source_verification"], "post_exec_parent_receipt": postexec}
        evidence.recheck()
    return result


prior.base.Evidence, prior.base.source_data = Evidence, source_data
export = prior.export


def __getattr__(name):
    return getattr(prior, name)


if __name__ == "__main__":
    prior.base.main()
