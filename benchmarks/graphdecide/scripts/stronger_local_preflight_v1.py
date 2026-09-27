#!/usr/bin/env python3
"""Read-only admission audit for the two proposed stronger local baselines.

This is not an inference coordinator. A failed admission must not download
weights, relocate the shared locks, or stop any service.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import full_matrix_gpu_resources as resources
from scripts import full_matrix_qwen27 as canonical960

runtime, suite = resources.runtime, resources.suite
LANES = {**resources.LANES, **resources.PROTECTED}
ROOT = REPO / "output/stronger-local-admission-20260926-v1"
PRIOR_ADMISSION = (
    REPO / "output/full-matrix-qwen27-20260925-v1/resources/admission.json")
HEADROOM_BYTES = 32 * 1024**3
MODELS = (
    {
        "lane": "qwen38_27b_bf16",
        "model": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "tensor_parallel_size": 2,
        "generation": "2026; newer-generation comparison",
        "thinking_policy": "enable_thinking=false",
    },
    {
        "lane": "qwen25_72b_bf16",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "revision": "495f39366efef23836d0cfae4fbe635880d2be31",
        "architecture": "Qwen2ForCausalLM",
        "tensor_parallel_size": 4,
        "generation": "2024; older generation, dense size comparison",
        "thinking_policy": "plain instruct, no hybrid thinking",
    },
)


def four_gpu_mapping(services, inventory):
    """Reject every unexpected occupant, including on physical GPUs 2 and 3."""
    by_gpu = {int(index): uuid for index, uuid in inventory["gpus"]}
    suite.require(set(by_gpu) >= {0, 1, 2, 3}, "Four GPUs are required.")
    suite.require(set(services) == set(LANES), "Missing original service.")
    allowed = {}
    for lane, (gpu, _, _) in LANES.items():
        original = services[lane]
        suite.require(original["cuda_environment"].get("CUDA_VISIBLE_DEVICES") == str(gpu),
                      "Original CUDA mapping differs.")
        children = {p["pid"] for p in original["process_tree"]}
        matches = [(int(pid), uuid) for pid, uuid, _ in inventory["processes"]
                   if int(pid) in children]
        suite.require(matches and all(uuid == by_gpu[gpu] for _, uuid in matches),
                      "Original engine GPU mapping differs.")
        for child in children:
            suite.require(child not in allowed, "Original service trees overlap.")
            allowed[child] = by_gpu[gpu]
    targets = [by_gpu[gpu] for gpu in range(4)]
    suite.require(all(allowed.get(int(pid)) == uuid
                      for pid, uuid, _ in inventory["processes"] if uuid in targets),
                  "Unexpected GPU occupant; no shutdown permitted.")
    return targets


def validate_metadata(spec, info, config, index):
    suite.require(info["sha"] == spec["revision"], "Pinned Hub revision mismatch.")
    suite.require(config.get("architectures") == [spec["architecture"]],
                  "Unexpected model architecture.")
    suite.require(not config.get("quantization_config"), "Quantized model forbidden.")
    dtype = config.get("dtype", config.get("torch_dtype"))
    text = config.get("text_config", {})
    dtype = dtype or text.get("dtype", text.get("torch_dtype"))
    suite.require(dtype == "bfloat16", "Config must advertise BF16.")
    files = {}
    for sibling in info["siblings"]:
        name = sibling["rfilename"]
        path = Path(name)
        suite.require(not path.is_absolute() and ".." not in path.parts,
                      "Unsafe remote filename.")
        files[name] = sibling
    shards = sorted(set(index["weight_map"].values()))
    suite.require(shards and set(shards) <= set(files), "Missing indexed shard.")
    result = {}
    for name in shards:
        f = files[name]
        lfs = f.get("lfs", {})
        digest = lfs.get("sha256", "")
        suite.require(name.endswith(".safetensors") and f["size"] > 0
                      and len(digest) == 64
                      and all(c in "0123456789abcdef" for c in digest)
                      and lfs.get("size") == f["size"], "Missing size/LFS SHA evidence.")
        result[name] = {"bytes": f["size"], "hub_lfs_sha256": digest}
    return {
        **spec, "advertised_dtype": dtype, "shard_count": len(shards),
        "payload_bytes": sum(f["bytes"] for f in result.values()),
        "repository_bytes": sum(f.get("size", 0) for f in files.values()),
        "shards": result, "trust_remote_code": False,
        "local_payload_sha_verified": False,
        "safetensors_header_dtypes_verified": False,
        "tokenizer_rendering_verified": False,
    }


def storage_admission(free, metadata, headroom=HEADROOM_BYTES):
    required = sum(m["repository_bytes"] for m in metadata)
    return {
        "available_bytes": free, "repository_bytes_upper_bound": required,
        "payload_bytes": sum(m["payload_bytes"] for m in metadata),
        "reserved_headroom_bytes": headroom,
        "remaining_after_upper_bound_bytes": free - required,
        "sufficient": free >= required + headroom,
        "download_policy": "One local-dir snapshot per pinned model; no duplicate weight cache.",
    }


def lock_evidence(prior, permitted_root):
    """Inspect project-resident historical evidence, never the lock directory."""
    known = {Path(v["path"]).parent for v in prior["leases"].values()}
    suite.require(len(known) == 1, "Historical standard lock namespaces disagree.")
    directory = next(iter(known))
    suite.require(directory.is_absolute(), "Historical lock root is not absolute.")
    paths = {lane: str(directory / f"jevgraphbench-paired-{os.getuid()}-{lane}.lock")
             for lane in LANES}
    allowed = directory.is_relative_to(permitted_root)
    return {
        "standard_paths": paths,
        "namespace_source": str(PRIOR_ADMISSION.relative_to(REPO)),
        "namespace_source_kind": "Historical namespace only; never historical PID identities.",
        "access_attempted": False, "relocated": False, "admissible": allowed,
        "reason": None if allowed else "standard_lock_namespace_outside_permitted_file_scope",
    }


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def audit(root):
    root = runtime.local(root)
    suite.require(not root.exists(), "Audit root exists; no overwrite.")
    root.mkdir(parents=True)
    actor = runtime.process(os.getpid())
    suite.write_json(root / "start.json", {"at_utc": runtime.now(), "identity": actor})
    runtime.status(root, status="admission_in_progress", pid=actor["pid"], model_calls=0)
    try:
        before = {lane: resources.service(lane) for lane in LANES}
        for original in before.values():
            base = Path(f"/proc/{original['pid']}")
            suite.require(base.stat().st_uid == os.getuid()
                          and original["cwd"] == str(REPO),
                          "Original is not owned by this user/project.")
        inventory = runtime.gpu_inventory()
        targets = four_gpu_mapping(before, inventory)
        configs = suite.shared.model_configs(models=tuple(LANES))
        asyncio.run(suite.shared.vllm_preflight(
            [SimpleNamespace(model=c) for c in configs.values()]))
        suite.write_json(root / "resources-before.json", {
            "at_utc": runtime.now(), "services": before, "inventory": inventory,
            "target_gpu_uuids": targets, "all_four_verified_idle": True,
            "restoration_settings_captured": True, "processes_stopped": 0,
        })
        metadata = []
        for spec in MODELS:
            base = f"https://huggingface.co/{spec['model']}"
            revision = spec["revision"]
            urls = {
                "hub": f"https://huggingface.co/api/models/{spec['model']}/revision/{revision}?blobs=true",
                "config": f"{base}/resolve/{revision}/config.json",
                "index": f"{base}/resolve/{revision}/model.safetensors.index.json",
            }
            documents = {key: get_json(url) for key, url in urls.items()}
            for key, document in documents.items():
                suite.write_json(root / f"{spec['lane']}-{key}.json", document)
            validated = validate_metadata(
                spec, documents["hub"], documents["config"], documents["index"])
            metadata.append({**validated, "sources": urls, "checked_at_utc": runtime.now()})
        suite.write_json(root / "model-metadata.json", metadata)
        storage = storage_admission(shutil.disk_usage(REPO).free, metadata)
        suite.write_json(root / "storage.json", storage)
        locks = lock_evidence(suite.read_json(PRIOR_ADMISSION), REPO)
        suite.write_json(root / "standard-lock-admission.json", locks)
        suite.write_json(root / "planned-protocol.json", {
            "state": "planned_not_registered_for_inference",
            "models": list(MODELS), "episodes_per_lane": 960,
            "exact_per_lane": 800, "construction_per_lane": 160,
            "maximum_calls_per_lane": 1920,
            "maximum_run_seconds_per_lane": 21600,
            "maximum_request_seconds": 180, "maximum_generated_tokens": 64,
            "task_counts": canonical960.EXPECTED,
            "constrain_choices": True, "thinking": False,
            "retry": False, "repair": False, "resume_inference": False,
            "history": "Model's own trajectory",
            "construction_primary": "Mean absolute objective gap on feasible cases, with coverage",
            "construction_supplemental": "Optimum count over all scheduled cases",
            "comparison_scope": "Separate inspected-bank expansion, not the original web10 matrix",
            "configuration_template": asdict(canonical960.configuration()),
            "template_warning": "Existing template only; new endpoints/checkpoints not registered.",
        })
        after = {lane: resources.service(lane) for lane in LANES}
        suite.require(all(resources.same_identity(before[lane], after[lane]) for lane in LANES),
                      "Original identity changed during audit.")
        four_gpu_mapping(after, runtime.gpu_inventory())
        asyncio.run(suite.shared.vllm_preflight(
            [SimpleNamespace(model=c) for c in configs.values()]))
        blockers = []
        if not storage["sufficient"]:
            blockers.append("insufficient_storage_headroom")
        if not locks["admissible"]:
            blockers.append(locks["reason"])
        result = {
            "status": "blocked" if blockers else "admission_only_no_inference",
            "at_utc": runtime.now(), "blockers": blockers, "model_calls": 0,
            "weight_bytes_downloaded": 0, "processes_stopped": 0,
            "original_services_unchanged": True, "original_services_verified_idle": True,
            "restoration_required": False, "coordinator_started": False,
            "completed_lanes": 0, "planned_lanes": 2,
            "inference_implementation_complete": False,
            "required_before_inference": [
                "Unchanged standard locks accessible within allowed file-operation scope",
                "Bounded coordinator/lease/runner safety tests and source freeze",
                "Payload LFS SHA and safetensors dtype verification",
                "Local and served tokenizer nonthinking rendering verification",
                "Exact canonical initial-request equality and independent replay",
            ],
        }
        suite.write_json(root / "resources-after.json", {
            "at_utc": runtime.now(), "services": after, "unchanged": True})
    except BaseException as exc:
        result = {
            "status": "blocked", "at_utc": runtime.now(),
            "blockers": ["admission_check_failed"], "error_type": type(exc).__name__,
            "message": str(exc)[:1000], "model_calls": 0, "processes_stopped": 0,
            "weight_bytes_downloaded": 0, "coordinator_started": False,
        }
    hashes = {
        **canonical960.source_hashes(),
        "scripts/full_matrix_gpu_resources.py": suite.digest(resources.__file__),
        "scripts/stronger_local_preflight_v1.py": suite.digest(__file__),
    }
    canonical960.invariance.freeze_sources(root, hashes)
    suite.write_json(root / "source-manifest.json", hashes)
    suite.write_json(root / "terminal.json", result)
    suite.write_json(root / "seal.json", {
        p.name: suite.digest(p) for p in sorted(root.glob("*.json"))
        if p.name not in {"seal.json", "status.json"}})
    runtime.status(root, **result)
    return result


def verify(root):
    for name, digest in suite.read_json(root / "seal.json").items():
        suite.require(suite.digest(root / name) == digest, "Audit artifact drift.")
    for name, digest in suite.read_json(root / "source-manifest.json").items():
        suite.require(suite.digest(root / "source-snapshot" / name) == digest,
                      "Audit source snapshot drift.")
    result = suite.read_json(root / "terminal.json")
    suite.require(result["model_calls"] == 0 and result["processes_stopped"] == 0
                  and result["weight_bytes_downloaded"] == 0
                  and not result["coordinator_started"],
                  "Admission-only audit must not claim model execution.")
    return {"verified": True, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root) if args.command == "audit" else verify(args.root)
    print(json.dumps(result, sort_keys=True))
    return 2 if args.command == "audit" and result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
