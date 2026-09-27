#!/usr/bin/env python3
"""Pinned BF16 artifacts and bounded canonical960 runners for two local expansions."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from scripts import full_matrix_qwen27 as base
from scripts import stronger_local_preflight_v1 as admission
from src.clients._llm_choice import ANSWER_ONLY_SYSTEM_MESSAGE, prompt_for
from src.clients.base import ClientTimeoutError

runtime, suite, canonical, release = base.runtime, base.suite, base.canonical, base.release
SPECS = {m["lane"]: {**m, "port": 8011 + i,
                    "served_name": m["model"].split("/", 1)[1]}
         for i, m in enumerate(admission.MODELS)}
ROOT = REPO / "output/stronger-local-execution-20260926-v1"
MATERIALS = REPO / "output/stronger-local-models-20260926-v1"
MAX_SECONDS = 21600
MAX_CALLS = 1920
METADATA_NAMES = {
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "merges.txt", "vocab.json", "preprocessor_config.json",
}


def configuration(lane):
    spec = SPECS[lane]
    return replace(base.configuration(), model=spec["served_name"],
                   base_url=f"http://127.0.0.1:{spec['port']}/v1",
                   think=False, max_tokens=64, timeout_seconds=180,
                   constrain_choices=True, output_format="answer_only")


def atomic_json(path, value):
    writing = path.with_name(f"{path.name}.writing-{os.getpid()}")
    with writing.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    writing.replace(path)


def stable_json(path, value, *, ignore=()):
    if path.exists():
        prior = suite.read_json(path)
        suite.require({k: v for k, v in prior.items() if k not in ignore} ==
                      {k: v for k, v in value.items() if k not in ignore},
                      "Existing pinned metadata differs; no overwrite.")
        return prior
    suite.write_json(path, value)
    return value


def source_hashes():
    names = (
        "scripts/stronger_local_preflight_v1.py",
        "scripts/stronger_local_runtime_v1.py",
        "scripts/stronger_local_coordinator_v1.py",
        "scripts/stronger_local_replay_v1.py",
        "scripts/full_matrix_gpu_resources.py",
        "tests/test_stronger_local_execution_v1.py",
    )
    return {**base.source_hashes(), **{
        name: suite.digest(REPO / name) for name in names}}


def check_sources(root, manifest):
    suite.require(source_hashes() == manifest["source_sha256"], "Live source drift.")
    for name, digest in manifest["source_sha256"].items():
        suite.require(suite.digest(root / "source-snapshot" / name) == digest,
                      "Source snapshot drift.")


def check_file(path, remote):
    suite.require(path.is_file() and not path.is_symlink()
                  and path.stat().st_size == remote["size"], "Checkpoint file size differs.")
    digest = suite.digest(path)
    if remote.get("lfs"):
        suite.require(digest == remote["lfs"]["sha256"], "Checkpoint LFS SHA differs.")
    else:
        h = hashlib.sha1(f"blob {remote['size']}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        suite.require(h.hexdigest() == remote["blobId"], "Checkpoint git blob SHA differs.")
    return {"bytes": remote["size"], "sha256": digest,
            "hub_lfs_sha256": remote.get("lfs", {}).get("sha256"),
            "hub_blob_id": remote.get("blobId")}


def fetch_file(spec, remote, snapshot):
    name = remote["rfilename"]
    suite.require(name in METADATA_NAMES or name.endswith(".safetensors"),
                  "Remote code or unselected file refused.")
    suite.require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                  "Unsafe checkpoint path.")
    path = snapshot / name
    suite.require(not snapshot.is_symlink() and path.resolve().is_relative_to(snapshot.resolve()),
                  "Checkpoint path escapes the selected snapshot.")
    if path.exists():
        return check_file(path, remote)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    suite.require(not partial.is_symlink(), "Symlink partial refused.")
    offset = partial.stat().st_size if partial.exists() else 0
    suite.require(offset <= remote["size"], "Oversize partial checkpoint.")
    if offset < remote["size"]:
        request = urllib.request.Request(
            f"https://huggingface.co/{spec['model']}/resolve/{spec['revision']}/{name}",
            headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset:
                suite.require(response.status == 206 and
                              response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"),
                              "Server refused exact partial-download continuation.")
            with partial.open("ab" if partial.exists() else "xb") as stream:
                while chunk := response.read(4 * 1024 * 1024):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    verified = check_file(partial, remote)
    suite.require(not path.exists(), "Concurrent checkpoint writer detected.")
    partial.rename(path)
    return verified


def tokenizer_check(lane, snapshot):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    instances = release.labeled_instances(base.BUNDLE, base.SPLIT,
                                         release.manifest_at(base.BUNDLE))
    request = canonical.public_request(instances[0], [])
    messages = [{"role": "system", "content": ANSWER_ONLY_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt_for(request)}]
    kwargs = {"add_generation_prompt": True, "enable_thinking": False}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
    suite.require(type(tokens) is list and all(type(token) is int for token in tokens),
                  "Tokenizer did not return a flat integer token sequence.")
    suffix = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    suite.require("<|im_start|>assistant" in rendered, "Assistant generation boundary absent.")
    if lane == "qwen38_27b_bf16":
        suite.require("<think>" in suffix and "</think>" in suffix
                      and not suffix.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                      and not suffix.split("</think>", 1)[1].strip(),
                      "Newer model does not render an empty closed thinking block.")
    else:
        suite.require("<think>" not in rendered and "</think>" not in rendered
                      and not suffix.strip(), "Plain-instruct tokenizer renders thinking.")
    suite.require(len(tokens) < 16000, "Tokenizer preflight exceeds context limit.")
    return {"checked_at_utc": runtime.now(), "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "token_ids": tokens, "token_count": len(tokens),
            "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "assistant_suffix": suffix, "nonthinking_verified": True,
            "trust_remote_code": False, "model_calls": 0}


def prepare_model(lane, materials=MATERIALS, *, weights=False):
    spec = SPECS[lane]
    root = runtime.local(materials) / lane
    suite.require(not root.is_symlink() and root.resolve().is_relative_to(materials.resolve()),
                  "Model directory escapes materials root.")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshot"
    suite.require(not snapshot.is_symlink(), "Symlink snapshot refused.")
    snapshot.mkdir(exist_ok=True)
    if weights and (root / "artifacts.json").exists():
        return verify_artifacts(lane, materials)
    info = (suite.read_json(root / "hub-metadata.json") if (root / "hub-metadata.json").exists()
            else admission.get_json(
                f"https://huggingface.co/api/models/{spec['model']}/revision/{spec['revision']}?blobs=true"))
    suite.require(info["sha"] == spec["revision"], "Hub revision drift.")
    files = {f["rfilename"]: f for f in info["siblings"]}
    for name in ("config.json", "model.safetensors.index.json"):
        fetch_file(spec, files[name], snapshot)
    metadata = admission.validate_metadata(
        spec, info, suite.read_json(snapshot / "config.json"),
        suite.read_json(snapshot / "model.safetensors.index.json"))
    stable_json(root / "hub-metadata.json", info)
    stable_json(root / "public-admission.json", metadata)
    selected = sorted(METADATA_NAMES & files.keys())
    if weights:
        selected += sorted(metadata["shards"])
    # Existing files are verified in place; partials share the final snapshot,
    # so there is no second HF cache containing another payload copy.
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(lambda name: (name, fetch_file(spec, files[name], snapshot)),
                                selected))
    tokenization = tokenizer_check(lane, snapshot)
    stable_json(root / "tokenizer-preflight.json", tokenization, ignore=("checked_at_utc",))
    if not weights:
        return {"lane": lane, "status": "metadata_and_tokenizer_prepared",
                "weight_bytes_downloaded": 0, "model_calls": 0}
    dtypes = Counter()
    for name in sorted(metadata["shards"]):
        with (snapshot / name).open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            suite.require(0 < length < 100_000_000, "Invalid safetensors header length.")
            header = json.loads(stream.read(length))
        dtypes.update(value["dtype"] for key, value in header.items() if key != "__metadata__")
    suite.require(dtypes["BF16"] > 0 and not (set(dtypes) - {"BF16", "F32"}),
                  "Payload is not BF16 with optional FP32 auxiliaries.")
    value = {
        "model": spec["model"], "revision": spec["revision"], "snapshot": str(snapshot),
        "downloaded_at_utc": runtime.now(), "files": dict(entries),
        "tensor_dtypes": dict(dtypes), "tokenizer_preflight_sha256":
            suite.digest(root / "tokenizer-preflight.json"),
        "trust_remote_code": False, "quantization": None,
    }
    suite.write_json(root / "artifacts.json", value)
    suite.write_json(root / "artifacts.sha256", suite.digest(root / "artifacts.json"))
    return value


def verify_artifacts(lane, materials=MATERIALS, *, payload=True):
    root = runtime.local(materials) / lane
    value = suite.read_json(root / "artifacts.json")
    suite.require(suite.digest(root / "artifacts.json") ==
                  suite.read_json(root / "artifacts.sha256"), "Artifact manifest drift.")
    spec = SPECS[lane]
    suite.require(value["model"] == spec["model"] and value["revision"] == spec["revision"]
                  and value["trust_remote_code"] is False and value["quantization"] is None,
                  "Checkpoint identity mismatch.")
    snapshot = root / "snapshot"
    suite.require(value["snapshot"] == str(snapshot), "Unexpected checkpoint directory.")
    info = suite.read_json(root / "hub-metadata.json")
    suite.require(info["sha"] == spec["revision"], "Saved Hub revision differs.")
    remote = {f["rfilename"]: f for f in info["siblings"]}
    public = admission.validate_metadata(
        spec, info, suite.read_json(snapshot / "config.json"),
        suite.read_json(snapshot / "model.safetensors.index.json"))
    suite.require(set(value["files"]) == (METADATA_NAMES & remote.keys()) | set(public["shards"]),
                  "Checkpoint manifest omits or adds selected files.")
    dtypes = value["tensor_dtypes"]
    suite.require(dtypes.get("BF16", 0) > 0 and not (dtypes.keys() - {"BF16", "F32"}),
                  "Saved payload dtype admission differs.")
    for name, entry in value["files"].items():
        path = snapshot / name
        suite.require(not path.is_symlink() and path.resolve().is_relative_to(snapshot)
                      and path.stat().st_size == entry["bytes"], "Checkpoint file drift.")
        if payload or not name.endswith(".safetensors"):
            suite.require(check_file(path, remote[name]) == entry, "Checkpoint hash drift.")
    suite.require(suite.digest(root / "tokenizer-preflight.json") ==
                  value["tokenizer_preflight_sha256"], "Tokenizer admission drift.")
    return value


def register_lane(root, lane, materials=MATERIALS):
    suite.require(not root.exists(), "Lane registration exists; no overwrite.")
    checkpoint = verify_artifacts(lane, materials)
    verified = release.verify(base.BUNDLE)
    instances = release.labeled_instances(base.BUNDLE, base.SPLIT, release.manifest_at(base.BUNDLE))
    base.check_instances(instances)
    reference_status = suite.read_json(base.REFERENCE / "status.json")
    suite.require(reference_status["status"] == "completed", "Canonical reference is incomplete.")
    for name, digest in reference_status["artifact_sha256"].items():
        suite.require(suite.digest(base.REFERENCE / name) == digest, "Reference artifact drift.")
    reference = {a["instance_id"]: a["request"] for a in
                 suite.read_rows(base.REFERENCE / "attempts.jsonl") if a["step"] == 0}
    requests = [{"instance_id": i["id"], "request": suite.json_value(
        asdict(canonical.public_request(i, [])))} for i in instances]
    suite.require(len(reference) == 960 and all(
        reference[r["instance_id"]] == r["request"] for r in requests),
        "Initial requests differ from canonical Jev.")
    root.mkdir(parents=True)
    base.invariance.write_rows(root / "initial-requests.jsonl", requests)
    hashes = source_hashes()
    base.invariance.freeze_sources(root, hashes)
    value = {
        "registered_at_utc": runtime.now(), "lane": lane, "model": asdict(configuration(lane)),
        "checkpoint_revision": checkpoint["revision"], "materials": str(materials),
        "artifact_manifest_sha256": suite.digest(materials / lane / "artifacts.json"),
        "bundle": str(base.BUNDLE), "bundle_manifest_sha256": suite.digest(base.BUNDLE / "manifest.json"),
        "source_sha256": hashes, "initial_requests_sha256": suite.digest(root / "initial-requests.jsonl"),
        "episodes": 960, "exact": 800, "construction": 160, "maximum_calls": MAX_CALLS,
        "maximum_run_seconds": MAX_SECONDS, "task_counts": base.EXPECTED,
        "release_verification": verified, "reference_status_sha256":
            suite.digest(base.REFERENCE / "status.json"),
        "policy": {"retry": False, "repair": False, "resume": False, "thinking": False,
                   "quantization": False, "own_history": True,
                   "construction_primary": "conditional mean absolute gap with feasible coverage",
                   "comparison": "Separate inspected-bank expansion, not fresh confirmation",
                   "response_storage": "No response bodies or hidden reasoning text"},
    }
    suite.write_json(root / "registration.json", value)
    suite.write_json(root / "registration.sha256", suite.digest(root / "registration.json"))
    runtime.status(root, status="registered", calls=0)
    return value


def load(root):
    value = suite.read_json(root / "registration.json")
    suite.require(suite.digest(root / "registration.json") ==
                  suite.read_json(root / "registration.sha256"), "Registration drift.")
    suite.require(value["model"] == asdict(configuration(value["lane"]))
                  and value["maximum_calls"] == MAX_CALLS
                  and value["maximum_run_seconds"] == MAX_SECONDS, "Configuration drift.")
    check_sources(root, value)
    bundle = Path(value["bundle"])
    suite.require(suite.digest(bundle / "manifest.json") == value["bundle_manifest_sha256"],
                  "Bundle manifest drift.")
    # Check all bundled source/data hashes, not just its manifest.
    for name in release.manifest_at(bundle)["sha256"]:
        release.checked_file(bundle, name, release.manifest_at(bundle))
    materials = Path(value["materials"])
    suite.require(suite.digest(materials / value["lane"] / "artifacts.json") ==
                  value["artifact_manifest_sha256"], "Checkpoint manifest drift.")
    instances = release.labeled_instances(bundle, base.SPLIT, release.manifest_at(bundle))
    base.check_instances(instances)
    requests = [{"instance_id": i["id"], "request": suite.json_value(
        asdict(canonical.public_request(i, [])))} for i in instances]
    suite.require(suite.digest(root / "initial-requests.jsonl") ==
                  value["initial_requests_sha256"]
                  and suite.read_rows(root / "initial-requests.jsonl") == requests,
                  "Canonical request drift.")
    return value, instances


async def run(root, *, client_factory=runtime.RecordingClient,
              preflight=suite.shared.vllm_preflight):
    registration, instances = load(root)
    output = root / "run"
    suite.require(not output.exists(), "Run exists; inference retry/resume forbidden.")
    output.mkdir()
    config = configuration(registration["lane"])
    suite.write_json(output / "manifest.json", {
        "model": asdict(config), "registration_sha256": suite.digest(root / "registration.json"),
        "injected_client": client_factory is not runtime.RecordingClient,
        "injected_preflight": preflight is not suite.shared.vllm_preflight,
        "episodes": 960, "maximum_calls": MAX_CALLS})
    attempts, client = [], None
    state = {"status": "running", "calls": 0, "pid": os.getpid()}
    started = time.monotonic()
    runtime.status(root, **state)
    with (output / "attempts.jsonl").open("x") as stream:
        try:
            async with asyncio.timeout(MAX_SECONDS):
                await preflight([SimpleNamespace(model=config)])
                client = client_factory(**config.client_kwargs())
                await asyncio.wait_for(client.initialize(), config.timeout_seconds)
                for instance in instances:
                    history = []
                    while (request := canonical.public_request(instance, history)) is not None:
                        suite.require(len(attempts) < MAX_CALLS
                                      and len(history) < instance["query_budget"], "Call budget exceeded.")
                        check_sources(root, registration)
                        tick, diagnostics, error_type, cancellation = time.monotonic(), None, None, None
                        try:
                            response = await asyncio.wait_for(client.predict(request), config.timeout_seconds)
                            diagnostics = runtime.response_diagnostics(response.raw_output)
                            record = suite.response_record(response, request, config)
                        except BaseException as exc:
                            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                                cancellation = exc
                            if isinstance(exc, TimeoutError):
                                exc = ClientTimeoutError("Request deadline exceeded.")
                            error_type = type(exc).__name__
                            diagnostics = getattr(exc, "response_diagnostics", None)
                            record = {"status": "failure", "selected_option_id": None,
                                      "resolved_model": None,
                                      "usage": suite.safe_usage(getattr(exc, "usage", None)),
                                      **suite.failure(exc)}
                            if cancellation is not None:
                                record["fatal"] = True
                        row = {"instance_id": instance["id"], "step": len(history),
                               "request": asdict(request), "response_diagnostics": diagnostics,
                               "exception_type": error_type, "latency_seconds": time.monotonic() - tick,
                               **record}
                        suite.append_row(stream, row)
                        stream.flush()
                        os.fsync(stream.fileno())
                        attempts.append(suite.json_value(row))
                        runtime.status(root, **{**state, "calls": len(attempts),
                                               "last_instance_id": instance["id"]})
                        if cancellation is not None:
                            raise cancellation
                        if record["status"] == "failure":
                            if record["fatal"]:
                                state.update(status="aborted", diagnostic_code=record["diagnostic_code"])
                            break
                        history.append(record["selected_option_id"])
                    if state["status"] != "running":
                        break
                check_sources(root, registration)
                if state["status"] == "running":
                    state["status"] = "completed_with_failures" if any(
                        a["status"] == "failure" for a in attempts) else "completed"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(
                exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)) else "aborted",
                diagnostic_code="run_deadline" if isinstance(exc, TimeoutError)
                else suite.failure(exc)["diagnostic_code"], exception_type=type(exc).__name__)
        finally:
            if client is not None:
                try:
                    await asyncio.wait_for(client.aclose(), 30)
                except BaseException as exc:
                    state.update(status="aborted", diagnostic_code="shutdown_failed",
                                 exception_type=type(exc).__name__)
            rows = canonical.replay(instances, attempts, config, native_action=False)
            base.invariance.write_rows(output / "scores.jsonl", rows)
            suite.write_json(output / "summary.json", suite.summarize(rows))
            state.update(calls=len(attempts), wall_seconds=time.monotonic() - started,
                         finished_at_utc=runtime.now(),
                         failure_counts=dict(Counter(a["diagnostic_code"] for a in attempts
                                                    if a["status"] == "failure")))
            suite.write_json(output / "terminal.json", state)
            suite.write_json(output / "seal.json", {
                name: suite.digest(output / name) for name in
                ("manifest.json", "attempts.jsonl", "scores.jsonl", "summary.json", "terminal.json")})
            runtime.status(root, **state)
    verify(root, allow_injected=client_factory is not runtime.RecordingClient
           or preflight is not suite.shared.vllm_preflight)
    return state


def verify(root, *, allow_injected=False):
    registration, instances = load(root)
    out = root / "run"
    for name, digest in suite.read_json(out / "seal.json").items():
        suite.require(suite.digest(out / name) == digest, "Run seal differs.")
    manifest = suite.read_json(out / "manifest.json")
    suite.require(allow_injected or not (
        manifest["injected_client"] or manifest["injected_preflight"]),
        "Injected execution is not real model evidence.")
    suite.require(manifest["model"] == registration["model"]
                  and manifest["registration_sha256"] == suite.digest(root / "registration.json"),
                  "Run registration identity differs.")
    attempts = suite.read_rows(out / "attempts.jsonl")
    suite.require(len(attempts) <= MAX_CALLS, "Call budget exceeded.")
    rows = canonical.replay(instances, attempts, configuration(registration["lane"]), native_action=False)
    suite.require(rows == suite.read_rows(out / "scores.jsonl"), "Score replay differs.")
    suite.require(suite.json_value(suite.summarize(rows)) == suite.read_json(out / "summary.json"),
                  "Summary replay differs.")
    terminal = suite.read_json(out / "terminal.json")
    suite.require(terminal["calls"] == len(attempts), "Terminal calls differ.")
    if terminal["status"].startswith("completed"):
        suite.require(all(r["feasible"] or r["failure_count"] for r in rows),
                      "Completed status conceals unattempted instances.")
    return {"verified": True, "status": terminal["status"], "episodes": len(rows),
            "calls": len(attempts), "model_calls": 0}


def preserve_unattempted(root, reason):
    registration, instances = load(root)
    output = root / "run"
    suite.require(not output.exists(), "Cannot replace an attempted run.")
    output.mkdir()
    config = configuration(registration["lane"])
    suite.write_json(output / "manifest.json", {
        "model": asdict(config), "registration_sha256": suite.digest(root / "registration.json"),
        "injected_client": False, "injected_preflight": False,
        "episodes": 960, "maximum_calls": MAX_CALLS, "execution_started": False})
    (output / "attempts.jsonl").touch(exist_ok=False)
    rows = canonical.replay(instances, [], config, native_action=False)
    base.invariance.write_rows(output / "scores.jsonl", rows)
    suite.write_json(output / "summary.json", suite.summarize(rows))
    state = {"status": "unattempted_due_coordinator_failure", "calls": 0,
             "reason": reason, "finished_at_utc": runtime.now()}
    suite.write_json(output / "terminal.json", state)
    suite.write_json(output / "seal.json", {
        name: suite.digest(output / name) for name in
        ("manifest.json", "attempts.jsonl", "scores.jsonl", "summary.json", "terminal.json")})
    runtime.status(root, **state)
    return state


async def cancellable(root):
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        return await run(root)
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("metadata", "download", "verify-artifacts", "run", "verify"))
    parser.add_argument("--lane", choices=tuple(SPECS))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--materials", type=Path, default=MATERIALS)
    args = parser.parse_args()
    if args.command in {"metadata", "download", "verify-artifacts"}:
        parser.error("--lane is required") if not args.lane else None
        result = (verify_artifacts(args.lane, args.materials) if args.command == "verify-artifacts"
                  else prepare_model(args.lane, args.materials, weights=args.command == "download"))
    elif args.command == "run":
        result = asyncio.run(cancellable(args.root))
    else:
        result = verify(args.root)
    print(json.dumps(result, sort_keys=True))
    return 2 if result.get("status") in {"aborted", "interrupted"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
