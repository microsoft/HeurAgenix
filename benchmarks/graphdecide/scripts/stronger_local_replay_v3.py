#!/usr/bin/env python3
"""Standalone bundled replay with pinned neutral contracts and blocked providers."""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.abc
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scripts import stronger_local_replay_v1 as core

CONTRACTS = {"src.clients": "src/clients/__init__.py",
             "src.clients.base": "src/clients/base.py"}
PROVIDER_ROOTS = (
    "vllm", "transformers", "httpx", "httpcore", "requests", "openai",
    "anthropic", "huggingface_hub", "github_copilot", "copilot",
)


def is_provider(name):
    return ((name == "src.clients" or name.startswith("src.clients."))
            and name not in CONTRACTS) or any(
                name == root or name.startswith(root + ".") for root in PROVIDER_ROOTS)


class NoProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if is_provider(fullname):
            raise ImportError(f"Provider import prohibited during independent replay: {fullname}")
        return None


def validate_python_files(bundle):
    manifest = core.read(bundle / "manifest.json")
    core.require(core.digest(bundle / "manifest.json") ==
                 core.read(bundle / "manifest.sha256"), "Bundle manifest seal differs.")
    for name, expected in manifest["sha256"].items():
        if not name.endswith(".py"):
            continue
        path = bundle / name
        core.require(not Path(name).is_absolute() and ".." not in Path(name).parts
                     and not path.is_symlink() and path.resolve().is_relative_to(bundle)
                     and core.digest(path) == expected, "Bundled Python source differs.")
    core.require(set(CONTRACTS.values()) <= set(manifest["sha256"]),
                 "Neutral contracts are absent from the sealed bundle.")
    return manifest


def import_audit(bundle, manifest, modules=None):
    modules = sys.modules if modules is None else modules
    providers = sorted(name for name in modules if is_provider(name))
    core.require(not providers, f"Independent replay imported a provider: {providers}")
    contracts = {}
    for name, relative in CONTRACTS.items():
        core.require(name in modules, "Bundled neutral contract module is missing.")
        path = Path(modules[name].__file__).resolve()
        core.require(path == bundle / relative and core.digest(path) == manifest["sha256"][relative],
                     "Neutral contracts came from outside the sealed bundle.")
        contracts[name] = {"path": str(path), "sha256": manifest["sha256"][relative],
                           "role": "provider-neutral decision contracts, not a backend"}
    return {"provider_modules_imported": providers, "bundled_contract_modules": contracts,
            "src_clients_namespace_entries": sorted(
                name for name in modules if name == "src.clients" or name.startswith("src.clients.")),
            "provider_import_blocker_active": True}


def verify(root, bundle):
    registration = core.read(root / "registration.json")
    core.require(core.digest(root / "registration.json") ==
                 core.read(root / "registration.sha256"), "Registration hash differs.")
    core.require(str(bundle) == registration["bundle"]
                 and core.digest(bundle / "manifest.json") == registration["bundle_manifest_sha256"],
                 "Bundle identity differs.")
    core.require(core.digest(Path(core.__file__)) ==
                 registration["source_sha256"]["scripts/stronger_local_replay_v1.py"],
                 "Frozen independent score reconstruction source differs.")
    manifest = validate_python_files(bundle)
    core.require(not any(is_provider(name) for name in sys.modules),
                 "Provider was already imported before isolated replay.")
    guard = NoProviders()
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, str(bundle))
    try:
        api = importlib.import_module("release_api")
        core.require(Path(api.__file__).resolve() == bundle / "release_api.py"
                     and Path(api.tasks.__file__).resolve() ==
                     bundle / "src/benchmark/extended_tasks.py", "Ambient scorer imports forbidden.")
        api.verify(bundle)
        instances = api.labeled_instances(bundle, "structural_challenge", api.manifest_at(bundle))
        out = root / "run"
        for name, expected in core.read(out / "seal.json").items():
            core.require(core.digest(out / name) == expected, "Run seal differs.")
        run_manifest = core.read(out / "manifest.json")
        core.require(run_manifest["model"] == registration["model"]
                     and run_manifest["registration_sha256"] == core.digest(root / "registration.json"),
                     "Run/registration identity differs.")
        core.require(not run_manifest["injected_client"] and not run_manifest["injected_preflight"],
                     "Injected execution cannot be real model evidence.")
        result = core.replay(
            instances, core.rows(out / "attempts.jsonl"), core.rows(out / "scores.jsonl"),
            api, registration["model"]["model"])
        terminal = core.read(out / "terminal.json")
        core.require(terminal["calls"] == result["calls"], "Terminal call count differs.")
        if terminal["status"].startswith("completed"):
            core.require(all(cell["feasible"] + cell["failed_episodes"] == cell["scheduled"]
                             for cell in result["by_task"].values()),
                         "Completed status hides incomplete/unattempted episodes.")
        result.update(import_audit(bundle, manifest))
        result.update(run_status=terminal["status"], input_root=str(root),
                      input_registration_sha256=core.digest(root / "registration.json"),
                      input_run_seal_sha256=core.digest(out / "seal.json"),
                      replay_version=3, input_artifacts_modified=False,
                      replay_source_sha256={
                          "scripts/stronger_local_replay_v1.py": core.digest(Path(core.__file__)),
                          "scripts/stronger_local_replay_v3.py": core.digest(Path(__file__))})
        return result
    finally:
        sys.meta_path.remove(guard)


def write_report(output, result):
    core.require(not output.exists(), "Recovered output exists; no overwrite.")
    output.mkdir(parents=True)
    with (output / "report.json").open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    fields = ("task", "scheduled", "feasible", "feasible_coverage", "unattempted",
              "failed_episodes", "mean_absolute_gap_feasible", "gap_denominator",
              "optimum_count_supplemental")
    with (output / "construction-gaps.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for task, metric in result["by_task"].items():
            if "mean_absolute_gap_feasible" in metric:
                writer.writerow({"task": task, **metric})
    with (output / "seal.json").open("x") as stream:
        json.dump({name: core.digest(output / name)
                   for name in ("report.json", "construction-gaps.csv")}, stream, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, bundle, output = args.root.resolve(), args.bundle.resolve(), args.output.resolve()
    core.require(output.is_relative_to(REPO) and output != root and not output.exists(),
                 "Use a new, separate project-local recovered-output directory.")
    result = verify(root, bundle)
    write_report(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
