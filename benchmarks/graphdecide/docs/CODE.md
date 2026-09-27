# GraphDecide public code guide

## Scope and working directory

Run every command below from **`benchmarks/graphdecide/`**, not from the
HeurAgenix repository root. Both projects use a Python package named `src`;
running from the wrong directory can import the other project's package.
Use a dedicated environment rather than installing both projects together.

This directory contains the task generators, exact evaluators, model-client
adapters, configuration templates, and historical experiment scripts.
The internal package/distribution name `jevgraphbench` is retained for
compatibility. Published result summaries are not a promise that every original
raw response, model checkpoint, or all-twelve-model replay artifact is included.
The original experiment source and frozen experiment outputs are not rewritten
by this source release.

## Installation

Python 3.11 or newer is required:

```bash
cd benchmarks/graphdecide
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Core dependencies are NetworkX, HTTPX, PyYAML, and tqdm; see `pyproject.toml`
for supported ranges. The offline tests use standard-library `unittest`.
No model weights, GPU runtime, or API credential is needed for the offline
examples. Installing dependencies is separate from executing these examples.

## Generate tasks and construct a request offline

```bash
python - <<'PY'
from dataclasses import asdict
from src.benchmark.structural_tasks import build_structural_bank
from src.benchmark.extended_tasks import next_request, score, validate_instances
from scripts.graph_task_prompt import canonical_graph_request

instances, provenance = build_structural_bank(
    seed=20260930, node_counts=(8,), repetitions=1,
)
print(validate_instances(instances))
instance = next(row for row in instances if row["task"] == "pair_connectivity")
request = canonical_graph_request(instance["task"], next_request(instance, []))
print(asdict(request))
# An illustrative supplied action, not a model prediction or benchmark result:
print(score(instance, [request.options[0].id]))
PY
```

Generation computes exact reference answers and reports realized coverage.
Infeasible generation quotas remain explicit; do not equate requested counts
with generated counts. Internal instance records contain private references:
send only the constructed request to a model, never the entire record or
generation provenance. Sequential construction tasks require repeated
`next_request(instance, history)` calls and scoring the complete action history.

For a durable, model-free experiment plan:

```bash
python scripts/extended_graph_suite.py plan \
  --root output/my-offline-plan --design structural_dev \
  --node-counts 8 --samples-per-size 1 --models jev
```

`plan` does not call the named model. It writes a new plan and code snapshot;
use a new output directory for each plan. The `run` subcommand is different:
it performs live model evaluation and must not be used as an offline smoke test.

## Portable data bundles and scoring

`scripts/build_graph_release.py` builds deterministic bundles from previously
generated, verified experiment plans. Its command-line interface takes
development, challenge, and TSP-confirmation plans; inspect `--help` for paths.
It does not manufacture the historical frozen splits from result summaries.
For a bundle you have built or separately obtained:

```bash
python scripts/graph_release_api.py verify --root PATH_TO_BUNDLE
python scripts/graph_release_api.py request --root PATH_TO_BUNDLE \
  --split development --id INSTANCE_ID --history '[]'
python scripts/graph_release_api.py score --root PATH_TO_BUNDLE \
  --split development --id INSTANCE_ID --history '["yes"]'
```

Replace the ID and action history with values belonging to that bundle.
Request construction reads public records only; scoring reads references.
Bundle hashes verify integrity, not authenticity. `explicit_actions` is the
recommended prompt protocol for new comparisons; historical protocols remain
available so old records can be interpreted without silently changing them.
`scripts/evaluate_graph_release.py` is a **live** model runner, not the offline
scorer. Its tests use fake clients.

## Model clients and real-network templates

- `src/clients/base.py` defines requests, options, responses, and typed failures.
- `src/clients/typesafe.py` reads `TYPESAFE_API_KEY` from the environment.
  Never put a real key into a YAML file or source code.
- `src/clients/vllm_client.py` uses an already running compatible model server.
  Server installation, weights, and serving resources are not bundled.
- `src/clients/github_copilot_client.py` requires the optional
  `python -m pip install -e '.[copilot]'` dependency and a separately configured,
  authenticated Copilot runtime. Installing the SDK alone is not authentication.

The YAML files in `configs/` are **live, paid-provider examples** for foundational
adjacency/distance tasks on separately downloaded real networks. They are not
the synthetic ten-task structural suite. `python run_benchmark.py --help` is
safe offline; running it with a configuration performs evaluation.

`data/real/` includes only a provenance README and checksum manifest. No raw
dataset archives are redistributed. Read its README and upstream terms before
using `python -m src.datasets.download`; downloading is an explicit network
operation. Datasets do not inherit this repository's code license.

## Portable offline validation

The following focused suite exercises task generation, exact scoring, dataset
loading with fixtures, client contracts with mock transports, the CLI, request
construction, and deterministic portable-bundle export. Temporary test files
stay inside this checkout:

```bash
mkdir -p .test-work
TMPDIR="$PWD/.test-work" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/tests:$PWD" python -m unittest -q \
  test_benchmark test_clients test_llm_clients test_datasets test_entrypoint \
  test_extended_tasks test_foundational_tasks test_structural_tasks \
  test_structural_integration test_extended_graph_suite test_graph_release \
  test_evaluate_graph_release test_sampling_controls test_response_diagnostics \
  test_paper_deadline test_paper_full_matrix test_paper_pilot_table \
  test_paper_laya test_paper_thinking_pilot test_paper_robustness \
  test_paper_legacy_astra test_paper_finalizer
rmdir .test-work
```

The curated checkout passed **263 tests** with this selection (232 core tests
and 31 paper-builder unit tests using synthetic fixtures). An optional SDK
signature test may be skipped when the Copilot extra is absent. No inference,
dataset download, or GPU change is part of this test command.
The documented generation example and plan command also produced and verified
22 instances across all ten tasks; a bundle built from that plan passed the
offline bundle verifier with zero model calls.

## Historical scripts and integration evidence

The complete Python orchestration and test modules are retained because they
import one another. They are **not** all fresh-clone runnable experiments:

- `full_matrix_*`, `stronger_local_*`, trained-model, Laya, and SEMIF scripts
  can require experiment-specific paths, frozen `output/` plans, local ports,
  Linux process controls, GPUs, external model checkouts, or model weights.
  Review and configure these prerequisites before executing them.
- Optional advanced imports include PyTorch, Transformers, Hugging Face Hub,
  psutil, ZeroMQ, and separately obtained `decider`, `kev`, `laya`, and
  `semif_phase1` packages. They are not core installation requirements.
  `scripts/trained_jevlike_requirements.txt` is a historical environment overlay,
  **not** a complete standalone model environment or permission to replace
  an existing PyTorch/GPU installation.
- Exporters and `paper/*.py` builders require their corresponding verified
  evidence files. Website/paper/evidence tests and some orchestration tests
  refer to unbundled historical outputs. Full test discovery is therefore not
  the portable smoke-test contract; missing evidence is not a passing replay.
- No external model implementation, checkpoint, raw provider response, local
  credential file, cache, environment, or raw network archive was copied as
  part of the code curation. Host-path defaults in historical scripts describe
  setup assumptions, not portable paths.

All `src/**/*.py`, `scripts/**/*.py`, and `tests/**/*.py` files were selected,
along with three YAML templates, the two root Python/build entrypoint files,
the historical requirements overlay, and the two dataset provenance files.
At curation time this was 21 core Python modules, 79 original script modules,
88 Python test modules, and `run_benchmark.py`. Nine historical paper builders
and their package initializer are additionally included under `paper/`
(199 Python files in total, all syntax-checked). The original builder locations
and imports are preserved; the original frozen source was not changed.
Non-Python browser tests, generated outputs, caches, and local environments
were excluded from this code inventory. Paper, website, figures, and public
result summaries are curated separately.

The credential-pattern inventory found only mocked test credentials,
deliberately invalid test URLs, and one manifest identifier matched by an
overbroad token pattern. Findings were inspected without emitting credential
values. The YAML templates contain provider identifiers and environment-key
instructions, not embedded API keys. This targeted inventory is not a
comprehensive security certification.
