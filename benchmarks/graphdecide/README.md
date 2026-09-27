# GraphDecide

## Benchmarking Decision Models on Graphs

- **Xianliang Yang** — Microsoft — [xianya@microsoft.com](mailto:xianya@microsoft.com)
- **Yapu Zhang** — Beijing University of Technology — [zhangyapu@bjut.edu.cn](mailto:zhangyapu@bjut.edu.cn)
- **Li Zhao** — Microsoft, corresponding author — [lizo@microsoft.com](mailto:lizo@microsoft.com)

GraphDecide evaluates whether valid local decisions become correct graph
answers and good complete solutions. It covers six exact tasks and four
irreversible construction tasks on small, exactly verifiable graphs.

**12 configurations · 10 tasks · 960 shared canonical instances · sizes 8–12**

[Paper: 8-page main text + appendices](web/assets/graphdecide-v17.pdf) ·
[All numeric results](docs/RESULTS.md) · [CSV](web/data/benchmark.csv) ·
[JSON and provenance](web/data/benchmark.json) · [Code guide](docs/CODE.md)

This directory is self-contained. The original HeurAgenix framework at the
repository root is unchanged; run GraphDecide commands **from this directory**
to avoid mixing the two projects' `src` packages.

## Results

### Exact graph decisions

![Exact-task macro-size accuracy for all twelve configurations](docs/figures/exact-accuracy.svg)

Accuracy is macro-averaged across graph sizes. In particular, degree accuracy
is **not** the pooled correct count divided by 400. Binary chance is 50%;
degree is multiclass. The lower GPT block uses default reasoning and is not
a matched-compute comparison.

### Sequential optimization

![Mean additive construction gaps and feasible coverage](docs/figures/construction-gaps.svg)

**Lower gap is better.** TSP uses solution minus optimum; maximization tasks
use optimum minus solution. Each gap averages feasible completed episodes,
with feasible/scheduled coverage shown underneath. Each column has its own
units and color scale; there is no blended optimization score. Optimum-hit
counts are supplementary, not the primary quality measure.

### What the results support

- Jev's 100% adjacency accuracy does not extend to uniformly reliable
  counting or structural reasoning.
- Jev has smaller mean gaps than the added Qwen3.8-27B and Qwen2.5-72B
  configurations on all four constructions. Qwen3.5-27B still has smaller
  MaxCut and LT gaps than Jev; Kev ties Jev's LT gap of 0.600.
- Larger/newer is not monotonically better under the tested bounded
  direct-answer protocol. The older-generation 72B is not a scale-only
  ablation, and these results do not establish general parameter equivalence.
- GPT-5.4's nine timeouts and Laya's nine unsupported LT inputs remain
  in scheduled denominators. Zero conditional gap does not mean zero failures.
- Relative neural advantages do not imply solver competitiveness. Classical
  reference gaps are 0.450 for MaxCut and 0.200 for TSP, versus Jev's
  8.150 and 45.850; those algorithms are not equal-budget online competitors.

Only the original four configurations received prospective new-graph
confirmation (952 realized cases). The twelve-configuration matrix uses an
already inspected bank, not a new blind holdout. Rankings and size profiles
are descriptive; no family-wise significance or fitted failure threshold is
claimed. Jev-like means open decision-oriented readouts, not verified
reconstructions of Jev's private architecture.

## Task inventory

| Track | Tasks | Primary measure |
|---|---|---|
| One-step exact decisions | Adjacency, degree, cycle, connectivity, distance threshold, articulation | Size-macro accuracy |
| Full own-history constructions | MaxCut, Manhattan TSP, deterministic normalized LT influence, at-most-two-community modularity | Mean additive gap + feasible coverage |

Every model sees the public state and legal actions, never the reference
answer or optimum. Deterministic transitions apply actual model choices;
there is no oracle-prefix substitution or answer repair. Graph sizes are
small to enable exact labels, objectives and continuation checks.

## Quick start

```bash
git clone --branch graph_benchmark https://github.com/microsoft/HeurAgenix.git
cd HeurAgenix/benchmarks/graphdecide
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Requires Python 3.11+. The original distribution/import names (`jevgraphbench`,
`src`) and frozen experiment identifiers are preserved for compatibility.
This release does not install models, start GPU servers or call paid APIs.

### View the interactive results

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory web
```

Open <http://localhost:8765/>. In a remote VS Code workspace, forward port
8765 first. The website is static, requires no credentials or model services,
and works under a project subpath. This branch does not activate GitHub Pages.

### Regenerate README visualizations

```bash
python scripts/render_readme_results.py
python scripts/render_readme_results.py --check
```

The SVGs and [numeric tables](docs/RESULTS.md) are generated from the included
public JSON, not manually entered scores. Generation makes no inference calls.
The JSON retains its sealed `canonical-20260927-v14` identity; later paper
versions are editorial revisions, not new experiments.

### Test and build

```bash
PYTHONPATH=tests python -m unittest test_extended_tasks test_paper_structure test_readme_results
node --test tests/test_web.mjs
cd paper
latexmk -g -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Node is needed only for frontend tests; TeX Live is needed only to rebuild
the PDF. [The code guide](docs/CODE.md) separates portable tests from
historical evidence-dependent checks and documents live execution boundaries.
Configuration templates use environment-based credentials: do not commit
keys or repurpose historical local experiment settings without reviewing them.

## Layout and release boundaries

| Path | Purpose |
|---|---|
| [`src/`](src/) | Task definitions, graph data helpers and model clients |
| [`configs/`](configs/) | Original real-network task configuration templates |
| [`scripts/`](scripts/) | Synthetic suite, oracles, diagnostics, verifiers and historical coordinators |
| [`tests/`](tests/) | Offline regression tests and explicitly evidence-dependent checks |
| [`paper/`](paper/) | LaTeX sources, generated tables and bibliography |
| [`web/`](web/) | Static result explorer, public aggregates and current PDF |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Accessible numerical tables and interpretation |

**Included:** project source code, configuration templates, tests, public
aggregate results, deterministic README figures and rebuildable manuscript.

**Not included:** model weights, API keys, raw provider responses, hidden
reasoning, downloaded real-network archives or the full local execution
ledger/replay bundles. Historical exporters require their corresponding
sealed local evidence; a missing `output/` directory is not a reason to rerun
paid experiments. The PDF and public aggregates can be used without it.
This code release is not a full independently redistributable twelve-model
raw-trace release or third-party reproduction.

The repository's [MIT license](../../LICENSE) and [security policy](../../SECURITY.md)
apply as provided by HeurAgenix. Upstream model/data licenses still apply;
no third-party model weights are redistributed.
