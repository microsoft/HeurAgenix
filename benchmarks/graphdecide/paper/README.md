# GraphDecide manuscript

**GraphDecide: Benchmarking Decision Models on Graphs**

Xianliang Yang (Microsoft), Yapu Zhang (Beijing University of Technology),
and Li Zhao (Microsoft; corresponding author).

[Current PDF](../web/assets/graphdecide-v17.pdf): 8 pages of main text,
references on pages 9--11, and appendices from page 12. This is a working
manuscript, not a claim of venue acceptance or third-party replication.

## Rebuild without inference

Run from this directory:

```bash
latexmk -g -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Requires TeX Live with `pgfplots`, `booktabs`, `amsmath`, `hyperref`,
`geometry`, `microtype`, `natbib` and `lmodern`. The output is `build/main.pdf`.
No Python, credentials, model servers or local experiment ledgers are needed
to compile the included sources.

- `main.tex`: document assembly, authors and appendix ordering.
- `main-text.tex`: eight-page main narrative and twelve-model result tables.
- `protocol-details.tex`: full contracts, generation, model settings and audits.
- `related_work.tex`, `capability-analysis.tex`: extended appendix discussions.
- `original-results.tex`, `confirmation-details.tex`, `interface-controls.tex`:
  preserved historical evidence.
- `generated*/`: precomputed LaTeX tables and figures; do not hand-edit scores.

Code is released at <https://github.com/microsoft/HeurAgenix.git>, branch
`graph_benchmark`, directory `benchmarks/graphdecide`.

The public JSON and CSV in `../web/data/` contain the canonical twelve-model
aggregates and safe provenance. Full raw ledgers, local source manifests,
provider response bodies and weights are not included. Rebuilding this PDF
is distinct from rerunning or independently reproducing all experiments.
