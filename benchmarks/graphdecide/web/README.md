# GraphDecide static result explorer

From `benchmarks/graphdecide/`:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory web
```

Open <http://localhost:8765/>; forward port 8765 if using Remote SSH.
No API calls, npm build, backend, analytics, external fonts or credentials
are needed. Serve over HTTP, not `file://`, for module and JSON loading.

The page contains all twelve configurations and 720 task/size metric cells.
It supports exact/construction tabs, size/family/model filters, task sorting,
model details, coverage and descriptive size profiles. GPT references remain
separately grouped. Gap is the primary construction metric; optimum hits are
supplementary. Missing quality is never fabricated as zero.

- [Paper](assets/graphdecide-v17.pdf): eight-page main text plus references
  and appendices; SHA-256
  `97f6ae0998a20aecac3e37b8c1cfb441b4019c96f2dbbdd4b8866a2f3cf7bf2d`.
- [JSON](data/benchmark.json): public aggregate metrics and safe source hashes.
- [CSV](data/benchmark.csv): identical metrics in tabular form.
- [README visualization](../docs/RESULTS.md): static figures and accessible tables.

The public JSON retains its sealed v14 identity and original internal title.
GraphDecide is the display name; subsequent PDF revisions are editorial.
This is an inspected-bank presentation, not a full raw-evidence replay package
or proof of third-party replication.

Test from `benchmarks/graphdecide/`:

```bash
node --test tests/test_web.mjs
```

The optional `tests/test_web_browser.cjs` requires an existing Playwright
installation and local server. It checks desktop and 320/390px layouts,
filters, references, error states, project subpaths and the PDF checksum.
Playwright is not a website dependency.

All local URLs are relative. This branch does not activate a GitHub Pages
deployment or modify HeurAgenix workflows. If separately configuring hosting,
publish only this `web/` directory, never local experiment outputs.
