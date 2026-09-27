"""Render deterministic README figures from the public aggregate snapshot."""
import argparse
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = (
    "jev_action", "qwen4_grammar", "qwen9_grammar", "qwen27_grammar",
    "qwen38_27b_bf16", "qwen25_72b_bf16", "qwen4_token_scores",
    "decider", "kev", "laya", "gpt54_default_reasoning", "gpt6astra_default_reasoning",
)
LABELS = (
    "Jev", "Qwen3.5-4B grammar", "Qwen3.5-9B grammar", "Qwen3.5-27B grammar",
    "Qwen3.8-27B BF16", "Qwen2.5-72B BF16", "Qwen3.5-4B scores",
    "Decider-2B v11", "Kev-4B", "Laya", "GPT-5.4 (reasoning)", "Astra (reasoning)",
)
EXACT = ("adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
         "distance_threshold", "articulation_point")
CONSTRUCTION = ("maxcut_construct", "tsp_construct", "lt_influence_construct",
                "community_bipartition_construct")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(data):
    require(data["schemaVersion"] == 1 and data["version"] == "canonical-20260927-v14",
            "Unexpected public snapshot version")
    require(len(data["models"]) == 12
            and {m["id"] for m in data["models"]} == set(EXPECTED_MODELS),
            "Expected twelve configurations")
    require(len(data["tasks"]) == 10
            and {t["id"] for t in data["tasks"]} == set(EXACT + CONSTRUCTION),
            "Expected all ten tasks")
    index = {(r["model"], r["task"], r["nodeCount"]): r for r in data["metrics"]}
    expected = {(m, t, n) for m in EXPECTED_MODELS for t in EXACT + CONSTRUCTION
                for n in ("all", 8, 9, 10, 11, 12)}
    require(len(data["metrics"]) == 720 and set(index) == expected,
            "Missing or duplicate metric cells")
    for model in EXPECTED_MODELS:
        for task in EXACT + CONSTRUCTION:
            total = index[model, task, "all"]
            cells = [index[model, task, n] for n in range(8, 13)]
            for row in [total, *cells]:
                fields = ("scheduled", "feasible", "hits", "unsupported", "failures", "unattempted")
                require(all(type(row[f]) is int and row[f] >= 0 for f in fields),
                        "Invalid coverage counts")
                require(row["scheduled"] > 0 and 0 <= row["hits"] <= row["feasible"]
                        and row["feasible"] + row["unsupported"] + row["failures"]
                        + row["unattempted"] == row["scheduled"], "Invalid coverage partition")
                if task in CONSTRUCTION:
                    gap = row["meanGap"]
                    require(gap is None if not row["feasible"] else
                            type(gap) in (float, int) and math.isfinite(gap) and gap >= 0,
                            "Missing or invalid feasible gap")
            for field in fields:
                require(total[field] == sum(c[field] for c in cells), "Aggregate count mismatch")
            if task in EXACT:
                macro = math.fsum(c["hits"] / c["scheduled"] for c in cells) / 5
                require(math.isclose(total["accuracy"], macro, abs_tol=1e-12, rel_tol=0),
                        "Exact macro accuracy mismatch")
            elif total["feasible"]:
                mean = math.fsum(c["meanGap"] * c["feasible"] for c in cells
                                 if c["feasible"]) / total["feasible"]
                require(math.isclose(total["meanGap"], mean, abs_tol=1e-12, rel_tol=0),
                        "Feasible gap weighting mismatch")
    return index


def color(value):
    value = min(1, max(0, value))
    start, end = (246, 247, 242), (143, 199, 183)
    return "#" + "".join(f"{round(a + value * (b - a)):02x}" for a, b in zip(start, end))


def figure(index, tasks, headings, exact):
    width, row_height, left, cell = 1120, 48, 270, (1120 - 294) / len(tasks)
    height = 790
    title = "Exact graph decisions" if exact else "Sequential construction: mean objective gap"
    note = ("Macro-size accuracy (%), higher is better. Counts are shared across models."
            if exact else "Lower is better; absolute task units. Color scales are separate per column.")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(note)} '
        'Twelve configurations on 960 shared cases. GPT uses a separate reasoning budget. '
        'Exact numeric tables are available in RESULTS.md.</desc>',
        '<rect width="1120" height="790" rx="12" fill="#faf9f6"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#172c35">',
        f'<text x="24" y="34" font-size="24" font-weight="bold">{escape(title)}</text>',
        f'<text x="24" y="60" font-size="14">{escape(note)}</text>',
    ]
    for j, header in enumerate(headings):
        parts.append(f'<text x="{left + cell * (j + .5):.1f}" y="93" text-anchor="middle" '
                     f'font-size="14" font-weight="bold">{escape(header)}</text>')
    maxima = [max(index[m, task, "all"]["meanGap"] or 0 for m in EXPECTED_MODELS)
              if not exact else 1 for task in tasks]
    for i, (model, label) in enumerate(zip(EXPECTED_MODELS, LABELS)):
        y = 110 + i * row_height + (30 if i >= 10 else 0)
        if i == 10:
            parts.append('<text x="24" y="608" font-size="13" font-weight="bold">'
                         'SEPARATE GPT REFERENCES — default reasoning, not matched compute</text>')
        parts.append(f'<text x="24" y="{y + 25}" font-size="15" '
                     f'font-weight="{"bold" if i == 0 else "normal"}">{escape(label)}</text>')
        for j, task in enumerate(tasks):
            row = index[model, task, "all"]
            value = row["accuracy"] if exact else row["meanGap"]
            quality = value if exact else (1 - value / maxima[j] if value is not None and maxima[j] else 0)
            text = (f"{100 * value:.2f}%" if exact else
                    "N/A" if value is None else
                    f"{value:.4f}" if task == "community_bipartition_construct" else f"{value:.3f}")
            x = left + cell * j
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{cell - 6:.1f}" height="43" '
                         f'rx="4" fill="{color(quality)}"/>')
            parts.append(f'<text x="{x + (cell - 6) / 2:.1f}" y="{y + (25 if exact else 19)}" '
                         f'text-anchor="middle" font-size="16">{text}</text>')
            if not exact:
                parts.append(f'<text x="{x + (cell - 6) / 2:.1f}" y="{y + 35}" '
                             f'text-anchor="middle" font-size="11">'
                             f'{row["feasible"]}/{row["scheduled"]} feasible</text>')
    foot = ("Binary chance reference: 50%; degree is multiclass. All-size degree is macro, not pooled."
            if exact else "Units: edges / Manhattan distance / activated nodes / modularity. No pooled OR score.")
    parts.extend([
        f'<text x="24" y="746" font-size="13">{escape(foot)}</text>',
        '<text x="24" y="770" font-size="13">GraphDecide · canonical inspected bank · '
        'n = 8–12 · point estimates, not significance claims</text>', '</g></svg>\n',
    ])
    return "\n".join(parts)


def render(data, source_digest):
    index = validate(data)
    lines = [
        "# GraphDecide results", "",
        "Twelve configurations, ten tasks, 960 shared instances. "
        "These are inspected-bank point estimates, not a universal ranking.", "",
        "## Exact decisions", "",
        "![Exact-task macro-size accuracy](figures/exact-accuracy.svg)", "",
        "| Configuration | Adjacency | Degree | Cycle | Connectivity | Distance | Articulation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in zip(EXPECTED_MODELS, LABELS):
        lines.append("| " + label + " | " + " | ".join(
            f'{100 * index[model, task, "all"]["accuracy"]:.2f}%'
            for task in EXACT) + " |")
    lines += ["", "Accuracy averages the five size-specific rates equally. "
              "Degree has 400 queries; other exact tasks have 80 each. "
              "The two GPT rows use different reasoning budgets.", "",
              "## Construction gaps", "",
              "![Mean additive gap with feasible coverage](figures/construction-gaps.svg)", "",
              "| Configuration | MaxCut (edges) | TSP (distance) | LT (nodes) | Modularity |",
              "|---|---:|---:|---:|---:|"]
    for model, label in zip(EXPECTED_MODELS, LABELS):
        values = []
        for task in CONSTRUCTION:
            row = index[model, task, "all"]
            gap = "N/A" if row["meanGap"] is None else f'{row["meanGap"]:.4f}'
            values.append(f'{gap} ({row["feasible"]}/{row["scheduled"]})')
        lines.append("| " + label + " | " + " | ".join(values) + " |")
    lines += [
        "", "Each cell is **mean gap (feasible/scheduled)**. Gap = solution minus optimum "
        "for TSP, optimum minus solution otherwise. Means condition on feasible completion; "
        "units differ and must not be pooled. SVG gap colors are scaled independently per column.", "",
        "## Interpretation and scope", "",
        "- Jev's perfect adjacency does not imply reliable counting or structural reasoning.",
        "- Jev has smaller construction gaps than both added Qwen3.8-27B and Qwen2.5-72B "
        "configurations. Qwen3.5-27B still has smaller MaxCut and LT gaps than Jev; "
        "Kev ties Jev on LT at 0.600.",
        "- Qwen2.5-72B is older-generation, not a scale-only ablation. All Qwen grammar "
        "rows use bounded direct answers, not full reasoning capacity.",
        "- Laya retains nine unsupported LT inputs; GPT-5.4 retains nine construction timeouts. "
        "Zero conditional gap is not zero failure rate.",
        "- Only the original four configurations received prospective new-graph confirmation. "
        "The twelve-model rankings have no full fresh-graph replication.",
        "- Original classical references achieve MaxCut gap 0.450 and TSP gap 0.200; "
        "uniform legal MaxCut has expected gap 5.325 versus Jev's 8.150. "
        "Neural relative advantages do not establish solver competitiveness. "
        "These references are not matched-budget online models.", "",
        "## Source", "",
        "[Public JSON](../web/data/benchmark.json) · [CSV](../web/data/benchmark.csv)", "",
        f"JSON SHA-256: `{source_digest}`. Figures and tables are generated without inference "
        "by `python scripts/render_readme_results.py`; use `--check` to detect stale files.", "",
    ]
    return {
        "docs/RESULTS.md": "\n".join(lines),
        "docs/figures/exact-accuracy.svg": figure(index, EXACT,
            ("Adjacency", "Degree", "Cycle", "Connectivity", "Distance", "Articulation"), True),
        "docs/figures/construction-gaps.svg": figure(index, CONSTRUCTION,
            ("MaxCut (edges)", "TSP (distance)", "LT (nodes)", "Modularity"), False),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = ROOT / "web/data/benchmark.json"
    outputs = render(json.loads(source.read_text()), sha256(source.read_bytes()).hexdigest())
    for name, content in outputs.items():
        path = ROOT / name
        if args.check:
            require(path.is_file() and path.read_text() == content, f"Stale README result: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f"{'Checked' if args.check else 'Rendered'} {len(outputs)} README assets; no inference.")


if __name__ == "__main__":
    main()
