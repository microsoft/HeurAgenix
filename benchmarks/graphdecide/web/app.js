import { validateData, metricKey, visibleModels, valueFor, formatMetric, coverageFor } from "./model.mjs";

const $ = selector => document.querySelector(selector);
const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const state = {
  kind: "exact", family: "all", size: "all", metric: "meanGap", anchors: false,
  search: "", sortTask: "adjacency", ascending: false, selectedModel: "jev_action",
  chartTask: "adjacency", chartModel: "qwen25_72b_bf16",
};
const familyClass = { Jev: "family-jev", "Jev-like": "family-like", Qwen: "family-qwen", GPT: "family-gpt" };
const taskLabels = {
  lt_influence_construct: "Deterministic LT",
  community_bipartition_construct: "Two-group modularity",
};
let data;
let index;
const rowFor = (model, task, size = state.size) => index.get(metricKey(model, task, size));
const currentTasks = () => data.tasks.filter(task => task.kind === state.kind);
const pill = model => element("span", `family-pill ${familyClass[model.family]}`, model.family);
const svgNS = "http://www.w3.org/2000/svg";

function renderTable() {
  const tasks = currentTasks();
  const head = element("tr");
  const modelHeader = element("th", "", "Model / configuration");
  modelHeader.scope = "col";
  head.append(modelHeader);
  for (const task of tasks) {
    const th = element("th");
    th.scope = "col";
    th.setAttribute("aria-sort", task.id === state.sortTask ? (state.ascending ? "ascending" : "descending") : "none");
    const button = element("button", "", taskLabels[task.id] || task.name);
    button.type = "button";
    button.title = `${task.description} Click to sort within model groups.`;
    button.append(element("span", "", task.id === state.sortTask ? (state.ascending ? "↑" : "↓") : "↕"));
    button.addEventListener("click", () => {
      state.ascending = state.sortTask === task.id ? !state.ascending : state.kind === "construction" && state.metric === "meanGap";
      state.sortTask = task.id;
      state.chartTask = task.id;
      $("#chart-task").value = task.id;
      renderTable();
      renderChart();
    });
    th.append(button);
    head.append(th);
  }
  $("#results-table thead").replaceChildren(head);
  const groups = visibleModels(data, state, index);
  const body = $("#results-table tbody");
  body.replaceChildren();
  for (const [group, models] of Object.entries(groups)) {
    if (group === "anchors" && models.length) {
      const tr = element("tr", "anchor-group");
      const th = element("th", "", "UPPER REFERENCES · Default reasoning · Separate inference budget");
      th.colSpan = tasks.length + 1;
      th.scope = "rowgroup";
      tr.append(th);
      body.append(tr);
    }
    for (const model of models) {
      const tr = element("tr", model.id === "jev_action" ? "reference-row" : "");
      tr.dataset.model = model.id;
      const nameCell = element("td");
      const button = element("button", "model-name", model.name);
      button.type = "button";
      button.setAttribute("aria-label", `Inspect ${model.name} configuration`);
      button.setAttribute("aria-pressed", String(state.selectedModel === model.id));
      button.addEventListener("click", () => {
        state.selectedModel = model.id;
        renderModel();
        renderTable();
      });
      const meta = element("span", "model-meta");
      meta.append(pill(model), element("span", "", model.parameters));
      nameCell.append(button, meta);
      tr.append(nameCell);
      for (const task of tasks) {
        const row = rowFor(model.id, task.id);
        const td = element("td");
        const value = valueFor(row, state.kind, state.metric);
        td.append(element("span", "metric-value", formatMetric(row, task, state.metric)));
        const sub = task.kind === "exact"
          ? `${row.hits}/${row.scheduled} ${task.id === "degree_exact" && state.size === "all" ? "pooled" : "correct"}`
          : state.metric === "optimalRate"
            ? `${(100 * row.optimalRate).toFixed(1)}% of scheduled`
            : `${row.feasible}/${row.scheduled} feasible`;
        td.append(element("span", "metric-sub", sub));
        td.title = `${task.description}\n${row.scheduled} scheduled; ${row.feasible} feasible; ${row.unsupported} unsupported; ${row.failures} execution failures.`;
        if (state.kind === "exact" || state.metric === "optimalRate") {
          const meter = element("span", "score-meter");
          meter.setAttribute("aria-hidden", "true");
          const fill = element("span");
          fill.style.width = `${100 * value}%`;
          meter.append(fill);
          td.append(meter);
        }
        tr.append(td);
      }
      body.append(tr);
    }
  }
  if (!groups.decision.length && !groups.anchors.length) {
    const tr = element("tr", "empty-state");
    const td = element("td", "", "No models match these filters. Clear the search or choose another family.");
    td.colSpan = tasks.length + 1;
    tr.append(td);
    body.append(tr);
  }
  $("#result-count").textContent = `${groups.decision.length} decision configurations · ${groups.anchors.length} upper references · ${state.size === "all" ? "all five sizes" : `${state.size} vertices`}`;
  const metric = state.kind === "exact" ? "Accuracy (%) · higher is better"
    : state.metric === "meanGap" ? "Primary: mean absolute gap · lower is better · feasible completions only"
      : "Supplementary: optimum hits / scheduled · higher is better";
  $("#metric-note").textContent = metric;
  $("#table-caption").textContent = `${state.kind === "exact" ? "Exact graph decisions" : "Sequential constructions"} — ${metric}. ${state.size === "all" ? "All sizes 8–12" : `${state.size} vertices`}.`;
  $("#denominator-note").textContent = state.kind === "exact"
    ? "All-sizes accuracy is macro-averaged across graph sizes. Degree counts below each percentage are pooled, not the macro estimator. Binary chance reference: 50%; degree is multiclass. Point estimates, not significance claims."
    : "Gap = solution − optimum for TSP; optimum − solution for MaxCut, LT and modularity. Units: edges / distance / activated nodes / modularity. These are absolute gaps, not percentages. Laya LT has 31/40 feasible cases; GPT-5.4 retains nine construction timeouts. Compare coverage as well as quality; optimum counts are supplementary.";
  $("#anchor-note").hidden = !state.anchors;
}

function renderModel() {
  const model = data.models.find(item => item.id === state.selectedModel);
  const totals = coverageFor(data, index, model.id, state.size);
  const root = $("#model-details");
  root.replaceChildren(element("h4", "", model.name), pill(model));
  const dl = element("dl");
  for (const [label, value] of [
    ["Parameters", model.parameters], ["Protocol", model.protocol],
    ["Comparison", model.group === "strong_anchors" ? "Upper reference, not matched compute" : "Direct-decision configuration"],
    ["Scope", state.size === "all" ? "All 960 canonical instances" : `${state.size}-vertex cells, all ten tasks`],
  ]) dl.append(element("dt", "", label), element("dd", "", value));
  root.append(dl);
  const coverage = element("div", "coverage-summary");
  for (const [value, label] of [[`${totals.feasible}/${totals.scheduled}`, "Feasible / scheduled"], [totals.unsupported, "Unsupported inputs"], [totals.failures, "Execution failures"]]) {
    const block = element("div");
    block.append(element("strong", "", value), element("span", "", label));
    coverage.append(block);
  }
  root.append(coverage);
  if (totals.unattempted) root.append(element("p", "model-caveat", `${totals.unattempted} scheduled episodes were unattempted.`));
  root.append(element("p", "model-caveat", model.note));
}

function svgElement(tag, attrs, text) {
  const node = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderChart() {
  const task = data.tasks.find(item => item.id === state.chartTask);
  const selected = data.models.find(item => item.id === state.chartModel);
  const models = [data.models.find(model => model.id === "jev_action"), selected].filter((model, i, all) => all.findIndex(item => item.id === model.id) === i);
  const series = models.map(model => ({
    model, rows: data.sizes.map(size => rowFor(model.id, task.id, size)),
    values: data.sizes.map(size => valueFor(rowFor(model.id, task.id, size), state.kind, state.metric)),
  }));
  const isRate = state.kind === "exact" || state.metric === "optimalRate";
  const values = series.flatMap(item => item.values).filter(value => value !== null);
  const maximum = isRate ? 1 : Math.max(1, ...values) * 1.12;
  const x = i => 46 + i * 105;
  const y = value => 170 - value / maximum * 140;
  const root = $("#size-chart");
  root.replaceChildren();
  const description = `${task.name}: ${state.kind === "exact" ? "accuracy" : state.metric === "meanGap" ? "mean objective gap" : "optimum frequency"} across sizes 8–12.`;
  const svg = svgElement("svg", { viewBox: "0 0 495 202", class: "size-plot", role: "img", "aria-labelledby": "chart-title chart-description" });
  svg.append(svgElement("title", { id: "chart-title" }, description));
  svg.append(svgElement("desc", { id: "chart-description" }, "Descriptive size profile. Exact values, coverage and missing values are available in the chart data table below."));
  for (let i = 0; i <= 4; i++) {
    const value = maximum * i / 4;
    svg.append(svgElement("line", { x1: 46, x2: 466, y1: y(value), y2: y(value), stroke: "#e5e8e3", "stroke-width": 1 }));
    const label = isRate ? `${Math.round(value * 100)}%` : value.toFixed(maximum < 2 ? 2 : 0);
    svg.append(svgElement("text", { x: 37, y: y(value) + 3, "text-anchor": "end" }, label));
  }
  if (state.kind === "exact" && task.id !== "degree_exact") {
    svg.append(svgElement("line", { x1: 46, x2: 466, y1: y(.5), y2: y(.5), stroke: "#8e989c", "stroke-dasharray": "4 4" }));
    svg.append(svgElement("text", { x: 466, y: y(.5) - 6, "text-anchor": "end" }, "50% binary reference"));
  }
  data.sizes.forEach((size, i) => svg.append(svgElement("text", { x: x(i), y: 190, "text-anchor": "middle" }, `n = ${size}`)));
  const legend = element("div", "chart-legend");
  series.forEach((item, s) => {
    const color = s === 0 ? "#b84c33" : "#286b65";
    let previous = null;
    item.values.forEach((value, i) => {
      if (value === null) { previous = null; return; }
      if (previous !== null) svg.append(svgElement("line", { x1: x(i - 1), y1: y(previous), x2: x(i), y2: y(value), stroke: color, "stroke-width": 2 }));
      const point = svgElement("circle", { cx: x(i), cy: y(value), r: 4, fill: color, stroke: "#fff", "stroke-width": 1.5 });
      point.append(svgElement("title", {}, `${item.model.name}, n=${data.sizes[i]}: ${formatMetric(item.rows[i], task, state.metric)}; ${item.rows[i].feasible}/${item.rows[i].scheduled} feasible`));
      svg.append(point);
      previous = value;
    });
    const label = element("span");
    label.append(element("span", `chart-key ${s === 0 ? "jev" : "comparison"}`), document.createTextNode(item.model.name));
    legend.append(label);
  });
  root.append(svg, legend, element("p", "small-note", `${task.description} ${isRate ? "Higher is better." : `Gap unit: ${task.unit}. Lower is better.`} This chart always shows all five sizes.`));
  const details = element("details", "chart-data");
  details.append(element("summary", "", "View chart values and coverage"));
  const scroll = element("div", "table-scroll");
  const table = element("table");
  table.append(element("caption", "sr-only", description));
  const thead = element("thead");
  const tr = element("tr");
  for (const title of ["Model", ...data.sizes.map(size => `n=${size}`)]) {
    const th = element("th", "", title);
    th.scope = "col";
    tr.append(th);
  }
  thead.append(tr);
  table.append(thead);
  const tbody = element("tbody");
  for (const item of series) {
    const tr = element("tr");
    const th = element("th", "", item.model.name);
    th.scope = "row";
    tr.append(th);
    for (const row of item.rows) {
      const td = element("td");
      td.append(element("span", "", formatMetric(row, task, state.metric)),
        element("span", "metric-sub", `${row.feasible}/${row.scheduled} feasible`));
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(tbody);
  scroll.append(table);
  details.append(scroll);
  root.append(details);
}

function populateChartControls() {
  const tasks = currentTasks();
  if (!tasks.some(task => task.id === state.chartTask)) state.chartTask = tasks[0].id;
  $("#chart-task").replaceChildren(...tasks.map(task => new Option(task.name, task.id)));
  $("#chart-task").value = state.chartTask;
  const models = data.models.filter(model => model.group === "decision_matrix" || state.anchors);
  if (!models.some(model => model.id === state.chartModel)) state.chartModel = "kev";
  $("#chart-model").replaceChildren(...models.map(model => new Option(model.name, model.id)));
  $("#chart-model").value = state.chartModel;
}

function changeKind(kind) {
  state.kind = kind;
  state.sortTask = currentTasks()[0].id;
  state.ascending = kind === "construction" && state.metric === "meanGap";
  for (const button of document.querySelectorAll(".tab")) {
    const active = button.dataset.kind === kind;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }
  $("#result-panel").setAttribute("aria-labelledby", `tab-${kind}`);
  $("#metric-field").hidden = kind === "exact";
  populateChartControls();
  renderTable();
  renderChart();
}

function wireControls() {
  for (const button of document.querySelectorAll(".tab")) {
    button.addEventListener("click", () => changeKind(button.dataset.kind));
    button.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const kind = event.key === "Home" ? "exact" : event.key === "End" ? "construction" : state.kind === "exact" ? "construction" : "exact";
      changeKind(kind);
      $(`#tab-${kind}`).focus();
    });
  }
  $("#model-search").addEventListener("input", event => { state.search = event.target.value; renderTable(); });
  $("#family-filter").addEventListener("change", event => { state.family = event.target.value; renderTable(); });
  $("#size-filter").addEventListener("change", event => {
    state.size = event.target.value === "all" ? "all" : Number(event.target.value);
    renderTable();
    renderModel();
  });
  $("#show-anchors").addEventListener("change", event => {
    state.anchors = event.target.checked;
    if (!state.anchors && data.models.find(model => model.id === state.selectedModel).group === "strong_anchors") {
      state.selectedModel = "jev_action";
    }
    populateChartControls();
    renderTable();
    renderModel();
    renderChart();
  });
  $("#metric-select").addEventListener("change", event => {
    state.metric = event.target.value;
    state.ascending = state.metric === "meanGap";
    renderTable();
    renderChart();
  });
  $("#chart-task").addEventListener("change", event => { state.chartTask = event.target.value; renderChart(); });
  $("#chart-model").addEventListener("change", event => { state.chartModel = event.target.value; renderChart(); });
}

async function initialize() {
  try {
    const response = await fetch(new URL("./data/benchmark.json", import.meta.url));
    if (!response.ok) throw new Error(`Data request failed: HTTP ${response.status}.`);
    data = await response.json();
    index = validateData(data);
    wireControls();
    populateChartControls();
    renderTable();
    renderChart();
    renderModel();
    const provenance = $("#provenance-details");
    for (const [label, value] of [
      ["Snapshot", data.version], ["Publication date", data.evidenceDate],
      ["Original ten-model summary SHA-256", data.provenance.summarySha256],
      ["Original ten-model export seal SHA-256", data.provenance.sealSha256],
      ["Original ten-model completion SHA-256", data.provenance.completionSha256],
      ["Original ten-model verified input hashes", data.provenance.verifiedInputHashes],
      ["New inference calls", data.provenance.modelCalls],
    ]) provenance.append(element("dt", "", label), element("dd", "", value));
    provenance.append(
      element("dt", "", "Recorded extension inference calls (already completed)"),
      element("dd", "", data.provenance.recordedExtensionModelCalls),
    );
    for (const id of ["qwen38_27b_bf16", "qwen25_72b_bf16"]) {
      const model = data.models.find(item => item.id === id);
      const source = data.provenance.strongerSources[id];
      for (const [label, value] of [
        ["verified source hashes", source.verifiedSourceHashes],
        ["registration SHA-256", source.registrationSha256],
        ["run seal SHA-256", source.runSealSha256],
        ["independent replay SHA-256 (recomputed and matched)", source.independentReplaySha256],
      ]) provenance.append(element("dt", "", `${model.name}: ${label}`), element("dd", "", value));
    }
    $("#explorer").hidden = false;
  } catch (error) {
    console.error("GraphDecide could not initialize:", error);
    $("#load-error").textContent = `The benchmark data could not be loaded or validated. ${error.message} Serve the web directory over HTTP (see web/README.md); do not open index.html with file://. No substitute results are displayed.`;
    $("#load-error").hidden = false;
  } finally {
    $("#loading").hidden = true;
  }
}

initialize();
