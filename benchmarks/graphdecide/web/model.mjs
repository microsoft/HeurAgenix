export const metricKey = (model, task, size) => `${model}|${task}|${size}`;

export function validateData(data) {
  if (data?.schemaVersion !== 1 || !Array.isArray(data.models) ||
      !Array.isArray(data.tasks) || !Array.isArray(data.metrics) ||
      data.version !== "canonical-20260927-v14" ||
      data.models.length !== 12 || data.tasks.length !== 10 ||
      JSON.stringify(data.sizes) !== "[8,9,10,11,12]") {
    throw new Error("Unsupported or incomplete benchmark snapshot.");
  }
  const modelIds = new Set(data.models.map(model => model.id));
  const taskIds = new Set(data.tasks.map(task => task.id));
  if (modelIds.size !== 12 || taskIds.size !== 10 ||
      !modelIds.has("qwen38_27b_bf16") || !modelIds.has("qwen25_72b_bf16") ||
      data.models.filter(model => model.group === "strong_anchors").length !== 2 ||
      data.models.filter(model => model.group === "decision_matrix").length !== 10) {
    throw new Error("Invalid model/task inventory or anchor grouping.");
  }
  const provenance = data.provenance;
  if (provenance?.verifiedInputHashes !== 1550 || provenance.modelCalls !== 0 ||
      provenance.recordedExtensionModelCalls !== 3840) {
    throw new Error("Invalid publication verification provenance.");
  }
  for (const id of ["qwen38_27b_bf16", "qwen25_72b_bf16"]) {
    const source = provenance.strongerSources?.[id];
    if (!source || !Number.isInteger(source.verifiedSourceHashes) ||
        source.verifiedSourceHashes <= 0 || source.recordedModelCalls !== 1920 ||
        source.modelCalls !== 0 ||
        ["registrationSha256", "runSealSha256", "independentReplaySha256",
          "independentReplayRecomputedSha256"].some(field =>
          typeof source[field] !== "string" || !/^[0-9a-f]{64}$/.test(source[field])) ||
        source.independentReplaySha256 !== source.independentReplayRecomputedSha256) {
      throw new Error(`Invalid stronger-model provenance: ${id}`);
    }
  }
  const index = new Map();
  for (const row of data.metrics) {
    const key = metricKey(row.model, row.task, row.nodeCount);
    const counts = ["scheduled", "feasible", "hits", "unsupported", "failures", "unattempted"];
    if (!modelIds.has(row.model) || !taskIds.has(row.task) ||
        !["all", ...data.sizes].includes(row.nodeCount) || index.has(key) ||
        counts.some(field => !Number.isInteger(row[field]) || row[field] < 0) ||
        row.scheduled <= 0 || row.hits > row.feasible ||
        row.feasible + row.unsupported + row.failures + row.unattempted !== row.scheduled) {
      throw new Error(`Invalid or duplicate metric cell: ${key}`);
    }
    const task = data.tasks.find(item => item.id === row.task);
    if (task.kind === "exact") {
      if (!Number.isFinite(row.accuracy) || row.accuracy < 0 || row.accuracy > 1 ||
          row.optimalRate !== null || row.meanGap !== null || row.meanObjective !== null) {
        throw new Error(`Invalid exact-task metric: ${key}`);
      }
    } else if (task.kind === "construction") {
      if (row.accuracy !== null || !Number.isFinite(row.optimalRate) ||
          Math.abs(row.optimalRate - row.hits / row.scheduled) > 1e-12 ||
          (row.feasible === 0
            ? row.meanGap !== null || row.meanObjective !== null
            : !Number.isFinite(row.meanGap) || row.meanGap < 0 || !Number.isFinite(row.meanObjective))) {
        throw new Error(`Invalid construction metric: ${key}`);
      }
    } else {
      throw new Error(`Unknown task kind: ${task.kind}`);
    }
    index.set(key, row);
  }
  if (index.size !== 720) throw new Error("Expected all 720 model/task/size cells.");
  for (const model of data.models) {
    for (const task of data.tasks) {
      const total = index.get(metricKey(model.id, task.id, "all"));
      const cells = data.sizes.map(size => index.get(metricKey(model.id, task.id, size)));
      if (!total || cells.some(cell => !cell) || total.scheduled !== task.scheduled) {
        throw new Error("Missing model/task/size coverage.");
      }
      for (const field of ["scheduled", "feasible", "hits", "unsupported", "failures", "unattempted"]) {
        if (total[field] !== cells.reduce((sum, cell) => sum + cell[field], 0)) {
          throw new Error(`Inconsistent aggregate ${field}.`);
        }
      }
      if (task.kind === "exact") {
        const macro = cells.reduce((sum, cell) => sum + cell.hits / cell.scheduled, 0) / 5;
        if (Math.abs(total.accuracy - macro) > 1e-12 ||
            cells.some(cell => Math.abs(cell.accuracy - cell.hits / cell.scheduled) > 1e-12)) {
          throw new Error("Exact accuracy does not preserve macro-size weighting.");
        }
      } else if (total.feasible > 0) {
        for (const field of ["meanGap", "meanObjective"]) {
          const mean = cells.reduce((sum, cell) =>
            sum + (cell.feasible ? cell[field] * cell.feasible : 0), 0) / total.feasible;
          if (Math.abs(total[field] - mean) > 1e-12) {
            throw new Error("Construction means do not preserve feasible weighting.");
          }
        }
      }
    }
  }
  return index;
}

export function valueFor(row, kind, metric) {
  return row[kind === "exact" ? "accuracy" : metric];
}

export function visibleModels(data, state, index) {
  const query = state.search.trim().toLowerCase();
  const filtered = data.models.filter(model =>
    (model.group === "strong_anchors" ? state.anchors : state.family === "all" || model.family === state.family) &&
    `${model.name} ${model.family} ${model.parameters}`.toLowerCase().includes(query));
  const compare = (left, right) => {
    const a = valueFor(index.get(metricKey(left.id, state.sortTask, state.size)), state.kind, state.metric);
    const b = valueFor(index.get(metricKey(right.id, state.sortTask, state.size)), state.kind, state.metric);
    if (a === null && b === null) return left.name.localeCompare(right.name);
    if (a === null) return 1;
    if (b === null) return -1;
    return (state.ascending ? a - b : b - a) || left.name.localeCompare(right.name);
  };
  return {
    decision: filtered.filter(model => model.group === "decision_matrix").sort(compare),
    anchors: filtered.filter(model => model.group === "strong_anchors").sort(compare),
  };
}

export function formatMetric(row, task, metric) {
  if (task.kind === "exact") return `${(100 * row.accuracy).toFixed(2)}%`;
  if (metric === "optimalRate") return `${row.hits}/${row.scheduled}`;
  return row.meanGap === null ? "—" : row.meanGap.toFixed(task.unit === "modularity" ? 4 : 3);
}

export function coverageFor(data, index, modelId, size = "all") {
  const totals = { scheduled: 0, feasible: 0, unsupported: 0, failures: 0, unattempted: 0 };
  for (const task of data.tasks) {
    const row = index.get(metricKey(modelId, task.id, size));
    for (const key of Object.keys(totals)) totals[key] += row[key];
  }
  return totals;
}
