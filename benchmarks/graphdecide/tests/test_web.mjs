import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateData, metricKey, valueFor, visibleModels, coverageFor, formatMetric } from "../web/model.mjs";

const data = JSON.parse(readFileSync(new URL("../web/data/benchmark.json", import.meta.url)));
const index = validateData(data);
const state = {
  kind: "exact", family: "all", size: "all", metric: "meanGap", anchors: false,
  search: "", sortTask: "adjacency", ascending: false,
};

test("the static snapshot contains all 720 validated task/size/model cells", () => {
  assert.equal(index.size, 720);
  assert.equal(data.provenance.verifiedInputHashes, 1550);
  assert.equal(data.provenance.modelCalls, 0);
});

test("default view has ten direct configurations with no GPT ranking", () => {
  const result = visibleModels(data, state, index);
  assert.equal(result.decision.length, 10);
  assert.equal(result.anchors.length, 0);
  assert.equal(result.decision[0].id, "jev_action");
});

test("GPT stays separate even when its score exceeds a direct model", () => {
  const result = visibleModels(data, { ...state, anchors: true, sortTask: "degree_exact" }, index);
  assert.equal(result.decision.length, 10);
  assert.equal(result.anchors.length, 2);
  assert.ok(result.decision.every(model => model.family !== "GPT"));
});

test("family filter applies to decision models; anchors remain opt-in references", () => {
  const result = visibleModels(data, { ...state, family: "Jev-like", anchors: true }, index);
  assert.equal(result.decision.length, 4);
  assert.equal(result.anchors.length, 2);
  assert.ok(result.decision.every(model => model.family === "Jev-like"));
});

test("search is case-insensitive and has an explicit empty result", () => {
  assert.equal(visibleModels(data, { ...state, search: "kEV" }, index).decision[0].id, "kev");
  assert.deepEqual(visibleModels(data, { ...state, search: "no-such-model" }, index), { decision: [], anchors: [] });
});

test("degree displays macro accuracy rather than pooled counts", () => {
  const task = data.tasks.find(task => task.id === "degree_exact");
  const row = index.get(metricKey("jev_action", task.id, "all"));
  assert.equal(formatMetric(row, task, "meanGap"), "61.43%");
  assert.equal(row.hits, 241);
  assert.equal(row.scheduled, 400);
  assert.notEqual(valueFor(row, "exact", "meanGap"), row.hits / row.scheduled);
  const sizeRow = index.get(metricKey("jev_action", task.id, 12));
  assert.equal(formatMetric(sizeRow, task, "meanGap"), "48.96%");
});

test("construction sorting reverses direction for gaps and optimum rates", () => {
  const gap = visibleModels(data, { ...state, kind: "construction", sortTask: "lt_influence_construct", ascending: true }, index);
  const rate = visibleModels(data, { ...state, kind: "construction", metric: "optimalRate", sortTask: "lt_influence_construct" }, index);
  assert.equal(gap.decision[0].id, "qwen27_grammar");
  assert.equal(rate.decision[0].id, "qwen27_grammar");
  assert.equal(rate.decision[1].id, "kev");
});

test("new BF16 rows preserve generation, macro accuracy, gap and complete coverage", () => {
  const degree = data.tasks.find(task => task.id === "degree_exact");
  const tsp = data.tasks.find(task => task.id === "tsp_construct");
  for (const [model, accuracy, pooled, gap] of [
    ["qwen38_27b_bf16", "45.50%", 178, 183.7],
    ["qwen25_72b_bf16", "41.24%", 162, 152.4],
  ]) {
    const row = index.get(metricKey(model, degree.id, "all"));
    assert.equal(formatMetric(row, degree, "meanGap"), accuracy);
    assert.equal(row.hits, pooled);
    assert.notEqual(row.accuracy, row.hits / row.scheduled);
    assert.equal(index.get(metricKey(model, tsp.id, "all")).meanGap, gap);
    assert.deepEqual(coverageFor(data, index, model), {
      scheduled: 960, feasible: 960, unsupported: 0, failures: 0, unattempted: 0,
    });

  }
  assert.equal(visibleModels(data, { ...state, search: "72B" }, index).decision[0].id, "qwen25_72b_bf16");
  assert.equal(visibleModels(data, { ...state, family: "Qwen" }, index).decision.length, 5);
});

test("page links and headline values belong to the current twelve-model snapshot", () => {
  const html = readFileSync(new URL("../web/index.html", import.meta.url), "utf8");
  assert.match(html, /Working paper v17/);
  assert.match(html, /v17 has 8 pages of main text/);
  assert.match(html, /href="assets\/graphdecide-v17\.pdf"/);
  assert.match(html, /<title>GraphDecide: Benchmarking Decision Models on Graphs<\/title>/);
  assert.match(html, /Li Zhao<\/a> \(corresponding author\)/);
  assert.match(html, /Xianliang Yang<\/a> — Microsoft/);
  assert.match(html, /Yapu Zhang<\/a> — Beijing University of Technology/);
  assert.match(html, /Li Zhao<\/a> \(corresponding author\) — Microsoft/);
  for (const email of ["xianya@microsoft.com", "zhangyapu@bjut.edu.cn", "lizo@microsoft.com"]) {
    assert.ok(html.includes(`href="mailto:${email}"`));
  }
  assert.match(html, /11,520 scheduled episode evaluations/);
  assert.match(html, /720 model × task × size metric cells/);
  assert.doesNotMatch(html, /jevgraphbench-v1[34]\.pdf|9,600 scheduled/);
  for (const id of ["jev_action", "qwen27_grammar", "qwen38_27b_bf16", "qwen25_72b_bf16"]) {
    const row = index.get(metricKey(id, "tsp_construct", "all"));
    assert.ok(html.includes(row.meanGap.toFixed(2)));
  }
});

test("Laya and GPT-5.4 coverage preserves nine distinct failure outcomes", () => {
  assert.deepEqual(coverageFor(data, index, "laya"), {
    scheduled: 960, feasible: 951, unsupported: 9, failures: 0, unattempted: 0,
  });
  assert.deepEqual(coverageFor(data, index, "gpt54_default_reasoning"), {
    scheduled: 960, feasible: 951, unsupported: 0, failures: 9, unattempted: 0,
  });
  const task = data.tasks.find(task => task.id === "lt_influence_construct");
  const row = index.get(metricKey("laya", task.id, "all"));
  assert.equal(formatMetric(row, task, "optimalRate"), "6/40");
  assert.equal(row.feasible, 31);
});

test("missing feasible quality is a dash, never a fabricated zero", () => {
  const task = data.tasks.find(task => task.id === "maxcut_construct");
  const row = { meanGap: null, hits: 0, scheduled: 8 };
  assert.equal(formatMetric(row, task, "meanGap"), "—");
  const changedIndex = new Map(index);
  changedIndex.set(metricKey("jev_action", task.id, "all"), row);
  const result = visibleModels(data, { ...state, kind: "construction", sortTask: task.id, ascending: true }, changedIndex);
  assert.equal(result.decision.at(-1).id, "jev_action");
});

test("partial, duplicate, malformed and pooled-as-macro snapshots fail explicitly", () => {
  for (const mutate of [
    data => data.metrics.pop(),
    data => { data.metrics[1] = data.metrics[0]; },
    data => { data.metrics[0].scheduled = 0; },
    data => { data.metrics.find(row => row.task === "degree_exact" && row.nodeCount === "all").accuracy = 241 / 400; },
    data => { data.metrics.find(row => row.task === "maxcut_construct").meanGap = null; },
    data => { data.models[0].group = "strong_anchors"; },
    data => { data.version = "canonical-20260925"; },
    data => { delete data.provenance.strongerSources.qwen25_72b_bf16; },
    data => { data.provenance.strongerSources.qwen38_27b_bf16.independentReplayRecomputedSha256 = "0".repeat(64); },
    data => { data.metrics.find(row => row.task === "tsp_construct" && row.nodeCount === "all").meanGap += 1; },
  ]) {
    const copy = structuredClone(data);
    mutate(copy);
    assert.throws(() => validateData(copy));
  }
});
