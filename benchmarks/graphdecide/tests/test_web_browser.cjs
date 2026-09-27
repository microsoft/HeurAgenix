/* Optional end-to-end check: needs Playwright and a running local web preview. */
const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { mkdir } = require("node:fs/promises");
const { join } = require("node:path");
const { chromium } = require("playwright");

async function main() {
  const base = process.env.WEB_TEST_URL || "http://127.0.0.1:8765/";
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await context.newPage();
  const errors = [];
  const externalRequests = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {
    if (new URL(request.url()).origin !== new URL(base).origin) externalRequests.push(request.url());
  });
  try {
    await page.goto(base);
    await page.locator("#explorer").waitFor({ state: "visible" });
    assert.equal(await page.title(), "GraphDecide: Benchmarking Decision Models on Graphs");
    assert.equal(await page.locator('a[href="mailto:xianya@microsoft.com"]').count(), 1);
    assert.equal(await page.locator('a[href="mailto:zhangyapu@bjut.edu.cn"]').count(), 1);
    assert.equal(await page.locator('a[href="mailto:lizo@microsoft.com"]').count(), 1);
    assert.match(await page.locator(".resources-section").innerText(), /Beijing University of Technology/);
    assert.equal(await page.locator("#results-table tr[data-model]").count(), 10);
    assert.equal(await page.locator(".anchor-group").count(), 0);
    assert.match(await page.locator('tr[data-model="jev_action"]').innerText(), /61\.43%/);
    assert.match(await page.locator('tr[data-model="qwen38_27b_bf16"]').innerText(), /45\.50%/);
    assert.match(await page.locator('tr[data-model="qwen25_72b_bf16"]').innerText(), /41\.24%/);
    assert.match(await page.locator("#provenance-details").textContent(), /3840/);
    assert.match(await page.locator("#provenance-details").textContent(), /Qwen2\.5-72B: independent replay SHA-256/);
    await page.locator("#model-search").fill("72B");
    assert.equal(await page.locator("#results-table tr[data-model]").count(), 1);
    await page.locator('tr[data-model="qwen25_72b_bf16"] .model-name').click();
    assert.match(await page.locator("#model-details").innerText(), /960\/960/);
    await page.locator("#model-search").fill("");
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    if (process.env.WEB_SCREENSHOTS) {
      await mkdir(process.env.WEB_SCREENSHOTS, { recursive: true });
      await page.screenshot({ path: join(process.env.WEB_SCREENSHOTS, "desktop.png"), fullPage: true });
    }
    await page.getByRole("button", { name: "Exact degree", exact: false }).click();
    assert.equal(await page.locator("#results-table tr[data-model]").first().getAttribute("data-model"), "jev_action");
    await page.getByRole("button", { name: "Exact degree", exact: false }).click();
    assert.equal(await page.locator("#results-table tr[data-model]").first().getAttribute("data-model"), "laya");
    await page.getByRole("button", { name: "Adjacency", exact: false }).click();
    await page.locator("#family-filter").selectOption("Jev-like");
    assert.equal(await page.locator("#results-table tr[data-model]").count(), 4);
    await page.locator("#model-search").fill("kEv");
    assert.equal(await page.locator("#results-table tr[data-model]").count(), 1);
    await page.locator("#model-search").fill("missing-model");
    assert.match(await page.locator(".empty-state").innerText(), /No models match/);
    await page.locator("#model-search").fill("");
    await page.locator("#family-filter").selectOption("all");
    await page.locator("#show-anchors").check();
    assert.equal(await page.locator("#results-table tr[data-model]").count(), 12);
    assert.equal(await page.locator(".anchor-group").count(), 1);
    assert.equal(await page.locator("#anchor-note").isVisible(), true);
    await page.locator("#size-filter").selectOption("12");
    assert.match(await page.locator('tr[data-model="jev_action"]').innerText(), /48\.96%/);
    await page.locator("#size-filter").selectOption("all");
    await page.locator("#tab-construction").click();
    assert.match(await page.locator("#metric-note").innerText(), /Primary: mean absolute gap/);
    assert.equal(await page.locator("#metric-select").inputValue(), "meanGap");
    assert.match(await page.locator('tr[data-model="jev_action"]').innerText(), /45\.850/);
    assert.match(await page.locator('tr[data-model="qwen38_27b_bf16"]').innerText(), /183\.700/);
    assert.match(await page.locator('tr[data-model="qwen25_72b_bf16"]').innerText(), /152\.400/);
    assert.match(await page.locator('tr[data-model="laya"]').innerText(), /31\/40 feasible/);
    await page.locator("#metric-select").selectOption("optimalRate");
    assert.match(await page.locator('tr[data-model="laya"]').innerText(), /6\/40/);
    await page.getByRole("button", { name: "Deterministic LT", exact: false }).click();
    assert.equal(await page.locator("#results-table tr[data-model]").first().getAttribute("data-model"), "qwen27_grammar");
    await page.locator('tr[data-model="gpt54_default_reasoning"] .model-name').click();
    assert.match(await page.locator("#model-details").innerText(), /951\/960/);
    await page.locator("#chart-task").selectOption("lt_influence_construct");
    await page.locator("#chart-model").selectOption("laya");
    assert.equal(await page.locator("#size-chart circle").count(), 10);
    await page.locator(".chart-data summary").click();
    assert.equal(await page.locator(".chart-data tbody tr").count(), 2);
    await page.locator("#tab-construction").focus();
    await page.keyboard.press("ArrowLeft");
    assert.equal(await page.locator("#tab-exact").getAttribute("aria-selected"), "true");
    await page.locator("#show-anchors").uncheck();
    assert.equal(await page.locator(".anchor-group").count(), 0);
    assert.match(await page.locator("#model-details h4").innerText(), /^Jev$/);

    for (const path of ["data/benchmark.csv", "data/benchmark.json", "assets/mark.svg", "assets/graphdecide-v17.pdf"]) {
      const response = await context.request.get(new URL(path, base).href);
      assert.equal(response.status(), 200, path);
      if (path.endsWith(".pdf")) {
        const bytes = await response.body();
        assert.equal(createHash("sha256").update(bytes).digest("hex"), "97f6ae0998a20aecac3e37b8c1cfb441b4019c96f2dbbdd4b8866a2f3cf7bf2d");
      }
    }
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.evaluate(() => window.scrollTo(0, 0));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, `${width}px viewport has page overflow`);
      await page.locator("#tab-construction").click();
      await page.locator("#metric-select").selectOption("meanGap");
      assert.equal(await page.locator('tr[data-model="jev_action"]').isVisible(), true);
      await page.locator("#tab-exact").click();
      if (process.env.WEB_SCREENSHOTS && width === 390) {
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({ path: join(process.env.WEB_SCREENSHOTS, "mobile.png"), fullPage: true });
      }
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(externalRequests, []);

    const nested = await context.newPage();
    await nested.route("**/JevGraphBench/**", async route => {
      const nestedUrl = new URL(route.request().url());
      const response = await context.request.get(new URL(nestedUrl.pathname.replace("/JevGraphBench/", ""), base).href);
      await route.fulfill({ response });
    });
    await nested.goto(new URL("JevGraphBench/", base).href);
    await nested.locator("#explorer").waitFor({ state: "visible" });
    assert.equal(await nested.locator("#results-table tr[data-model]").count(), 10);
    const offline = await context.newPage();
    await offline.route("**/data/benchmark.json", route => route.fulfill({ status: 404, body: "missing" }));
    await offline.goto(base);
    await offline.locator("#load-error").waitFor({ state: "visible" });
    assert.match(await offline.locator("#load-error").innerText(), /HTTP 404/);
    assert.equal(await offline.locator("#explorer").isVisible(), false);
    const malformed = await context.newPage();
    await malformed.route("**/data/benchmark.json", route => route.fulfill({ contentType: "application/json", body: '{"schemaVersion":1}' }));
    await malformed.goto(base);
    await malformed.locator("#load-error").waitFor({ state: "visible" });
    assert.match(await malformed.locator("#load-error").innerText(), /incomplete benchmark snapshot/);
    console.log("Browser checks passed: desktop, 390px/320px mobile, filters, sorting, tabs, anchors, chart, failure states, assets, project subpath, and no external requests.");
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
