from __future__ import annotations

import unittest
import json
import shutil
import subprocess
from pathlib import Path


ASSETS = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "autoresearch"
    / "scripts"
    / "autoresearch_runtime"
    / "dashboard_assets"
)


class DashboardObserverUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = (ASSETS / "live_app.js").read_text(encoding="utf-8")
        cls.html = (ASSETS / "index.html").read_text(encoding="utf-8")

    def test_overview_and_lifecycle_follow_the_read_only_contract(self) -> None:
        self.assertIn('let lifecycleMode = "active"', self.app)
        self.assertIn("&lifecycle=", self.app)
        self.assertIn("dashboard_summary", self.app)
        self.assertIn("current_epoch_id", self.app)
        self.assertIn("active_run", self.app)
        self.assertIn("trend.slice(-5)", self.app)
        self.assertIn("confirm_lab_id", self.app)
        self.assertIn("window.confirm(", self.app)
        for tab in ("overview", "runs", "evidence"):
            self.assertIn(tab, self.app)
        self.assertIn('["evidence", "Свидетельства"]', self.app)

    def test_run_history_is_epoch_scoped_and_paginated_without_controls(self) -> None:
        self.assertIn("limit=20", self.app)
        self.assertIn("epoch_id", self.app)
        self.assertIn("next_offset", self.app)
        self.assertNotIn("/commands", self.app)
        self.assertNotIn("startOperation", self.app)
        self.assertNotIn("pollJob", self.app)

    def test_evidence_live_refresh_search_theme_and_same_origin_access_remain(self) -> None:
        self.assertIn("/evidence?limit=20", self.app)
        self.assertIn("/evidence/", self.app)
        self.assertIn("new EventSource(", self.app)
        self.assertIn("If-None-Match", self.app)
        self.assertIn('getElementById("lab-search")', self.app)
        self.assertIn('getElementById("theme-toggle")', self.app)
        self.assertNotIn("bootstrapBrowser", self.app)
        self.assertNotIn("Authorization", self.app)
        self.assertIn('id="navigation"', self.html)
        self.assertGreaterEqual(self.app.count('applySearch(document.getElementById("lab-search").value)'), 2)
        self.assertIn("Активных лабораторий пока нет.", self.app)
        self.assertIn("Запусков в этой эпохе пока нет.", self.app)

    def _run_js(self, harness: str) -> dict[str, object]:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the dashboard state test")
        result = subprocess.run(
            [node, "-e", harness], capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_failed_lifecycle_switch_keeps_current_list_and_reports_the_error(self) -> None:
        start = self.app.index("async function setLifecycleMode(mode) {")
        end = self.app.index("\nasync function loadLabPage(offset)", start)
        function = self.app[start:end]
        view = self._run_js(f"""
let lifecycleMode = "active";
let selected = "lab-a";
let detailRequest = 0;
let labs = [{{uid: "lab-a"}}];
let message = "";
async function reloadLabs() {{ throw new Error("API offline"); }}
function renderLabs() {{}}
function setMessage(value) {{ message = value; }}
{function}
(async () => {{
  let escaped = false;
  try {{ await setLifecycleMode("archived"); }} catch (error) {{ escaped = true; }}
  process.stdout.write(JSON.stringify({{lifecycleMode, selected, lab: labs[0].uid, message, escaped}}));
}})();
""")
        self.assertEqual(view, {
            "lifecycleMode": "active", "selected": "lab-a", "lab": "lab-a",
            "message": "API offline", "escaped": False,
        })



    def test_both_list_sources_failing_keeps_dashboard_unavailable(self) -> None:
        start = self.app.index("async function reloadLabs(options = {}) {")
        end = self.app.index("\nfunction markLive()", start)
        reload_labs = self.app[start:end]
        start = self.app.index("function markLive() {")
        end = self.app.index("\nasync function refreshRevision()", start)
        mark_live = self.app[start:end]
        view = self._run_js(f"""
let lifecycleMode = "active";
let selected = "";
let labOffset = 0;
let labPage = {{offset: 0, total: 0, next_offset: null}};
let labRequestVersion = 0;
let detailRequest = 0;
let labs = [];
let sqliteLabs = [];
let legacyLabs = [];
let sourceWarning = "";
let sourcesUnavailable = false;
const LAB_PAGE_SIZE = 100;
const boot = {{root_id: "root"}};
const selectedKey = "selected";
const localStorage = {{removeItem() {{}}}};
const classes = new Set();
const updated = {{classList: {{add(name) {{ classes.add(name); }}, remove(name) {{ classes.delete(name); }}}}}};
const modeText = {{textContent: ""}};
let message = "";
function labKey() {{ return ""; }}
function oldLegacyLabKey() {{ return ""; }}
function renderLabs() {{}}
function renderPortfolio() {{}}
function setMessage(value) {{ message = value; }}
async function reloadLab() {{}}
async function request() {{ throw new Error("SQLite offline"); }}
async function requestLegacy() {{ throw new Error("Legacy offline"); }}
{reload_labs}
{mark_live}
(async () => {{
  await reloadLabs();
  markLive();
  process.stdout.write(JSON.stringify({{live: classes.has("is-live"), down: classes.has("is-down"),
    mode: modeText.textContent, message, unavailable: sourcesUnavailable}}));
}})();
""")
        self.assertFalse(view["live"])
        self.assertTrue(view["down"])
        self.assertEqual(view["mode"], "Нет связи")
        self.assertTrue(view["unavailable"])
        self.assertIn("недоступны", view["message"].lower())

    def test_sqlite_run_page_uses_list_metrics_without_per_run_requests(self) -> None:
        start = self.app.index("async function reloadLab() {")
        end = self.app.index("\nasync function setLifecycleMode(mode)", start)
        function = self.app[start:end]
        view = self._run_js(f"""
const lab = {{uid: "sqlite-lab", source: "sqlite", dashboard_summary: {{current_epoch: 2,
  metric_definitions: {{primary: {{name: "score", direction: "max"}},
    guards: [{{name: "safety", direction: "min"}}]}}}}}};
let labs = [lab];
let selected = "sqlite-lab";
let detailRequest = 0;
let selectedEpoch = null;
let epochIds = [];
let runOffset = 0;
let runPage = {{items: [], next_offset: null, offset: 0, loading: false, error: ""}};
let runMetrics = [];
let evidence = [];
let calls = [];
function labId(value) {{ return value.uid; }}
function selectedLab() {{ return lab; }}
function summaryFor(value) {{ return value.dashboard_summary; }}
function renderLabDetail() {{}}
function setMessage() {{}}
async function loadEpochIds() {{ return [2]; }}
async function loadRunMetrics(base, items) {{
  const values = [];
  for (const item of items) values.push(await request(base + "/runs/" + item.run_id));
  return values;
}}
async function request(path) {{
  calls.push(path);
  if (path.includes("/evidence")) return {{items: []}};
  if (path.includes("/runs?") && path.includes("limit=20")) return {{items: [
    {{run_id: "run-a", epoch_id: 2, metrics: {{score: {{decision_worst: 0.75, mean: 0.8, completed_repeats: 3}},
      safety: {{decision_worst: 0.9, mean: 0.92, completed_repeats: 3}}}}}},
    {{run_id: "run-b", epoch_id: 2, metrics: {{score: {{decision_worst: 0.85, mean: 0.9, completed_repeats: 3}},
      safety: {{decision_worst: 0.95, mean: 0.96, completed_repeats: 3}}}}}},
  ], next_offset: null}};
  if (path.includes("/runs/")) return {{run_id: "run-detail"}};
  return {{items: []}};
}}
{function}
(async () => {{
  await reloadLab();
  process.stdout.write(JSON.stringify({{calls, runCount: runMetrics.length, selectedEpoch,
    scores: runMetrics.map(item => item.metrics && item.metrics.score && item.metrics.score.decision_worst),
    safety: runMetrics.map(item => item.metrics && item.metrics.safety && item.metrics.safety.decision_worst)}}));
}})();
""")
        self.assertFalse(any("/runs/" in call and "/runs?" not in call for call in view["calls"]))
        self.assertEqual(view["runCount"], 2)
        self.assertEqual(view["selectedEpoch"], 2)
        self.assertTrue(any("epoch_id=2" in call for call in view["calls"]))
        self.assertEqual(view["scores"], [0.75, 0.85])
        self.assertEqual(view["safety"], [0.9, 0.95])

    def test_sqlite_summary_metric_roles_are_normalized_for_the_ui(self) -> None:
        start = self.app.index("function metricDefinitions(summary, lab) {")
        end = self.app.index("\nfunction primaryAndGuards(summary, lab)", start)
        function = self.app[start:end]
        view = self._run_js(f"""
{function}
const definitions = metricDefinitions({{
  metric_definitions: {{
    primary: {{name: "score", direction: "max"}},
    guards: [{{name: "safety", direction: "min"}}],
  }},
}}, {{}});
process.stdout.write(JSON.stringify(definitions.map(item => ({{name: item.name, role: item.role}}))));
""")
        self.assertEqual(view, [
            {"name": "score", "role": "primary"},
            {"name": "safety", "role": "guard"},
        ])

    def test_sqlite_summary_metric_means_render_values_and_chart_points(self) -> None:
        value_start = self.app.index("function metricValue(snapshot, metric) {")
        value_end = self.app.index("\nfunction formatValue(value)", value_start)
        chart_start = self.app.index("function renderMetricChart(metric, points, options = {}) {")
        chart_end = self.app.index("\nfunction renderMiniMetric", chart_start)
        svg_start = self.app.index("function svgElement(tag, attributes = {}) {")
        svg_end = self.app.index("\nfunction renderMetricChart", svg_start)
        view = self._run_js(f"""
const UNAVAILABLE = "Недоступно";
class Element {{
  constructor(tag) {{ this.tag = tag; this.children = []; this.attributes = {{}}; }}
  append(child) {{ this.children.push(child); }}
  setAttribute(key, value) {{ this.attributes[key] = String(value); }}
}}
const document = {{createElementNS(_ns, tag) {{ return new Element(tag); }}}};
{self.app[value_start:value_end]}
{self.app[svg_start:svg_end]}
{self.app[chart_start:chart_end]}
const summaryValue = metricValue({{metric_means: {{score: 0.84}}}}, "score");
const chart = renderMetricChart("score", [
  {{run_id: "a", status: "accepted", metric_means: {{score: 0.72}}}},
  {{run_id: "b", status: "accepted", metric_means: {{score: 0.84}}}},
]);
process.stdout.write(JSON.stringify({{
  summaryValue,
  chartPoints: chart.children.filter(item => item.tag === "circle").length,
  hasLine: chart.children.some(item => item.attributes.class === "v04-metric-line"),
}}));
""")
        self.assertEqual(view, {"summaryValue": 0.84, "chartPoints": 2, "hasLine": True})






    def test_epoch_change_clears_old_runs_and_marks_failed_page(self) -> None:
        start = self.app.index("async function loadRunPage(epoch, offset = 0) {")
        end = self.app.index("\nasync function selectLab(id)", start)
        function = self.app[start:end]
        view = self._run_js(f"""
let selected = "lab-a";
let selectedEpoch = 1;
let runOffset = 0;
let detailRequest = 0;
let runPage = {{items: [{{run_id: "old-run"}}], offset: 0}};
let runMetrics = [{{run_id: "old-run"}}];
let rendered = [];
const lab = {{uid: "lab-a", dashboard_summary: {{current_epoch_id: 1}}}};
function selectedLab() {{ return lab; }}
function labId(value) {{ return value.uid; }}
function summaryFor() {{ return lab.dashboard_summary; }}
function renderLabDetail() {{ rendered.push({{count: runPage.items.length, loading: !!runPage.loading}}); }}
function setMessage() {{}}
async function request() {{ throw new Error("API offline"); }}
async function loadRunMetrics() {{ return []; }}
{function}
(async () => {{
  await loadRunPage(2, 0);
  process.stdout.write(JSON.stringify({{rendered, epoch: selectedEpoch, count: runPage.items.length,
    loading: !!runPage.loading, error: runPage.error || "", metrics: runMetrics.length}}));
}})();
""")
        self.assertEqual(view["rendered"][0], {"count": 0, "loading": True})
        self.assertEqual(view["epoch"], 2)
        self.assertEqual(view["count"], 0)
        self.assertFalse(view["loading"])
        self.assertEqual(view["error"], "API offline")
        self.assertEqual(view["metrics"], 0)

    def test_run_recovery_controls_remain_visible_while_loading_or_failed(self) -> None:
        start = self.app.index("function renderRunList() {")
        end = self.app.index("\nfunction renderRuns()", start)
        function = self.app[start:end]
        view = self._run_js(f"""
class FakeNode {{
  constructor(tag, cls = "", text = "") {{
    this.tag = tag; this.className = cls; this.textContent = text;
    this.children = []; this.listeners = {{}}; this.attributes = {{}}; this.value = "";
  }}
  append(...items) {{ this.children.push(...items); }}
  setAttribute(name, value) {{ this.attributes[name] = value; }}
  addEventListener(name, handler) {{ this.listeners[name] = handler; }}
}}
function node(tag, cls = "", text = "") {{ return new FakeNode(tag, cls, text); }}
function section(title, help) {{
  const host = node("section", "v04-card");
  host.append(node("h3", "", title), node("p", "v04-muted", help));
  return host;
}}
const document = {{createElement: tag => new FakeNode(tag)}};
let selectedEpoch = 3;
let epochIds = [3, 2];
let runOffset = 20;
let loadCalls = [];
async function loadRunPage(epoch, offset) {{ loadCalls.push([epoch, offset]); }}
function allNodes(root) {{ return [root, ...root.children.flatMap(allNodes)]; }}
function snapshot(loading, error) {{
  runPage = {{items: [{{run_id: "stale-run"}}], next_offset: 40, offset: 20, loading, error}};
  const tree = renderRunList();
  const nodes = allNodes(tree);
  const select = nodes.find(item => item.tag === "select");
  const newer = nodes.find(item => item.textContent === "Более новые");
  const older = nodes.find(item => item.textContent === "Более старые");
  if (select) {{ select.value = "2"; select.listeners.change(); }}
  if (older) older.listeners.click();
  return {{
    picker: !!select, newer: !!newer, older: !!older,
    message: nodes.find(item => item.textContent === (loading ? "Загрузка запусков…" : error))?.textContent || "",
    staleRunVisible: nodes.some(item => item.textContent === "stale-run"),
  }};
}}
let runPage = {{}};
{function}
const loading = snapshot(true, "");
const failed = snapshot(false, "API offline");
process.stdout.write(JSON.stringify({{loading, failed, loadCalls}}));
""")
        self.assertEqual(view["loading"], {
            "picker": True, "newer": True, "older": True,
            "message": "Загрузка запусков…", "staleRunVisible": False,
        })
        self.assertEqual(view["failed"], {
            "picker": True, "newer": True, "older": True,
            "message": "API offline", "staleRunVisible": False,
        })
        self.assertEqual(view["loadCalls"], [[2, 0], [3, 40], [2, 0], [3, 40]])

    def test_metric_value_and_render_metric_chart_respect_statistic_mean_median_worst(self) -> None:
        value_start = self.app.index("function metricValue(snapshot, metric) {")
        value_end = self.app.index("\nfunction formatValue(value)", value_start)
        chart_start = self.app.index("function renderMetricChart(metric, points, options = {}) {")
        chart_end = self.app.index("\nfunction renderMiniMetric", chart_start)
        svg_start = self.app.index("function svgElement(tag, attributes = {}) {")
        svg_end = self.app.index("\nfunction renderMetricChart", svg_start)
        view = self._run_js(f"""
const UNAVAILABLE = "Недоступно";
class Element {{
  constructor(tag) {{ this.tag = tag; this.children = []; this.attributes = {{}}; }}
  append(...child) {{ this.children.push(...child); }}
  setAttribute(k, v) {{ this.attributes[k] = String(v); }}
}}
const document = {{ createElementNS: (_ns, tag) => new Element(tag) }};
{self.app[value_start:value_end]}
{self.app[svg_start:svg_end]}
{self.app[chart_start:chart_end]}

const metrics = {{ score: {{ decision_worst: 0.5, mean: 0.8, median: 0.7 }} }};
const vWorst = metricValue({{ metrics, statistic: "worst" }}, "score");
const vMean = metricValue({{ metrics, statistic: "mean" }}, "score");
const vMedian = metricValue({{ metrics, statistic: "median" }}, "score");

const points = [
  {{ run: {{ run_id: "r1", status: "accepted" }}, data: {{ metrics }} }},
];
const chartWorst = renderMetricChart("score", points, {{ statistic: "worst" }});
const chartMean = renderMetricChart("score", points, {{ statistic: "mean" }});
const chartMedian = renderMetricChart("score", points, {{ statistic: "median" }});

const getCircleLabel = c => c.children.find(el => el.tag === "circle").attributes["aria-label"];

process.stdout.write(JSON.stringify({{
  vWorst, vMean, vMedian,
  chartWorst: getCircleLabel(chartWorst),
  chartMean: getCircleLabel(chartMean),
  chartMedian: getCircleLabel(chartMedian),
}}));
""")
        self.assertEqual(view["vWorst"], 0.5)
        self.assertEqual(view["vMean"], 0.8)
        self.assertEqual(view["vMedian"], 0.7)
        self.assertIn("0.5", view["chartWorst"])
        self.assertIn("0.8", view["chartMean"])
        self.assertIn("0.7", view["chartMedian"])

    def test_format_delta_computes_signed_deltas_and_direction_classes(self) -> None:
        delta_start = self.app.index("function formatDelta(")
        delta_end = self.app.index("\nfunction renderOverview", delta_start)
        view = self._run_js(f"""
{self.app[delta_start:delta_end]}
const res = {{
  maxGood: formatDelta(10, 15, "max"),
  maxBad: formatDelta(15, 10, "max"),
  maxNeutral: formatDelta(10, 10, "max"),
  minGood: formatDelta(15, 10, "min"),
  minBad: formatDelta(10, 15, "min"),
  minNeutral: formatDelta(10, 10, "min"),
  nullBase: formatDelta(null, 10),
  undefBest: formatDelta(10, undefined),
  nanVal: formatDelta("invalid", 10),
}};
process.stdout.write(JSON.stringify(res));
""")
        self.assertEqual(view["maxGood"], {"text": "+5", "class": "good", "delta": 5})
        self.assertEqual(view["maxBad"], {"text": "-5", "class": "bad", "delta": -5})
        self.assertEqual(view["maxNeutral"], {"text": "0", "class": "neutral", "delta": 0})
        self.assertEqual(view["minGood"], {"text": "-5", "class": "good", "delta": -5})
        self.assertEqual(view["minBad"], {"text": "+5", "class": "bad", "delta": 5})
        self.assertEqual(view["minNeutral"], {"text": "0", "class": "neutral", "delta": 0})
        self.assertIsNone(view["nullBase"])
        self.assertIsNone(view["undefBest"])
        self.assertIsNone(view["nanVal"])

    def test_reload_lab_preserves_offset_and_epoch_without_loading_flash_on_refresh(self) -> None:
        start = self.app.index("async function reloadLab() {")
        end = self.app.index("\nasync function setLifecycleMode", start)
        function = self.app[start:end]
        view = self._run_js(f"""
let selected = "sqlite-lab";
const lab = {{ uid: "sqlite-lab", source: "sqlite", dashboard_summary: {{ current_epoch: 3 }} }};
function selectedLab() {{ return lab; }}
function labId(l) {{ return l.uid; }}
function summaryFor(l) {{ return l.dashboard_summary; }}
let detailRequest = 0;
let selectedEpoch = 2;
let runOffset = 40;
let runPage = {{ items: [{{ run_id: "existing-run" }}], next_offset: 60, offset: 40, loading: false, error: "" }};
let runMetrics = [{{ run_id: "existing-run" }}];
let evidence = [];
let epochIds = [3, 2, 1];
let renderCalls = [];
function renderLabDetail() {{
  renderCalls.push({{ itemsLen: runPage.items.length, loading: runPage.loading, offset: runPage.offset }});
}}
function setMessage() {{}}
let requests = [];
async function request(path) {{
  requests.push(path);
  if (path.includes("/runs?")) {{
    return {{ items: [{{ run_id: "new-run" }}], next_offset: 60, offset: 40, total: 100 }};
  }}
  return {{ items: [] }};
}}
async function loadEpochIds() {{ return [3, 2, 1]; }}

{function}

(async () => {{
  const promise = reloadLab();
  const midFlight = {{ itemsLen: runPage.items.length, loading: runPage.loading, selectedEpoch, runOffset }};
  await promise;
  const after = {{ itemsLen: runPage.items.length, loading: runPage.loading, selectedEpoch, runOffset, item0: runPage.items[0].run_id }};
  process.stdout.write(JSON.stringify({{ midFlight, after, requests, renderCalls }}));
}})();
""")
        self.assertEqual(view["midFlight"], {"itemsLen": 1, "loading": False, "selectedEpoch": 2, "runOffset": 40})
        self.assertEqual(view["after"], {"itemsLen": 1, "loading": False, "selectedEpoch": 2, "runOffset": 40, "item0": "new-run"})
        self.assertTrue(any("epoch_id=2" in r and "offset=40" in r for r in view["requests"]))
        self.assertTrue(all(not call["loading"] and call["itemsLen"] > 0 for call in view["renderCalls"]))

    def test_render_run_inspector_card_renders_structured_hypothesis_regressions_metrics_and_raw_json(self) -> None:
        start = self.app.index("function renderRunInspectorCard(")
        end = self.app.index("\nasync function showRun", start)
        function = self.app[start:end]
        view = self._run_js(f"""
class FakeNode {{
  constructor(tag, cls = "", text = "") {{
    this.tag = tag; this.className = cls; this.textContent = text;
    this.children = []; this.classList = {{ add: c => this.className += " " + c }};
    this.listeners = {{}}; this.style = {{}};
  }}
  append(...items) {{ this.children.push(...items); }}
  addEventListener(evt, fn) {{ this.listeners[evt] = fn; }}
  remove() {{}}
}}
function node(tag, cls = "", text = "") {{ return new FakeNode(tag, cls, text); }}
function section(title, help) {{
  const host = node("section", "v04-card");
  host.append(node("h3", "", title), node("p", "v04-muted", help));
  return host;
}}
const document = {{ createElement: tag => new FakeNode(tag) }};
const UNAVAILABLE = "Недоступно";

{function}

const info = {{
  run: {{ run_id: "run-10", alias: "trial-10", status: "accepted" }},
  data: {{
    run_id: "run-10",
    title: "Optimized prompt structuring",
    hypothesis: {{
      observation: "Observed hallucination on edge cases",
      expected_effect: "Reduce hallucinations by 50%"
    }},
    reason: "Significant improvement in accuracy",
    regressed_case_ids: ["case-01", "case-05"],
    metrics: {{
      accuracy: {{
        decision_worst: 0.85,
        mean: 0.90,
        median: 0.89,
        completed_repeats: 3
      }}
    }}
  }}
}};

const tree = renderRunInspectorCard(info);
function allNodes(n) {{ return [n, ...n.children.flatMap(allNodes)]; }}
const nodes = allNodes(tree);

const texts = nodes.map(n => n.textContent).filter(Boolean);
const detailsEl = nodes.find(n => n.tag === "details" && n.className.includes("v04-raw-details"));
const summaryEl = detailsEl ? detailsEl.children.find(n => n.tag === "summary") : null;
const preEl = detailsEl ? detailsEl.children.find(n => n.tag === "pre") : null;

process.stdout.write(JSON.stringify({{
  hasTitle: texts.some(t => t.includes("Optimized prompt structuring")),
  hasObs: texts.some(t => t.includes("Observed hallucination on edge cases")),
  hasExp: texts.some(t => t.includes("Reduce hallucinations by 50%")),
  hasReason: texts.some(t => t.includes("Significant improvement in accuracy")),
  hasRegressed: texts.some(t => t.includes("Регрессировавшие кейсы: case-01, case-05")),
  hasWorst: texts.some(t => t.includes("decision_worst: 0.85")),
  hasMean: texts.some(t => t.includes("mean: 0.9")),
  hasMedian: texts.some(t => t.includes("median: 0.89")),
  hasRepeats: texts.some(t => t.includes("completed_repeats: 3")),
  hasDetails: !!detailsEl,
  summaryText: summaryEl ? summaryEl.textContent : null,
  preHasJson: preEl ? preEl.textContent.includes("run-10") : false,
}}));
""")
        self.assertTrue(view["hasTitle"])
        self.assertTrue(view["hasObs"])
        self.assertTrue(view["hasExp"])
        self.assertTrue(view["hasReason"])
        self.assertTrue(view["hasRegressed"])
        self.assertTrue(view["hasWorst"])
        self.assertTrue(view["hasMean"])
        self.assertTrue(view["hasMedian"])
        self.assertTrue(view["hasRepeats"])
        self.assertTrue(view["hasDetails"])
        self.assertEqual(view["summaryText"], "Показать сырой JSON")
        self.assertTrue(view["preHasJson"])

    def test_find_sqlite_diff_artifact_and_render_run_list_diff_button(self) -> None:
        f_start = self.app.index("function findSqliteDiffArtifact(")
        f_end = self.app.index("\nfunction renderRunList", f_start)
        r_start = self.app.index("function renderRunList() {")
        r_end = self.app.index("\nfunction renderRuns()", r_start)
        view = self._run_js(f"""
class FakeNode {{
  constructor(tag, cls = "", text = "") {{
    this.tag = tag; this.className = cls; this.textContent = text;
    this.children = []; this.listeners = {{}}; this.attributes = {{}};
  }}
  append(...items) {{ this.children.push(...items); }}
  setAttribute(k, v) {{ this.attributes[k] = String(v); }}
  addEventListener(evt, fn) {{ this.listeners[evt] = fn; }}
}}
function node(tag, cls = "", text = "") {{ return new FakeNode(tag, cls, text); }}
function section(title, help) {{
  const host = node("section", "v04-card");
  host.append(node("h3", "", title), node("p", "v04-muted", help));
  return host;
}}
const document = {{ createElement: tag => new FakeNode(tag) }};

let evidence = [
  {{ artifact_id: "art-1", run_id: "run-1", name: "changes.diff" }},
  {{ artifact_id: "art-2", run_id: "run-2", name: "summary.txt" }},
  {{ artifact_id: "art-3", evidence_key: "patch-run-3", name: "output.bin" }}
];

{self.app[f_start:f_end]}
{self.app[r_start:r_end]}

const match1 = findSqliteDiffArtifact({{ run_id: "run-1" }});
const match2 = findSqliteDiffArtifact({{ run_id: "run-2" }});
const match3 = findSqliteDiffArtifact({{ run_id: "run-3" }});

const lab = {{ uid: "sql-lab", source: "sqlite" }};
function selectedLab() {{ return lab; }}
let selectedEpoch = 1;
let epochIds = [1];
let runOffset = 0;
let runPage = {{
  items: [
    {{ run_id: "run-1", status: "accepted" }},
    {{ run_id: "run-2", status: "accepted" }}
  ],
  loading: false,
  error: ""
}};
let evidenceOpened = null;
function loadEvidence(art) {{ evidenceOpened = art.artifact_id; }}
function showRun() {{}}

const tree = renderRunList();
function allNodes(n) {{ return [n, ...n.children.flatMap(allNodes)]; }}
const nodes = allNodes(tree);
const diffButtons = nodes.filter(n => n.tag === "button" && n.textContent === "Показать diff");

diffButtons[0].listeners.click();

process.stdout.write(JSON.stringify({{
  match1Id: match1 ? match1.artifact_id : null,
  match2: match2,
  match3Id: match3 ? match3.artifact_id : null,
  diffButtonCount: diffButtons.length,
  opened: evidenceOpened
}}));
""")
        self.assertEqual(view["match1Id"], "art-1")
        self.assertIsNone(view["match2"])
        self.assertEqual(view["match3Id"], "art-3")
        self.assertEqual(view["diffButtonCount"], 1)
        self.assertEqual(view["opened"], "art-1")




if __name__ == "__main__":
    unittest.main()
