const boot = JSON.parse(document.getElementById("snapshot").textContent);
const nav = document.getElementById("navigation");
const detail = document.getElementById("detail");
const modeText = document.getElementById("mode-text");
const updated = document.getElementById("updated");
const banner = document.getElementById("connection");
const selectedKey = "autoresearch:selected:" + boot.root_id;
const UNAVAILABLE = "Недоступно";
const LAB_PAGE_SIZE = 100;

let labs = [];
let sqliteLabs = [];
const lifecycleMode = "active";
let selected = "";
let labOffset = 0;
let labPage = {offset: 0, total: 0, next_offset: null};
let labRequestVersion = 0;
let sourceWarning = "";
let sourcesUnavailable = false;
let activeTab = "overview";
let selectedEpoch = null;
let epochIds = [];
let runOffset = 0;
let runPage = {items: [], next_offset: null, offset: 0, loading: false, error: ""};
let runMetrics = [];
let evidence = [];
let revision = null;
let timer = 0;
let events = null;
let refreshing = false;
let detailRequest = 0;
let inspectedRun = null;

function node(tag, cls, value) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (value !== undefined && value !== null) el.textContent = String(value);
  return el;
}

function labId(lab) {
  if (!lab) return "";
  return String(lab.uid || lab.lab_id || lab.id || "");
}

function labKey(lab) {
  return labId(lab);
}

function selectedLab() {
  return labs.find(item => labKey(item) === selected) || null;
}

function summaryFor(lab = selectedLab()) {
  if (!lab) return {};
  if (lab.dashboard_summary && typeof lab.dashboard_summary === "object") return lab.dashboard_summary;
  return {};
}


function setMessage(text, kind = "info") {
  banner.hidden = !text;
  banner.className = "banner " + kind;
  banner.textContent = text || "";
}

async function request(path) {
  const response = await fetch(path, {credentials: "same-origin", cache: "no-store"});
  if (response.status === 304) return {notModified: true, headers: response.headers};
  let payload = null;
  try { payload = await response.json(); } catch (error) {}
  if (!response.ok || !payload || payload.ok !== true) {
    const error = payload && payload.error;
    throw new Error((error && error.message) ||
      "Локальный API вернул ошибку (HTTP " + response.status + ")");
  }
  return payload.data;
}

function metricDefinitions(summary, lab) {
  const raw = summary.metric_definitions;
  let definitions = [];
  if (Array.isArray(raw)) {
    definitions = raw.filter(item => item && typeof item === "object");
  } else if (raw && typeof raw === "object") {
    if (raw.primary !== undefined || raw.guards !== undefined) {
      const primary = raw.primary;
      if (primary && typeof primary === "object") definitions.push({...primary, role: primary.role || "primary"});
      else if (typeof primary === "string" && primary) definitions.push({name: primary, role: "primary"});
      if (Array.isArray(raw.guards)) {
        definitions.push(...raw.guards.map(item => typeof item === "string" ? {name: item, role: "guard"} :
          (item && typeof item === "object" ? {...item, role: item.role || "guard"} : null)).filter(Boolean));
      }
    } else {
      definitions = Object.entries(raw).map(([name, value]) =>
        value && typeof value === "object" ? {name, ...value} : {name});
    }
  }
  if (!definitions.length && lab && (lab.primary_metric || lab.metric)) {
    definitions.push({name: lab.primary_metric || lab.metric, role: "primary"});
  }
  return definitions.map(item => ({
    ...item,
    name: String(item.metric_name || item.name || item.key || ""),
    role: String(item.role || item.kind || "").toLowerCase(),
  })).filter(item => item.name);
}

function primaryAndGuards(summary, lab) {
  const definitions = metricDefinitions(summary, lab);
  const primary = definitions.find(item => item.role === "primary") || definitions[0] || null;
  const guards = definitions.filter(item => item.role === "guard");
  return {primary, guards};
}

function metricValue(snapshot, metric) {
  if (!snapshot || typeof snapshot !== "object") return null;
  const metrics = snapshot.metrics && typeof snapshot.metrics === "object" ? snapshot.metrics :
    (snapshot.metric_means && typeof snapshot.metric_means === "object" ? snapshot.metric_means : {});
  let value = metrics[metric];
  if (value === undefined) value = snapshot[metric];
  if (value === undefined && (snapshot.score !== undefined || snapshot.value !== undefined)) {
    value = snapshot.score !== undefined ? snapshot.score : snapshot.value;
  }
  const stat = snapshot && typeof snapshot === "object" && snapshot.statistic ? snapshot.statistic : null;
  const keys = stat === "mean"
    ? ["mean", "decision_mean", "value", "score", "decision_worst", "median"]
    : (stat === "median"
      ? ["median", "mean", "decision_mean", "value", "score", "decision_worst"]
      : ["decision_worst", "value", "score", "decision_mean", "mean", "median"]);
  if (value && typeof value === "object") {
    for (const key of keys) {
      if (value[key] !== undefined) { value = value[key]; break; }
    }
  }
  if (value === null || value === undefined || value === "") return null;
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return UNAVAILABLE;
  return typeof value === "number" && Number.isFinite(value) ? String(value) : String(value);
}

function summaryValue(summary, key, metric, statistic = null) {
  const snap = summary && summary[key];
  const stat = statistic || (summary && summary.statistic) || null;
  if (snap && typeof snap === "object" && stat && !snap.statistic) {
    return metricValue({...snap, statistic: stat}, metric);
  }
  return metricValue(snap, metric);
}

function section(title, help) {
  const host = node("section", "v04-card");
  host.append(node("h3", "", title));
  if (help) host.append(node("p", "v04-muted", help));
  return host;
}

function metricCard(title, value, cls = "") {
  const card = node("div", "v04-metric" + (cls ? " " + cls : ""));
  card.append(node("span", "v04-muted", title));
  card.append(node("strong", "", value === null || value === undefined ? UNAVAILABLE : value));
  return card;
}

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  return element;
}

function renderMetricChart(metric, points, options = {}) {
  const sample = points.map((point, index) => {
    const run = point && point.run ? point.run : point || {};
    const data = point && point.data ? point.data : run;
    const values = data && data.metrics && typeof data.metrics === "object" ? data.metrics :
      (data && data.metric_means && typeof data.metric_means === "object" ? data.metric_means : {});
    let metricValue = values[metric];
    if (metricValue === undefined) metricValue = data && data[metric];
    const stat = options.statistic || (data && data.statistic) || null;
    const keys = stat === "mean"
      ? ["mean", "decision_mean", "value", "score", "decision_worst", "median"]
      : (stat === "median"
        ? ["median", "mean", "decision_mean", "value", "score", "decision_worst"]
        : ["decision_worst", "value", "score", "decision_mean", "mean", "median"]);
    if (metricValue && typeof metricValue === "object") {
      for (const key of keys) {
        if (metricValue[key] !== undefined) { metricValue = metricValue[key]; break; }
      }
    }
    const numericValue = metricValue === null || metricValue === undefined || metricValue === ""
      ? null : (typeof metricValue === "number" ? metricValue : Number(metricValue));
    return {
      run,
      value: Number.isFinite(numericValue) ? numericValue : null,
      parent: run.parent_run_id ? String(run.parent_run_id) : "",
      id: String(run.run_id || ""),
      status: String(run.status || "unknown"),
      baseline: String(run.run_id || run.alias || "") === String(options.baselineId || "") ||
        run.kind === "baseline",
      index,
    };
  }).sort((left, right) => {
    const leftTime = Date.parse(left.run.created_at || "");
    const rightTime = Date.parse(right.run.created_at || "");
    return Number.isFinite(leftTime) && Number.isFinite(rightTime)
      ? leftTime - rightTime : left.index - right.index;
  });
  if (!sample.some(point => point.value !== null)) return null;

  const values = sample.filter(point => point.value !== null).map(point => point.value);
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const pad = Math.abs(min || 1) * 0.08;
    min -= pad;
    max += pad;
  } else {
    const pad = (max - min) * 0.12;
    min -= pad;
    max += pad;
  }
  const width = 900;
  const height = options.mini ? 90 : 230;
  const left = options.mini ? 5 : 54;
  const right = options.mini ? 5 : 18;
  const top = options.mini ? 6 : 16;
  const bottom = options.mini ? 8 : 38;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const x = index => sample.length <= 1 ? left + plotWidth / 2 : left + index * plotWidth / (sample.length - 1);
  const y = value => top + (max - value) * plotHeight / (max - min);
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": metric + " по запускам",
    class: options.mini ? "v04-sparkline" : "v04-history-chart",
  });
  if (!options.mini) {
    svg.append(svgElement("line", {
      x1: left, y1: height - bottom, x2: width - right, y2: height - bottom, class: "v04-axis",
    }));
    const earlier = svgElement("text", {x: left, y: height - 10, class: "v04-axis-label"});
    earlier.textContent = "раньше";
    svg.append(earlier);
    const later = svgElement("text", {
      x: width - right, y: height - 10, "text-anchor": "end", class: "v04-axis-label",
    });
    later.textContent = "позже";
    svg.append(later);
    const baseline = sample.find(point => point.baseline && point.value !== null);
    if (baseline) {
      svg.append(svgElement("line", {
        x1: left, y1: y(baseline.value), x2: width - right, y2: y(baseline.value),
        class: "v04-baseline-line",
      }));
    }
    const byId = new Map(sample.map((point, index) => [point.id, index]));
    for (let index = 0; index < sample.length; index++) {
      const point = sample[index];
      const parentIndex = byId.get(point.parent);
      if (point.value !== null && parentIndex !== undefined && sample[parentIndex].value !== null) {
        svg.append(svgElement("line", {
          x1: x(parentIndex), y1: y(sample[parentIndex].value), x2: x(index), y2: y(point.value),
          class: "v04-parent-line",
        }));
      }
    }
  }
  let path = "";
  let drawing = false;
  for (let index = 0; index < sample.length; index++) {
    const point = sample[index];
    if (point.value === null) { drawing = false; continue; }
    path += `${drawing ? " L" : "M"}${x(index)} ${y(point.value)}`;
    drawing = true;
  }
  if (path) svg.append(svgElement("path", {d: path, class: "v04-metric-line"}));
  for (let index = 0; index < sample.length; index++) {
    const point = sample[index];
    const statusClass = point.status.toLowerCase().replace(/[^a-z0-9_-]/g, "");
    const label = (point.run.alias || point.id || UNAVAILABLE) + " · " + point.status +
      (point.value === null ? " · значение отсутствует" : " · " + point.value);
    const circle = svgElement("circle", {
      cx: x(index), cy: point.value === null ? height - bottom - 4 : y(point.value),
      r: options.mini ? 3 : (point.baseline ? 7 : 5),
      class: "v04-point status-" + statusClass +
        (point.baseline ? " is-baseline" : "") + (point.value === null ? " has-no-value" : ""),
      tabindex: 0, "aria-label": label,
    });
    const tooltip = svgElement("title");
    tooltip.textContent = label;
    circle.append(tooltip);
    svg.append(circle);
  }
  return svg;
}

function renderMiniMetric(host, summary, definition, trend, label) {
  const name = definition ? definition.name : UNAVAILABLE;
  const card = node("div", "v04-mini-metric");
  card.append(node("span", "v04-muted", label + " · " + name));
  const latest = definition ? summaryValue(summary, "latest", name) : null;
  const best = definition ? summaryValue(summary, "best", name) : null;
  card.append(node("span", "v04-mini-value", "Последнее " + formatValue(latest) + " · лучший " + formatValue(best)));
  const svg = definition && trend.length ? renderMetricChart(name, trend, {mini: true}) : null;
  if (svg) card.append(svg);
  else card.append(node("span", "v04-chart-empty", UNAVAILABLE));
  host.append(card);
}

function activeRunText(summary) {
  if (!Object.prototype.hasOwnProperty.call(summary, "active_run")) return UNAVAILABLE;
  if (!summary.active_run) return "Нет активного запуска";
  const run = summary.active_run;
  const label = run.alias || run.run_id || run.id || UNAVAILABLE;
  const status = run.status || UNAVAILABLE;
  const progress = run.progress && typeof run.progress === "object" ? run.progress : {};
  const entries = Object.keys(progress).length ? Object.entries(progress) : Object.entries(run).filter(([key]) =>
    ["phase", "completed_repeats", "expected_repeats", "completed_cases", "total_cases", "current_case", "current_repeat"].includes(key));
  const details = entries.slice(0, 3).map(([key, value]) => {
    const safeKey = key.replace(/_/g, " ");
    return safeKey + ": " + (value === null || value === undefined ? UNAVAILABLE : String(value));
  });
  return label + " · " + status + (details.length ? " · " + details.join(" · ") : "");
}

function renderLabPageControls(host) {
  const controls = node("div", "v04-page-controls");
  if (labOffset > 0) {
    const previous = node("button", "button", "Предыдущие 100");
    previous.type = "button";
    previous.addEventListener("click", () => loadLabPage(Math.max(0, labOffset - LAB_PAGE_SIZE)));
    controls.append(previous);
  }
  if (labPage.next_offset !== null && labPage.next_offset !== undefined) {
    const next = node("button", "button", "Более лабораторий");
    next.type = "button";
    next.addEventListener("click", () => loadLabPage(Number(labPage.next_offset)));
    controls.append(next);
  }
  if (controls.childNodes.length) {
    controls.append(node("span", "v04-muted", "На странице " + labs.length +
      (labPage.total ? " · всего лабораторий: " + labPage.total : "")));
    host.append(controls);
  }
}

function renderLabs() {
  nav.replaceChildren();
  const selector = document.getElementById("top-lab-select");
  selector.replaceChildren();
  const overviewOption = node("option", "", "Обзор лабораторий");
  overviewOption.value = "";
  selector.append(overviewOption);
  const overview = node("button", "lab lab-overview" + (!selected ? " selected" : ""), "Обзор лабораторий");
  overview.type = "button";
  overview.addEventListener("click", showPortfolio);
  nav.append(overview);

  if (!labs.length) {
    nav.append(node("p", "empty", "Активных лабораторий пока нет."));
  }
  for (const lab of labs) {
    const id = labId(lab);
    const key = labKey(lab);
    const summary = summaryFor(lab);
    const {primary} = primaryAndGuards(summary, lab);
    const name = lab.goal || lab.name || (primary && primary.name) || lab.primary_metric || lab.metric || "Лаборатория";
    const option = node("option", "", name);
    option.value = key;
    selector.append(option);
    const button = node("button", "lab" + (key === selected ? " selected" : ""));
    button.type = "button";
    button.append(node("span", "lab-name", name));
    button.append(node("span", "lab-meta", [lab.status || lifecycleMode,
      "SQLite", primary && primary.name].filter(Boolean).join(" · ")));
    button.addEventListener("click", () => selectLab(key));
    nav.append(button);
  }
  renderLabPageControls(nav);
  selector.value = selected;
  applySearch(document.getElementById("lab-search").value);
}

function renderPortfolio() {
  detail.replaceChildren();
  detail.append(node("h2", "v04-page-title", "Активные лаборатории"));
  renderLabPageControls(detail);
  if (!labs.length) {
    const empty = labOffset > 0 ? "На этой странице лабораторий нет." : "Активных лабораторий пока нет.";
    detail.append(node("div", "empty", empty));
    applySearch(document.getElementById("lab-search").value);
    return;
  }
  for (const lab of labs) {
    const key = labKey(lab);
    const summary = summaryFor(lab);
    const {primary, guards} = primaryAndGuards(summary, lab);
    const trend = Array.isArray(summary.trend) ? summary.trend.slice(-5) : [];
    const card = node("button", "v04-card v04-lab-card");
    card.type = "button";
    card.addEventListener("click", () => selectLab(key));
    const heading = node("div", "v04-lab-card-head");
    heading.append(node("strong", "v04-lab-title", lab.goal || lab.name || UNAVAILABLE));
    heading.append(node("span", "v04-status", [lab.status || lifecycleMode,
      "SQLite"].join(" · ")));
    card.append(heading);
    card.append(node("p", "v04-muted", "Эпоха " + formatValue(summary.current_epoch_id ?? summary.current_epoch)));
    const metricGrid = node("div", "v04-mini-grid");
    if (primary) renderMiniMetric(metricGrid, summary, primary, trend, "Основная");
    else renderMiniMetric(metricGrid, summary, null, trend, "Основная");
    if (guards.length) {
      for (const guard of guards) renderMiniMetric(metricGrid, summary, guard, trend, "Защитная");
    } else {
      renderMiniMetric(metricGrid, summary, null, trend, "Защитная");
    }
    card.append(metricGrid);
    const progress = node("p", "v04-active-progress");
    progress.append(node("span", "v04-muted", "Текущий запуск · "));
    progress.append(node("span", "", activeRunText(summary)));
    card.append(progress);
    detail.append(card);
  }
  applySearch(document.getElementById("lab-search").value);
}

function renderTabs() {
  const tabs = node("div", "v04-tabs");
  tabs.setAttribute("role", "tablist");
  for (const [key, label] of [["overview", "Обзор"], ["runs", "Запуски"], ["evidence", "Свидетельства"]]) {
    const button = node("button", "v04-tab" + (activeTab === key ? " is-active" : ""), label);
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(activeTab === key));
    button.dataset.tab = key;
    button.addEventListener("click", () => { activeTab = key; renderLabDetail(); });
    tabs.append(button);
  }
  return tabs;
}

function formatDelta(baseVal, bestVal, direction = "max") {
  if (baseVal === null || baseVal === undefined || bestVal === null || bestVal === undefined) return null;
  const baseNum = Number(baseVal);
  const bestNum = Number(bestVal);
  if (!Number.isFinite(baseNum) || !Number.isFinite(bestNum)) return null;
  const delta = bestNum - baseNum;
  const formatted = Math.abs(delta) < 1e-6 ? "0" : (delta > 0 ? "+" : "") + Number(delta.toFixed(4));
  const isGood = direction === "min" ? delta < -1e-6 : delta > 1e-6;
  const isBad = direction === "min" ? delta > 1e-6 : delta < -1e-6;
  const statusClass = isGood ? "good" : (isBad ? "bad" : "neutral");
  return {text: formatted, class: statusClass, delta};
}

function renderOverview(summary, lab) {
  const host = node("div", "v04-tab-panel");
  host.setAttribute("role", "tabpanel");
  const {primary, guards} = primaryAndGuards(summary, lab);
  const definitions = [primary, ...guards].filter(Boolean);
  const metrics = section("Результаты", "Значения берутся из сохранённой сводки лаборатории.");
  if (!definitions.length) metrics.append(node("p", "v04-muted", UNAVAILABLE));
  const stat = summary && summary.statistic ? summary.statistic : null;
  for (const definition of definitions) {
    const row = node("div", "v04-summary-metric");
    row.append(node("strong", "", definition.name));
    row.append(node("span", "v04-muted", definition.role === "guard" ? "Защитная метрика" : "Основная метрика"));
    const baseVal = summaryValue(summary, "baseline", definition.name, stat);
    const bestVal = summaryValue(summary, "best", definition.name, stat);
    const latestVal = summaryValue(summary, "latest", definition.name, stat);
    row.append(metricCard("Baseline", formatValue(baseVal)));
    row.append(metricCard("Лучший результат", formatValue(bestVal)));
    row.append(metricCard("Последний запуск", formatValue(latestVal)));
    const deltaInfo = formatDelta(baseVal, bestVal, definition.direction);
    if (deltaInfo) {
      const deltaCard = metricCard("Дельта к Baseline", deltaInfo.text, deltaInfo.class);
      row.append(deltaCard);
    } else {
      row.append(metricCard("Дельта к Baseline", UNAVAILABLE));
    }
    metrics.append(row);
  }
  if (typeof summary.confirmed_cost_usd === "number" && summary.confirmed_cost_usd > 0) {
    metrics.append(node("p", "v04-muted", "Подтверждённый расход: $" + summary.confirmed_cost_usd.toFixed(2)));
  }
  host.append(metrics);

  const derivedCaseSignals = (typeof runPage !== "undefined" && runPage && Array.isArray(runPage.items))
    ? runPage.items
      .filter(r => Array.isArray(r.regressed_case_ids) && r.regressed_case_ids.length > 0)
      .slice(0, 3)
      .map(r => ({
        kind: "case_regression",
        message: "Регрессия кейсов в " + (r.alias || r.run_id) + ": " + r.regressed_case_ids.slice(0, 5).join(", "),
        action: r.reason || "Проверьте чекпоинты кейсов в сводке запуска.",
      }))
    : [];
  const signals = (Array.isArray(summary.signals) && summary.signals.length ? summary.signals : [])
    .concat(derivedCaseSignals)
;
  if (signals.length) {
    const diagSection = section("Сигналы и диагностика", "Диагностические сигналы и предупреждения.");
    for (const item of signals) {
      const row = node("div", "v04-run");
      const kind = item.kind || item.type || "signal";
      const isBad = kind === "error" || kind === "fail" || kind === "bad";
      row.append(node("span", "badge " + (isBad ? "bad" : "warn"), kind));
      const title = item.message || item.title || "";
      if (title) row.append(node("strong", "", title));
      const action = item.action || item.detail || "";
      if (action) row.append(node("span", "v04-muted", action));
      diagSection.append(row);
    }
    host.append(diagSection);
  }

  const progress = section("Текущий запуск", "Ход выполнения доступен только для чтения.");
  progress.append(node("p", "v04-active-progress", activeRunText(summary)));
  host.append(progress);
  return host;
}

function renderRunCharts() {
  const lab = selectedLab();
  const summary = summaryFor(lab);
  const definitions = metricDefinitions(summary, lab);
  const legacy = lab && lab.source === "legacy";
  const host = section("Метрики запусков", legacy
    ? "Графики строятся из сводки legacy-лаборатории; список можно листать назад."
    : "Графики показывают запуски выбранной эпохи; список можно листать назад.");
  if (!runMetrics.length) {
    host.append(node("p", "v04-muted", UNAVAILABLE));
    detail.append(host);
    return;
  }
  const baseline = summary.baseline || {};
  const baselineId = baseline.run_id || baseline.id || baseline.alias || "";
  const metrics = definitions.length ? definitions : [{name: lab && (lab.primary_metric || lab.metric) || "Метрика", role: "primary"}];
  let chartCount = 0;
  for (const definition of metrics) {
    const card = node("div", "v04-chart-card");
    card.append(node("h4", "", (definition.role === "guard" ? "Защитная метрика · " : "Основная метрика · ") + definition.name));
    const svg = renderMetricChart(definition.name, runMetrics, {baselineId, statistic: summary && summary.statistic});
    if (svg) {
      card.append(svg);
      chartCount++;
    } else {
      card.append(node("p", "v04-chart-empty", UNAVAILABLE));
    }
    host.append(card);
  }
  if (!chartCount) host.append(node("p", "v04-chart-legend", "Для этой эпохи значения метрик отсутствуют."));
  detail.append(host);
}

function findSqliteDiffArtifact(run) {
  if (!run || typeof evidence === "undefined" || !Array.isArray(evidence)) return null;
  const runId = String(run.run_id || run.id || "").toLowerCase();
  return evidence.find(item => {
    if (!item) return false;
    const itemRunId = String(item.run_id || "").toLowerCase();
    const name = String(item.name || "").toLowerCase();
    const key = String(item.evidence_key || "").toLowerCase();
    const matchesRun = (itemRunId && runId && itemRunId === runId) ||
      (runId && (name.includes(runId) || key.includes(runId)));
    const isDiff = name.includes("diff") || name.includes("patch") || key.includes("diff") || key.includes("patch");
    return matchesRun && isDiff;
  }) || null;
}

function renderRunList() {
  const lab = typeof selectedLab === "function" ? selectedLab() : null;
  const legacy = lab && lab.source === "legacy";
  const host = section("История запусков", legacy
    ? "Запуски legacy-лаборатории; список листается локально."
    : "Первые 20 запусков выбраны для текущей эпохи.");
  if (selectedEpoch || legacy) {
    const toolbar = node("div", "v04-run-toolbar");
    if (!legacy) {
      const label = node("label", "v04-field", "Эпоха");
      const select = document.createElement("select");
      select.setAttribute("aria-label", "Выбрать эпоху");
      const options = epochIds.length ? epochIds : [selectedEpoch];
      for (const epoch of options.filter(value => value !== null && value !== undefined)) {
        const option = node("option", "", "Эпоха " + epoch);
        option.value = String(epoch);
        select.append(option);
      }
      select.value = String(selectedEpoch || "");
      select.addEventListener("change", () => loadRunPage(Number(select.value), 0));
      label.append(select);
      toolbar.append(label);
    }
    if (runOffset > 0) {
      const newer = node("button", "button", "Более новые");
      newer.type = "button";
      newer.addEventListener("click", () => loadRunPage(legacy ? null : selectedEpoch, Math.max(0, runOffset - 20)));
      toolbar.append(newer);
    }
    let nextOffset = runPage.next_offset ?? runPage.nextOffset;
    if ((nextOffset === null || nextOffset === undefined) &&
        Number(runPage.total) > runOffset + runPage.items.length) {
      nextOffset = runOffset + runPage.items.length;
    }
    const numericNextOffset = Number(nextOffset);
    if (Number.isFinite(numericNextOffset) && numericNextOffset > runOffset) {
      const older = node("button", "button", "Более старые");
      older.type = "button";
      older.addEventListener("click", () => loadRunPage(legacy ? null : selectedEpoch, numericNextOffset));
      toolbar.append(older);
    }
    host.append(toolbar);
  }
  if (runPage.loading) {
    host.append(node("p", "v04-muted", "Загрузка запусков…"));
    return host;
  }
  if (runPage.error) {
    host.append(node("p", "v04-error", runPage.error));
    return host;
  }
  if (!selectedEpoch && !legacy) {
    host.append(node("p", "v04-muted", "Эпоха недоступна."));
    return host;
  }
  if (!runPage.items.length) {
    host.append(node("p", "v04-muted", legacy ? "История запусков пока пуста." : "Запусков в этой эпохе пока нет."));
    return host;
  }
  const unavail = typeof UNAVAILABLE !== "undefined" ? UNAVAILABLE : "Недоступно";
  for (const run of runPage.items) {
    const row = node("div", "v04-run");
    row.append(node("strong", "", run.alias || run.run_id || unavail));
    if (run.title) {
      row.append(node("span", "v04-run-title", run.title));
    }
    const statusText = (run.status || unavail) + (legacy
      ? " · файловая история" : " · эпоха " + (typeof formatValue === "function" ? formatValue(run.epoch_id) : run.epoch_id));
    row.append(node("span", "badge", statusText));
    if (run.reason) {
      row.append(node("span", "v04-muted v04-run-reason", run.reason));
    }
    if (Array.isArray(run.regressed_case_ids) && run.regressed_case_ids.length) {
      row.append(node("span", "badge bad", "Регрессия кейсов: " + run.regressed_case_ids.slice(0, 3).join(", ")));
    }
    row.append(node("span", "v04-muted", run.created_at || unavail));
    const button = node("button", "ghost", "Сводка");
    button.type = "button";
    button.addEventListener("click", () => showRun(run));
    row.append(button);
    if (!legacy && typeof findSqliteDiffArtifact === "function") {
      const diffArtifact = findSqliteDiffArtifact(run);
      if (diffArtifact) {
        const diffBtn = node("button", "ghost", "Показать diff");
        diffBtn.type = "button";
        diffBtn.addEventListener("click", () => loadEvidence(diffArtifact));
        row.append(diffBtn);
      }
    }
    host.append(row);
  }
  return host;
}

function renderRuns() {
  const host = node("div", "v04-tab-panel");
  host.setAttribute("role", "tabpanel");
  host.append(renderRunList());
  detail.append(host);
  if (!runPage.loading && !runPage.error && runPage.items.length) renderRunCharts();
}

function renderEvidence() {
  const legacy = selectedLab() && selectedLab().source === "legacy";
  const host = section("Свидетельства", legacy
    ? "Сведения и артефакты, возвращённые detail API legacy-лаборатории."
    : "Список содержит зарегистрированные артефакты лаборатории.");
  host.setAttribute("role", "tabpanel");
  if (!evidence.length) host.append(node("p", "v04-muted", UNAVAILABLE));
  for (const item of evidence) {
    const row = node("div", "v04-run");
    const nameStr = !legacy && item.evidence_key && item.name
      ? item.name + " (" + item.evidence_key + ")"
      : (item.name || item.title || item.evidence_key || item.artifact || String(item.artifact_id || UNAVAILABLE));
    row.append(node("span", "", nameStr));
    if (legacy) {
      row.append(node("span", "v04-muted", item.conclusion || item.observation || item.expected_effect || item.context || UNAVAILABLE));
    } else {
      row.append(node("span", "v04-muted", (item.media_type || UNAVAILABLE) + " · " + formatValue(item.size_bytes) + " bytes"));
      const button = node("button", "ghost", "Открыть свидетельство");
      button.type = "button";
      button.addEventListener("click", () => loadEvidence(item));
      row.append(button);
    }
    host.append(row);
  }
  detail.append(host);
}

function renderLabDetail() {
  const lab = selectedLab();
  const prevScrollTop = detail ? detail.scrollTop : 0;
  const main = typeof document !== "undefined" ? document.querySelector("main") : null;
  const prevMainScroll = main ? main.scrollTop : 0;
  detail.replaceChildren();
  if (!lab) { renderPortfolio(); return; }
  const summary = summaryFor(lab);
  const heading = node("section", "hero neutral v04-hero");
  heading.append(node("h2", "hero-title", lab.goal || lab.name || UNAVAILABLE));
  const verdict = node("p", "verdict", (lab.status || "активная") + " · эпоха " +
    formatValue(summary.current_epoch_id ?? summary.current_epoch));
  heading.append(verdict);
  if (summary.goal_status === "met") {
    heading.append(node("span", "badge good", "Цель достигнута"));
  } else if (summary.goal_status === "not_met") {
    heading.append(node("span", "badge bad", "Цель не достигнута"));
  } else if (summary.goal_status === "unconfirmed") {
    heading.append(node("span", "badge warn", "Цель: ожидание подтверждения"));
  }
  const {primary} = primaryAndGuards(summary, lab);
  heading.append(node("p", "v04-muted", primary ? "Основная метрика · " + primary.name : UNAVAILABLE));
  detail.append(heading, renderTabs());
  if (activeTab === "overview") detail.append(renderOverview(summary, lab));
  if (activeTab === "runs") {
    renderRuns();
    if (inspectedRun) {
      detail.append(renderRunInspectorCard(inspectedRun));
    }
  }
  if (activeTab === "evidence") renderEvidence();
  if (detail) detail.scrollTop = prevScrollTop;
  if (main) main.scrollTop = prevMainScroll;
}

function renderEvidenceDialog(item, value) {
  const trigger = typeof document !== "undefined" ? document.activeElement : null;
  const dialog = document.createElement("dialog");
  dialog.className = "v04-dialog";
  dialog.append(node("h3", "", (item && (item.name || item.title || item.evidence_key)) || "Evidence"));
  const strValue = String(value || "");
  const isDiff = Boolean(
    (item && /\.(diff|patch)$/i.test(item.name || item.evidence_key || "")) ||
    strValue.startsWith("diff --git") ||
    strValue.startsWith("--- ")
  );
  if (isDiff) {
    const pre = document.createElement("pre");
    pre.className = "v04-code diff";
    const lines = strValue.slice(0, 250000).split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      let cls = "";
      if (line.startsWith("+") && !line.startsWith("+++")) cls = "diff-add";
      else if (line.startsWith("-") && !line.startsWith("---")) cls = "diff-del";
      else if (line.startsWith("@@")) cls = "diff-hunk";
      const span = document.createElement("span");
      if (cls) span.className = cls;
      span.style.display = "block";
      span.textContent = line;
      pre.append(span);
    }
    dialog.append(pre);
  } else {
    dialog.append(node("pre", "v04-code", strValue.slice(0, 250000)));
  }
  const actions = node("div", "v04-actions");
  const copyBtn = node("button", "button", "Копировать");
  copyBtn.type = "button";
  copyBtn.addEventListener("click", () => {
    if (typeof navigator !== "undefined" && navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      navigator.clipboard.writeText(strValue).then(() => {
        copyBtn.textContent = "Скопировано";
        setTimeout(() => { copyBtn.textContent = "Копировать"; }, 1800);
      }).catch(() => {});
    }
  });
  const close = node("button", "button", "Закрыть");
  close.type = "button";
  close.addEventListener("click", () => dialog.close());
  actions.append(copyBtn, close);
  dialog.append(actions);
  document.body.append(dialog);
  dialog.addEventListener("close", () => {
    dialog.remove();
    if (trigger && typeof trigger.focus === "function") trigger.focus();
  }, {once: true});
  dialog.showModal();
}

async function loadEvidence(item) {
  try {
    const data = await request("/api/v1/labs/" + encodeURIComponent(selected) +
      "/evidence/" + encodeURIComponent(item.artifact_id));
    const binary = atob(data.content_base64 || "");
    const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
    let value;
    try { value = new TextDecoder("utf-8", {fatal: true}).decode(bytes); }
    catch (error) { value = "Base64 evidence: " + data.content_base64; }
    renderEvidenceDialog(item, value);
  } catch (error) { setMessage("Не удалось открыть evidence: " + error.message, "warn"); }
}

function renderRunInspectorCard(info) {
  const run = info && info.run ? info.run : (info || {});
  const data = info && info.data ? info.data : run;
  const unavail = typeof UNAVAILABLE !== "undefined" ? UNAVAILABLE : "Недоступно";
  const label = run.alias || run.title || run.run_id || data.run_id || unavail;
  const panel = section("Сводка запуска " + label,
    "Решение: " + (run.status || data.status || unavail) +
    (run.parent_run_id || data.parent_run_id ? " · parent: " + (run.parent_run_id || data.parent_run_id) : ""));
  panel.classList.add("v04-inspector");

  const closeBtn = node("button", "button ghost", "Закрыть сводку");
  closeBtn.type = "button";
  closeBtn.addEventListener("click", () => {
    inspectedRun = null;
    panel.remove();
  });
  panel.append(closeBtn);

  const hyp = (data && typeof data.hypothesis === "object" && data.hypothesis) ||
    (run && typeof run.hypothesis === "object" && run.hypothesis) || {};
  const title = data.title || run.title || hyp.title;
  if (title) {
    const titleP = node("p", "v04-hyp-title");
    titleP.append(node("strong", "", "Гипотеза: "), node("span", "", title));
    panel.append(titleP);
  }
  const observation = data.observation || run.observation || hyp.observation;
  if (observation) {
    const obsP = node("p", "v04-hyp-obs");
    obsP.append(node("strong", "", "Наблюдение: "), node("span", "", observation));
    panel.append(obsP);
  }
  const expected = data.expected_effect || run.expected_effect || hyp.expected_effect;
  if (expected) {
    const expP = node("p", "v04-hyp-exp");
    expP.append(node("strong", "", "Ожидаемый эффект: "), node("span", "", expected));
    panel.append(expP);
  }

  const reason = data.reason || run.reason;
  const statusLine = node("p", "v04-muted", "Статус: " + (run.status || data.status || unavail) +
    (reason ? " (" + reason + ")" : ""));
  panel.append(statusLine);

  const regressed = (Array.isArray(run.regressed_case_ids) && run.regressed_case_ids.length ? run.regressed_case_ids :
    (Array.isArray(data.regressed_case_ids) && data.regressed_case_ids.length ? data.regressed_case_ids : []));
  if (regressed.length) {
    panel.append(node("p", "v04-error", "Регрессировавшие кейсы: " + regressed.join(", ")));
  }

  const metricsObj = (data.metrics && typeof data.metrics === "object" ? data.metrics :
    (run.metrics && typeof run.metrics === "object" ? run.metrics : {}));
  const metricEntries = Object.entries(metricsObj);
  if (metricEntries.length) {
    const metricsGrid = node("div", "v04-mini-grid");
    for (const [mName, mVal] of metricEntries) {
      const mCard = node("div", "v04-metric");
      mCard.append(node("strong", "", mName));
      if (mVal && typeof mVal === "object") {
        if (mVal.decision_worst !== undefined) {
          mCard.append(node("span", "v04-muted", "decision_worst: " + (typeof formatValue === "function" ? formatValue(mVal.decision_worst) : mVal.decision_worst)));
        }
        if (mVal.mean !== undefined || mVal.decision_mean !== undefined) {
          mCard.append(node("span", "v04-muted", "mean: " + (typeof formatValue === "function" ? formatValue(mVal.mean ?? mVal.decision_mean) : (mVal.mean ?? mVal.decision_mean))));
        }
        if (mVal.median !== undefined) {
          mCard.append(node("span", "v04-muted", "median: " + (typeof formatValue === "function" ? formatValue(mVal.median) : mVal.median)));
        }
        if (mVal.completed_repeats !== undefined) {
          mCard.append(node("span", "v04-muted", "completed_repeats: " + (typeof formatValue === "function" ? formatValue(mVal.completed_repeats) : mVal.completed_repeats)));
        }
      } else {
        mCard.append(node("span", "v04-muted", "значение: " + (typeof formatValue === "function" ? formatValue(mVal) : mVal)));
      }
      metricsGrid.append(mCard);
    }
    panel.append(metricsGrid);
  }

  const details = document.createElement("details");
  details.className = "v04-raw-details";
  const summaryEl = document.createElement("summary");
  summaryEl.textContent = "Показать сырой JSON";
  summaryEl.style.cursor = "pointer";
  details.append(summaryEl);
  details.append(node("pre", "v04-code", JSON.stringify(data, null, 2)));
  panel.append(details);

  return panel;
}

async function showRun(run) {
  try {
    const lab = selectedLab();
    const cached = runMetrics.find(item => String(item.run_id) === String(run.run_id));
    const data = cached && cached.data ? cached.data : await request("/api/v1/labs/" + encodeURIComponent(selected) +
      "/runs/" + encodeURIComponent(run.run_id));
    inspectedRun = {run, data};
    const existing = detail.querySelector(".v04-inspector");
    const card = renderRunInspectorCard(inspectedRun);
    if (existing) existing.replaceWith(card);
    else detail.append(card);
    card.scrollIntoView?.({behavior: "smooth", block: "nearest"});
  } catch (error) { setMessage(error.message, "warn"); }
}

async function loadEpochIds(base, currentEpoch) {
  const found = new Set(currentEpoch === null ? [] : [Number(currentEpoch)]);
  let offset = 0;
  while (true) {
    const page = await request(base + "/runs?limit=100&offset=" + offset);
    for (const run of Array.isArray(page.items) ? page.items : []) {
      const epoch = Number(run.epoch_id);
      if (Number.isInteger(epoch) && epoch > 0) found.add(epoch);
    }
    const nextOffset = page.next_offset ?? page.nextOffset ??
      (page.next_cursor ? offset + (Array.isArray(page.items) ? page.items.length : 0) : null);
    if (nextOffset === null || nextOffset === undefined || Number(nextOffset) <= offset) break;
    offset = Number(nextOffset);
  }
  return Array.from(found).sort((left, right) => right - left);
}

async function loadRunPage(epoch, offset = 0) {
  const lab = selectedLab();
  if (!lab) return;
  const id = labId(lab);
  selectedEpoch = Number(epoch);
  runOffset = Number(offset) || 0;
  const requestedEpoch = selectedEpoch;
  const requestedOffset = runOffset;
  const version = detailRequest;
  runPage = {items: [], next_offset: null, offset: requestedOffset, loading: true, error: ""};
  runMetrics = [];
  renderLabDetail();
  try {
    const base = "/api/v1/labs/" + encodeURIComponent(id);
    const page = await request(base + "/runs?limit=20&epoch_id=" + encodeURIComponent(selectedEpoch) +
      "&offset=" + encodeURIComponent(runOffset));
    const items = Array.isArray(page.items) ? page.items : [];
    const metrics = items;
    if (selected !== id || version !== detailRequest || selectedEpoch !== requestedEpoch || runOffset !== requestedOffset) return;
    runPage = {...page, items, offset: runOffset, loading: false, error: ""};
    runMetrics = metrics;
    const observed = items.map(run => Number(run.epoch_id)).filter(value => Number.isInteger(value) && value > 0);
    const currentEpoch = summaryFor(lab).current_epoch_id ?? summaryFor(lab).current_epoch;
    epochIds = Array.from(new Set([...epochIds, ...observed, Number(currentEpoch)]))
      .filter(value => Number.isInteger(value) && value > 0).sort((left, right) => right - left);
    renderLabDetail();
  } catch (error) {
    if (selected !== id || version !== detailRequest || selectedEpoch !== requestedEpoch || runOffset !== requestedOffset) return;
    runPage = {items: [], next_offset: null, offset: requestedOffset, loading: false, error: error.message};
    runMetrics = [];
    renderLabDetail();
    setMessage(error.message, "warn");
  }
}

async function selectLab(id) {
  const lab = labs.find(item => labKey(item) === String(id) || labId(item) === String(id));
  detailRequest++;
  selected = lab ? labKey(lab) : String(id || "");
  activeTab = "overview";
  inspectedRun = null;
  selectedEpoch = null;
  epochIds = [];
  runOffset = 0;
  runPage = {items: [], next_offset: null, offset: 0, loading: false, error: ""};
  runMetrics = [];
  evidence = [];
  localStorage.setItem(selectedKey, selected);
  renderLabs();
  await reloadLab();
}

function showPortfolio() {
  detailRequest++;
  selected = "";
  inspectedRun = null;
  localStorage.removeItem(selectedKey);
  renderLabs();
  renderPortfolio();
}

async function reloadLab() {
  const lab = selectedLab();
  if (!lab) { renderPortfolio(); return; }
  const requestId = ++detailRequest;
  const id = labId(lab);
  const summary = summaryFor(lab);
  const hasExistingRuns = Boolean(runPage && Array.isArray(runPage.items) && runPage.items.length > 0);
  const defaultEpoch = Number(summary.current_epoch_id ?? summary.current_epoch) || null;
  selectedEpoch = (hasExistingRuns && Number.isInteger(selectedEpoch) && selectedEpoch > 0)
    ? selectedEpoch
    : defaultEpoch;
  const targetOffset = hasExistingRuns ? (Number(runOffset) || 0) : 0;
  if (!hasExistingRuns) {
    runPage = {items: [], next_offset: null, offset: 0, loading: Boolean(selectedEpoch), error: ""};
    runMetrics = [];
    evidence = [];
    renderLabDetail();
  }
  try {
    const base = "/api/v1/labs/" + encodeURIComponent(id);
    const runsRequest = selectedEpoch
      ? request(base + "/runs?limit=20&epoch_id=" + encodeURIComponent(selectedEpoch) + "&offset=" + encodeURIComponent(targetOffset))
      : Promise.resolve({items: [], next_offset: null});
    const [page, artifactPage] = await Promise.all([
      runsRequest,
      request(base + "/evidence?limit=20"),
    ]);
    if (selected !== id || requestId !== detailRequest) return;
    runPage = {...page, items: Array.isArray(page.items) ? page.items : [], offset: targetOffset, loading: false, error: ""};
    evidence = Array.isArray(artifactPage.items) ? artifactPage.items : [];
    const initialEpochs = runPage.items.map(run => Number(run.epoch_id)).filter(value => Number.isInteger(value) && value > 0);
    epochIds = Array.from(new Set([Number(selectedEpoch), ...initialEpochs]))
      .filter(value => Number.isInteger(value) && value > 0).sort((left, right) => right - left);
    renderLabDetail();
    runMetrics = selectedEpoch ? runPage.items : [];
    if (selected === id && requestId === detailRequest) renderLabDetail();
    if (selectedEpoch) {
      loadEpochIds(base, selectedEpoch).then(found => {
        if (selected !== id || requestId !== detailRequest) return;
        epochIds = found;
        if (typeof activeTab !== "undefined" && activeTab === "runs") renderLabDetail();
      }).catch(() => {});
    }
  } catch (error) {
    if (selected === id && requestId === detailRequest) {
      runPage = {items: [], next_offset: null, offset: targetOffset, loading: false, error: error.message};
      runMetrics = [];
      evidence = [];
      renderLabDetail();
      setMessage(error.message, "warn");
    }
  }
}

async function loadLabPage(offset) {
  try { await reloadLabs({offset, clearSelection: true}); }
  catch (error) { /* reloadLabs shows the error and retains the current page */ }
}

async function reloadLabs(options = {}) {
  const requestedOffset = Math.max(0, Number(options.offset === undefined ? labOffset : options.offset) || 0);
  const sourcePageSize = Math.floor(LAB_PAGE_SIZE / 2);
  const pageNumber = Math.floor(requestedOffset / LAB_PAGE_SIZE);
  const offset = pageNumber * LAB_PAGE_SIZE;
  const sourceOffset = pageNumber * sourcePageSize;
  const requestId = ++labRequestVersion;
  const path = "/api/v1/labs?limit=" + sourcePageSize + "&lifecycle=" + lifecycleMode + "&offset=" + sourceOffset;
  const [sqliteResult] = await Promise.allSettled([request(path)]);
  if (requestId !== labRequestVersion) return;

  const errors = [];
  let sqliteTotal = 0;
  let sqliteNextOffset = null;
  if (sqliteResult.status === "fulfilled") {
    const page = sqliteResult.value || {};
    const items = Array.isArray(page.items) ? page.items : [];
    sqliteLabs = items.map(item => ({...item, source: "sqlite"}));
    sqliteTotal = Number.isFinite(Number(page.total)) ? Number(page.total) : items.length;
    const explicitNext = page.next_offset ?? page.nextOffset;
    sqliteNextOffset = explicitNext === undefined || explicitNext === null ? null : Number(explicitNext);
    if (sqliteNextOffset === null && sqliteTotal > sourceOffset + items.length) {
      sqliteNextOffset = sourceOffset + (Number(page.limit) || sourcePageSize);
    }
    if (!Number.isFinite(sqliteNextOffset)) sqliteNextOffset = null;
  } else {
    sqliteLabs = [];
    errors.push("SQLite API: " + (sqliteResult.reason && sqliteResult.reason.message || "недоступен"));
  }


  labOffset = offset;
  labs = [...sqliteLabs];
  const total = sqliteTotal;
  const sqliteHasNext = Number.isFinite(sqliteNextOffset) && sqliteNextOffset > sourceOffset;
  labPage = {offset: labOffset, total,
    next_offset: sqliteHasNext ? labOffset + LAB_PAGE_SIZE : null};
  sourcesUnavailable = errors.length >= 1;
  sourceWarning = errors.length
    ? (sourcesUnavailable ? "Источники лабораторий недоступны: " : "Часть источников недоступна: ") + errors.join(" · ")
    : "";
  if (sourceWarning) setMessage(sourceWarning, "warn");
  else setMessage("", "info");

  if (options.clearSelection || !labs.some(item => labKey(item) === selected)) {
    selected = "";
    localStorage.removeItem(selectedKey);
    detailRequest++;
  }
  renderLabs();
  if (selected) await reloadLab();
  else renderPortfolio();
}

function markLive() {
  if (sourcesUnavailable) {
    updated.classList.remove("is-live");
    updated.classList.add("is-down");
    modeText.textContent = "Нет связи";
    setMessage(sourceWarning || "Источники лабораторий недоступны.", "bad");
    return;
  }
  updated.classList.add("is-live");
  updated.classList.remove("is-down");
  modeText.textContent = "В эфире";
  setMessage(sourceWarning, sourceWarning ? "warn" : "info");
}

async function refreshRevision() {
  if (refreshing) return;
  refreshing = true;
  try {
    const headers = revision === null ? {} : {"If-None-Match": '"' + revision + '"'};
    const response = await fetch("/api/v1/revision", {headers, credentials: "same-origin", cache: "no-store"});
    if (response.status !== 304) {
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error("revision");
      revision = String(payload.data.revision);
    }
  } catch (error) {
    // The next lab list request reports an unavailable API.
  }
  try {
    await reloadLabs();
    markLive();
  } catch (error) {
    updated.classList.add("is-down");
    modeText.textContent = "Нет связи";
    setMessage("Нет связи с локальным API. Панель повторит проверку автоматически.", "bad");
  } finally { refreshing = false; }
}

function scheduleRefresh() {
  clearTimeout(timer);
  timer = setTimeout(async () => {
    await refreshRevision();
    scheduleRefresh();
  }, document.hidden ? 20000 : 10000);
}

function initSSE() {
  if (typeof EventSource === "undefined") return;
  events = new EventSource("/api/events");
  events.addEventListener("update", () => refreshRevision());
  events.onerror = () => { /* revision polling remains active as fallback */ };
}

function applySearch(term) {
  const query = String(term || "").trim().toLocaleLowerCase("ru");
  for (const button of nav.querySelectorAll("button.lab")) {
    button.hidden = !button.textContent.toLocaleLowerCase("ru").includes(query);
  }
  for (const card of detail.querySelectorAll(".v04-lab-card")) {
    card.hidden = !card.textContent.toLocaleLowerCase("ru").includes(query);
  }
}

document.getElementById("lab-search").addEventListener("input", event => applySearch(event.target.value));
document.getElementById("top-lab-select").addEventListener("change", event => {
  if (event.target.value) selectLab(event.target.value);
  else showPortfolio();
});
document.getElementById("theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark";
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  localStorage.setItem("autoresearch:theme", document.documentElement.dataset.theme);
});
document.getElementById("nav-toggle").addEventListener("click", () => {
  const open = document.body.classList.toggle("nav-open");
  document.getElementById("nav-toggle").setAttribute("aria-expanded", String(open));
  document.getElementById("backdrop").hidden = !open;
});
document.getElementById("backdrop").addEventListener("click", () => {
  document.body.classList.remove("nav-open");
  document.getElementById("backdrop").hidden = true;
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    document.body.classList.remove("nav-open");
    const backdrop = document.getElementById("backdrop");
    if (backdrop) backdrop.hidden = true;
    const navToggle = document.getElementById("nav-toggle");
    if (navToggle) navToggle.setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshRevision(); });

async function start() {
  try {
    await refreshRevision();
    initSSE();
    scheduleRefresh();
  } catch (error) {
    setMessage(error.message, "warn");
    detail.replaceChildren(node("div", "empty", error.message));
  }
}

start();
