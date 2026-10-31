const boot = JSON.parse(document.getElementById("snapshot").textContent);
const POLL_ACTIVE_MS = 2000;
const POLL_IDLE_MS = 10000;
const SVG_NS = "http://www.w3.org/2000/svg";

let summary = boot.summary || {projects: [], archived: []};
let details = boot.details || {};
let selected = localStorage.getItem("autoresearch:selected:" + boot.root_id) || boot.selected || "";
let detailVersion = "";
let model = null;
let navFilter = "all";
let hypFilter = "all";
let newestFirst = true;
let chartGuard = "";
let acceptedOnly = false;
let historyQuery = "";
let historyLimit = 6;
let historyOpen = false;
let showEmptyArchive = false;
let hotId = "";
let timer = 0;
let sseSource = null;
let sseConnected = false;
const openHyp = new Set();

const nav = document.getElementById("navigation");
const detail = document.getElementById("detail");
const updated = document.getElementById("updated");
const modeText = document.getElementById("mode-text");
const main = document.getElementById("main");

function plural(count, one, few, many) {
  const value = Math.abs(Number(count) || 0);
  const tens = value % 10;
  const hundreds = value % 100;
  if (tens === 1 && hundreds !== 11) return one;
  if (tens >= 2 && tens <= 4 && (hundreds < 12 || hundreds > 14)) return few;
  return many;
}
function fmtScore(value, asPercent) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  if (asPercent) {
    const percent = number * 100;
    const digits = Math.abs(percent) >= 10 ? 1 : Math.abs(percent) >= 1 ? 2 : 3;
    return percent.toLocaleString("ru-RU", {minimumFractionDigits: digits, maximumFractionDigits: digits}) + "%";
  }
  return number.toLocaleString("ru-RU", {maximumSignificantDigits: 4});
}
function fmtDelta(value, asPercent) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  if (asPercent) return sign + (number * 100).toLocaleString("ru-RU", {maximumFractionDigits: 1}) + " п.п.";
  return sign + number.toLocaleString("ru-RU", {maximumSignificantDigits: 4});
}
function fmtMoney(value) {
  const number = Number(value || 0);
  if (number > 0 && number < 0.01) return "< $0,01";
  return "$" + number.toLocaleString("ru-RU", {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function fmtDuration(seconds) {
  const total = Math.round(Number(seconds) || 0);
  if (total < 60) return total + " с";
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return minutes + " мин";
  return Math.floor(minutes / 60) + " ч " + (minutes % 60) + " мин";
}
function prettyTitle(title, id) {
  const raw = String(title || id || "Гипотеза");
  const match = raw.match(/^H(\d+)_(.+)$/i);
  if (match) return "H" + Number(match[1]) + " · " + match[2].split("_").join(" ");
  if (String(id || "").startsWith("baseline") && raw === id) return "База";
  return raw.split("_").join(" ");
}
function axisLabel(row) {
  if (String(row.id).startsWith("baseline")) return "База";
  const match = String(row.id).match(/trial-(\d+)/);
  return match ? "H" + Number(match[1]) : String(row.index + 1);
}
function make(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function svgEl(name) {
  return document.createElementNS(SVG_NS, name);
}
function icon(kind) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", "ico");
  svg.setAttribute("aria-hidden", "true");
  const path = svgEl("path");
  const shapes = {
    check: "M3 8.5 6.2 11.5 13 4.5",
    x: "M4 4l8 8M12 4l-8 8",
    alert: "M8 2.8 14.2 13.5H1.8L8 2.8z",
    dot: "M8 5.2a2.8 2.8 0 1 1 0 5.6 2.8 2.8 0 0 1 0-5.6z",
  };
  path.setAttribute("d", shapes[kind] || shapes.dot);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "1.6");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  return svg;
}
function badge(kind, label) {
  const node = make("span", "badge " + (kind || ""), "");
  const mark = kind === "accepted" || kind === "good" || kind === "ready" || kind === "completed" ? "check"
    : kind === "failed" || kind === "bad" || kind === "config" ? "x"
    : kind === "warn" || kind === "incomparable" || kind === "running" ? "alert" : "dot";
  node.append(icon(mark), document.createTextNode(label || "—"));
  return node;
}

function activeLabs() {
  return (summary.projects || []).flatMap(project => (project.labs || []).map(lab => ({project, lab})));
}
function archivedLabs() {
  return (summary.archived || []).filter(item => item.project_id && item.id).map(item => ({
    project: {id: item.project_id, name: item.project || ""},
    lab: item,
  }));
}
function allLabs() {
  return activeLabs().concat(archivedLabs());
}
function firstActive() {
  return activeLabs()[0] || null;
}
function keyOf(project, lab) { return (lab.root_id || "") + "::" + project.id + "::" + lab.id; }
function groupOf(lab) {
  if (lab.status === "running" || lab.current_run) return "active";
  if (lab.status === "completed") return "done";
  return "ready";
}
function sparkline(values) {
  const points = (values || []).map((value, index) => ({index, value})).filter(point => point.value !== null && point.value !== undefined && Number.isFinite(Number(point.value)));
  if (points.length < 2) return null;
  const nums = points.map(point => Number(point.value));
  const low = Math.min(...nums);
  const high = Math.max(...nums);
  const width = 86;
  const height = 22;
  const svg = svgEl("svg");
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.setAttribute("class", "spark");
  svg.setAttribute("width", "86");
  svg.setAttribute("height", "22");
  const line = svgEl("polyline");
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "var(--accent)");
  line.setAttribute("stroke-width", "1.6");
  line.setAttribute("points", points.map((point, i) => {
    const x = points.length === 1 ? width / 2 : (i / ((points.length - 1) || 1)) * (width - 2) + 1;
    const y = high === low ? height / 2 : 3 + (1 - (Number(point.value) - low) / (high - low)) * (height - 6);
    return x.toFixed(1) + "," + y.toFixed(1);
  }).join(" "));
  svg.append(line);
  return svg;
}

function updateTopLabSelect() {
  const sel = document.getElementById("top-lab-select");
  if (!sel) return;
  sel.replaceChildren();
  const all = allLabs();
  if (!all.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Нет лабораторий";
    sel.append(opt);
    return;
  }
  for (const item of all) {
    const key = keyOf(item.project, item.lab);
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = (item.project.name || "Проект") + " / " + (item.lab.short_label || item.lab.name);
    if (key === selected) opt.selected = true;
    sel.append(opt);
  }
  sel.value = selected;
}

function renderFilters() {
  const host = document.getElementById("status-filters");
  host.replaceChildren();
  const archivedCount = (summary.archived || []).length;
  const filters = [["all", "Все"], ["active", "В работе"], ["ready", "Готовы"], ["done", "Завершены"],
    ["archived", "Архивные" + (archivedCount ? " (" + archivedCount + ")" : "")]];
  for (const [id, label] of filters) {
    const button = make("button", navFilter === id ? "is-on" : "", label);
    button.type = "button";
    button.addEventListener("click", () => { navFilter = id; renderFilters(); renderNav(); });
    host.append(button);
  }
}
function labActions(buttons) {
  const column = make("div", "lab-actions");
  for (const [className, label, handler] of buttons) {
    const button = make("button", className, label);
    button.type = "button";
    button.addEventListener("click", handler);
    column.append(button);
  }
  return column;
}
function renderLabCard(project, lab, buttons) {
  const button = make("button", "lab" + (keyOf(project, lab) === selected ? " selected" : ""));
  button.type = "button";
  button.dataset.key = keyOf(project, lab);
  button.addEventListener("click", () => select(project.id, lab.id));
  const top = make("div", "lab-top");
  top.append(make("span", "lab-name", lab.short_label || lab.name));
  top.append(badge(lab.status, lab.stage || lab.status));
  button.append(top);
  const meta = [lab.metric ? fmtScore(lab.best_score, metricPercent(lab)) : null,
    (lab.experiments || 0) + " " + plural(lab.experiments || 0, "опыт", "опыта", "опытов"),
    fmtMoney(lab.cost_usd)].filter(Boolean).join(" · ");
  button.append(make("span", "lab-meta", meta));
  if (lab.latest_trial_label) button.append(make("span", "lab-meta", "последний опыт: " + String(lab.latest_trial_label).toLowerCase()));
  if (lab.response_repro && Number.isFinite(lab.response_repro.total)) {
    const matched = lab.response_repro.total - (lab.response_repro.mismatched || 0);
    button.append(make("span", "lab-meta", "ответы: " + matched + " из " + lab.response_repro.total));
  }
  const spark = sparkline(lab.sparkline);
  if (spark) button.append(spark);
  const row = make("div", "lab-row");
  row.append(button);
  if (boot.live && buttons.length) row.append(labActions(buttons));
  nav.append(row);
}
function renderArchivedNav(query) {
  const totals = summary.archived_totals || {};
  const all = summary.archived || [];
  const hidden = all.filter(item => item.hidden).length;
  nav.append(make("p", "group-label", (totals.requests || 0) + " запросов · " + fmtMoney(totals.cost_usd || 0)));
  const groups = new Map();
  for (const item of all) {
    if (!showEmptyArchive && item.hidden) continue;
    const blob = [item.short_label, item.name, item.project, item.metric, item.stage].join(" ").toLowerCase();
    if (query && !blob.includes(query)) continue;
    const name = item.project || "Проект";
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(item);
  }
  for (const [name, labs] of groups) {
    nav.append(make("div", "project-title", name));
    for (const lab of labs) {
      const project = {id: lab.project_id, name};
      const buttons = [];
      if (lab.project_id && lab.id) {
        buttons.push(["lab-archive", "Вернуть", () => restoreLab(project.id, lab.id, lab.root_id)]);
        buttons.push(["lab-delete", "Удалить", () => deleteLab(project.id, lab.id, lab.root_id)]);
      }
      renderLabCard(project, lab, buttons);
    }
  }
  if (!groups.size) nav.append(make("p", "empty", query ? "Ничего не найдено" : "Архив пуст"));
  if (hidden) {
    const toggle = make("button", "ghost", showEmptyArchive ? "Скрыть пустые" : "Показать пустые (" + hidden + ")");
    toggle.type = "button";
    toggle.addEventListener("click", () => { showEmptyArchive = !showEmptyArchive; renderNav(); });
    nav.append(toggle);
  }
}
function renderNav() {
  const query = document.getElementById("lab-search").value.trim().toLowerCase();
  nav.replaceChildren();
  updateTopLabSelect();
  if (navFilter === "archived") {
    renderArchivedNav(query);
    return;
  }
  const groups = [["active", "Сейчас идут"], ["ready", "Готовы к опытам"], ["done", "Завершены"]];
  for (const project of summary.projects || []) {
    const labs = (project.labs || []).filter(lab => {
      const blob = (lab.short_label + " " + lab.name + " " + lab.metric + " " + lab.stage).toLowerCase();
      if (query && !blob.includes(query)) return false;
      return navFilter === "all" || groupOf(lab) === navFilter;
    });
    if (!labs.length) continue;
    nav.append(make("div", "project-title", project.name));
    for (const [id, label] of groups) {
      const bucket = labs.filter(lab => groupOf(lab) === id);
      if (!bucket.length) continue;
      if (navFilter === "all") nav.append(make("div", "group-label", label));
      for (const lab of bucket) {
        renderLabCard(project, lab, [
          ["lab-archive", "Архивировать", () => archiveLab(project.id, lab.id, lab.root_id)],
          ["lab-delete", "Удалить", () => deleteLab(project.id, lab.id, lab.root_id)],
        ]);
      }
    }
  }
  if (!nav.childElementCount) nav.append(make("p", "empty", "Ничего не найдено"));
}
function metricPercent(lab) {
  const name = String(lab.metric || "");
  if (/error|latency|duration|cost|time/i.test(name)) return false;
  const value = Number(lab.best_score);
  if (Number.isFinite(value) && (value < 0 || value > 1)) return false;
  return /f1|accuracy|precision|recall|rate|score|validity/i.test(name) || (Number.isFinite(value) && value >= 0 && value <= 1);
}

function primaryMeta() {
  return (model && model.metrics && model.metrics.primary) || {label: "Метрика", as_percent: false, direction: "max"};
}

function renderProblemsInto(host, problems) {
  host.replaceChildren();
  for (const problem of problems || []) {
    if (problem.level === "info") {
      const note = make("details", "footnote");
      note.append(make("summary", "", problem.title || "Справка"));
      note.append(make("p", "", problem.text || ""));
      if (problem.detail) note.append(make("p", "muted", problem.detail));
      host.append(note);
      continue;
    }
    const banner = make("section", "banner " + (problem.level || "warn"));
    banner.append(make("strong", "", problem.title || "Замечание"));
    banner.append(make("p", "", problem.text || ""));
    if (problem.detail) {
      const more = make("details");
      more.append(make("summary", "", "Технические подробности"));
      more.append(make("p", "", problem.detail));
      banner.append(more);
    }
    host.append(banner);
  }
}

function renderDetail(next) {
  model = next;
  const lab = next.lab || {};
  const meta = primaryMeta();
  const verdict = next.verdict || {level: "neutral", text: ""};
  const project = allLabs().find(item => keyOf(item.project, item.lab) === selected);

  const crumbs = document.getElementById("crumbs");
  if (crumbs) {
    crumbs.textContent = project ? project.project.name + " / " + (project.lab.short_label || project.lab.name) : "Лаборатория";
  }
  document.title = (project ? project.project.name : "AutoResearch") + " · AutoResearch";

  const currentKey = detail.dataset.selectedKey;
  const isSameLab = (currentKey === selected && detail.querySelector(".hero"));

  if (isSameLab) {
    updateDetailInPlace(next, meta);
    return;
  }

  const scroll = main.scrollTop;
  detail.dataset.selectedKey = selected;
  detail.replaceChildren();

  const hero = make("section", "hero " + (verdict.level || "neutral"));
  hero.id = "detail-hero";
  const heading = make("div", "heading");
  const title = make("div");
  title.append(make("h2", "hero-title", lab.name || "Лаборатория"));
  title.append(make("p", "verdict", verdict.text || "Нет резюме."));
  if (lab.archived) title.append(make("p", "muted", "В архиве. Просмотр без изменений."));
  if (lab.latest_trial_label) title.append(make("p", "muted hero-trial", "последний опыт: " + String(lab.latest_trial_label).toLowerCase()));
  heading.append(title, badge(lab.status, lab.stage || lab.status));
  hero.append(heading);
  if (next.current) {
    const current = make("div", "current");
    current.append(make("strong", "", "Сейчас выполняется " + prettyTitle(next.current.title, next.current.id)));
    current.append(make("div", "muted", "Текущий повтор: " + (lab.current_repeat || "подготовка") + " · завершено повторов: " + (lab.repeats_completed ?? "—")));
    hero.append(current);
  }
  detail.append(hero);

  const compHost = make("div");
  compHost.id = "comparison-host";
  if (lab.probe_comparison) compHost.append(renderComparisonPanel(lab.probe_comparison));
  detail.append(compHost);

  const probHost = make("div");
  probHost.id = "problems-host";
  renderProblemsInto(probHost, next.problems || []);
  detail.append(probHost);

  const kpisHost = make("div");
  kpisHost.id = "kpis-host";
  kpisHost.append(renderHeroKpis(next, meta));
  detail.append(kpisHost);

  detail.append(renderChart(next, meta));
  detail.append(renderHypotheses(next, meta));
  detail.append(renderHistory(next));
  detail.append(renderConditions(next));
  main.scrollTop = scroll;
}

function updateDetailInPlace(next, meta) {
  const lab = next.lab || {};
  const verdict = next.verdict || {level: "neutral", text: ""};

  const hero = document.getElementById("detail-hero");
  if (hero) {
    hero.className = "hero " + (verdict.level || "neutral");
    const titleEl = hero.querySelector(".hero-title");
    if (titleEl) titleEl.textContent = lab.name || "Лаборатория";
    const verdEl = hero.querySelector(".verdict");
    if (verdEl) verdEl.textContent = verdict.text || "Нет резюме.";
    const trialEl = hero.querySelector(".hero-trial");
    if (trialEl && lab.latest_trial_label) trialEl.textContent = "последний опыт: " + String(lab.latest_trial_label).toLowerCase();

    const head = hero.querySelector(".heading");
    if (head) {
      const oldBadge = head.querySelector(".badge");
      if (oldBadge) oldBadge.replaceWith(badge(lab.status, lab.stage || lab.status));
    }

    let currentBanner = hero.querySelector(".current");
    if (next.current) {
      if (!currentBanner) {
        currentBanner = make("div", "current");
        hero.append(currentBanner);
      }
      currentBanner.replaceChildren();
      currentBanner.append(make("strong", "", "Сейчас выполняется " + prettyTitle(next.current.title, next.current.id)));
      currentBanner.append(make("div", "muted", "Текущий повтор: " + (lab.current_repeat || "подготовка") + " · завершено повторов: " + (lab.repeats_completed ?? "—")));
    } else if (currentBanner) {
      currentBanner.remove();
    }
  }

  const compHost = document.getElementById("comparison-host");
  if (compHost) {
    compHost.replaceChildren();
    if (lab.probe_comparison) compHost.append(renderComparisonPanel(lab.probe_comparison));
  }

  const probHost = document.getElementById("problems-host");
  if (probHost) {
    renderProblemsInto(probHost, next.problems || []);
  }

  const kpisHost = document.getElementById("kpis-host");
  if (kpisHost) {
    kpisHost.replaceChildren(renderHeroKpis(next, meta));
  }

  const chartHost = document.getElementById("chart-host");
  if (chartHost) {
    drawChart(chartHost, next.series || [], meta);
  }

  updateHypothesesInPlace(next, meta);
}

function renderComparisonPanel(data) {
  const section = make("section", "provider-comparison");
  const metric = String(data.primary || "score");
  const asPercent = metric.toLowerCase().includes("f1") || metric.toLowerCase().includes("accuracy");
  for (const group of data.groups || []) {
    const head = make("div", "section-head");
    head.append(make("h3", "", "Сравнение · " + String(group.label || group.id || "группа")));
    if (group.baseline_label) head.append(badge("accepted", String(group.baseline_label)));
    section.append(head);
    section.append(make("p", "muted", "Метрика: " + metric + " · худший повтор · данные из probe-прогонов."));
    const chart = make("div", "provider-chart");
    chart.append(make("p", "provider-chart-title", "Худшее значение по метке"));
    for (const entry of group.entries || []) {
      const line = make("div", "provider-chart-line");
      line.append(make("span", "provider-chart-label", String(entry.label || "—")));
      const track = make("div", "provider-chart-track");
      const fill = document.createElement("span");
      fill.className = entry.label === group.baseline_label ? "selected" : "";
      const worst = Number(entry.worst);
      fill.style.width = Number.isFinite(worst) ? Math.max(0, Math.min(100, worst * (asPercent ? 100 : 1))) + "%" : "0%";
      track.append(fill);
      line.append(track, make("strong", "provider-chart-value", fmtScore(entry.worst, asPercent)));
      chart.append(line);
    }
    section.append(chart);
    const tableWrap = make("div", "provider-table-wrap");
    const table = make("table", "provider-table");
    const columns = ["Метка", "Худший", "Средний", "Разброс", "Запросы", "Стоимость"];
    const thead = document.createElement("thead");
    const heading = document.createElement("tr");
    columns.forEach(label => heading.append(make("th", "", label)));
    thead.append(heading);
    const tbody = document.createElement("tbody");
    for (const entry of group.entries || []) {
      const row = document.createElement("tr");
      if (entry.label === group.baseline_label) row.className = "provider-baseline-row";
      const name = make("td", "", String(entry.label || "—"));
      if (entry.label === group.baseline_label) name.append(make("small", "provider-current", "Лучший в группе"));
      row.append(name);
      row.append(make("td", "", fmtScore(entry.worst, asPercent)));
      row.append(make("td", "", fmtScore(entry.mean, asPercent)));
      row.append(make("td", "", fmtDelta(entry.spread, asPercent)));
      row.append(make("td", "", String((entry.usage || {}).requests || 0)));
      row.append(make("td", "", fmtMoney((entry.usage || {}).cost_usd || 0)));
      tbody.append(row);
    }
    table.append(thead, tbody);
    tableWrap.append(table);
    section.append(tableWrap);
  }
  return section;
}

function comparisonOutcome(lab) {
  const comparison = lab && lab.probe_comparison;
  const metric = comparison && comparison.primary;
  const group = comparison && (comparison.groups || [])[0];
  const baseline = group && (group.entries || [])[0];
  if (!comparison || !metric || !baseline) return null;
  const score = Number(baseline.worst);
  return Number.isFinite(score) ? {score, label: baseline.label || ""} : null;
}
function labQuery(rootId) {
  return rootId ? "?root=" + encodeURIComponent(rootId) : "";
}
async function postLabAction(projectId, labId, action, failureText, rootId) {
  try {
    const response = await fetch("/api/labs/" + encodeURIComponent(projectId) + "/" + encodeURIComponent(labId) + "/" + action + labQuery(rootId), {
      method: "POST",
      headers: {"X-AutoResearch-Confirm": labId},
    });
    if (response.status === 409) {
      let code = "";
      try { code = (await response.json()).error || ""; } catch (ignore) { code = ""; }
      window.alert(code === "ambiguous"
        ? "Несколько лабораторий с таким именем. Обновите страницу и выберите нужный корень."
        : code === "exists"
        ? (action === "restore" ? "Прежнее место лаборатории уже занято" : "В архиве уже есть такая лаборатория")
        : "Сначала остановите текущий опыт");
      return;
    }
    if (!response.ok) {
      window.alert(failureText);
      return;
    }
    if (action === "delete" && selected.endsWith("::" + labId)) {
      selected = "";
      localStorage.removeItem("autoresearch:selected:" + boot.root_id);
      detailVersion = "";
    }
    await refresh();
  } catch (failure) {
    window.alert(failureText);
  }
}
function deleteLab(projectId, labId, rootId) {
  return postLabAction(projectId, labId, "delete", "Удалить не удалось", rootId);
}
function archiveLab(projectId, labId, rootId) {
  return postLabAction(projectId, labId, "archive", "Архивировать не удалось", rootId);
}
function restoreLab(projectId, labId, rootId) {
  return postLabAction(projectId, labId, "restore", "Вернуть не удалось", rootId);
}

function renderHeroKpis(next, meta) {
  const lab = next.lab || {};
  const base = next.baseline || {};
  const best = next.best || {};
  const verdict = next.verdict || {};
  const budget = next.budget || {};
  const outcome = comparisonOutcome(lab);

  const delta = (base.score === null || base.score === undefined || best.score === null || best.score === undefined)
    ? null : (Number(best.score) - Number(base.score)) * ((next.conditions || {}).direction === "min" ? -1 : 1);

  const grid = make("div", "hero-kpis");

  // 1) Primary Metric (Best vs Baseline with delta)
  const card1 = make("div", "kpi-hero-card");
  card1.append(make("span", "kpi-label", outcome ? "Рекорд прошлых опытов" : (meta.label || "Главная метрика")));
  const card1Val = make("strong", "kpi-main-val", fmtScore(best.score, meta.as_percent));
  const card1Sub = make("div", "kpi-sub-row");
  const baseLabel = outcome ? "Probe: " : "База: ";
  const baseScore = outcome ? outcome.score : base.score;
  card1Sub.append(make("span", "muted", baseLabel + fmtScore(baseScore, meta.as_percent)));
  const dCard = deltaCard(delta, meta.as_percent);
  if (dCard) card1Sub.append(dCard);
  card1.append(card1Val, card1Sub);
  grid.append(card1);

  // 2) Current Stage (Ready / Running trial-XXXX / Completed)
  const card2 = make("div", "kpi-hero-card");
  card2.append(make("span", "kpi-label", "Текущий этап"));
  const stageKind = lab.status || "ready";
  const stageText = lab.stage || (stageKind === "running" ? "Выполняется" : stageKind === "completed" ? "Завершена" : "Готова к опытам");
  const card2Val = make("div", "kpi-main-val");
  card2Val.append(badge(stageKind, stageText));
  const card2Sub = make("div", "kpi-sub-row muted");
  if (next.current) {
    card2Sub.textContent = String(next.current.id || "") + (lab.current_repeat ? " · " + lab.current_repeat : "");
  } else if (lab.latest_trial_label) {
    card2Sub.textContent = "последний опыт: " + String(lab.latest_trial_label).toLowerCase();
  } else if (stageKind === "completed") {
    card2Sub.textContent = "Все опыты завершены";
  } else {
    card2Sub.textContent = "Ожидание запуска";
  }
  card2.append(card2Val, card2Sub);
  grid.append(card2);

  // 3) Hypotheses stats (Accepted count / total)
  const card3 = make("div", "kpi-hero-card");
  card3.append(make("span", "kpi-label", "Статистика гипотез"));
  const accepted = verdict.accepted || 0;
  const totalTrials = verdict.trials ?? lab.experiments ?? 0;
  const card3Val = make("strong", "kpi-main-val", accepted + " из " + totalTrials);
  const stackTotal = (verdict.accepted || 0) + (verdict.rejected || 0) + (verdict.failed || 0) || 1;
  const stack = make("div", "stack");
  stack.append(bar("accepted", verdict.accepted, stackTotal), bar("rejected", verdict.rejected, stackTotal), bar("failed", verdict.failed, stackTotal));
  const card3Sub = make("div", "kpi-sub-row muted", "принято " + accepted + " · откл. " + (verdict.rejected || 0) + " · ош. " + (verdict.failed || 0));
  card3.append(card3Val, stack, card3Sub);
  grid.append(card3);

  // 4) Resources (Requests & cost / time)
  const card4 = make("div", "kpi-hero-card");
  card4.append(make("span", "kpi-label", "Ресурсы и затраты"));
  const costUsed = (budget.cost && budget.cost.used) || lab.cost_usd || 0;
  const reqUsed = (budget.requests && budget.requests.used) || lab.requests || 0;
  const timeUsed = (budget.duration_seconds ?? lab.duration_seconds) || 0;
  const card4Val = make("strong", "kpi-main-val", fmtMoney(costUsed));
  const card4Sub = make("div", "kpi-sub-row muted", reqUsed + " запр. · " + fmtDuration(timeUsed));
  card4.append(card4Val, card4Sub);
  if (budget.cost && budget.cost.limit) {
    const ratio = Number(costUsed) / Number(budget.cost.limit);
    const meter = make("div", "meter" + (ratio > 1 ? " over" : ""));
    const fill = document.createElement("span");
    fill.style.width = Math.min(100, Math.max(0, ratio * 100)) + "%";
    meter.append(fill);
    card4.append(meter);
  }
  grid.append(card4);

  return grid;
}

function kpi(label, value, extra) {
  const card = make("div", "kpi");
  card.append(make("span", "", label), make("strong", "", value));
  if (extra) card.append(extra);
  return card;
}
function deltaCard(delta, asPercent) {
  if (delta === null || delta === undefined || !Number.isFinite(Number(delta))) return null;
  const tone = delta > 1e-9 ? "up" : delta < -1e-9 ? "down" : "flat";
  const arrow = tone === "up" ? "↑ " : tone === "down" ? "↓ " : "";
  return make("span", "delta " + tone, arrow + fmtDelta(delta, asPercent));
}
function bar(cls, value, total) {
  const node = document.createElement("i");
  node.className = cls;
  node.style.width = ((Number(value) || 0) / total * 100) + "%";
  return node;
}
function meterKpi(label, bucket, format) {
  const card = kpi(label, format(bucket.used));
  if (bucket.limit) {
    const ratio = Number(bucket.used) / Number(bucket.limit);
    const meter = make("div", "meter" + (ratio > 1 ? " over" : ""));
    const fill = document.createElement("span");
    fill.style.width = Math.min(100, Math.max(0, ratio * 100)) + "%";
    meter.append(fill);
    card.append(meter, make("span", "muted", format(bucket.used) + " из " + format(bucket.limit)));
  }
  return card;
}

function renderChart(next, meta) {
  const card = make("section", "chart-card");
  card.id = "detail-chart-card";
  const head = make("div", "chart-head");
  head.append(make("h3", "", "Динамика по гипотезам"));
  const seg = make("div", "seg");
  seg.append(toggle("Все опыты", !acceptedOnly, () => { acceptedOnly = false; redrawChart(); }),
    toggle("Только принятые", acceptedOnly, () => { acceptedOnly = true; redrawChart(); }));
  const guards = (next.metrics && next.metrics.guards) || [];
  if (guards.length) {
    seg.append(toggle(meta.label || "Главная", !chartGuard, () => { chartGuard = ""; redrawChart(); }));
    for (const guard of guards) {
      seg.append(toggle(guard.label || guard.name, chartGuard === guard.name, () => { chartGuard = guard.name; redrawChart(); }));
    }
  }
  head.append(seg);
  card.append(head);
  card.append(make("p", "muted", "Линия соединяет фактический результат каждой гипотезы: вверх — удачнее, вниз — хуже. Пунктир — база, зелёная ступень — лучшее на тот момент."));
  if ((next.series || []).some(row => row.recheck_source)) {
    card.append(make("p", "muted", "Точки с пометкой «Перепроверено» получены на исходном коде и настройках исторической серии. Первоначальные записи об ошибках сохранены."));
  }
  const wrap = make("div", "chart-wrap");
  wrap.id = "chart-host";
  card.append(wrap);
  const legend = make("div", "legend");
  legend.append(swatch("wave", "Результат гипотезы"), swatch("base", "База"), swatch("record", "Лучшее к моменту"));
  card.append(legend);
  drawChart(wrap, next.series || [], meta);
  if (!chartGuard && guards.length) card.append(renderGuardCharts(next));
  return card;
}
function toggle(label, on, action) {
  const button = make("button", on ? "is-on" : "", label);
  button.type = "button";
  button.addEventListener("click", action);
  return button;
}
function swatch(cls, label) {
  const node = make("span", "", "");
  const mark = document.createElement("i");
  mark.className = cls;
  node.append(mark, document.createTextNode(label));
  return node;
}
function redrawChart() {
  if (!model) return;
  const host = document.getElementById("chart-host");
  if (host) drawChart(host, model.series || [], primaryMeta());
  document.querySelectorAll(".chart-head .seg button").forEach(button => {
    const text = button.textContent;
    const on = (text === "Все опыты" && !acceptedOnly) || (text === "Только принятые" && acceptedOnly)
      || (text === (primaryMeta().label || "Главная") && !chartGuard)
      || text === (((model.metrics || {}).guards || []).find(guard => guard.name === chartGuard) || {}).label;
    button.classList.toggle("is-on", Boolean(on));
  });
}
function viewOf(row, meta) {
  if (!chartGuard) return row;
  const guard = (row.guards || {})[chartGuard] || {};
  const guardMeta = ((model.metrics || {}).guards || []).find(item => item.name === chartGuard) || {};
  return Object.assign({}, row, {
    worst: guard.worst, low: guard.low, high: guard.high, has_value: Boolean(guard.has_value),
    best_so_far: null, values: [], direction: guardMeta.direction || meta.direction,
  });
}

function drawChart(host, series, meta, compact) {
  host.replaceChildren();
  const rawRows = (series || []).map(row => viewOf(row, meta));
  const rows = rawRows.map(row => {
    if (!acceptedOnly || String(row.id).startsWith("baseline") || row.status === "accepted") return row;
    return Object.assign({}, row, {has_value: false, worst: null, low: null, high: null});
  });

  // 0 points edge case: render clean empty state placeholder
  if (!rows.length) {
    host.append(make("div", "chart-empty", "Результатов пока нет."));
    return;
  }

  const direction = chartGuard ? (rows[0].direction || meta.direction) : meta.direction;
  const asPercent = chartGuard ? Boolean((((model.metrics || {}).guards || []).find(item => item.name === chartGuard) || {}).as_percent) : meta.as_percent;
  
  // Exclude null/NaN/undefined values before path generation
  const valued = rows.filter(row => row.has_value && row.worst !== null && row.worst !== undefined && Number.isFinite(Number(row.worst)));
  const outcome = compact || chartGuard ? null : comparisonOutcome((model && model.lab) || {});

  const samples = [];
  valued.forEach(row => {
    for (const v of [row.low, row.high, row.worst]) {
      if (v !== null && v !== undefined && Number.isFinite(Number(v))) {
        samples.push(Number(v));
      }
    }
  });
  if (outcome && Number.isFinite(Number(outcome.score))) samples.push(Number(outcome.score));

  let floor = samples.length ? Math.min(...samples) : 0;
  let ceiling = samples.length ? Math.max(...samples) : 1;
  if (!Number.isFinite(floor) || !Number.isFinite(ceiling)) {
    floor = 0;
    ceiling = 1;
  }

  let span = ceiling - floor;
  // Edge case handling:
  // All points equal (span === 0) or 1 point: symmetric margin Math.max(Math.abs(floor) * 0.1, 0.05)
  // Negative numbers: correctly handle negative metrics (e.g. loss or error)
  if (span === 0 || !Number.isFinite(span) || Math.abs(span) < 1e-12) {
    const symMargin = Math.max(Math.abs(floor) * 0.1, 0.05);
    floor = floor - symMargin;
    ceiling = ceiling + symMargin;
  } else {
    const margin = span * 0.12 || Math.max(Math.abs(floor) * 0.08, 0.05);
    floor = floor - margin;
    ceiling = ceiling + margin;
  }
  span = ceiling - floor;
  if (span === 0 || !Number.isFinite(span)) span = 1;

  const width = Math.max(compact ? 520 : 720, rows.length * (compact ? 48 : 72));
  const height = compact ? 180 : 320;
  const pad = {l: 64, r: 36, t: 28, b: 42};
  const plotW = Math.max(1, width - pad.l - pad.r);
  const plotH = Math.max(1, height - pad.t - pad.b);

  // 1 point: center point at width / 2
  const x = index => {
    if (rows.length === 1) return width / 2;
    const pos = pad.l + index * plotW / (rows.length - 1);
    return Number.isFinite(pos) ? pos : width / 2;
  };

  const y = value => {
    if (value === null || value === undefined) return null;
    const num = Number(value);
    if (!Number.isFinite(num)) return null;
    const ratio = (num - floor) / (span || 1);
    const pos = pad.t + (direction === "min" ? ratio : 1 - ratio) * plotH;
    return Number.isFinite(pos) ? pos : pad.t + plotH / 2;
  };

  const svg = svgEl("svg");
  svg.setAttribute("viewBox", "0 0 " + width + " " + height);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Волна результатов по всем гипотезам");

  // Grid lines
  for (const fraction of [0, 0.5, 1]) {
    const py = pad.t + fraction * plotH;
    if (!Number.isFinite(py)) continue;
    const grid = svgEl("line");
    grid.setAttribute("x1", pad.l); grid.setAttribute("x2", width - pad.r);
    grid.setAttribute("y1", py); grid.setAttribute("y2", py);
    grid.setAttribute("stroke", "var(--line)");
    svg.append(grid);
    const label = svgEl("text");
    const shown = direction === "min" ? floor + fraction * span : ceiling - fraction * span;
    label.setAttribute("x", pad.l - 8); label.setAttribute("y", py + 4); label.setAttribute("text-anchor", "end");
    label.setAttribute("fill", "var(--muted)"); label.setAttribute("font-size", "11");
    label.textContent = fmtScore(shown, asPercent);
    svg.append(label);
  }

  // Baseline reference
  const baseline = valued.find(row => String(row.id).startsWith("baseline"));
  if (baseline) {
    const baseY = y(baseline.worst);
    if (baseY !== null && Number.isFinite(baseY)) {
      const line = svgEl("line");
      line.setAttribute("x1", pad.l); line.setAttribute("x2", width - pad.r);
      line.setAttribute("y1", baseY); line.setAttribute("y2", baseY);
      line.setAttribute("stroke", "var(--muted)"); line.setAttribute("stroke-dasharray", "4 5");
      svg.append(line);
    }
  }

  // Provider outcome
  if (outcome) {
    const py = y(outcome.score);
    if (py !== null && Number.isFinite(py)) {
      const mark = svgEl("line");
      mark.setAttribute("x1", pad.l); mark.setAttribute("x2", width - pad.r);
      mark.setAttribute("y1", py); mark.setAttribute("y2", py);
      mark.setAttribute("stroke", "var(--accent)"); mark.setAttribute("stroke-dasharray", "2 4");
      svg.append(mark);
      const caption = svgEl("text");
      caption.setAttribute("x", width - pad.r); caption.setAttribute("y", Math.max(pad.t + 12, py - 6));
      caption.setAttribute("text-anchor", "end"); caption.setAttribute("fill", "var(--accent)");
      caption.setAttribute("font-size", "11");
      caption.textContent = "текущий провайдер · худший F1";
      svg.append(caption);
    }
  }

  // Segments for spline and area
  const segments = [];
  let segment = [];
  rows.forEach((row, index) => {
    if (row.has_value && row.worst !== null && row.worst !== undefined && Number.isFinite(Number(row.worst))) {
      const px = x(index);
      const py = y(row.worst);
      if (Number.isFinite(px) && Number.isFinite(py)) {
        segment.push({row, index, x: px, y: py});
      } else if (segment.length) {
        segments.push(segment);
        segment = [];
      }
    } else if (segment.length) {
      segments.push(segment);
      segment = [];
    }
  });
  if (segment.length) segments.push(segment);

  // Gradient area
  const area = segments.filter(part => part.length > 1).map(part => {
    const curve = monotone(part);
    if (!curve) return "";
    const last = part[part.length - 1];
    const first = part[0];
    return curve + ` L ${last.x.toFixed(2)} ${(pad.t + plotH).toFixed(2)} L ${first.x.toFixed(2)} ${(pad.t + plotH).toFixed(2)} Z`;
  }).filter(Boolean).join(" ");
  if (area) {
    const fill = svgEl("path");
    fill.setAttribute("d", area);
    fill.setAttribute("fill", "var(--accent)");
    fill.setAttribute("opacity", "0.12");
    svg.append(fill);
  }

  // Splines
  for (let index = 0; index < segments.length; index += 1) {
    const part = segments[index];
    if (part.length > 1) {
      const curve = monotone(part);
      if (curve) {
        const path = svgEl("path");
        path.setAttribute("d", curve);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "var(--accent)");
        path.setAttribute("stroke-width", "2.5");
        path.setAttribute("stroke-linejoin", "round");
        path.setAttribute("stroke-linecap", "round");
        svg.append(path);
      }
    }
    const next = segments[index + 1];
    if (next && next.length) {
      const dash = svgEl("line");
      dash.setAttribute("x1", part[part.length - 1].x.toFixed(2)); dash.setAttribute("y1", part[part.length - 1].y.toFixed(2));
      dash.setAttribute("x2", next[0].x.toFixed(2)); dash.setAttribute("y2", next[0].y.toFixed(2));
      dash.setAttribute("stroke", "var(--accent)"); dash.setAttribute("stroke-dasharray", "3 4"); dash.setAttribute("opacity", "0.7");
      svg.append(dash);
    }
  }

  // Record line
  if (!chartGuard) {
    const record = rows.filter(row => row.best_so_far !== null && row.best_so_far !== undefined && Number.isFinite(Number(row.best_so_far)));
    if (record.length > 1) {
      let step = "";
      for (let i = 0; i < record.length; i++) {
        const row = record[i];
        const px = x(rows.indexOf(row));
        const py = y(row.best_so_far);
        if (!Number.isFinite(px) || py === null || !Number.isFinite(py)) continue;
        if (!step) {
          step = `M ${px.toFixed(2)} ${py.toFixed(2)}`;
        } else {
          const prevRow = record[i - 1];
          const prevPy = y(prevRow.best_so_far);
          if (prevPy !== null && Number.isFinite(prevPy)) {
            step += ` L ${px.toFixed(2)} ${prevPy.toFixed(2)} L ${px.toFixed(2)} ${py.toFixed(2)}`;
          } else {
            step += ` L ${px.toFixed(2)} ${py.toFixed(2)}`;
          }
        }
      }
      if (step) {
        const path = svgEl("path");
        path.setAttribute("d", step);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "var(--good)");
        path.setAttribute("stroke-width", "1.5");
        path.setAttribute("stroke-dasharray", "2 3");
        svg.append(path);
      }
    }
  }

  const finalBest = rows.reduce((best, row) => row.best_so_far === null || row.best_so_far === undefined ? best : row.best_so_far, null);
  const recordId = [...rows].reverse().find(row => row.has_value && row.worst === finalBest);

  rows.forEach((row, index) => {
    const px = x(index);
    if (!Number.isFinite(px)) return;
    if (row.has_value && row.low !== null && row.high !== null && Number.isFinite(Number(row.low)) && Number.isFinite(Number(row.high)) && row.low !== row.high) {
      const yLow = y(row.low);
      const yHigh = y(row.high);
      if (yLow !== null && yHigh !== null && Number.isFinite(yLow) && Number.isFinite(yHigh)) {
        const whisker = svgEl("line");
        whisker.setAttribute("x1", px); whisker.setAttribute("x2", px);
        whisker.setAttribute("y1", yHigh); whisker.setAttribute("y2", yLow);
        whisker.setAttribute("stroke", "var(--muted)"); whisker.setAttribute("stroke-width", "2");
        svg.append(whisker);
      }
    }
    const py = row.has_value ? y(row.worst) : pad.t + plotH;
    if (py === null || !Number.isFinite(py)) return;
    const kind = row.status_kind || row.status;
    const mark = pointMarker(kind, px, py);
    mark.setAttribute("data-point", row.id);
    mark.classList.add("marker");
    if (kind === "running") mark.classList.add("is-live");
    mark.addEventListener("mouseenter", event => showTip(host, event, row, asPercent));
    mark.addEventListener("mouseleave", () => hideTip(host));
    mark.addEventListener("click", () => focusHypothesis(row.id));
    svg.append(mark);

    const tick = svgEl("text");
    tick.setAttribute("x", px); tick.setAttribute("y", height - 16); tick.setAttribute("text-anchor", "middle");
    tick.setAttribute("fill", "var(--muted)"); tick.setAttribute("font-size", "11");
    if (rows.length <= 14 || index % 2 === 0) tick.textContent = axisLabel(row);
    svg.append(tick);

    const labelThis = row === recordId || index === rows.length - 1 || String(row.id).startsWith("baseline");
    if (labelThis && row.has_value) {
      const text = svgEl("text");
      const atEnd = index >= rows.length - 1;
      const atStart = index === 0;
      text.setAttribute("x", atEnd ? px - 8 : atStart ? px + 8 : px);
      text.setAttribute("y", py - 12);
      text.setAttribute("text-anchor", atEnd ? "end" : atStart ? "start" : "middle");
      text.setAttribute("fill", "var(--ink)"); text.setAttribute("font-size", "11"); text.setAttribute("font-weight", "700");
      text.textContent = (row === recordId && !atEnd ? "Лучший " : "") + fmtScore(row.worst, asPercent);
      svg.append(text);
    }
    if (!row.has_value) {
      const note = svgEl("text");
      note.setAttribute("x", px); note.setAttribute("y", py - 10); note.setAttribute("text-anchor", "middle");
      note.setAttribute("fill", "var(--bad)"); note.setAttribute("font-size", "10");
      note.textContent = "нет данных";
      svg.append(note);
    }
  });

  const providerNote = outcome
    ? make("p", "muted", "Опорная линия — итог выбранного провайдера"
      + (outcome.provider ? " (" + outcome.provider + ")" : "")
      + ": худший F1 " + fmtScore(outcome.score, asPercent)
      + ". Точки волны — рекорд прошлых опытов, отчёт их не переписывает.")
    : null;

  if (valued.length < 2) {
    const note = make("p", "muted", valued.length ? "Пока одна точка — волна появится со следующим опытом." : "На графике только опыты без числового результата.");
    host.append(svg, note);
    if (providerNote) host.append(providerNote);
    return;
  }
  host.append(svg);
  if (providerNote) host.append(providerNote);
}

function monotone(points) {
  const valid = (points || []).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  const count = valid.length;
  if (count < 2) return "";
  const slopes = [];
  for (let index = 0; index < count - 1; index += 1) {
    const dx = valid[index + 1].x - valid[index].x;
    if (Math.abs(dx) < 1e-9) {
      slopes.push(0);
    } else {
      const slope = (valid[index + 1].y - valid[index].y) / dx;
      slopes.push(Number.isFinite(slope) ? slope : 0);
    }
  }
  const tangent = new Array(count);
  tangent[0] = slopes[0] || 0;
  tangent[count - 1] = slopes[count - 2] || 0;
  for (let index = 1; index < count - 1; index += 1) {
    tangent[index] = slopes[index - 1] * slopes[index] <= 0 ? 0 : (slopes[index - 1] + slopes[index]) / 2;
  }
  for (let index = 0; index < count - 1; index += 1) {
    if (Math.abs(slopes[index]) < 1e-9) { tangent[index] = 0; tangent[index + 1] = 0; continue; }
    const a = tangent[index] / slopes[index];
    const b = tangent[index + 1] / slopes[index];
    const hypot = a * a + b * b;
    if (hypot > 9) {
      const scale = 3 / Math.sqrt(hypot);
      tangent[index] = scale * a * slopes[index];
      tangent[index + 1] = scale * b * slopes[index];
    }
  }
  let path = "M " + valid[0].x.toFixed(2) + " " + valid[0].y.toFixed(2);
  for (let index = 0; index < count - 1; index += 1) {
    const dx = (valid[index + 1].x - valid[index].x) / 3;
    const cp1x = valid[index].x + dx;
    const cp1y = valid[index].y + (tangent[index] || 0) * dx;
    const cp2x = valid[index + 1].x - dx;
    const cp2y = valid[index + 1].y - (tangent[index + 1] || 0) * dx;
    const endx = valid[index + 1].x;
    const endy = valid[index + 1].y;
    if ([cp1x, cp1y, cp2x, cp2y, endx, endy].every(Number.isFinite)) {
      path += " C " + cp1x.toFixed(2) + " " + cp1y.toFixed(2)
        + ", " + cp2x.toFixed(2) + " " + cp2y.toFixed(2)
        + ", " + endx.toFixed(2) + " " + endy.toFixed(2);
    } else {
      path += " L " + endx.toFixed(2) + " " + endy.toFixed(2);
    }
  }
  return path;
}

function pointMarker(kind, x, y) {
  const color = kind === "accepted" ? "var(--good)"
    : kind === "rechecked" ? "var(--accent)"
    : kind === "failed" || kind === "config" ? "var(--bad)"
    : kind === "invalid" || kind === "incomparable" ? "var(--warn)"
    : kind === "running" ? "var(--accent)" : "var(--neutral)";
  let node;
  if (kind === "rejected") {
    node = svgEl("polygon");
    node.setAttribute("points", `${x},${y - 7} ${x + 7},${y} ${x},${y + 7} ${x - 7},${y}`);
  } else if (kind === "invalid" || kind === "incomparable") {
    node = svgEl("polygon");
    node.setAttribute("points", `${x},${y - 8} ${x + 7},${y + 6} ${x - 7},${y + 6}`);
  } else if (kind === "failed" || kind === "config") {
    node = svgEl("g");
    [[-5, -5, 5, 5], [-5, 5, 5, -5]].forEach(pair => {
      const line = svgEl("line");
      line.setAttribute("x1", x + pair[0]); line.setAttribute("y1", y + pair[1]);
      line.setAttribute("x2", x + pair[2]); line.setAttribute("y2", y + pair[3]);
      line.setAttribute("stroke", color); line.setAttribute("stroke-width", "2");
      node.append(line);
    });
  } else {
    node = svgEl("circle");
    node.setAttribute("cx", x); node.setAttribute("cy", y); node.setAttribute("r", "6");
  }
  node.setAttribute("fill", kind === "failed" || kind === "config" ? "none" : color);
  node.setAttribute("stroke", "var(--panel)");
  node.setAttribute("stroke-width", "2");
  const hit = svgEl("circle");
  hit.setAttribute("cx", x); hit.setAttribute("cy", y); hit.setAttribute("r", "12");
  hit.setAttribute("fill", "transparent");
  const group = svgEl("g");
  group.append(node, hit);
  group.style.cursor = "pointer";
  return group;
}

function showTip(host, event, row, asPercent) {
  highlight(row.id);
  let tip = host.querySelector(".chart-tip");
  if (!tip) { tip = make("div", "chart-tip"); host.append(tip); }
  tip.replaceChildren();

  const hyp = ((model && model.hypotheses) || []).find(h => h.id === row.id) || {};
  const kind = row.status_kind || row.status || hyp.status_kind || hyp.status || "neutral";
  const outcomeText = hyp.outcome || row.outcome || row.status || "—";

  const head = make("div", "tip-head");
  head.append(make("b", "tip-title", prettyTitle(row.title || hyp.title, row.id)));
  head.append(badge(kind, outcomeText));
  tip.append(head);

  const grid = make("div", "tip-grid");

  const valRow = make("div", "tip-row");
  valRow.append(make("span", "tip-lbl", "Значение:"));
  valRow.append(make("strong", "tip-val", row.has_value ? fmtScore(row.worst, asPercent) : "Нет данных"));
  grid.append(valRow);

  const deltaRow = make("div", "tip-row");
  deltaRow.append(make("span", "tip-lbl", "К базе:"));
  const dVal = row.delta_vs_baseline ?? hyp.delta_vs_baseline;
  deltaRow.append(deltaCard(dVal, asPercent) || make("span", "", "—"));
  grid.append(deltaRow);

  const durSec = row.duration_seconds ?? hyp.duration_seconds ?? (hyp.runs ? hyp.runs.reduce((s, r) => s + (r.duration_seconds || 0), 0) : null);
  if (durSec !== null && durSec !== undefined && Number(durSec) > 0) {
    const durRow = make("div", "tip-row");
    durRow.append(make("span", "tip-lbl", "Длительность:"));
    durRow.append(make("span", "tip-val", fmtDuration(durSec)));
    grid.append(durRow);
  }

  if (row.delta_vs_previous !== null && row.delta_vs_previous !== undefined) {
    const prevRow = make("div", "tip-row");
    prevRow.append(make("span", "tip-lbl", "К предыд.:"));
    prevRow.append(make("span", "tip-val", fmtDelta(row.delta_vs_previous, asPercent)));
    grid.append(prevRow);
  }

  tip.append(grid);

  if (row.reason_label) tip.append(make("div", "tip-reason", row.reason_label));
  if (row.recheck_source) tip.append(make("div", "tip-recheck", "Историческая перепроверка · " + row.recheck_source));

  const bounds = host.getBoundingClientRect();
  const tipWidth = tip.offsetWidth || 220;
  let left = event.clientX - bounds.left;
  let top = event.clientY - bounds.top;
  if (left + tipWidth / 2 > bounds.width) left = bounds.width - tipWidth / 2 - 8;
  if (left - tipWidth / 2 < 8) left = tipWidth / 2 + 8;
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}

function hideTip(host) {
  host.querySelector(".chart-tip")?.remove();
}
function renderGuardCharts(next) {
  const wrap = make("div", "guards");
  for (const guard of next.metrics.guards || []) {
    const card = make("div", "guard-card");
    card.append(make("h4", "", "Защитная метрика · " + (guard.label || guard.name)));
    const host = make("div", "chart-wrap");
    const saved = chartGuard;
    chartGuard = guard.name;
    drawChart(host, next.series || [], primaryMeta(), true);
    chartGuard = saved;
    card.append(host);
    wrap.append(card);
  }
  return wrap;
}

function renderHypCard(item, next, meta, low, high) {
  const kind = item.status_kind || item.status;
  const isOpen = openHyp.has(item.id);
  const card = make("article", "hyp " + kind + (item.improved_best ? " is-record" : "") + (isOpen ? " is-open" : "") + (hotId === item.id ? " is-hot" : ""));
  card.id = "hyp-" + item.id;
  card.dataset.id = item.id;

  const button = make("button", "hyp-main");
  button.type = "button";
  button.setAttribute("aria-expanded", isOpen ? "true" : "false");

  const name = make("div");
  name.append(make("div", "hyp-title", prettyTitle(item.title, item.id)));
  if (item.reason_label) name.append(make("small", "", item.reason_label));

  const score = make("div");
  const point = (next.series || []).find(row => row.id === item.id);
  const rangeText = point && point.has_value
    ? (point.low === point.high ? fmtScore(point.worst, meta.as_percent) : fmtScore(point.low, meta.as_percent) + " – " + fmtScore(point.high, meta.as_percent))
    : "—";
  score.append(make("div", "score", rangeText));
  const range = make("div", "range");
  if (item.primary_score !== null && item.primary_score !== undefined && high > low) {
    const fill = document.createElement("i");
    const start = ((Number(item.primary_score) - low) / (high - low)) * 100;
    fill.style.left = "0";
    fill.style.width = Math.max(4, start) + "%";
    const mark = document.createElement("b");
    mark.style.left = start + "%";
    range.append(fill, mark);
  }
  score.append(range);

  const dots = make("div", "guard-dots");
  for (const guard of item.guard_summary || []) {
    const dot = document.createElement("span");
    dot.title = guard.name + ": " + guard.value;
    if ((item.guard_failures || []).includes(guard.name)) dot.className = "bad";
    dots.append(dot);
  }

  const spend = item.cost_status === "partial" ? "от " + fmtMoney(item.cost_usd || 0) : fmtMoney(item.cost_usd || 0);
  button.append(name, badge(kind, item.outcome || item.status), deltaNode(item.delta_vs_baseline, meta.as_percent), score, make("div", "cost", (item.requests || 0) + " запр. · " + spend), dots);

  button.addEventListener("click", () => {
    if (openHyp.has(item.id)) {
      openHyp.delete(item.id);
      card.classList.remove("is-open");
      button.setAttribute("aria-expanded", "false");
    } else {
      openHyp.add(item.id);
      card.classList.add("is-open");
      button.setAttribute("aria-expanded", "true");
      const diffPre = card.querySelector(".diff");
      if (item.diff_available && !item.diff && diffPre && boot.live) {
        loadDiff(item, diffPre);
      }
    }
  });
  button.addEventListener("mouseenter", () => highlight(item.id));

  const body = make("div", "hyp-body");
  if (item.observation) body.append(make("p", "", item.observation));
  if (item.expected_effect) body.append(make("p", "muted", "Ожидание: " + item.expected_effect));
  if (item.recheck_source) body.append(make("p", "muted", "Измерено заново на историческом коде. Первоначальный статус: " + item.original_outcome + ". Источник: " + item.recheck_source));
  if (item.unknown_cost_attempts || item.prior_unknown_cost_attempts) {
    const unknown = Number(item.unknown_cost_attempts || 0) + Number(item.prior_unknown_cost_attempts || 0);
    body.append(make("p", "muted", "Стоимость " + unknown + " отклонённых запросов не подтверждена. Сумма на карточке относится к завершённым повторам и показывает только подтверждённые списания."));
  }
  if (item.diff_available) body.append(renderDiff(item));

  card.append(button, body);
  return card;
}

function updateHypCard(card, item, next, meta, low, high) {
  const kind = item.status_kind || item.status;
  const isOpen = openHyp.has(item.id);
  card.className = "hyp " + kind + (item.improved_best ? " is-record" : "") + (isOpen ? " is-open" : "") + (hotId === item.id ? " is-hot" : "");
  const btn = card.querySelector(".hyp-main");
  if (!btn) return;
  btn.setAttribute("aria-expanded", isOpen ? "true" : "false");

  const titleDiv = btn.querySelector(".hyp-title");
  if (titleDiv) titleDiv.textContent = prettyTitle(item.title, item.id);
  const reasonSmall = btn.querySelector("small");
  if (reasonSmall && item.reason_label) reasonSmall.textContent = item.reason_label;

  const oldBadge = btn.querySelector(".badge");
  if (oldBadge) oldBadge.replaceWith(badge(kind, item.outcome || item.status));

  const oldDelta = btn.querySelector(".delta");
  if (oldDelta) oldDelta.replaceWith(deltaNode(item.delta_vs_baseline, meta.as_percent));

  const scoreDiv = btn.querySelector(".score");
  const point = (next.series || []).find(row => row.id === item.id);
  const rangeText = point && point.has_value
    ? (point.low === point.high ? fmtScore(point.worst, meta.as_percent) : fmtScore(point.low, meta.as_percent) + " – " + fmtScore(point.high, meta.as_percent))
    : "—";
  if (scoreDiv) scoreDiv.textContent = rangeText;

  const costDiv = btn.querySelector(".cost");
  const spend = item.cost_status === "partial" ? "от " + fmtMoney(item.cost_usd || 0) : fmtMoney(item.cost_usd || 0);
  if (costDiv) costDiv.textContent = (item.requests || 0) + " запр. · " + spend;

  const guardDiv = btn.querySelector(".guard-dots");
  if (guardDiv) {
    guardDiv.replaceChildren();
    for (const guard of item.guard_summary || []) {
      const dot = document.createElement("span");
      dot.title = guard.name + ": " + guard.value;
      if ((item.guard_failures || []).includes(guard.name)) dot.className = "bad";
      guardDiv.append(dot);
    }
  }
}

function updateHypothesesInPlace(next, meta) {
  const section = document.getElementById("hypotheses-section");
  if (!section) return;

  const trials = next.hypotheses || [];
  const headTitle = section.querySelector("h3");
  if (headTitle) headTitle.textContent = "Гипотезы и решения (" + trials.length + ")";

  const list = section.querySelector(".hyp-list");
  if (!list) return;

  let rows = trials.filter(item => hypFilter === "all" || item.status === hypFilter || item.status_kind === hypFilter || (hypFilter === "failed" && (item.status_kind === "failed" || item.status_kind === "config" || item.status_kind === "invalid")));
  rows = newestFirst ? rows.slice().reverse() : rows.slice();

  if (!rows.length) {
    list.replaceChildren(make("p", "empty", "Результатов пока нет."));
    return;
  }

  const emptyP = list.querySelector(".empty");
  if (emptyP) emptyP.remove();

  const scores = trials.map(item => item.primary_score).filter(value => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  const low = scores.length ? Math.min(...scores) : 0;
  const high = scores.length ? Math.max(...scores) : 1;

  const existingElements = new Map();
  list.querySelectorAll(".hyp").forEach(el => {
    if (el.dataset.id) existingElements.set(el.dataset.id, el);
  });

  const rowIds = new Set(rows.map(r => r.id));
  for (const [id, el] of existingElements) {
    if (!rowIds.has(id)) el.remove();
  }

  rows.forEach((item, targetIndex) => {
    let card = existingElements.get(item.id);
    if (!card) {
      card = renderHypCard(item, next, meta, low, high);
      const currentChildren = list.children;
      if (targetIndex < currentChildren.length) {
        list.insertBefore(card, currentChildren[targetIndex]);
      } else {
        list.append(card);
      }
      existingElements.set(item.id, card);
    } else {
      updateHypCard(card, item, next, meta, low, high);
      const currentChildren = list.children;
      if (currentChildren[targetIndex] !== card) {
        if (targetIndex < currentChildren.length) {
          list.insertBefore(card, currentChildren[targetIndex]);
        } else {
          list.append(card);
        }
      }
    }
  });
}

function renderHypotheses(next, meta) {
  const section = make("section");
  section.id = "hypotheses-section";
  const head = make("div", "section-head");
  const trials = next.hypotheses || [];
  head.append(make("h3", "", "Гипотезы и решения (" + trials.length + ")"));
  const seg = make("div", "seg");
  for (const [id, label] of [["all", "Все"], ["accepted", "Принятые"], ["rejected", "Отклонённые"], ["failed", "Ошибки"]]) {
    seg.append(toggle(label, hypFilter === id, () => { hypFilter = id; updateHypothesesInPlace(model, primaryMeta()); }));
  }
  seg.append(toggle(newestFirst ? "Сначала новые" : "Сначала старые", newestFirst, () => { newestFirst = !newestFirst; updateHypothesesInPlace(model, primaryMeta()); }));
  head.append(seg);
  section.append(head);
  const list = make("div", "hyp-list");
  section.append(list);

  let rows = trials.filter(item => hypFilter === "all" || item.status === hypFilter || item.status_kind === hypFilter || (hypFilter === "failed" && (item.status_kind === "failed" || item.status_kind === "config" || item.status_kind === "invalid")));
  rows = newestFirst ? rows.slice().reverse() : rows.slice();
  if (!rows.length) {
    list.append(make("p", "empty", "Результатов пока нет."));
    return section;
  }
  const scores = trials.map(item => item.primary_score).filter(value => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  const low = scores.length ? Math.min(...scores) : 0;
  const high = scores.length ? Math.max(...scores) : 1;
  for (const item of rows) {
    list.append(renderHypCard(item, next, meta, low, high));
  }
  return section;
}

function deltaNode(value, asPercent) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return make("div", "delta flat", "—");
  const tone = value > 1e-9 ? "up" : value < -1e-9 ? "down" : "flat";
  return make("div", "delta " + tone, fmtDelta(value, asPercent));
}
function fillNode(pre, text) {
  pre.replaceChildren();
  String(text || "").split("\n").forEach(line => {
    const row = make("div", line.startsWith("+") && !line.startsWith("+++") ? "diff-add" : line.startsWith("-") && !line.startsWith("---") ? "diff-del" : line.startsWith("@@") ? "diff-hunk" : "", line || " ");
    pre.append(row);
  });
}
function loadDiff(item, pre) {
  const lab = model.lab || {};
  fetch("/api/labs/" + encodeURIComponent(lab.project_id) + "/" + encodeURIComponent(lab.id) + "/runs/" + encodeURIComponent(item.id) + "/diff" + labQuery(lab.root_id))
    .then(response => response.ok ? response.text() : Promise.reject())
    .then(text => {
      item.diff = text;
      fillNode(pre, text);
    })
    .catch(() => {
      fillNode(pre, "Diff недоступен");
    });
}
function renderDiff(item) {
  const box = make("div", "diff-box");
  const bar = make("div", "diff-bar");
  const copy = make("button", "ghost", "Копировать");
  copy.type = "button";
  const pre = make("pre", "diff");
  if (item.diff) fillNode(pre, item.diff);
  else pre.append(make("div", "", "Загрузка diff…"));
  copy.addEventListener("click", async event => {
    event.stopPropagation();
    const text = item.diff || pre.innerText;
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = "Скопировано";
      setTimeout(() => { copy.textContent = "Копировать"; }, 2000);
    } catch {
      copy.textContent = "Не удалось скопировать";
    }
  });
  bar.append(make("strong", "", "Изменение"), copy);
  box.append(bar, pre);
  if (!item.diff && boot.live && openHyp.has(item.id)) {
    loadDiff(item, pre);
  }
  return box;
}

function renderHistory(next) {
  const items = next.historical_hypotheses || [];
  const section = make("section", "history panel");
  section.id = "detail-history";
  const head = make("div", "section-head");
  head.append(make("h3", "", "Прежние гипотезы (" + items.length + ")"));
  const toggleHistory = make("button", "ghost", historyOpen ? "Свернуть" : "Показать");
  toggleHistory.type = "button";
  toggleHistory.addEventListener("click", () => {
    historyOpen = !historyOpen;
    section.replaceWith(renderHistory(model));
  });
  head.append(toggleHistory);
  section.append(head, make("p", "muted", "Исторические оценки показываются отдельно и не меняют текущую базу."));
  if (!historyOpen) return section;
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Поиск по прошлым гипотезам";
  search.value = historyQuery;
  search.setAttribute("aria-label", "Поиск по прошлым гипотезам");
  search.addEventListener("input", () => {
    historyQuery = search.value;
    historyLimit = 6;
    section.replaceWith(renderHistory(model));
    const newSearch = document.querySelector("#detail-history input[type='search']");
    if (newSearch) newSearch.focus();
  });
  section.append(search);
  const query = historyQuery.trim().toLowerCase();
  const matched = items.filter(item => !query || (item.title + " " + item.source + " " + item.conclusion).toLowerCase().includes(query));
  const list = make("div", "history-list");
  matched.slice(0, historyLimit).forEach(item => {
    const card = make("article", "history-card");
    card.append(make("b", "", prettyTitle(item.title, item.id)));
    const source = make("div", "muted", "Источник: " + (item.source || "—"));
    if (item.artifact) source.append(make("small", "muted", "артефакт: " + item.artifact));
    card.append(source, make("div", "", "Прежний результат: " + (item.score || "—")), make("div", "", item.conclusion || "—"));
    card.append(make("div", "muted", [item.context, item.comparison, item.recheck].filter(Boolean).join(" · ")));
    list.append(card);
  });
  section.append(list);
  if (matched.length > historyLimit) {
    const more = make("button", "ghost pager", "Показать ещё");
    more.type = "button";
    more.addEventListener("click", () => {
      historyLimit += 8;
      section.replaceWith(renderHistory(model));
    });
    section.append(more);
  }
  return section;
}

function renderConditions(next) {
  const lab = next.lab || {};
  const conditions = next.conditions || {};
  const box = make("details", "conditions");
  box.append(make("summary", "", "Условия измерения"));
  const chips = make("div", "chips");
  const meta = primaryMeta();
  const bits = [
    "Метрика: " + (meta.label || lab.metric || "—"),
    "Направление: " + (conditions.direction === "min" ? "меньше — лучше" : "больше — лучше"),
    "Повторов: " + (conditions.repeats ?? "—"),
    "Лимит времени: " + (conditions.timeout_seconds ? conditions.timeout_seconds + " с" : "—"),
    "Версия: " + (lab.id || "—"),
    "Входы: " + (conditions.input_fingerprint || "не указаны"),
  ];
  bits.forEach(text => chips.append(make("span", "chip", text)));
  box.append(chips);
  return box;
}

function highlight(id) {
  hotId = id;
  document.querySelectorAll(".hyp.is-hot").forEach(node => node.classList.remove("is-hot"));
  const row = document.getElementById("hyp-" + id);
  if (row) row.classList.add("is-hot");
}
function focusHypothesis(id) {
  openHyp.add(id);
  const card = document.getElementById("hyp-" + id);
  if (card) {
    card.classList.add("is-open");
    const btn = card.querySelector(".hyp-main");
    if (btn) btn.setAttribute("aria-expanded", "true");
    const diffPre = card.querySelector(".diff");
    const hyp = ((model && model.hypotheses) || []).find(h => h.id === id) || {};
    if (hyp.diff_available && !hyp.diff && diffPre && boot.live) {
      loadDiff(hyp, diffPre);
    }
    card.scrollIntoView({block: "center"});
  } else if (model) {
    renderDetail(model);
    const newCard = document.getElementById("hyp-" + id);
    if (newCard) newCard.scrollIntoView({block: "center"});
  }
}

function select(projectId, labId) {
  const match = allLabs().find(item => item.project.id === projectId && item.lab.id === labId);
  selected = match ? keyOf(match.project, match.lab) : "";
  if (!selected) return;
  localStorage.setItem("autoresearch:selected:" + boot.root_id, selected);
  detailVersion = "";
  openHyp.clear();
  document.body.classList.remove("nav-open");
  document.getElementById("backdrop").hidden = true;
  renderNav();
  loadSelected(true);
}

async function loadSelected(force) {
  const chosen = allLabs().find(item => keyOf(item.project, item.lab) === selected);
  if (!chosen) {
    const first = firstActive();
    if (first && keyOf(first.project, first.lab) !== selected) {
      select(first.project.id, first.lab.id);
      return;
    }
    detail.replaceChildren(make("div", "empty", "Пока нет лабораторий"));
    return;
  }
  const version = chosen.lab.detail_version || "";
  if (!force && version === detailVersion && model) return;
  let next = details[selected];
  if (boot.live) {
    // Only show skeleton if no lab rendered yet or switching to a different lab!
    const currentRenderedKey = detail.dataset.selectedKey;
    if (!model || currentRenderedKey !== selected) {
      detail.replaceChildren(make("div", "skeleton"));
    }
    try {
      const response = await fetch("/api/labs/" + encodeURIComponent(chosen.project.id) + "/" + encodeURIComponent(chosen.lab.id) + labQuery(chosen.lab.root_id), {cache: "no-store"});
      if (!response.ok) throw new Error("details");
      next = await response.json();
      details[selected] = next;
    } catch (error) {
      setMode(false);
      if (!next) return;
    }
  }
  if (!next) {
    detail.replaceChildren(make("div", "empty", "Нет данных по лаборатории"));
    return;
  }
  detailVersion = version;
  renderDetail(next);
}

function labIsRunning() {
  return activeLabs().some(item => item.lab.status === "running" || item.lab.current_run);
}

function setMode(online, stamp) {
  updated.classList.toggle("is-live", Boolean(boot.live && online));
  updated.classList.toggle("is-down", Boolean(boot.live && !online));
  updated.classList.toggle("is-sse", Boolean(boot.live && online && sseConnected));
  const banner = document.getElementById("connection");
  if (!online && boot.live) {
    banner.hidden = false;
    banner.textContent = "Нет связи с сервером. Показаны последние загруженные данные.";
    modeText.textContent = "Нет связи";
    return;
  }
  banner.hidden = true;
  if (!boot.live) { modeText.textContent = "Офлайн-снимок"; return; }
  const time = stamp ? new Date(stamp * 1000).toLocaleTimeString("ru-RU") : "";
  const sseTag = sseConnected ? " · SSE" : "";
  modeText.textContent = "В эфире" + sseTag + (time ? " · обновлено " + time : "");
}

function acceptSummary(next) {
  summary = next;
  renderFilters();
  renderNav();
  if (!selected && firstActive()) selected = keyOf(firstActive().project, firstActive().lab);
  const current = allLabs().find(item => keyOf(item.project, item.lab) === selected);
  if (!current || current.lab.detail_version !== detailVersion) loadSelected(true);
}

async function refresh() {
  if (!boot.live) return;
  try {
    const headers = summary.revision ? {"If-None-Match": summary.revision} : {};
    const response = await fetch("/api/status", {cache: "no-store", headers});
    if (response.status === 304) { setMode(true); return; }
    if (!response.ok) throw new Error("status");
    const next = await response.json();
    acceptSummary(next);
    setMode(true, next.updated_at);
  } catch (error) {
    setMode(false);
  }
}

function scheduleRefresh() {
  clearTimeout(timer);
  if (!boot.live) return;
  const delay = document.hidden ? 20000 : (labIsRunning() ? POLL_ACTIVE_MS : POLL_IDLE_MS);
  timer = setTimeout(() => { refresh().finally(scheduleRefresh); }, delay);
}

function initSSE() {
  if (!boot.live || typeof EventSource === "undefined") return;
  try {
    if (sseSource) {
      sseSource.close();
      sseSource = null;
    }
    sseSource = new EventSource("/api/events");
    sseSource.onopen = () => {
      sseConnected = true;
      setMode(true);
    };
    const onEvent = () => {
      // Instantly refresh data without waiting for the 10s polling interval!
      refresh();
    };
    sseSource.onmessage = onEvent;
    sseSource.addEventListener("update", onEvent);
    sseSource.onerror = () => {
      sseConnected = false;
      // Seamlessly fallback to polling
      setMode(true);
    };
  } catch (err) {
    sseConnected = false;
  }
}

document.getElementById("lab-search").addEventListener("input", renderNav);

const topLabSelect = document.getElementById("top-lab-select");
if (topLabSelect) {
  topLabSelect.addEventListener("change", (e) => {
    const key = e.target.value;
    const match = allLabs().find(item => keyOf(item.project, item.lab) === key);
    if (match) select(match.project.id, match.lab.id);
  });
}

document.getElementById("theme-toggle").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme === "dark"
    || (!document.documentElement.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "light" : "dark";
  localStorage.setItem("autoresearch:theme", document.documentElement.dataset.theme);
});
document.getElementById("nav-toggle").addEventListener("click", () => {
  const open = document.body.classList.toggle("nav-open");
  document.getElementById("nav-toggle").setAttribute("aria-expanded", open ? "true" : "false");
  document.getElementById("backdrop").hidden = !open;
});
document.getElementById("backdrop").addEventListener("click", () => {
  document.body.classList.remove("nav-open");
  document.getElementById("backdrop").hidden = true;
});
document.addEventListener("visibilitychange", () => { if (!document.hidden && boot.live) refresh().finally(scheduleRefresh); });

renderFilters();
renderNav();
if (boot.live) {
  updated.classList.add("is-live");
  modeText.textContent = "В эфире";
  initSSE();
} else {
  modeText.textContent = "Офлайн-снимок";
}
if (!selected && firstActive()) selected = keyOf(firstActive().project, firstActive().lab);
loadSelected(true);
if (boot.live) scheduleRefresh();
