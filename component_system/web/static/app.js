document.body.addEventListener("htmx:responseError", (event) => {
  const target = event.detail.target;
  if (!target) {
    return;
  }
  target.innerHTML = `<div class="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">Request failed.</div>`;
});

function selectedSeedIdFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get("seed_id");
}

function applySelectedSeed(seedId) {
  const cards = document.querySelectorAll(".seed-card[data-seed-id]");
  cards.forEach((card) => {
    const isSelected = seedId !== null && card.dataset.seedId === seedId;
    card.classList.toggle("is-selected", isSelected);
    card.setAttribute("aria-current", isSelected ? "true" : "false");
  });
}

let dashboardPollInFlight = false;
let seedDetailPollInFlight = false;
let seedVersionsPollInFlight = false;
const lastSeedVersions = {};
const savedScrollPositions = { runs: null, timeline: null };
const INTERACTION_DEBOUNCE_MS = 3000;
let lastRunsInteraction = 0;
let lastTimelineInteraction = 0;

function seedDetailUrl(seedId) {
  const detail = document.getElementById("seed-detail");
  const template = detail?.dataset.seedDetailUrlTemplate;
  if (!template || !seedId) {
    return null;
  }
  return template.replace("__SEED_ID__", encodeURIComponent(seedId));
}

function seedVersionsUrl(seedId) {
  const detail = document.getElementById("seed-detail");
  const template = detail?.dataset.seedVersionsUrlTemplate;
  if (!template || !seedId) return null;
  return template.replace("__SEED_ID__", encodeURIComponent(seedId));
}

function seedRunsUrl(seedId) {
  const detail = document.getElementById("seed-detail");
  const template = detail?.dataset.seedRunsUrlTemplate;
  if (!template || !seedId) return null;
  return template.replace("__SEED_ID__", encodeURIComponent(seedId));
}

function seedTimelineUrl(seedId) {
  const detail = document.getElementById("seed-detail");
  const template = detail?.dataset.seedTimelineUrlTemplate;
  if (!template || !seedId) return null;
  return template.replace("__SEED_ID__", encodeURIComponent(seedId));
}

function isLogViewerOpen() {
  const target = document.getElementById("seed-detail");
  if (!target) {
    return false;
  }
  if (target.querySelector('[data-log-viewer-open="true"]')) {
    return true;
  }
  if (target.querySelector("[data-log-stream]")) {
    return true;
  }
  const seedId = selectedSeedIdFromUrl();
  return Boolean(seedId && localStorage.getItem(`seed-active-run-${seedId}`));
}

function dashboardBoardUrl() {
  const board = document.getElementById("dashboard-board");
  const base = board?.dataset.dashboardPartialUrl;
  if (!base) {
    return null;
  }
  const seedId = selectedSeedIdFromUrl();
  if (!seedId) {
    return base;
  }
  const separator = base.includes("?") ? "&" : "?";
  return `${base}${separator}seed_id=${encodeURIComponent(seedId)}`;
}

function pollDashboardBoard() {
  const target = document.getElementById("dashboard-board");
  const url = dashboardBoardUrl();
  if (!target || !url || dashboardPollInFlight) {
    return;
  }
  dashboardPollInFlight = true;
  htmx
    .ajax("GET", url, { target: "#dashboard-board", swap: "outerHTML" })
    .finally(() => {
      dashboardPollInFlight = false;
    });
}

function pollSeedDetail() {
  const seedId = selectedSeedIdFromUrl();
  const target = document.getElementById("seed-detail");
  const url = seedDetailUrl(seedId);
  if (!target || !url || seedDetailPollInFlight) {
    return;
  }
  if (isLogViewerOpen()) {
    return;
  }
  seedDetailPollInFlight = true;
  htmx.ajax("GET", url, { target: "#seed-detail", swap: "innerHTML" }).finally(() => {
    seedDetailPollInFlight = false;
  });
}

function applyRunsPartial(seedId) {
  const listEl = document.getElementById("seed-runs-list");
  const paneEl = document.getElementById("seed-runs-scroll-pane");
  const url = seedRunsUrl(seedId);
  if (!listEl || !url) return Promise.resolve();
  savedScrollPositions.runs = paneEl ? paneEl.scrollTop : null;
  return htmx.ajax("GET", url, { target: "#seed-runs-list", swap: "innerHTML" });
}

function applyTimelinePartial(seedId) {
  const listEl = document.getElementById("seed-timeline-list");
  const paneEl = document.getElementById("seed-timeline-scroll-pane");
  const url = seedTimelineUrl(seedId);
  if (!listEl || !url) return Promise.resolve();
  savedScrollPositions.timeline = paneEl ? paneEl.scrollTop : null;
  return htmx.ajax("GET", url, { target: "#seed-timeline-list", swap: "innerHTML" });
}

function pollSeedDetailSections() {
  const seedId = selectedSeedIdFromUrl();
  if (!seedId || isLogViewerOpen()) return;
  const versionsUrl = seedVersionsUrl(seedId);
  if (!versionsUrl || seedVersionsPollInFlight) return;
  seedVersionsPollInFlight = true;
  fetch(versionsUrl)
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      if (!data) return;
      const prev = lastSeedVersions[seedId] || {};
      const runsChanged = data.runs_version !== prev.runs_version;
      const timelineChanged = data.timeline_version !== prev.timeline_version;
      lastSeedVersions[seedId] = {
        runs_version: data.runs_version,
        timeline_version: data.timeline_version,
      };
      const now = Date.now();
      const runsIdle = now - lastRunsInteraction >= INTERACTION_DEBOUNCE_MS;
      const timelineIdle = now - lastTimelineInteraction >= INTERACTION_DEBOUNCE_MS;
      const promises = [];
      if (runsChanged && runsIdle) promises.push(applyRunsPartial(seedId));
      if (timelineChanged && timelineIdle) promises.push(applyTimelinePartial(seedId));
      return Promise.all(promises);
    })
    .finally(() => {
      seedVersionsPollInFlight = false;
    });
}

function attachScrollPaneInteractionGuards() {
  const runsPane = document.getElementById("seed-runs-scroll-pane");
  const timelinePane = document.getElementById("seed-timeline-scroll-pane");
  function onRunsActivity() {
    lastRunsInteraction = Date.now();
  }
  function onTimelineActivity() {
    lastTimelineInteraction = Date.now();
  }
  runsPane?.addEventListener("scroll", onRunsActivity, { passive: true });
  runsPane?.addEventListener("mouseenter", onRunsActivity);
  runsPane?.addEventListener("focusin", onRunsActivity);
  timelinePane?.addEventListener("scroll", onTimelineActivity, { passive: true });
  timelinePane?.addEventListener("mouseenter", onTimelineActivity);
  timelinePane?.addEventListener("focusin", onTimelineActivity);
}

function pollDashboard() {
  if (document.hidden) return;
  if (isLogViewerOpen()) return;
  pollDashboardBoard();
  const seedId = selectedSeedIdFromUrl();
  if (seedId && document.getElementById("seed-runs-list")) {
    pollSeedDetailSections();
  } else if (seedId && !document.getElementById("seed-runs-list")) {
    pollSeedDetail();
  }
}

document.body.addEventListener("htmx:beforeRequest", (event) => {
  const target = event.detail?.target;
  if (!target || !isLogViewerOpen()) {
    return;
  }
  // Pause daemon status auto-refresh while viewing logs.
  if (target.id === "daemon-status-panel") {
    event.preventDefault();
  }
});

document.body.addEventListener("click", (event) => {
  const card = event.target.closest(".seed-card[data-seed-id]");
  if (!card) {
    return;
  }
  applySelectedSeed(card.dataset.seedId);
});

document.body.addEventListener("htmx:afterSettle", (event) => {
  const target = event.detail?.target;
  if (!target) return;
  if (target.id === "seed-detail") {
    applySelectedSeed(selectedSeedIdFromUrl());
    attachScrollPaneInteractionGuards();
    return;
  }
  if (target.id === "seed-runs-list") {
    const pane = document.getElementById("seed-runs-scroll-pane");
    if (pane && savedScrollPositions.runs != null) {
      pane.scrollTop = savedScrollPositions.runs;
      savedScrollPositions.runs = null;
    }
    initializeLogStreams(target.closest("#seed-detail") || document);
    return;
  }
  if (target.id === "seed-timeline-list") {
    const pane = document.getElementById("seed-timeline-scroll-pane");
    if (pane && savedScrollPositions.timeline != null) {
      pane.scrollTop = savedScrollPositions.timeline;
      savedScrollPositions.timeline = null;
    }
    return;
  }
});

window.addEventListener("popstate", () => {
  applySelectedSeed(selectedSeedIdFromUrl());
});

applySelectedSeed(selectedSeedIdFromUrl());
attachScrollPaneInteractionGuards();
window.setInterval(pollDashboard, 5000);

const logStreamIntervals = new Map();
const logStreamState = new Map();
const ansiCtor = window.AnsiUp || window.ansi_up?.AnsiUp || null;
const ansiRenderer = ansiCtor ? new ansiCtor() : null;

if (ansiRenderer && Object.prototype.hasOwnProperty.call(ansiRenderer, "escape_html")) {
  ansiRenderer.escape_html = true;
}

function stripAnsiSequences(value) {
  // CSI: \x1b[...m, OSC: \x1b]...\x07 or \x1b\ ; then any remaining ESC controls.
  return (value || "")
    .replace(/\u001b\][^\u0007]*(?:\u0007|\u001b\\)/g, "")
    .replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "")
    .replace(/\u001b[@-_]/g, "");
}

function isRunComplete(status) {
  return status === "succeeded" || status === "failed";
}

function updateLogStatus(runId, text) {
  const nodes = document.querySelectorAll(`[data-log-status][data-run-id="${runId}"]`);
  nodes.forEach((node) => {
    node.textContent = text;
  });
}

function updateCopyButtonState(runId, stream, enabled) {
  const buttons = document.querySelectorAll(
    `[data-log-copy][data-run-id="${runId}"][data-stream="${stream}"]`
  );
  buttons.forEach((button) => {
    button.disabled = !enabled;
  });
}

function appendLogContent(pre, chunk) {
  const currentRaw = pre.dataset.rawLog || "";
  const nextRaw = currentRaw + (chunk || "");

  // Keep the viewer responsive for very large logs.
  const maxChars = 200_000;
  const trimmedRaw =
    nextRaw.length > maxChars ? nextRaw.slice(nextRaw.length - maxChars) : nextRaw;

  pre.dataset.rawLog = trimmedRaw;
  if (ansiRenderer) {
    pre.innerHTML = ansiRenderer.ansi_to_html(trimmedRaw);
  } else {
    pre.textContent = stripAnsiSequences(trimmedRaw);
  }

  pre.scrollTop = pre.scrollHeight;
}

async function pollLogStream(pre) {
  const runId = pre.dataset.runId;
  const stream = pre.dataset.stream || "stdout";
  if (!runId) {
    return;
  }

  const state = logStreamState.get(pre) || { offset: 0, complete: false };
  const response = await fetch(
    `/component-system/api/runs/${encodeURIComponent(runId)}/log?stream=${encodeURIComponent(stream)}&offset=${state.offset}`
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch logs for ${runId}: ${response.status}`);
  }

  const payload = await response.json();
  const chunk = payload.chunk || "";
  const nextOffset = Number(payload.next_offset || 0);
  const complete = Boolean(payload.complete);

  appendLogContent(pre, chunk);
  updateCopyButtonState(runId, stream, pre.textContent.length > 0);
  logStreamState.set(pre, { offset: nextOffset, complete });

  if (complete) {
    updateLogStatus(runId, "Completed");
    const intervalId = logStreamIntervals.get(pre);
    if (intervalId) {
      clearInterval(intervalId);
      logStreamIntervals.delete(pre);
    }
    return;
  }

  if (chunk) {
    updateLogStatus(runId, "Streaming...");
  } else {
    updateLogStatus(runId, "Waiting for log output...");
  }
}

function cleanupDetachedLogStreams() {
  for (const [pre, intervalId] of logStreamIntervals.entries()) {
    if (!document.body.contains(pre)) {
      clearInterval(intervalId);
      logStreamIntervals.delete(pre);
      logStreamState.delete(pre);
    }
  }
}

function initializeLogCopyButtons(root) {
  root.querySelectorAll("[data-log-copy]").forEach((button) => {
    if (button.dataset.logCopyReady === "true") {
      return;
    }
    button.dataset.logCopyReady = "true";
    button.addEventListener("click", async () => {
      const runId = button.dataset.runId;
      if (!runId) {
        return;
      }
      const stream = button.dataset.stream || "stdout";
      const pre = root.querySelector(
        `[data-log-stream][data-run-id="${runId}"][data-stream="${stream}"]`
      );
      if (!pre || !pre.textContent) {
        return;
      }
      try {
        await navigator.clipboard.writeText(pre.textContent);
        const labelBefore = button.textContent;
        button.textContent = "Copied!";
        setTimeout(() => {
          button.textContent = labelBefore || "Copy";
        }, 1200);
      } catch (error) {
        console.error("Failed to copy log output", error);
      }
    });
  });
}

async function loadPromptContent(pre) {
  const runId = pre.dataset.runId;
  if (!runId) return;
  try {
    const response = await fetch(
      `/component-system/api/runs/${encodeURIComponent(runId)}/prompt`
    );
    if (!response.ok) return;
    const payload = await response.json();
    const content = payload.content ?? "";
    pre.textContent = content;
    const copyBtn = document.querySelector(
      `[data-prompt-copy][data-run-id="${runId}"]`
    );
    if (copyBtn) copyBtn.disabled = false;
  } catch (err) {
    console.error("Failed to load prompt for run", runId, err);
  }
}

function initializePromptDisplays(root) {
  root.querySelectorAll("[data-prompt-content]").forEach((pre) => {
    if (pre.dataset.promptLoaded === "true") return;
    pre.dataset.promptLoaded = "true";
    loadPromptContent(pre);
  });
  root.querySelectorAll("[data-prompt-copy]").forEach((button) => {
    if (button.dataset.promptCopyReady === "true") return;
    button.dataset.promptCopyReady = "true";
    button.addEventListener("click", async () => {
      const runId = button.dataset.runId;
      if (!runId) return;
      const pre = root.querySelector(
        `[data-prompt-content][data-run-id="${runId}"]`
      );
      if (!pre || !pre.textContent) return;
      try {
        await navigator.clipboard.writeText(pre.textContent);
        const labelBefore = button.textContent;
        button.textContent = "Copied!";
        setTimeout(() => {
          button.textContent = labelBefore || "Copy";
        }, 1200);
      } catch (err) {
        console.error("Failed to copy prompt", err);
      }
    });
  });
}

function initializeLogStreams(root = document) {
  cleanupDetachedLogStreams();
  initializeLogCopyButtons(root);
  initializePromptDisplays(root);

  root.querySelectorAll("[data-log-stream]").forEach((pre) => {
    if (pre.dataset.logStreamReady === "true") {
      return;
    }
    pre.dataset.logStreamReady = "true";
    const runStatus = pre.dataset.runStatus || "";
    const runId = pre.dataset.runId;
    if (!runId) {
      return;
    }

    if (isRunComplete(runStatus)) {
      updateLogStatus(runId, "Completed");
    } else {
      updateLogStatus(runId, "Connecting...");
    }

    const runPoll = async () => {
      try {
        await pollLogStream(pre);
      } catch (error) {
        updateLogStatus(runId, "Log fetch failed");
        console.error(error);
      }
    };

    runPoll();
    const intervalId = window.setInterval(runPoll, 2000);
    logStreamIntervals.set(pre, intervalId);
  });
}

function observeLogStreamMounts() {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type !== "childList" || mutation.addedNodes.length === 0) {
        continue;
      }
      for (const node of mutation.addedNodes) {
        if (!(node instanceof Element)) {
          continue;
        }
        if (
          node.matches?.("[data-log-stream], [data-log-copy], [data-prompt-content], [data-prompt-copy]") ||
          node.querySelector?.("[data-log-stream], [data-log-copy], [data-prompt-content], [data-prompt-copy]")
        ) {
          initializeLogStreams(node);
          return;
        }
      }
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
}

document.body.addEventListener("htmx:afterSettle", (event) => {
  const target = event.detail?.target;
  if (!target) {
    return;
  }
  if (target.id === "seed-detail") {
    initializeLogStreams(target);
  }
});

initializeLogStreams(document);
observeLogStreamMounts();
