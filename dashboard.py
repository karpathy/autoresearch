"""
Autoresearch Dashboard — single-file FastAPI server.
Reads ~/.hermes/autoresearch (or REPO_DIR env var).

Usage:
    cd ~/.hermes/autoresearch
    uv run dashboard.py

Then open http://localhost:7788
"""

import asyncio
import csv
import io
import json
import os
import re
import signal
import subprocess
import textwrap
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_DIR = Path(os.environ.get("REPO_DIR", str(Path(__file__).parent)))
RESULTS_TSV = REPO_DIR / "results.tsv"
RUN_LOG = REPO_DIR / "run.log"
TRAIN_PY = REPO_DIR / "train.py"
PROGRAM_MD = REPO_DIR / "program.md"

app = FastAPI()

# ---------------------------------------------------------------------------
# API — Results
# ---------------------------------------------------------------------------

@app.get("/api/results")
def get_results():
    if not RESULTS_TSV.exists():
        return []
    rows = []
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                r["val_bpb"] = float(r.get("val_bpb", 0))
            except (ValueError, TypeError):
                pass
            try:
                r["memory_gb"] = float(r.get("memory_gb", 0))
            except (ValueError, TypeError):
                pass
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# API — Live training data (from run.log)
# ---------------------------------------------------------------------------

@app.get("/api/live")
def get_live():
    if not RUN_LOG.exists():
        return {"steps": [], "summary": None, "running": False, "raw_tail": ""}

    text = RUN_LOG.read_text(errors="replace")

    # Parse step lines
    step_pat = re.compile(
        r"step (\d+) \(([\d.]+)%\) \| loss: ([\d.]+) \| lrm: ([\d.]+) \| "
        r"dt: (\d+)ms \| tok/sec: ([\d,]+) \| mfu: ([\d.]+)% \| epoch: (\d+) \| remaining: ([\d.]+)s"
    )
    steps = []
    for m in step_pat.finditer(text):
        steps.append({
            "step": int(m.group(1)),
            "progress": float(m.group(2)),
            "loss": float(m.group(3)),
            "lrm": float(m.group(4)),
            "dt_ms": int(m.group(5)),
            "tok_sec": int(m.group(6).replace(",", "")),
            "mfu": float(m.group(7)),
            "epoch": int(m.group(8)),
            "remaining": float(m.group(9)),
        })

    # Parse summary block
    summary = None
    sum_pat = re.compile(
        r"^(val_bpb|training_seconds|total_seconds|peak_vram_mb|mfu_percent|"
        r"total_tokens_M|num_steps|num_params_M|depth):\s+(.+)$",
        re.MULTILINE,
    )
    matches = dict(sum_pat.findall(text))
    if matches:
        summary = {k: float(v) for k, v in matches.items()}

    # Check if train.py is actually running
    try:
        r = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, timeout=5)
        running = r.returncode == 0
    except Exception:
        running = False

    # Last 60 lines for raw log stream
    lines = text.splitlines()
    raw_tail = "\n".join(lines[-60:]) if lines else ""

    return {
        "steps": steps[-500:],   # cap for JSON size
        "summary": summary,
        "running": running,
        "raw_tail": raw_tail,
    }


# ---------------------------------------------------------------------------
# API — SSE log stream (live tail of run.log)
# ---------------------------------------------------------------------------

@app.get("/api/stream")
async def stream_log():
    async def event_gen():
        offset = 0
        if RUN_LOG.exists():
            offset = RUN_LOG.stat().st_size
        while True:
            await asyncio.sleep(1)
            if not RUN_LOG.exists():
                continue
            size = RUN_LOG.stat().st_size
            if size > offset:
                with open(RUN_LOG, "rb") as f:
                    f.seek(offset)
                    chunk = f.read(size - offset).decode("utf-8", errors="replace")
                offset = size
                for line in chunk.splitlines():
                    line = line.strip()
                    if line:
                        yield f"data: {json.dumps(line)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# API — Git log
# ---------------------------------------------------------------------------

@app.get("/api/git-log")
def get_git_log():
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "--no-decorate", "-40"],
            capture_output=True, text=True, cwd=REPO_DIR, timeout=10,
        )
        commits = []
        for line in r.stdout.strip().splitlines():
            if line:
                parts = line.split(" ", 1)
                commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
        return commits
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/diff/{commit}")
def get_diff(commit: str):
    if not re.match(r"^[a-f0-9]{7,40}$", commit):
        return JSONResponse({"error": "invalid commit"}, status_code=400)
    try:
        r = subprocess.run(
            ["git", "diff", f"{commit}~1", commit, "--", "train.py"],
            capture_output=True, text=True, cwd=REPO_DIR, timeout=10,
        )
        return {"commit": commit, "diff": r.stdout or "(no changes to train.py)"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# API — GPU / system status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    gpu_info = []
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpu_info.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_pct": int(parts[2]),
                    "mem_used_mb": int(parts[3]),
                    "mem_total_mb": int(parts[4]),
                    "temp_c": int(parts[5]),
                })
    except Exception:
        pass

    branch = ""
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=REPO_DIR, timeout=5,
        )
        branch = r.stdout.strip()
    except Exception:
        pass

    exp_count = 0
    best_bpb = None
    if RESULTS_TSV.exists():
        with open(RESULTS_TSV) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                exp_count += 1
                try:
                    v = float(row.get("val_bpb", 0))
                    if v > 0 and (best_bpb is None or v < best_bpb):
                        best_bpb = v
                except (ValueError, TypeError):
                    pass

    try:
        running = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        running = False

    return {
        "gpus": gpu_info,
        "branch": branch,
        "experiment_count": exp_count,
        "best_bpb": best_bpb,
        "running": running,
        "repo_dir": str(REPO_DIR),
    }


# ---------------------------------------------------------------------------
# API — Hyperparams (parsed from train.py)
# ---------------------------------------------------------------------------

@app.get("/api/hyperparams")
def get_hyperparams():
    if not TRAIN_PY.exists():
        return {}
    text = TRAIN_PY.read_text()
    # Extract UPPER_CASE = value assignments from the hyperparams section
    params = {}
    pat = re.compile(r"^([A-Z_]{3,})\s*=\s*(.+?)(?:\s*#.*)?$", re.MULTILINE)
    for m in pat.finditer(text):
        name, val = m.group(1), m.group(2).strip()
        # Skip internal torch/os stuff
        if name.startswith("os") or name.startswith("torch"):
            continue
        try:
            params[name] = json.loads(val)
        except Exception:
            params[name] = val
    return params


# ---------------------------------------------------------------------------
# API — Launch / stop controls
# ---------------------------------------------------------------------------

_agent_proc = None


@app.post("/api/launch")
async def launch_agent(req: Request):
    global _agent_proc
    if _agent_proc and _agent_proc.poll() is None:
        return {"status": "already_running", "pid": _agent_proc.pid}

    body = await req.json()
    run_tag = body.get("run_tag", "")
    max_experiments = body.get("max_experiments", 0)
    model = body.get("model", "")

    if not run_tag:
        return JSONResponse({"error": "run_tag required"}, status_code=400)

    # Build a prompt that tells the agent to set up and start the loop
    prompt = f"Read program.md and set up the autoresearch run with tag '{run_tag}'. Then start the experiment loop immediately and never stop."
    if max_experiments:
        prompt += f" Run at most {max_experiments} experiments."
    if model:
        prompt += f" Use model {model}."

    # Launch hermes as a subprocess so this server stays alive
    env = os.environ.copy()
    try:
        _agent_proc = subprocess.Popen(
            ["hermes", "--no-banner", "--quiet", prompt],
            cwd=REPO_DIR,
            env=env,
            stdout=open(RUN_LOG.parent / "agent.log", "w"),
            stderr=subprocess.STDOUT,
        )
        return {"status": "launched", "pid": _agent_proc.pid}
    except FileNotFoundError:
        # Fall back to just running train.py directly for demo
        _agent_proc = subprocess.Popen(
            ["uv", "run", "train.py"],
            cwd=REPO_DIR,
            env=env,
            stdout=open(RUN_LOG, "w"),
            stderr=subprocess.STDOUT,
        )
        return {"status": "launched_train_direct", "pid": _agent_proc.pid}


@app.post("/api/stop")
def stop_agent():
    global _agent_proc
    killed = []
    # Kill any running train.py processes
    try:
        r = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True, timeout=5)
        for pid in r.stdout.strip().splitlines():
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed.append(int(pid))
            except Exception:
                pass
    except Exception:
        pass
    if _agent_proc and _agent_proc.poll() is None:
        _agent_proc.terminate()
        killed.append(_agent_proc.pid)
    return {"status": "stopped", "killed": killed}


# ---------------------------------------------------------------------------
# API — Read / write train.py and program.md
# ---------------------------------------------------------------------------

@app.get("/api/file/{filename}")
def read_file(filename: str):
    if filename == "train.py":
        path = TRAIN_PY
    elif filename == "program.md":
        path = PROGRAM_MD
    else:
        return JSONResponse({"error": "not allowed"}, status_code=403)
    if not path.exists():
        return {"content": ""}
    return {"content": path.read_text()}


@app.post("/api/file/{filename}")
async def write_file(filename: str, req: Request):
    if filename not in ("program.md",):
        return JSONResponse({"error": "only program.md can be edited here"}, status_code=403)
    body = await req.json()
    content = body.get("content", "")
    path = REPO_DIR / filename
    path.write_text(content)
    return {"status": "saved"}


# ---------------------------------------------------------------------------
# Dashboard HTML (fully embedded, no CDN required — uses vanilla JS + Chart.js CDN)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Autoresearch Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --dim: #7d8590; --green: #3fb950; --red: #f85149;
    --amber: #d29922; --cyan: #58a6ff; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font: 14px/1.5 'Segoe UI', system-ui, sans-serif; }
  a { color: var(--cyan); text-decoration: none; }

  /* Layout */
  #app { display: flex; height: 100vh; overflow: hidden; }
  #sidebar { width: 230px; min-width: 180px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto; }
  #main { flex: 1; overflow-y: auto; padding: 20px; }

  /* Sidebar */
  .sb-title { padding: 18px 16px 8px; font-size: 15px; font-weight: 700; color: var(--cyan); letter-spacing: .5px; }
  .sb-section { padding: 12px 16px 4px; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--dim); }
  .sb-nav a { display: block; padding: 7px 20px; color: var(--dim); font-size: 13px; cursor: pointer; border-left: 3px solid transparent; transition: .15s; }
  .sb-nav a:hover, .sb-nav a.active { color: var(--text); background: rgba(255,255,255,.04); border-left-color: var(--cyan); }
  .sb-kpi { padding: 8px 16px; }
  .sb-kpi .label { font-size: 11px; color: var(--dim); }
  .sb-kpi .value { font-size: 16px; font-weight: 600; }
  #status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; background: var(--dim); }
  #status-dot.running { background: var(--green); box-shadow: 0 0 6px var(--green); }

  /* Panels */
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; overflow: hidden; }
  .panel-head { padding: 12px 16px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 13px; display: flex; align-items: center; gap: 8px; justify-content: space-between; }
  .panel-body { padding: 16px; }
  .page { display: none; }
  .page.active { display: block; }

  /* KPIs */
  .kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
  .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; min-width: 130px; flex: 1; }
  .kpi .k-label { font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: .5px; }
  .kpi .k-val { font-size: 22px; font-weight: 700; margin-top: 2px; }
  .kpi .k-sub { font-size: 11px; color: var(--dim); margin-top: 2px; }

  /* Table */
  .exp-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .exp-table th { padding: 8px 12px; text-align: left; color: var(--dim); font-weight: 500; border-bottom: 1px solid var(--border); font-size: 12px; }
  .exp-table td { padding: 8px 12px; border-bottom: 1px solid rgba(48,54,61,.5); }
  .exp-table tr:last-child td { border-bottom: none; }
  .exp-table tr:hover td { background: rgba(255,255,255,.02); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .badge.keep { background: rgba(63,185,80,.15); color: var(--green); }
  .badge.discard { background: rgba(248,81,73,.15); color: var(--red); }
  .badge.crash { background: rgba(210,153,34,.15); color: var(--amber); }
  .best-row td { background: rgba(63,185,80,.06) !important; }
  .commit-link { font-family: monospace; font-size: 12px; color: var(--cyan); cursor: pointer; }

  /* Log stream */
  #log-output { background: #010409; border-radius: 6px; padding: 12px; height: 400px; overflow-y: auto; font-family: monospace; font-size: 12px; white-space: pre-wrap; word-break: break-all; }
  .log-step { color: var(--amber); }
  .log-err { color: var(--red); }
  .log-git { color: var(--cyan); }
  .log-sum { color: var(--green); font-weight: 600; }

  /* Charts */
  .chart-wrap { position: relative; height: 260px; }

  /* Controls */
  .ctrl-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .form-group { display: flex; flex-direction: column; gap: 4px; }
  label { font-size: 12px; color: var(--dim); }
  input, select, textarea { background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 8px 10px; font-size: 13px; width: 100%; outline: none; transition: border-color .15s; }
  input:focus, select:focus, textarea:focus { border-color: var(--cyan); }
  textarea { font-family: monospace; font-size: 12px; resize: vertical; }
  .btn { padding: 8px 18px; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity .15s; }
  .btn:hover { opacity: .85; }
  .btn-green { background: var(--green); color: #0d1117; }
  .btn-red { background: var(--red); color: #fff; }
  .btn-dim { background: var(--border); color: var(--text); }
  .btn-row { display: flex; gap: 8px; flex-wrap: wrap; }

  /* GPU bar */
  .gpu-bar-wrap { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-top: 4px; }
  .gpu-bar { height: 100%; background: var(--green); border-radius: 4px; transition: width .5s; }
  .gpu-bar.hot { background: var(--red); }

  /* Diff */
  #diff-output { background: #010409; border-radius: 6px; padding: 12px; max-height: 500px; overflow-y: auto; font-family: monospace; font-size: 12px; white-space: pre; }
  .diff-add { color: var(--green); }
  .diff-del { color: var(--red); }
  .diff-meta { color: var(--cyan); }

  /* Progress bar */
  .progress-wrap { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .progress-bar { height: 100%; background: var(--cyan); border-radius: 3px; transition: width .5s; }

  /* Responsive */
  @media (max-width: 700px) {
    #sidebar { width: 52px; }
    .sb-title, .sb-section, .sb-nav a span, .sb-kpi .label, .sb-kpi .value { display: none; }
  }
</style>
</head>
<body>
<div id="app">

<!-- Sidebar -->
<div id="sidebar">
  <div class="sb-title">⚗ AutoResearch</div>
  <div class="sb-kpi">
    <div class="label">Status</div>
    <div class="value" id="sb-status"><span id="status-dot"></span><span id="sb-status-text">—</span></div>
  </div>
  <div class="sb-kpi">
    <div class="label">Best val_bpb</div>
    <div class="value" id="sb-best">—</div>
  </div>
  <div class="sb-kpi">
    <div class="label">Experiments</div>
    <div class="value" id="sb-count">—</div>
  </div>
  <div class="sb-kpi">
    <div class="label">Branch</div>
    <div class="value" style="font-size:12px;word-break:break-all" id="sb-branch">—</div>
  </div>
  <div class="sb-section">Navigate</div>
  <div class="sb-nav">
    <a onclick="showPage('overview')" class="active">📊 Overview</a>
    <a onclick="showPage('live')">⚡ Live Training</a>
    <a onclick="showPage('experiments')">🔬 Experiments</a>
    <a onclick="showPage('git')">🌿 Git History</a>
    <a onclick="showPage('controls')">⚙ Controls</a>
    <a onclick="showPage('settings')">📝 Settings</a>
  </div>
</div>

<!-- Main -->
<div id="main">

  <!-- Overview -->
  <div id="page-overview" class="page active">
    <div class="kpi-row" id="kpi-row"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
      <div class="panel">
        <div class="panel-head">val_bpb over time</div>
        <div class="panel-body"><div class="chart-wrap"><canvas id="chart-bpb"></canvas></div></div>
      </div>
      <div class="panel">
        <div class="panel-head">Keep / Discard distribution</div>
        <div class="panel-body"><div class="chart-wrap"><canvas id="chart-status"></canvas></div></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head">GPU Status</div>
      <div class="panel-body" id="gpu-overview"></div>
    </div>
  </div>

  <!-- Live -->
  <div id="page-live" class="page">
    <div class="panel">
      <div class="panel-head">
        <span>Live Training Progress</span>
        <label style="font-size:12px;color:var(--dim)">
          <input type="checkbox" id="autoscroll" checked> autoscroll
        </label>
      </div>
      <div class="panel-body">
        <div id="live-progress-wrap" style="margin-bottom:12px;display:none">
          <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--dim);margin-bottom:4px">
            <span id="live-step-label">step 0</span>
            <span id="live-pct-label">0%</span>
          </div>
          <div class="progress-wrap"><div class="progress-bar" id="live-progress-bar" style="width:0%"></div></div>
          <div style="display:flex;gap:16px;margin-top:8px;font-size:12px;flex-wrap:wrap" id="live-stats"></div>
        </div>
        <div class="chart-wrap" style="height:200px;margin-bottom:12px"><canvas id="chart-live-loss"></canvas></div>
        <div id="log-output"></div>
      </div>
    </div>
  </div>

  <!-- Experiments -->
  <div id="page-experiments" class="page">
    <div class="panel">
      <div class="panel-head">
        <span>All Experiments</span>
        <input type="text" id="exp-search" placeholder="filter..." style="width:160px;padding:4px 8px;font-size:12px">
      </div>
      <div class="panel-body" style="overflow-x:auto">
        <table class="exp-table">
          <thead><tr>
            <th>#</th><th>Commit</th><th>val_bpb</th><th>VRAM GB</th><th>Status</th><th>Description</th>
          </tr></thead>
          <tbody id="exp-tbody"></tbody>
        </table>
      </div>
    </div>
    <div class="panel" id="diff-panel" style="display:none">
      <div class="panel-head">
        <span id="diff-title">Diff</span>
        <button class="btn btn-dim" style="padding:3px 10px;font-size:12px" onclick="document.getElementById('diff-panel').style.display='none'">✕</button>
      </div>
      <div class="panel-body"><div id="diff-output"></div></div>
    </div>
  </div>

  <!-- Git -->
  <div id="page-git" class="page">
    <div class="panel">
      <div class="panel-head">Git History (last 40 commits)</div>
      <div class="panel-body" style="overflow-x:auto">
        <table class="exp-table">
          <thead><tr><th>Hash</th><th>Message</th></tr></thead>
          <tbody id="git-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Controls -->
  <div id="page-controls" class="page">
    <div class="panel">
      <div class="panel-head">Launch Autonomous Research Loop</div>
      <div class="panel-body">
        <div class="ctrl-grid" style="margin-bottom:16px">
          <div class="form-group">
            <label>Run tag (e.g. mar18)</label>
            <input type="text" id="ctrl-tag" placeholder="mar18">
          </div>
          <div class="form-group">
            <label>Model</label>
            <select id="ctrl-model">
              <option value="">Default (from config)</option>
              <option value="anthropic/claude-opus-4.6">claude-opus-4.6</option>
              <option value="anthropic/claude-sonnet-4-20250514">claude-sonnet-4</option>
              <option value="openai/gpt-4o">gpt-4o</option>
              <option value="cerebras/qwen-3-235b-a22b-instruct-2507">Qwen3-235B (Cerebras)</option>
            </select>
          </div>
          <div class="form-group">
            <label>Max experiments (0 = unlimited)</label>
            <input type="number" id="ctrl-max-exp" value="0" min="0">
          </div>
        </div>
        <div class="btn-row">
          <button class="btn btn-green" onclick="launchAgent()">▶ Launch Agent</button>
          <button class="btn btn-red" onclick="stopAgent()">■ Stop</button>
        </div>
        <div id="ctrl-result" style="margin-top:12px;font-size:13px;color:var(--dim)"></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">Current Hyperparameters (live from train.py)</div>
      <div class="panel-body">
        <table class="exp-table" id="hparam-table">
          <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody id="hparam-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Settings -->
  <div id="page-settings" class="page">
    <div class="panel">
      <div class="panel-head">program.md</div>
      <div class="panel-body">
        <textarea id="settings-program" rows="30" style="width:100%"></textarea>
        <div class="btn-row" style="margin-top:12px">
          <button class="btn btn-green" onclick="saveSettings()">Save</button>
          <div id="settings-result" style="color:var(--dim);font-size:12px;align-self:center;margin-left:8px"></div>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head">train.py (read-only)</div>
      <div class="panel-body">
        <textarea id="settings-train" rows="30" style="width:100%" readonly></textarea>
      </div>
    </div>
  </div>

</div><!-- /main -->
</div><!-- /app -->

<script>
// ---- State ----
let chartBpb = null, chartStatus = null, chartLiveLoss = null;
let liveSteps = [];
let allResults = [];
let logBuffer = [];
const MAX_LOG = 500;
let eventSource = null;

// ---- Page nav ----
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sb-nav a').forEach(a => a.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  const links = document.querySelectorAll('.sb-nav a');
  for (const l of links) {
    if (l.getAttribute('onclick') && l.getAttribute('onclick').includes(name)) l.classList.add('active');
  }
  if (name === 'git') loadGit();
  if (name === 'settings') loadSettings();
  if (name === 'controls') loadHparams();
}

// ---- Init charts ----
function initCharts() {
  const darkBg = 'transparent';
  const gridColor = 'rgba(48,54,61,.6)';
  const baseOpts = {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { labels: { color: '#7d8590', font: { size: 11 } } } },
    scales: {
      x: { grid: { color: gridColor }, ticks: { color: '#7d8590', font: { size: 10 } } },
      y: { grid: { color: gridColor }, ticks: { color: '#7d8590', font: { size: 10 } } },
    }
  };

  chartBpb = new Chart(document.getElementById('chart-bpb'), {
    type: 'scatter',
    data: { datasets: [
      { label: 'KEEP', data: [], pointBackgroundColor: '#3fb950', pointRadius: 6 },
      { label: 'DISCARD', data: [], pointBackgroundColor: '#f85149', pointRadius: 4, pointStyle: 'crossRot' },
    ]},
    options: { ...baseOpts, plugins: { ...baseOpts.plugins, tooltip: { callbacks: {
      label: ctx => {
        const p = ctx.raw;
        return `#${p.idx} ${p.desc || ''}: ${p.y?.toFixed(6)}`;
      }
    }}}}
  });

  chartStatus = new Chart(document.getElementById('chart-status'), {
    type: 'doughnut',
    data: { labels: ['keep', 'discard', 'crash'], datasets: [{ data: [0,0,0], backgroundColor: ['#3fb950','#f85149','#d29922'], borderWidth: 0 }]},
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#7d8590' } } } }
  });

  chartLiveLoss = new Chart(document.getElementById('chart-live-loss'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'loss', data: [], borderColor: '#d29922', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
      { label: 'mfu%', data: [], borderColor: '#58a6ff', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y2' },
    ]},
    options: {
      ...baseOpts,
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: '#7d8590', font: { size: 10 }, maxTicksLimit: 8 } },
        y: { grid: { color: gridColor }, ticks: { color: '#7d8590', font: { size: 10 } }, title: { display: true, text: 'loss', color: '#d29922' } },
        y2: { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#58a6ff', font: { size: 10 } }, title: { display: true, text: 'mfu%', color: '#58a6ff' } },
      }
    }
  });
}

// ---- Results ----
async function loadResults() {
  const res = await fetch('/api/results');
  allResults = await res.json();
  renderResultsTable(allResults);
  updateOverviewCharts(allResults);
  updateKpis(allResults);
}

function updateKpis(results) {
  const keeps = results.filter(r => r.status === 'keep' || r.status === 'baseline');
  const discards = results.filter(r => r.status === 'discard');
  const crashes = results.filter(r => r.status === 'crash');
  const bpbs = keeps.map(r => r.val_bpb).filter(v => v > 0);
  const best = bpbs.length ? Math.min(...bpbs) : null;
  const baseline = bpbs.length ? bpbs[0] : null;
  const pctImprove = best && baseline ? ((baseline - best) / baseline * 100).toFixed(2) : null;

  document.getElementById('kpi-row').innerHTML = `
    <div class="kpi"><div class="k-label">Best val_bpb</div><div class="k-val" style="color:var(--green)">${best ? best.toFixed(6) : '—'}</div></div>
    <div class="kpi"><div class="k-label">Baseline val_bpb</div><div class="k-val">${baseline ? baseline.toFixed(6) : '—'}</div></div>
    <div class="kpi"><div class="k-label">Improvement</div><div class="k-val" style="color:var(--cyan)">${pctImprove ? pctImprove + '%' : '—'}</div></div>
    <div class="kpi"><div class="k-label">Total Runs</div><div class="k-val">${results.length}</div><div class="k-sub">${keeps.length} kept · ${discards.length} discarded · ${crashes.length} crashed</div></div>
  `;
}

function updateOverviewCharts(results) {
  let keepPts = [], discardPts = [];
  let keep = 0, discard = 0, crash = 0;
  results.forEach((r, i) => {
    const pt = { x: i + 1, y: r.val_bpb, idx: i + 1, desc: r.description };
    if (r.status === 'keep' || r.status === 'baseline') { keepPts.push(pt); keep++; }
    else if (r.status === 'discard') { discardPts.push(pt); discard++; }
    else crash++;
  });
  chartBpb.data.datasets[0].data = keepPts;
  chartBpb.data.datasets[1].data = discardPts;
  chartBpb.update('none');
  chartStatus.data.datasets[0].data = [keep, discard, crash];
  chartStatus.update('none');
}

function renderResultsTable(results) {
  const bpbs = results.filter(r => r.val_bpb > 0).map(r => r.val_bpb);
  const bestBpb = bpbs.length ? Math.min(...bpbs) : null;
  const tbody = document.getElementById('exp-tbody');
  tbody.innerHTML = '';
  const filter = document.getElementById('exp-search').value.toLowerCase();
  const filtered = results.filter(r =>
    !filter || JSON.stringify(r).toLowerCase().includes(filter)
  );
  [...filtered].reverse().forEach((r, i) => {
    const realIdx = results.length - i;
    const isBest = r.val_bpb === bestBpb;
    const tr = document.createElement('tr');
    if (isBest) tr.className = 'best-row';
    tr.innerHTML = `
      <td>${realIdx}</td>
      <td><span class="commit-link" onclick="showDiff('${r.commit}')">${r.commit}</span></td>
      <td><b>${r.val_bpb > 0 ? r.val_bpb.toFixed(6) : '—'}</b> ${isBest ? '★' : ''}</td>
      <td>${r.memory_gb > 0 ? r.memory_gb.toFixed(1) : '—'}</td>
      <td><span class="badge ${r.status}">${r.status}</span></td>
      <td style="color:var(--dim)">${r.description || ''}</td>
    `;
    tbody.appendChild(tr);
  });
}

document.getElementById('exp-search').addEventListener('input', () => renderResultsTable(allResults));

// ---- Diff ----
async function showDiff(commit) {
  const res = await fetch(`/api/diff/${commit}`);
  const data = await res.json();
  const panel = document.getElementById('diff-panel');
  document.getElementById('diff-title').textContent = `Diff — ${commit}`;
  panel.style.display = '';
  const out = document.getElementById('diff-output');
  out.innerHTML = '';
  (data.diff || '').split('\n').forEach(line => {
    const span = document.createElement('span');
    if (line.startsWith('+')) span.className = 'diff-add';
    else if (line.startsWith('-')) span.className = 'diff-del';
    else if (line.startsWith('@@') || line.startsWith('diff')) span.className = 'diff-meta';
    span.textContent = line + '\n';
    out.appendChild(span);
  });
  panel.scrollIntoView({ behavior: 'smooth' });
}

// ---- Live training ----
function initSSE() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource('/api/stream');
  eventSource.onmessage = e => {
    const line = JSON.parse(e.data);
    appendLog(line);
  };
  eventSource.onerror = () => {
    setTimeout(initSSE, 3000);
  };
}

function appendLog(line) {
  logBuffer.push(line);
  if (logBuffer.length > MAX_LOG) logBuffer.shift();
  const el = document.getElementById('log-output');
  const span = document.createElement('span');
  if (/^step \d+/.test(line)) span.className = 'log-step';
  else if (/val_bpb|training_seconds|peak_vram_mb/.test(line)) span.className = 'log-sum';
  else if (/error|Error|FAIL/.test(line)) span.className = 'log-err';
  else if (/git|branch|commit/.test(line)) span.className = 'log-git';
  span.textContent = line + '\n';
  el.appendChild(span);
  if (document.getElementById('autoscroll').checked) el.scrollTop = el.scrollHeight;
  if (el.childNodes.length > MAX_LOG) el.removeChild(el.firstChild);
}

async function pollLive() {
  const res = await fetch('/api/live');
  const data = await res.json();
  liveSteps = data.steps;
  updateLiveChart(liveSteps);
  if (liveSteps.length) {
    const last = liveSteps[liveSteps.length - 1];
    const wrap = document.getElementById('live-progress-wrap');
    wrap.style.display = '';
    document.getElementById('live-step-label').textContent = `step ${last.step}`;
    document.getElementById('live-pct-label').textContent = `${last.progress.toFixed(1)}%`;
    document.getElementById('live-progress-bar').style.width = last.progress + '%';
    document.getElementById('live-stats').innerHTML = `
      <span style="color:var(--amber)">loss: <b>${last.loss.toFixed(4)}</b></span>
      <span style="color:var(--cyan)">mfu: <b>${last.mfu.toFixed(1)}%</b></span>
      <span style="color:var(--purple)">tok/s: <b>${last.tok_sec.toLocaleString()}</b></span>
      <span style="color:var(--dim)">remaining: <b>${last.remaining.toFixed(0)}s</b></span>
    `;
  }
  // Update status
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('sb-status-text');
  if (data.running) { dot.className = 'running'; txt.textContent = 'Running'; }
  else { dot.className = ''; txt.textContent = 'Idle'; }
  if (data.summary) {
    const s = data.summary;
    document.getElementById('sb-best').textContent = s.val_bpb ? s.val_bpb.toFixed(6) : '—';
  }
}

function updateLiveChart(steps) {
  if (!steps.length) return;
  const labels = steps.map(s => s.step);
  chartLiveLoss.data.labels = labels;
  chartLiveLoss.data.datasets[0].data = steps.map(s => s.loss);
  chartLiveLoss.data.datasets[1].data = steps.map(s => s.mfu);
  chartLiveLoss.update('none');
}

// ---- Status polling ----
async function pollStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  document.getElementById('sb-branch').textContent = data.branch || '—';
  document.getElementById('sb-count').textContent = data.experiment_count;
  if (data.best_bpb) document.getElementById('sb-best').textContent = data.best_bpb.toFixed(6);
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('sb-status-text');
  if (data.running) { dot.className = 'running'; txt.textContent = 'Running'; }
  else { dot.className = ''; txt.textContent = 'Idle'; }
  renderGpuOverview(data.gpus);
}

function renderGpuOverview(gpus) {
  const el = document.getElementById('gpu-overview');
  if (!gpus.length) { el.innerHTML = '<span style="color:var(--dim)">No GPU info available</span>'; return; }
  el.innerHTML = gpus.map(g => `
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
        <span><b>GPU ${g.index}</b> ${g.name}</span>
        <span style="color:var(--dim)">${g.util_pct}% util · ${(g.mem_used_mb/1024).toFixed(1)}/${(g.mem_total_mb/1024).toFixed(1)} GB · ${g.temp_c}°C</span>
      </div>
      <div class="gpu-bar-wrap"><div class="gpu-bar ${g.temp_c > 80 ? 'hot' : ''}" style="width:${g.util_pct}%"></div></div>
      <div class="gpu-bar-wrap" style="margin-top:3px"><div class="gpu-bar" style="width:${g.mem_used_mb/g.mem_total_mb*100}%;background:var(--purple)"></div></div>
    </div>
  `).join('');
}

// ---- Git ----
async function loadGit() {
  const res = await fetch('/api/git-log');
  const commits = await res.json();
  const tbody = document.getElementById('git-tbody');
  tbody.innerHTML = commits.map(c => `
    <tr>
      <td><span class="commit-link" onclick="showDiff('${c.hash}')">${c.hash}</span></td>
      <td style="color:var(--dim)">${c.message}</td>
    </tr>
  `).join('');
}

// ---- Controls ----
async function launchAgent() {
  const tag = document.getElementById('ctrl-tag').value.trim();
  const model = document.getElementById('ctrl-model').value;
  const maxExp = parseInt(document.getElementById('ctrl-max-exp').value) || 0;
  const res = await fetch('/api/launch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_tag: tag, model, max_experiments: maxExp }),
  });
  const data = await res.json();
  const el = document.getElementById('ctrl-result');
  if (data.error) el.innerHTML = `<span style="color:var(--red)">${data.error}</span>`;
  else el.innerHTML = `<span style="color:var(--green)">Launched (PID ${data.pid})</span>`;
}

async function stopAgent() {
  const res = await fetch('/api/stop', { method: 'POST' });
  const data = await res.json();
  document.getElementById('ctrl-result').innerHTML =
    `<span style="color:var(--amber)">Stopped. Killed PIDs: ${data.killed.join(', ') || 'none'}</span>`;
}

async function loadHparams() {
  const res = await fetch('/api/hyperparams');
  const params = await res.json();
  const tbody = document.getElementById('hparam-tbody');
  tbody.innerHTML = Object.entries(params).map(([k, v]) =>
    `<tr><td style="font-family:monospace;color:var(--cyan)">${k}</td><td style="font-family:monospace">${JSON.stringify(v)}</td></tr>`
  ).join('');
}

// ---- Settings ----
async function loadSettings() {
  const [prog, train] = await Promise.all([
    fetch('/api/file/program.md').then(r => r.json()),
    fetch('/api/file/train.py').then(r => r.json()),
  ]);
  document.getElementById('settings-program').value = prog.content;
  document.getElementById('settings-train').value = train.content;
}

async function saveSettings() {
  const content = document.getElementById('settings-program').value;
  const res = await fetch('/api/file/program.md', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  const data = await res.json();
  document.getElementById('settings-result').textContent =
    data.status === 'saved' ? '✓ Saved' : '✗ Error';
}

// ---- Init ----
initCharts();
initSSE();
loadResults();
pollStatus();
pollLive();

setInterval(loadResults, 10_000);
setInterval(pollStatus, 5_000);
setInterval(pollLive, 2_000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7788))
    print(f"Dashboard starting at http://localhost:{port}")
    print(f"Repo dir: {REPO_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
