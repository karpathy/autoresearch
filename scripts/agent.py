"""
Autonomous research agent for autoresearch.
Follows program.md: modifies train.py code structurally, runs experiments,
keeps improvements, discards failures. Uses an LLM to propose code changes.

Usage:
    uv run scripts/agent.py                      # use Claude API
    uv run scripts/agent.py --local              # use local LM Studio
    uv run scripts/agent.py --max-runs 50
    uv run scripts/agent.py --resume             # continue from existing branch
    uv run scripts/agent.py --no-dashboard       # text-only mode

Requires ANTHROPIC_API_KEY env var for Claude, or --local for LM Studio.
"""

import os
import sys
import shutil
import re
import json
import time
import argparse
import subprocess
import threading
from datetime import datetime
from collections import deque

from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from rich.console import Console

# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

_nvml_available = False
_nvml_handle = None
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    _nvml_available = True
except Exception:
    pass

GPU_TEMP_MAX_START = 70
GPU_TEMP_ABORT = 85
COOLDOWN_SECONDS = 45

# Auto-detect GPU VRAM (leave 500MB headroom for OS/desktop)
if _nvml_available:
    _gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
    VRAM_TOTAL_MB = _gpu_mem_info.total // (1024 * 1024)
    VRAM_LIMIT_MB = VRAM_TOTAL_MB - 500
    GPU_NAME = pynvml.nvmlDeviceGetName(_nvml_handle)
    if isinstance(GPU_NAME, bytes):
        GPU_NAME = GPU_NAME.decode()
else:
    VRAM_TOTAL_MB = 8192
    VRAM_LIMIT_MB = 7500
    GPU_NAME = "Unknown GPU"


def get_gpu_stats():
    if not _nvml_available:
        return {"temp": None, "vram_used_mb": 0, "vram_total_mb": 8192, "gpu_util": 0}
    try:
        temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
        return {
            "temp": temp,
            "vram_used_mb": mem.used / 1024 / 1024,
            "vram_total_mb": mem.total / 1024 / 1024,
            "gpu_util": util.gpu,
        }
    except Exception:
        return {"temp": None, "vram_used_mb": 0, "vram_total_mb": 8192, "gpu_util": 0}


def get_gpu_temp():
    if not _nvml_available:
        return None
    try:
        return pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None


def wait_for_cool_gpu(max_wait=300):
    start = time.time()
    while time.time() - start < max_wait:
        temp = get_gpu_temp()
        if temp is None or temp <= GPU_TEMP_MAX_START:
            return True
        if temp >= GPU_TEMP_ABORT:
            return False
        time.sleep(10)
    return False


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PY = os.path.join(PROJECT_ROOT, "train.py")

# These get set by _init_dataset_paths() after arg parsing
RESULTS_TSV = ""
LOG_FILE = ""
STATE_FILE = ""


def _init_dataset_paths(dataset_name):
    """Set results/log/state file paths based on dataset. Default dataset uses
    original filenames for backwards compatibility."""
    global RESULTS_TSV, LOG_FILE, STATE_FILE
    if dataset_name and dataset_name != "default":
        suffix = f"_{dataset_name}"
    else:
        suffix = ""
    RESULTS_TSV = os.path.join(PROJECT_ROOT, f"agent_results{suffix}.tsv")
    LOG_FILE = os.path.join(PROJECT_ROOT, f"agent{suffix}.log")
    STATE_FILE = os.path.join(PROJECT_ROOT, f"agent_state{suffix}.json")


# Default until arg parsing overrides
_init_dataset_paths(os.environ.get("AUTORESEARCH_DATASET", "default"))

def archive_previous_results(tag):
    """Archive existing results/log/state files into an archives folder.
    Files are moved to archives/<tag>/ with incrementing suffix if needed."""
    archive_dir = os.path.join(PROJECT_ROOT, "archives")
    os.makedirs(archive_dir, exist_ok=True)

    for filepath in [RESULTS_TSV, LOG_FILE, STATE_FILE]:
        if not filepath or not os.path.exists(filepath):
            continue
        fname = os.path.basename(filepath)
        # Find next available slot: archives/fname, archives/fname.1, .2, ...
        dest = os.path.join(archive_dir, fname)
        counter = 1
        while os.path.exists(dest):
            dest = os.path.join(archive_dir, f"{fname}.{counter}")
            counter += 1
        shutil.move(filepath, dest)
        log_to_file(f"Archived {fname} -> archives/{os.path.basename(dest)}")




def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def log_to_file(msg):
    """Append a timestamped message to agent.log for review."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
        f.flush()
        os.fsync(f.fileno())


def write_crash_state(experiment_num, description, phase, extra=None):
    """Write current experiment state to disk so we know where a crash happened."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "experiment_num": experiment_num,
        "description": description,
        "phase": phase,
    }
    if extra:
        state.update(extra)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def clear_crash_state():
    """Remove state file on clean completion."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def check_previous_crash():
    """Check if a previous run crashed and report what happened."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args):
    result = subprocess.run(["git"] + list(args), cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    return result.stdout.strip(), result.returncode


def git_commit(msg):
    subprocess.run(["git", "add", "train.py"], cwd=PROJECT_ROOT)
    subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT,
                  capture_output=True, text=True)
    sha, _ = git("rev-parse", "--short", "HEAD")
    return sha


def git_revert():
    git("checkout", "HEAD~1", "--", "train.py")
    git("commit", "-m", "revert: discard failed experiment")
    return git("rev-parse", "--short", "HEAD")[0]


def git_push():
    """Push current branch to origin. Fails silently if no remote."""
    out, rc = git("push", "origin", "HEAD")
    if rc != 0:
        err, _ = git("push", "origin", "HEAD", "--set-upstream")
        log_to_file(f"git push: {err if rc != 0 else 'ok'}")
    else:
        log_to_file("git push: ok")


# ---------------------------------------------------------------------------
# Training output parser
# ---------------------------------------------------------------------------

STEP_RE = re.compile(
    r"step\s+(\d+)\s+\((\d+\.\d+)%\)\s*\|\s*loss:\s*([\d.]+)\s*\|\s*lrm:\s*([\d.]+)\s*\|\s*"
    r"dt:\s*(\d+)ms\s*\|\s*tok/sec:\s*([\d,]+)\s*\|\s*mfu:\s*([\d.]+)%\s*\|\s*"
    r"epoch:\s*(\d+)\s*\|\s*remaining:\s*(\d+)s"
)


def parse_step_line(line):
    m = STEP_RE.search(line)
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "progress": float(m.group(2)),
        "loss": float(m.group(3)),
        "lrm": float(m.group(4)),
        "dt_ms": int(m.group(5)),
        "tok_sec": int(m.group(6).replace(",", "")),
        "mfu": float(m.group(7)),
        "epoch": int(m.group(8)),
        "remaining": int(m.group(9)),
    }


# ---------------------------------------------------------------------------
# Run experiment with live streaming
# ---------------------------------------------------------------------------

TRAIN_TIMEOUT = 900  # kill training if it takes longer than 15 min wall clock
                     # (torch.compile can take 3-5 min on first run + 5 min training)


def validate_train_py():
    """Quick syntax check on train.py before running. Returns error string or None."""
    try:
        import ast
        source = read_file(TRAIN_PY)
        ast.parse(source)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


def run_training_live(on_line=None):
    """Run train.py, stream output via callback. Returns parsed results dict."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    all_output = []
    try:
        proc = subprocess.Popen(
            [sys.executable, TRAIN_PY],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,
            bufsize=0,
        )

        t_start_train = time.time()
        _last_hw_check = 0.0
        buf = b""
        while True:
            # Hard timeout: kill if wall clock exceeds limit (protects against sleep/hang)
            if time.time() - t_start_train > TRAIN_TIMEOUT:
                proc.kill()
                log_to_file(f"TIMEOUT: training killed after {TRAIN_TIMEOUT}s wall clock")
                return {"error": f"timeout ({TRAIN_TIMEOUT}s wall clock)"}

            # Hardware safety: check VRAM and temperature every 3 seconds
            now = time.time()
            if now - _last_hw_check >= 3.0 and now - t_start_train > 30:  # skip first 30s for init
                _last_hw_check = now
                hw = get_gpu_stats()
                hw_temp = hw.get("temp")
                hw_vram = hw.get("vram_used_mb", 0)
                hw_vram_total = hw.get("vram_total_mb", 1)
                if hw_temp is not None and hw_temp >= GPU_TEMP_ABORT:
                    proc.kill()
                    log_to_file(f"SAFETY: training killed — GPU temp {hw_temp}C >= {GPU_TEMP_ABORT}C abort threshold")
                    return {"error": f"GPU temperature critical ({hw_temp}C) — killed to protect hardware"}
                # VRAM kill removed: pynvml reports stale CUDA contexts from prior crashes,
                # causing false positives. train.py has its own VRAM guard.

            chunk = proc.stdout.read(1)
            if not chunk:
                break
            if chunk in (b"\r", b"\n"):
                if buf:
                    line = buf.decode("utf-8", errors="replace").strip()
                    if line:
                        all_output.append(line)
                        if on_line:
                            on_line(line)
                    buf = b""
            else:
                buf += chunk

        if buf:
            line = buf.decode("utf-8", errors="replace").strip()
            if line:
                all_output.append(line)
                if on_line:
                    on_line(line)

        proc.wait(timeout=30)
        output = "\n".join(all_output)

        # Parse results block
        results = {}
        in_results = False
        for line in all_output:
            line = line.strip()
            if line == "---":
                in_results = True
                continue
            if in_results and ":" in line:
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val

        if results:
            return results
        if proc.returncode != 0:
            # Find the actual error: look for Traceback or Error lines
            error_lines = []
            capture = False
            for line in all_output:
                if "Traceback" in line or "Error" in line or "assert" in line.lower():
                    capture = True
                if capture:
                    error_lines.append(line)
            if error_lines:
                error_msg = "\n".join(error_lines[-20:])  # last 20 error lines
            else:
                error_msg = "\n".join(all_output[-20:])  # fallback: last 20 lines
            return {"error": error_msg}
        return None

    except subprocess.TimeoutExpired:
        proc.kill()
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results():
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\tsample_text\n")


def log_result(commit, val_bpb, memory_gb, status, description, sample_text=""):
    # Sanitize sample text for TSV (no tabs/newlines)
    clean_sample = sample_text.replace("\t", " ").replace("\n", " ").replace("\r", "")[:300]
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\t{clean_sample}\n")
    log_to_file(f"RESULT: {status} | val_bpb={val_bpb:.6f} | mem={memory_gb:.1f}GB | {description}")


def get_results_history():
    if not os.path.exists(RESULTS_TSV):
        return []
    rows = []
    with open(RESULTS_TSV, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                rows.append({
                    "commit": parts[0],
                    "val_bpb": float(parts[1]) if parts[1] != "0.000000" else 999.0,
                    "memory_gb": float(parts[2]),
                    "status": parts[3],
                    "description": parts[4],
                    "sample_text": parts[5] if len(parts) > 5 else "",
                })
    return rows


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def _count_recent_failures(results_history):
    """Count consecutive non-keep results from the end of history."""
    count = 0
    for r in reversed(results_history):
        if r["status"] == "keep":
            break
        count += 1
    return count


def _categorize_experiment(desc):
    """Categorize an experiment by its description for compact summary."""
    d = desc.lower()
    # Check specific/compound terms before broad ones
    if any(w in d for w in ("tying", "embed", "vocab")):
        return "embedding"
    if any(w in d for w in ("z-loss", "loss function", "cross entropy", "label smooth")):
        return "loss"
    if any(w in d for w in ("adam", "beta1", "beta2", "epsilon", "learning rate", "lr ", "warmup", "weight decay", "optimizer", "momentum")):
        return "optimizer"
    if any(w in d for w in ("batch size", "accumulation", "gradient clip", "grad clip")):
        return "batch/grad"
    if any(w in d for w in ("head", "attention", "softcap", "qk", "window")):
        return "attention"
    if any(w in d for w in ("dropout", "regulariz", "stochastic", "unfreezing")):
        return "regularization"
    if any(w in d for w in ("layer", "depth", "width", "block", "dim ", "dimension", "channel")):
        return "architecture"
    if any(w in d for w in ("sequence", "seq len")):
        return "data"
    if "loss" in d:
        return "loss"
    return "other"


def build_prompt(train_py_source, results_history, best_bpb):
    history_section = ""
    near_misses_str = ""
    if results_history:
        # Build compact table: one line per experiment, no redundant lists
        lines = []
        categories = {}
        has_crashes = False
        near_misses = []  # experiments within 5% of baseline
        for r in results_history:
            status = r["status"]
            bpb = r["val_bpb"]
            desc = r["description"]
            cat = _categorize_experiment(desc)

            if status == "crash":
                lines.append(f"  CRASH  {desc}")
                has_crashes = True
            else:
                gap = bpb - best_bpb
                marker = " <-- BEST" if bpb == best_bpb and status == "keep" else ""
                lines.append(f"  {bpb:.4f} (+{gap:.4f}) {desc}{marker}")
                # Track near misses (within 5% of best)
                if status == "discard" and bpb < 999 and gap < best_bpb * 0.05:
                    near_misses.append((gap, desc))

            # Track category stats
            if cat not in categories:
                categories[cat] = {"count": 0, "best_gap": 999.0}
            categories[cat]["count"] += 1
            if status == "discard" and bpb < 999:
                gap = bpb - best_bpb
                categories[cat]["best_gap"] = min(categories[cat]["best_gap"], gap)

        # Category summary - shows what's been exhausted
        cat_summary = "\n## Category summary (attempts / closest to baseline):\n"
        for cat, stats in sorted(categories.items(), key=lambda x: -x[1]["count"]):
            gap_str = f"+{stats['best_gap']:.4f}" if stats["best_gap"] < 999 else "n/a"
            cat_summary += f"  {cat}: {stats['count']} tried, best gap {gap_str}\n"

        # Near-miss section — these almost worked, try combining or tweaking
        if near_misses:
            near_misses.sort(key=lambda x: x[0])
            near_misses_str = "\n## Near misses (closest to beating baseline — consider combining or tweaking these):\n"
            for gap, desc in near_misses[:5]:
                near_misses_str += f"  +{gap:.4f}  {desc}\n"

        # Compact experiment log
        history_section = cat_summary + near_misses_str
        history_section += f"\n## All {len(results_history)} experiments (val_bpb / gap to best):\n"
        history_section += "\n".join(lines) + "\n"

        if has_crashes:
            history_section += "\nCRASH NOTE: flash-attn3 crashes on ANY normalization change. torch.compile times out on large structural changes.\n"

    # Adaptive strategy guidance
    fail_streak = _count_recent_failures(results_history) if results_history else 0
    streak_str = ""
    if fail_streak >= 15:
        streak_str = f"""
## CRITICAL: {fail_streak} consecutive failures!
Everything tried so far has failed. Try something CREATIVE and UNCONVENTIONAL:
- Combine 2-3 near-miss ideas above (they almost worked individually)
- Try known-good techniques from ML research: z-loss regularization, embedding weight tying, cosine LR schedule
- OPPOSITE direction from failed experiments (if increasing X failed, try decreasing it)
- Training throughput: halving TOTAL_BATCH_SIZE doubles step count in 5 min — more steps often beats bigger batches
DO NOT repeat anything from the history.
"""
    elif fail_streak >= 5:
        streak_str = f"""
## WARNING: {fail_streak} consecutive failures. Change strategy.
Look at the category summary — avoid exhausted categories. Focus on:
- Under-explored categories with fewest attempts
- Combining near-miss experiments that were close to baseline
- Training throughput improvements (more steps in 5 min = lower val_bpb)
"""
    elif fail_streak >= 3:
        streak_str = f"""
## CAUTION: {fail_streak} consecutive failures. Try simpler changes.
"""

    # Known-good directions from community research
    known_good = """
## Proven techniques from other autoresearch sessions:
- Weight decay on embeddings/value embeddings (0.001-0.003) — often a big win
- Halving TOTAL_BATCH_SIZE for 2x more training steps — more steps in 5 min usually helps
- WARMDOWN_RATIO=0.75 (more aggressive LR cooldown)
- Cosine LR schedule instead of linear warmdown
- z-loss: add small penalty on log-partition function (1e-4 * logits.logsumexp(-1).square().mean())
These are NOT guaranteed to work here but are worth trying if not yet attempted.
"""

    return f"""You are an autonomous ML researcher. Your goal: minimize val_bpb on this training script.

## Constraints
- You can ONLY modify train.py. prepare.py is read-only.
- Training runs for a fixed 5-minute time budget. Lower val_bpb is better.
- GPU: {GPU_NAME} with {VRAM_TOTAL_MB}MB VRAM. Don't exceed ~{VRAM_LIMIT_MB}MB peak.
- One idea per experiment. Available packages: torch, numpy only.
- flash-attn3 is fragile: Do NOT change normalization (RMSNorm, LayerNorm, QKNorm, etc.) - WILL crash.
- Do NOT modify TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE, or DEPTH - these are hardware-dependent and changing them crashes the training loop.
- Do NOT modify the optimizer class (MuonAdamW) or its parameter groups - torch.compile will segfault.
- Avoid large structural changes that trigger torch.compile recompilation (causes timeouts).
- TOTAL_BATCH_SIZE must be divisible by (DEVICE_BATCH_SIZE * MAX_SEQ_LEN). MAX_SEQ_LEN=2048 from prepare.py.

## Current best val_bpb: {best_bpb:.6f}
{streak_str}{history_section}{known_good}
## Current train.py:
```python
{train_py_source}
```

## Your task
Propose ONE modification to lower val_bpb. Do NOT repeat or closely variant anything from the experiment history.
Think creatively — what hasn't been tried yet?

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "description": "short description of the change",
  "changes": [
    {{
      "old": "exact string to find in train.py",
      "new": "replacement string"
    }}
  ]
}}

Each change is a find-and-replace on train.py. The "old" string must be unique in the file.
Keep changes minimal and surgical. One idea at a time."""


# AgentGuard47: auto-trace all Anthropic API calls if key is set
_agentguard_initialized = False
def _init_agentguard():
    global _agentguard_initialized
    if _agentguard_initialized:
        return
    _agentguard_initialized = True
    ag_key = os.environ.get("AGENTGUARD_API_KEY", "")
    if not ag_key:
        return
    try:
        from agentguard import Tracer, BudgetGuard, patch_anthropic
        from agentguard.sinks.http import HttpSink
        sink = HttpSink(
            url="https://app.agentguard47.com/api/ingest",
            api_key=ag_key,
        )
        tracer = Tracer(sink=sink, service="autoresearch")
        budget = BudgetGuard(max_cost_usd=50.00)
        patch_anthropic(tracer, budget_guard=budget)
        log_to_file("AgentGuard47: tracing + budget enforcement enabled ($50 limit)")
    except Exception as e:
        log_to_file(f"AgentGuard47 init failed: {e}")


def call_claude(prompt, temperature=None):
    try:
        import anthropic
        _init_agentguard()  # patch before first client creation
        client = anthropic.Anthropic(timeout=180.0)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=temperature if temperature else anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except ImportError:
        return None
    except Exception as e:
        log_to_file(f"ERROR calling Claude: {e}")
        return None


def call_local(prompt, temperature=None):
    try:
        import requests
        resp = requests.post(
            "http://127.0.0.1:1234/v1/chat/completions",
            json={
                "model": "deepseek/deepseek-r1-0528-qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": temperature if temperature else 0.7,
            },
            timeout=120,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log_to_file(f"ERROR calling LM Studio: {e}")
        return None


def parse_llm_response(response_text):
    if not response_text:
        return None
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', response_text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def apply_changes(source, changes):
    modified = source
    for change in changes:
        old = change.get("old", "")
        new = change.get("new", "")
        if not old:
            continue
        if old not in modified:
            log_to_file(f"APPLY FAIL: could not find: {old[:80]}...")
            return None
        if modified.count(old) > 1:
            log_to_file(f"APPLY FAIL: ambiguous match ({modified.count(old)}x): {old[:80]}...")
            return None
        modified = modified.replace(old, new, 1)
    return modified


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

SPARK_CHARS = list(" .:-=+*#@")
GPU_TEMP_WARN = 75
GPU_TEMP_PAUSE = 80
# GPU_NAME, VRAM_TOTAL_MB, VRAM_LIMIT_MB defined in GPU monitoring section above

PHASE_STYLES = {
    "THINKING":  ("THINKING",  "bold magenta"),
    "TRAINING":  ("TRAINING",  "bold yellow"),
    "APPLYING":  ("APPLYING",  "bold blue"),
    "BASELINE":  ("BASELINE",  "bold yellow"),
    "KEEP":      ("IMPROVED",  "bold green"),
    "DISCARD":   ("DISCARDED", "bold red"),
    "CRASH":     ("CRASHED",   "bold red"),
    "COOLING":   ("COOLING",   "bold cyan"),
    "DONE":      ("COMPLETE",  "bold green"),
    "STARTING":  ("STARTING",  "dim"),
}


def sparkline(values, width=50):
    if not values:
        return ""
    recent = list(values)[-width:]
    lo, hi = min(recent), max(recent)
    rng = hi - lo if hi > lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / rng * 8), 8)] for v in recent)


def bar(pct, width=30):
    """Unicode block progress bar."""
    pct = max(0.0, min(100.0, float(pct)))
    full_blocks = int(width * pct / 100)
    remainder = (width * pct / 100) - full_blocks
    blocks = ["", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589", "\u2588"]
    partial = blocks[min(int(remainder * 8), 8)] if full_blocks < width else ""
    empty = width - full_blocks - (1 if partial else 0)
    return f"\u2588" * full_blocks + partial + "\u2591" * empty


def build_dashboard(state):
    """Build the Rich layout from current state dict."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=16),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="training", ratio=2),
        Layout(name="leaderboard", ratio=3),
    )
    layout["right"].split_column(
        Layout(name="gpu_panel", ratio=3),
        Layout(name="sample_panel", ratio=2),
    )

    # --- Common state ---
    phase = state.get("phase", "IDLE")
    experiment_num = state.get("experiment_num", 0)
    max_runs = state.get("max_runs", 0)
    best_bpb = state.get("best_bpb", 999.0)
    llm_name = state.get("llm_name", "Claude")
    branch = state.get("branch", "")
    elapsed = state.get("total_elapsed", 0)
    em, es = divmod(int(elapsed), 60)
    eh, em = divmod(em, 60)
    history = state.get("history", [])
    kept = [r for r in history if r["status"] == "keep"]
    dataset = os.environ.get("AUTORESEARCH_DATASET", "default")

    # --- Header (compact) ---
    phase_label, phase_style = PHASE_STYLES.get(phase, (phase, "dim"))
    h = Text()
    h.append(" autoresearch ", style="bold white on bright_blue")
    h.append("  ")
    h.append(f" {phase_label} ", style=phase_style)
    h.append("  ")
    h.append(f"Exp {experiment_num}/{max_runs}", style="bold")
    h.append("  ")
    if best_bpb < 999:
        h.append(f"Best {best_bpb:.6f}", style="bold green")
        h.append("  ")
    h.append(f"{len(history)} run  ", style="dim")
    h.append(f"{len(kept)} kept", style="green")
    h.append("  ")
    h.append(f"{eh}:{em:02d}:{es:02d}", style="bold")
    h.append(f"  {dataset}", style="dim italic")
    h.append(f"  {branch}", style="dim")

    layout["header"].update(Panel(h, border_style="bright_blue"))

    # --- Training panel ---
    metrics = state.get("metrics")
    loss_history = state.get("loss_history", [])
    current_idea = state.get("current_idea", "")

    t = Text()
    if current_idea:
        idea_short = current_idea[:90] + "..." if len(current_idea) > 90 else current_idea
        t.append(f"  {idea_short}\n", style="italic white")
        t.append(f"\n")

    if metrics:
        step = metrics["step"]
        pct = metrics["progress"]
        loss = metrics["loss"]
        tok_s = metrics["tok_sec"]
        mfu = metrics["mfu"]
        remaining = metrics["remaining"]
        mins, secs = divmod(remaining, 60)

        t.append(f"  Progress  ", style="dim")
        t.append(f"{bar(pct)} ", style="bright_blue")
        t.append(f"{pct:.1f}%\n", style="bold")
        t.append(f"  Loss      ", style="dim")
        t.append(f"{loss:.6f}", style="bold yellow")
        t.append(f"   Step ", style="dim")
        t.append(f"{step:,}\n", style="bold")
        t.append(f"  Speed     ", style="dim")
        t.append(f"{tok_s:,} tok/s", style="bold")
        t.append(f"  {mfu:.1f}% MFU\n", style="dim")
        t.append(f"  ETA       ", style="dim")
        t.append(f"{mins:.0f}m {secs:.0f}s\n", style="bold")
        t.append(f"\n")
        t.append(f"  {sparkline(loss_history)}\n", style="bright_yellow")
    elif phase == "THINKING":
        te = state.get("phase_elapsed", 0)
        dots = "." * (int(te) % 4 + 1)
        t.append(f"  Querying {llm_name}{dots}\n\n", style="italic magenta")
        t.append(f"  Elapsed   ", style="dim")
        t.append(f"{int(te)}s\n", style="bold")
        if best_bpb < 999:
            t.append(f"  Target    ", style="dim")
            t.append(f"< {best_bpb:.6f} val_bpb\n", style="bold green")
        streak = _count_recent_failures(history)
        if streak > 0:
            t.append(f"  Streak    ", style="dim")
            streak_style = "bold red" if streak >= 5 else "yellow" if streak >= 3 else "dim"
            t.append(f"{streak} failures\n", style=streak_style)
    elif phase == "COOLING":
        ce = state.get("phase_elapsed", 0)
        gpu = state.get("gpu", {})
        temp_now = gpu.get("temp")
        pct = min(ce / COOLDOWN_SECONDS * 100, 100)
        t.append(f"  Cooldown  ", style="dim")
        t.append(f"{bar(pct)} ", style="cyan")
        t.append(f"{int(ce)}/{COOLDOWN_SECONDS}s\n", style="bold")
        if temp_now is not None:
            temp_style = "bold red" if temp_now >= 80 else "yellow" if temp_now >= 70 else "green"
            t.append(f"  GPU       ", style="dim")
            t.append(f"{temp_now}C", style=temp_style)
            t.append(f"  target <{GPU_TEMP_MAX_START}C\n", style="dim")
    elif phase == "APPLYING":
        t.append(f"  Applying changes & committing...\n", style="italic bright_blue")
    elif phase in ("TRAINING", "BASELINE") and not metrics:
        te = state.get("phase_elapsed", 0)
        if te < 15:
            t.append(f"  Loading model and data...\n", style="italic yellow")
        elif te < 60:
            t.append(f"  Compiling kernels...\n", style="italic yellow")
        else:
            t.append(f"  Training in progress...\n", style="italic yellow")
        t.append(f"  Elapsed   ", style="dim")
        t.append(f"{int(te)}s\n", style="bold")
    elif phase == "KEEP":
        t.append(f"  IMPROVED — change kept\n", style="bold green")
    elif phase == "DISCARD":
        t.append(f"  No improvement — reverted\n", style="bold yellow")
    elif phase == "CRASH":
        t.append(f"  Run failed — reverted\n", style="bold red")
    else:
        t.append(f"  Initializing...\n", style="dim")

    training_title = "Training" if phase in ("TRAINING", "BASELINE") else phase.title()
    border = "bright_green" if phase in ("TRAINING", "BASELINE") else "bright_magenta" if phase == "THINKING" else "cyan" if phase == "COOLING" else "green" if phase == "KEEP" else "dim"
    layout["training"].update(Panel(t, title=f"[bold]{training_title}[/]", border_style=border))

    # --- Experiment Log ---
    table = Table(expand=True, show_lines=False, padding=(0, 1), show_header=True,
                  header_style="bold dim", box=None)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("SHA", width=7, style="dim")
    table.add_column("Status", width=6)
    table.add_column("val_bpb", width=10, justify="right")
    table.add_column("VRAM", width=6, justify="right")
    table.add_column("Description", ratio=1, no_wrap=True)

    for idx, r in enumerate(history[-15:], 1):
        row_num = len(history) - min(15, len(history)) + idx
        status_map = {"keep": ("KEEP", "bold green"), "discard": ("SKIP", "yellow"), "crash": ("FAIL", "red")}
        icon, sty = status_map.get(r["status"], (r["status"], "white"))
        bpb_str = f"{r['val_bpb']:.4f}" if r["val_bpb"] < 999 else "—"
        is_best = r["val_bpb"] == best_bpb and r["status"] == "keep" and best_bpb < 999
        bpb_style = "bold bright_green" if is_best else "white" if r["val_bpb"] < 999 else "dim"
        commit = r["commit"][:7] if r["commit"] != "-------" else "—"
        vram_str = f"{r['memory_gb']:.1f}G" if r["memory_gb"] > 0 else "—"

        table.add_row(
            str(row_num), commit, Text(icon, style=sty),
            Text(bpb_str, style=bpb_style), vram_str,
            Text(r["description"][:60], style="white" if r["status"] == "keep" else "dim"),
        )

    if not history:
        table.add_row("", "", Text("—", style="dim"), "—", "—", Text("Waiting for baseline...", style="dim"))

    layout["leaderboard"].update(Panel(table, title="[bold]Experiments[/]", border_style="bright_blue"))

    # --- Hardware Monitor ---
    gpu = state.get("gpu", {})
    temp = gpu.get("temp")
    vram_used = gpu.get("vram_used_mb", 0)
    vram_total = gpu.get("vram_total_mb", 8192)
    util = gpu.get("gpu_util", 0)
    vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
    crashed = [r for r in history if r["status"] == "crash"]
    discarded = [r for r in history if r["status"] == "discard"]

    g = Text()
    g.append(f"  {GPU_NAME}\n\n", style="dim")

    # GPU bars
    if temp is not None:
        temp_style = "bold red" if temp >= GPU_TEMP_ABORT else "bold yellow" if temp >= GPU_TEMP_PAUSE else "green"
        g.append(f"  Temp  ", style="dim")
        g.append(f"{temp:3d}C ", style=temp_style)
        g.append(f"{bar(min(temp, 100), width=18)}\n", style=temp_style)
    else:
        g.append(f"  Temp  N/A\n", style="dim")

    vram_style = "bold red" if vram_used >= VRAM_LIMIT_MB else "yellow" if vram_used >= VRAM_LIMIT_MB * 0.9 else "green"
    g.append(f"  VRAM  ", style="dim")
    g.append(f"{vram_used:5.0f}", style=vram_style)
    g.append(f"/{vram_total:.0f}MB ", style="dim")
    g.append(f"{bar(vram_pct, width=10)}\n", style=vram_style)

    g.append(f"  Load  ", style="dim")
    g.append(f"{util:3d}%  ", style="bold")
    g.append(f"{bar(util, width=18)}\n", style="bright_blue")

    # Safety limits
    g.append(f"\n")
    g.append(f"  Safety limits:\n", style="dim")
    g.append(f"  Pause {GPU_TEMP_PAUSE}C ", style="dim")
    g.append(f"Abort {GPU_TEMP_ABORT}C ", style="dim")
    g.append(f"VRAM {VRAM_LIMIT_MB}MB\n", style="dim")

    # Stats
    g.append(f"\n")
    g.append(f"  Runs ", style="dim")
    g.append(f"{len(history):3d}  ", style="bold")
    g.append(f"Kept ", style="dim")
    g.append(f"{len(kept)}", style="bold green")
    if len(history) > 0:
        g.append(f" ({100*len(kept)/len(history):.0f}%)", style="dim")
    g.append(f"\n")
    g.append(f"  Skip ", style="dim")
    g.append(f"{len(discarded):3d}  ", style="yellow")
    g.append(f"Fail ", style="dim")
    g.append(f"{len(crashed)}\n", style="red")

    if len(kept) >= 2:
        baseline_bpb = next((r["val_bpb"] for r in history if r["status"] == "keep"), 999)
        if baseline_bpb < 999 and best_bpb < baseline_bpb:
            improvement = baseline_bpb - best_bpb
            g.append(f"  Gain ", style="dim")
            g.append(f"-{improvement:.6f}\n", style="bold green")

    layout["gpu_panel"].update(Panel(g, title="[bold]GPU[/]", border_style="bright_blue"))

    # --- Model Output ---
    sample = state.get("sample_text", "")
    s = Text()
    if sample:
        # Word-wrap to panel width (~38 chars)
        words = sample.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 40:
                s.append(f"{line}\n", style="italic bright_white")
                line = "  "
            line += word + " "
        if line.strip():
            s.append(f"{line}\n", style="italic bright_white")
    else:
        s.append(f"\n  Waiting for training...\n", style="dim")

    layout["sample_panel"].update(Panel(s, title="[bold bright_green]Model Output[/]", border_style="bright_green"))

    # --- Footer (log) ---
    log_lines = state.get("log_lines", deque())
    recent_logs = list(log_lines)[-13:]
    footer = Text()
    for line in recent_logs:
        footer.append(f"{line}\n", style="dim")
    if not recent_logs:
        footer.append("  Waiting...\n", style="dim")

    layout["footer"].update(Panel(footer, title="[dim]Activity Log[/]", border_style="dim"))

    return layout


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # Clear stale Triton/torch compile caches on startup (prevents sm_120 segfaults)
    import shutil
    for _cache in [
        os.path.expanduser("~/.triton"),
        os.path.expanduser("~/.cache/triton"),
        os.path.expanduser("~/.cache/torch_extensions"),
    ]:
        if os.path.exists(_cache):
            shutil.rmtree(_cache, ignore_errors=True)

    parser = argparse.ArgumentParser(description="Autonomous research agent")
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--local", action="store_true", help="Use local LM Studio instead of Claude")
    parser.add_argument("--resume", action="store_true", help="Resume from existing branch")
    parser.add_argument("--tag", type=str, default=None, help="Branch tag (default: date-based)")
    parser.add_argument("--no-dashboard", action="store_true", help="Text-only mode (no Rich TUI)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (default, pubmed)")
    args = parser.parse_args()

    if args.dataset:
        os.environ["AUTORESEARCH_DATASET"] = args.dataset
        _init_dataset_paths(args.dataset)

    _call_llm_base = call_local if args.local else call_claude
    llm_name = "LM Studio (local)" if args.local else "Claude Sonnet"

    def call_llm(prompt, fail_streak=0):
        """Call LLM with adaptive temperature — more creative after more failures."""
        if args.local:
            return _call_llm_base(prompt)
        # Ramp temperature: 0 for first few, up to 0.8 after 15+ failures
        temp = None
        if fail_streak >= 15:
            temp = 0.8
        elif fail_streak >= 10:
            temp = 0.6
        elif fail_streak >= 5:
            temp = 0.4
        return _call_llm_base(prompt, temperature=temp)

    # Setup branch
    tag = args.tag or datetime.now().strftime("%b%d").lower()
    branch = f"autoresearch/{tag}"

    existing, _ = git("branch", "--list", branch)
    if existing:
        # Branch exists — just check it out (resume or crash recovery)
        current, _ = git("branch", "--show-current")
        if current.strip() != branch:
            git("checkout", branch)
        log_to_file(f"Resuming on branch: {branch}")
    else:
        # New branch — archive old results and start fresh
        if args.tag:
            archive_previous_results(args.tag)
        git("checkout", "-b", branch)
        log_to_file(f"Created branch: {branch}")

    init_results()

    # Check for previous crash
    prev_crash = check_previous_crash()
    if prev_crash:
        log_to_file(f"PREVIOUS CRASH DETECTED: {json.dumps(prev_crash)}")
        print(f"  Previous crash detected:")
        print(f"    Time:       {prev_crash.get('timestamp', '?')}")
        print(f"    Experiment: {prev_crash.get('description', '?')}")
        print(f"    Phase:      {prev_crash.get('phase', '?')}")
        if prev_crash.get("last_step"):
            print(f"    Last step:  {prev_crash['last_step']} ({prev_crash.get('last_progress', '?')}%)")
            print(f"    Last loss:  {prev_crash.get('last_loss', '?')}")
        clear_crash_state()

    # Load existing history on resume
    prior_history = get_results_history()
    prior_count = len(prior_history)
    prior_best = 999.0
    valid = [r["val_bpb"] for r in prior_history if r["val_bpb"] < 999]
    if valid:
        prior_best = min(valid)

    if args.resume and prior_history:
        log_to_file(f"Resumed: {prior_count} prior experiments, best={prior_best:.6f}")
    log_to_file(f"Agent started: llm={llm_name}, max_runs={args.max_runs}, branch={branch}")

    # Shared state for dashboard
    state = {
        "phase": "STARTING",
        "experiment_num": prior_count,
        "max_runs": args.max_runs,
        "best_bpb": prior_best,
        "llm_name": llm_name,
        "branch": branch,
        "metrics": None,
        "loss_history": deque(maxlen=200),
        "current_idea": "",
        "history": prior_history,
        "gpu": get_gpu_stats(),
        "log_lines": deque(maxlen=50),
        "total_elapsed": 0,
        "phase_start": time.time(),
        "phase_elapsed": 0,
        "sample_text": "",
    }
    t_agent_start = time.time()

    def add_log(msg):
        state["log_lines"].append(f"  {msg}")
        log_to_file(msg)

    _last_gpu_poll = [0.0]
    def update_gpu():
        now = time.time()
        if now - _last_gpu_poll[0] >= 2.0:  # poll GPU every 2s, not every frame
            state["gpu"] = get_gpu_stats()
            _last_gpu_poll[0] = now

    def on_training_line(line):
        """Callback for each line of training output."""
        parsed = parse_step_line(line)
        if parsed:
            state["metrics"] = parsed
            state["loss_history"].append(parsed["loss"])
        # Show non-step lines in log
        if not parsed and line.strip():
            state["log_lines"].append(f"  {line[:120]}")

    # ---- Text-only fallback ----
    if args.no_dashboard:
        _run_text_mode(args, state, call_llm, add_log, on_training_line, t_agent_start)
        return

    # ---- Dashboard mode ----
    console = Console()
    console.clear()
    # Disable mouse tracking to prevent Windows Terminal from freezing
    # on click events (entering "mark mode" which pauses output)
    print("\033[?1000l\033[?1003l\033[?1006l", end="", flush=True)

    with Live(build_dashboard(state), console=console, refresh_per_second=4, screen=True) as live:

        def refresh():
            try:
                state["total_elapsed"] = time.time() - t_agent_start
                state["phase_elapsed"] = time.time() - state["phase_start"]
                update_gpu()
                live.update(build_dashboard(state))
            except Exception as e:
                log_to_file(f"Dashboard render error: {e}")

        def set_phase(phase_name):
            state["phase"] = phase_name
            state["phase_start"] = time.time()
            state["phase_elapsed"] = 0
            if phase_name not in ("TRAINING", "BASELINE"):
                state["metrics"] = None

        # Establish baseline if needed
        history = get_results_history()
        state["history"] = history

        if not history:
            set_phase("BASELINE")
            add_log("Running baseline training...")
            refresh()

            if not wait_for_cool_gpu():
                add_log("GPU too hot for baseline. Aborting.")
                set_phase("DONE")
                refresh()
                time.sleep(3)
                return

            state["metrics"] = None
            state["current_idea"] = "Establishing baseline (unmodified train.py)"

            # Run baseline in thread so dashboard keeps updating
            baseline_result = [None]
            def baseline_thread():
                baseline_result[0] = run_training_live(on_line=on_training_line)
            bt = threading.Thread(target=baseline_thread, daemon=True)
            bt.start()
            while bt.is_alive():
                refresh()
                time.sleep(0.5)
            bt.join()

            results = baseline_result[0]
            refresh()

            if results and "val_bpb" in results:
                sha = git_commit("baseline")
                sample = str(results.get("sample_text", ""))[:300]
                log_result(sha, results["val_bpb"], float(results.get("peak_vram_mb", 0)) / 1024, "keep", "baseline", sample)
                if sample:
                    state["sample_text"] = sample
                add_log(f"Baseline: val_bpb = {results['val_bpb']:.6f}")
                history = get_results_history()
                state["history"] = history
            else:
                add_log(f"Baseline failed: {results}")
                set_phase("DONE")
                refresh()
                time.sleep(5)
                return

        best_bpb = min(r["val_bpb"] for r in history if r["val_bpb"] < 999)
        state["best_bpb"] = best_bpb
        # Show best result's sample text
        best_sample = next((r["sample_text"] for r in history if r["val_bpb"] == best_bpb and r.get("sample_text")), "")
        if best_sample:
            state["sample_text"] = best_sample
        add_log(f"Current best: val_bpb = {best_bpb:.6f} | {len(history)} experiments so far")
        refresh()

        remaining = max(0, args.max_runs - prior_count)
        for i in range(remaining):
          try:
            state["experiment_num"] = prior_count + i + 1
            state["metrics"] = None

            # Parallel: cool GPU + ask LLM at the same time
            set_phase("THINKING")
            state["current_idea"] = ""
            add_log(f"Experiment {prior_count+i+1}/{args.max_runs}: asking {llm_name}...")
            write_crash_state(i + 1, "querying LLM", "THINKING")
            refresh()

            train_source = read_file(TRAIN_PY)
            fail_streak = _count_recent_failures(history)
            if fail_streak >= 5:
                add_log(f"STREAK: {fail_streak} consecutive failures — forcing strategy change")
                log_to_file(f"STREAK: {fail_streak} consecutive failures — forcing strategy change")
            elif fail_streak >= 3:
                add_log(f"STREAK: {fail_streak} consecutive failures — suggesting simpler approach")
                log_to_file(f"STREAK: {fail_streak} consecutive failures — suggesting simpler approach")
            prompt = build_prompt(train_source, history, best_bpb)

            # Launch LLM call in background while GPU cools
            llm_result = [None]
            def _llm_thread():
                llm_result[0] = call_llm(prompt, fail_streak=fail_streak)
            llm_t = threading.Thread(target=_llm_thread, daemon=True)
            llm_t.start()

            # Cool GPU while LLM thinks
            if not wait_for_cool_gpu():
                add_log("GPU too hot. Stopping agent.")
                break

            # Wait for LLM if it's still thinking
            llm_t.join()
            response = llm_result[0]
            proposal = parse_llm_response(response)

            if not proposal or "changes" not in proposal:
                add_log("LLM gave unparseable response. Skipping.")
                log_to_file(f"RAW RESPONSE: {(response or '')[:500]}")
                time.sleep(3)
                refresh()
                continue

            description = proposal.get("description", "unknown change")
            state["current_idea"] = description
            add_log(f"Idea: {description}")

            # Apply changes
            set_phase("APPLYING")
            refresh()

            modified = apply_changes(train_source, proposal["changes"])
            if modified is None:
                add_log(f"Could not apply changes. Skipping.")
                log_result("-------", 0.0, 0.0, "crash", f"FAILED TO APPLY: {description}")
                history = get_results_history()
                state["history"] = history
                refresh()
                continue

            write_file(TRAIN_PY, modified)

            # Validate before wasting time on training
            syntax_err = validate_train_py()
            if syntax_err:
                add_log(f"Patch broke syntax: {syntax_err}")
                log_to_file(f"SYNTAX ERROR after patch: {syntax_err}")
                git("checkout", "--", "train.py")  # revert
                log_result("-------", 0.0, 0.0, "crash", f"SYNTAX: {description} [{syntax_err}]")
                history = get_results_history()
                state["history"] = history
                refresh()
                continue

            sha = git_commit(description)
            add_log(f"Committed: {sha}")

            # Train
            set_phase("TRAINING")
            state["metrics"] = None
            # state["sample_text"] preserved — shows best result
            state["loss_history"] = deque(maxlen=200)
            write_crash_state(i + 1, description, "TRAINING")
            refresh()

            t0 = time.time()
            _step_log_counter = [0]

            def _on_training_line_logged(line):
                """Wraps on_training_line to also log progress to disk."""
                on_training_line(line)
                parsed = parse_step_line(line)
                if parsed:
                    _step_log_counter[0] += 1
                    if _step_log_counter[0] % 10 == 0:
                        write_crash_state(i + 1, description, "TRAINING", {
                            "last_step": parsed["step"],
                            "last_progress": parsed["progress"],
                            "last_loss": parsed["loss"],
                        })
                        log_to_file(f"step {parsed['step']} | loss {parsed['loss']:.6f} | {parsed['progress']:.1f}%")

            # Run training in a thread so we can keep refreshing the dashboard
            result_holder = [None]

            def train_thread():
                result_holder[0] = run_training_live(on_line=_on_training_line_logged)

            t = threading.Thread(target=train_thread, daemon=True)
            t.start()

            while t.is_alive():
                refresh()
                time.sleep(0.5)
            t.join()

            results = result_holder[0]
            elapsed = time.time() - t0

            if results and "val_bpb" in results:
                val_bpb = results["val_bpb"]
                memory_gb = float(results.get("peak_vram_mb", 0)) / 1024
                sample = str(results.get("sample_text", ""))[:300]
                improved = 0 < val_bpb < best_bpb
                if sample and improved:
                    state["sample_text"] = sample  # val_bpb must be positive and better than best

                if improved:
                    best_bpb = val_bpb
                    state["best_bpb"] = best_bpb
                    log_result(sha, val_bpb, memory_gb, "keep", description, sample)
                    set_phase("KEEP")
                    add_log(f"KEEP: val_bpb={val_bpb:.6f} NEW BEST!")
                    git_push()
                else:
                    log_result(sha, val_bpb, memory_gb, "discard", description, sample)
                    set_phase("DISCARD")
                    add_log(f"DISCARD: val_bpb={val_bpb:.6f} (best: {best_bpb:.6f})")
                    git_revert()
            elif results and "error" in results:
                err = str(results["error"])[:100]
                log_result(sha, 0.0, 0.0, "crash", f"{description} [{err}]")
                set_phase("CRASH")
                add_log(f"CRASH: {err}")
                git_revert()
            else:
                log_result(sha, 0.0, 0.0, "crash", f"{description} [no output]")
                set_phase("CRASH")
                add_log("CRASH: no output")
                git_revert()

            history = get_results_history()
            state["history"] = history
            clear_crash_state()
            add_log(f"Elapsed: {elapsed:.0f}s")
            refresh()
          except Exception as _loop_err:
            log_to_file(f"EXPERIMENT EXCEPTION: {_loop_err}")
            add_log(f"Exception: {str(_loop_err)[:80]}")
            try:
                git_revert()
            except Exception:
                pass
            try:
                clear_crash_state()
            except Exception:
                pass
            refresh()

          # Brief pause -- main cooling happens during next THINKING phase
          if i < args.max_runs - 1:
              for _ in range(10):
                  refresh()
                  time.sleep(1)

        # Final
        set_phase("DONE")
        refresh()
        time.sleep(3)

    # Print summary outside Live
    console.print()
    history = get_results_history()
    kept = [r for r in history if r["status"] == "keep"]
    console.print(f"[bold]Total experiments:[/] {len(history)}")
    console.print(f"[bold]Kept:[/] {len(kept)}")
    console.print(f"[bold]Best val_bpb:[/] {best_bpb:.6f}")
    console.print(f"[bold]Results:[/] {RESULTS_TSV}")
    console.print(f"[bold]Log:[/] {LOG_FILE}")
    console.print(f"[bold]Branch:[/] {branch}")


def _run_text_mode(args, state, call_llm, add_log, on_training_line, t_start):
    """Fallback text-only mode (no Rich)."""
    history = get_results_history()

    if not history:
        print("  Running baseline...")
        if not wait_for_cool_gpu():
            print("  GPU too hot. Aborting.")
            return
        results = run_training_live(on_line=lambda l: print(f"\r  {l[:100]}", end="", flush=True))
        print()
        if results and "val_bpb" in results:
            sha = git_commit("baseline")
            sample = str(results.get("sample_text", ""))[:300]
            log_result(sha, results["val_bpb"], float(results.get("peak_vram_mb", 0)) / 1024, "keep", "baseline", sample)
            print(f"  Baseline: val_bpb = {results['val_bpb']:.6f}")
            history = get_results_history()
        else:
            print(f"  Baseline failed: {results}")
            return

    best_bpb = min(r["val_bpb"] for r in history if r["val_bpb"] < 999)
    text_prior = len(history)
    print(f"  Best: {best_bpb:.6f} | {text_prior} experiments")

    remaining = max(0, args.max_runs - text_prior)
    for i in range(remaining):
        print(f"\n{'='*60}")
        print(f"  Experiment {text_prior+i+1}/{args.max_runs}")
        print(f"{'='*60}")

        if not wait_for_cool_gpu():
            print("  GPU too hot. Stopping.")
            break

        train_source = read_file(TRAIN_PY)
        fail_streak = _count_recent_failures(history)
        prompt = build_prompt(train_source, history, best_bpb)
        print("  Asking LLM...", flush=True)

        response = call_llm(prompt, fail_streak=fail_streak)
        proposal = parse_llm_response(response)

        if not proposal or "changes" not in proposal:
            print("  Unparseable response. Skipping.")
            continue

        description = proposal.get("description", "unknown change")
        print(f"  Idea: {description}")

        modified = apply_changes(train_source, proposal["changes"])
        if modified is None:
            print("  Could not apply. Skipping.")
            log_result("-------", 0.0, 0.0, "crash", f"FAILED TO APPLY: {description}")
            history = get_results_history()
            continue

        write_file(TRAIN_PY, modified)
        sha = git_commit(description)

        print(f"  Training...", flush=True)
        write_crash_state(i + 1, description, "TRAINING")
        t0 = time.time()
        _text_step_count = [0]

        def _text_on_line(l):
            print(f"\r  {l[:100]}", end="", flush=True)
            parsed = parse_step_line(l)
            if parsed:
                _text_step_count[0] += 1
                if _text_step_count[0] % 10 == 0:
                    write_crash_state(i + 1, description, "TRAINING", {
                        "last_step": parsed["step"],
                        "last_progress": parsed["progress"],
                        "last_loss": parsed["loss"],
                    })
                    log_to_file(f"step {parsed['step']} | loss {parsed['loss']:.6f} | {parsed['progress']:.1f}%")

        results = run_training_live(on_line=_text_on_line)
        print()
        elapsed = time.time() - t0

        if results and "val_bpb" in results:
            val_bpb = results["val_bpb"]
            memory_gb = float(results.get("peak_vram_mb", 0)) / 1024
            sample = str(results.get("sample_text", ""))[:300]
            if 0 < val_bpb < best_bpb:  # val_bpb must be positive and better than best
                best_bpb = val_bpb
                log_result(sha, val_bpb, memory_gb, "keep", description, sample)
                print(f"  KEEP: {val_bpb:.6f} NEW BEST!")
                git_push()
            else:
                log_result(sha, val_bpb, memory_gb, "discard", description, sample)
                print(f"  DISCARD: {val_bpb:.6f} (best: {best_bpb:.6f})")
                git_revert()
        elif results and "error" in results:
            err = str(results["error"])[:100]
            log_result(sha, 0.0, 0.0, "crash", f"{description} [{err}]")
            print(f"  CRASH: {err}")
            git_revert()
        else:
            log_result(sha, 0.0, 0.0, "crash", f"{description} [no output]")
            print(f"  CRASH: no output")
            git_revert()

        clear_crash_state()
        print(f"  Elapsed: {elapsed:.0f}s")
        history = get_results_history()

        if i < args.max_runs - 1:
            print(f"  Cooling 10s...")
            time.sleep(10)

    print(f"\n  DONE: Best val_bpb = {best_bpb:.6f} | {len(history)} experiments")
    print(f"  Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
