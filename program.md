# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

---

## Observability & Intelligence Modules (Optional)

Four optional modules enhance the experiment loop with monitoring, memory,
resilience, and proactive guidance. These are **non-breaking additions** —
the core `train.py` loop works without them. They help you make smarter
experiment decisions and help humans understand what happened overnight.

All modules use only stdlib (json, time, math, statistics) — no new
dependencies beyond what's already in `pyproject.toml`.

### monitor.py — Experiment Metrics & Observability

Tracks per-experiment and per-step metrics across the session. Inspired by
production observability systems with Prometheus-compatible export.

**Key features:**
- Per-experiment tracking: val_bpb, VRAM, MFU, step count, loss curves
- Session-level aggregates: keep rate, improvement velocity (BPB/hour)
- Real-time alerting: loss spikes, VRAM pressure, crash streaks, plateaus
- Export: JSON for dashboards, Prometheus text exposition format
- Terminal dashboard: `format_dashboard(tracker)` for quick-glance status

**Usage in the experiment loop:**

```python
from monitor import ExperimentTracker, format_dashboard

tracker = ExperimentTracker()

# At the start of each experiment:
tracker.start_experiment(commit, description, change_tags=["learning_rate"])

# During training (optional — for per-step loss curves):
tracker.record_step(step, train_loss, lr, vram_mb, tokens_per_sec)

# After each experiment:
tracker.end_experiment(val_bpb=0.995, status="keep", peak_vram_mb=44000)

# Periodic dashboard output:
print(format_dashboard(tracker))

# Export for external dashboards:
tracker.export_json()
```

**Environment variables for alert thresholds:**
- `AUTORESEARCH_ALERT_LOSS_SPIKE_RATIO` — loss spike detection multiplier (default: 3.0)
- `AUTORESEARCH_ALERT_VRAM_WARNING_MB` — VRAM warning threshold (default: 75000)
- `AUTORESEARCH_ALERT_CONSECUTIVE_CRASH_LIMIT` — crashes before alert (default: 3)
- `AUTORESEARCH_ALERT_PLATEAU_WINDOW` — experiments without improvement (default: 15)

### memory.py — Cross-Session Experiment Memory

Persists experiment knowledge across research sessions using Hebbian
association learning. Remembers what types of changes tend to improve BPB
and which are dead ends.

**Key features:**
- Hebbian associations: strengthens category-outcome connections over time
- Temporal decay: older experiments contribute less (keeps system responsive)
- Surprise-weighted storage: unexpected results are remembered longer
- Auto-tagging: automatically categorizes experiments by description keywords
- Pattern extraction: identifies promising directions and dead ends

**Change categories** (the taxonomy for experiment classification):
`architecture`, `attention`, `activation`, `optimizer`, `learning_rate`,
`schedule`, `batch_size`, `initialization`, `regularization`, `normalization`,
`embedding`, `numerical`, `simplification`, `combination`, `radical`

**Usage in the experiment loop:**

```python
from memory import ExperimentMemory

memory = ExperimentMemory()

# After each experiment:
memory.store_experiment(
    commit="a1b2c3d",
    description="increase matrix LR to 0.06",
    val_bpb=0.993, delta_bpb=-0.004, status="keep",
    change_tags=["learning_rate", "optimizer"],
)

# Before choosing the next experiment:
promising = memory.get_promising_directions(top_k=5)
dead_ends = memory.get_dead_ends()
plateau = memory.get_plateaus()

# Periodically apply temporal decay (e.g., every 20 experiments):
memory.decay()
```

### resilience.py — Circuit Breakers & Anomaly Detection

Prevents wasting GPU time on repeated failures and detects problematic
patterns across experiments.

**Key features:**
- Circuit breaker: blocks experiments after N consecutive crashes, with
  half-open probing and exponential backoff recovery
- Anomaly detection: plateaus, VRAM creep, systematic regression, crash clusters
- Backpressure monitoring: tracks VRAM trends, warns before OOM
- Unified ExperimentGuard: single pre/post experiment interface

**Usage in the experiment loop:**

```python
from resilience import ExperimentGuard

guard = ExperimentGuard(gpu_vram_mb=81920)  # Set to your GPU's VRAM

# Before each experiment:
verdict = guard.pre_experiment(description="double model width")
if verdict.blocked:
    print(f"BLOCKED: {verdict.reason}")
    # Skip this experiment, try something else
for warning in verdict.warnings:
    print(f"WARNING: {warning}")

# After each experiment:
guard.post_experiment(val_bpb=0.995, status="keep", peak_vram_mb=44000)
```

### guidance.py — Proactive Experiment Suggestions

Synthesizes signals from memory, monitoring, and resilience to suggest
the most promising next experiment. Detects contradictions and assesses
overall strategy effectiveness.

**Key features:**
- Ranked experiment suggestions based on Hebbian memory associations
- Plateau breaker: suggests radical changes when stuck
- Contradiction detection: finds conflicting results that need investigation
- Strategy assessment: evaluates phase (exploring/exploiting/plateaued)
- Formatted guidance: ready-to-inject context for agent decisions

**Usage in the experiment loop:**

```python
from guidance import ExperimentAdvisor

advisor = ExperimentAdvisor(memory=memory, tracker=tracker, guard=guard)

# Before choosing the next experiment:
guidance = advisor.get_guidance()
print(guidance["formatted"])  # Human-readable guidance
# Use guidance["suggestions"] to inform your next experiment choice
```

### Recommended Integration Pattern

Here's the full recommended pattern for the experiment loop with all modules:

```python
from monitor import ExperimentTracker, format_dashboard
from memory import ExperimentMemory
from resilience import ExperimentGuard
from guidance import ExperimentAdvisor

tracker = ExperimentTracker()
memory = ExperimentMemory()
guard = ExperimentGuard()
advisor = ExperimentAdvisor(memory=memory, tracker=tracker, guard=guard)

# LOOP FOREVER:
while True:
    # 1. Get guidance for next experiment
    guidance = advisor.get_guidance()
    # (Use suggestions to pick your next experiment)

    # 2. Pre-experiment safety check
    verdict = guard.pre_experiment(description=description)
    if verdict.blocked:
        # Wait or try a different approach
        continue

    # 3. Modify train.py and commit
    # ...

    # 4. Run experiment
    tracker.start_experiment(commit, description, change_tags)
    # uv run train.py > run.log 2>&1
    tracker.end_experiment(val_bpb, status, peak_vram_mb)

    # 5. Record results
    memory.store_experiment(commit, description, val_bpb, delta_bpb, status)
    guard.post_experiment(val_bpb, status, peak_vram_mb)

    # 6. Periodic maintenance
    if tracker.session.total_experiments % 20 == 0:
        memory.decay()
        tracker.export_json()
        print(format_dashboard(tracker))
```

### State directories

All modules persist state to `.autoresearch/` (gitignored):
- `.autoresearch/metrics/` — monitor session data and JSON exports
- `.autoresearch/memory/` — experiment memory and Hebbian associations
- `.autoresearch/resilience/` — circuit breaker and anomaly state
