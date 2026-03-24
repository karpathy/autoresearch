# Phase 2: Experiment Infrastructure - Research

**Researched:** 2026-03-25
**Domain:** Experiment loop harness for autonomous ReID model training
**Confidence:** HIGH

## Summary

Phase 2 builds the infrastructure that enables an AI agent to autonomously run experiments on `train.py`. The core deliverables are: (1) `train.py` writes a `metrics.json` file after each run with all sub-metrics, (2) the agent manages git state (commit before run, reset on failure), (3) `results.tsv` logs every experiment with all required columns, (4) OOM and runtime crashes are caught and recovered from, (5) VRAM is tracked per-experiment, and (6) a fixed 10-epoch budget is enforced.

This phase does NOT modify the agent instructions (`program.md`) -- that is Phase 3. This phase ensures the mechanical infrastructure exists so a loop-driving agent has reliable primitives: run, measure, log, keep-or-discard, recover.

**Primary recommendation:** Implement metrics.json output in train.py (written by train.py itself), have the agent read metrics.json and write results.tsv (agent owns logging), and wrap the main() call in train.py with try/except for OOM and general exceptions with structured error output to metrics.json.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Agent (Claude Code) drives the loop directly via shell commands. No run.sh wrapper. Agent calls `python train.py`, reads output, manages git, appends to results.tsv. Follows original autoresearch pattern.
- **D-02:** train.py writes metrics to a JSON file (e.g., `metrics.json`) after training completes. Agent reads this file for structured metric parsing instead of grepping stdout. More reliable than stdout parsing.
- **D-03:** Claude's discretion on whether agent or train.py writes to results.tsv. The key requirement is that every experiment gets a row with: commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status, description.

### Claude's Discretion
- Whether results.tsv is written by train.py (automatic) or by agent (after reading metrics.json)
- Exact git workflow (commit before run vs after, branch strategy)
- How to implement 3-consecutive-crash skip logic (in program.md instructions vs code)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFRA-01 | Each experiment is a git commit; improvement = keep, regression = `git reset --hard` | Git workflow pattern from program.md; agent commits train.py before running, resets HEAD~1 on regression |
| INFRA-02 | results.tsv logs every experiment: commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status, description | Agent reads metrics.json then appends TSV row; format matches program.md pattern |
| INFRA-03 | OOM and runtime crashes are caught, logged as "crash" in results.tsv, git reset performed, loop continues | try/except in train.py __main__ block; writes crash status to metrics.json; agent reads and logs |
| INFRA-04 | After 3 consecutive crashes on the same idea, agent skips that direction | Agent-side logic reading results.tsv crash streak; encoded in program.md instructions (Phase 3) but infrastructure must support it |
| INFRA-05 | Peak VRAM tracked and logged per experiment | `torch.cuda.max_memory_allocated()` in train.py, written to metrics.json |
| INFRA-06 | Run output includes decomposed sub-metrics in greppable format | metrics.json contains all sub-metrics; train.py also prints summary block to stdout |
| INFRA-07 | Fixed budget of 10 epochs per experiment, teacher cache build time excluded | EPOCHS constant in prepare.py (immutable); train.py imports it; cache build happens in prepare.py before epoch loop |
</phase_requirements>

## Standard Stack

### Core

No new libraries needed. This phase uses only what is already in the project.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `json` | 3.12 | Write/read metrics.json | Zero dependencies, structured output |
| Python stdlib `sys` | 3.12 | Exit codes for crash signaling | Already imported |
| `torch.cuda` | (installed) | `max_memory_allocated()` for VRAM tracking | Already used in monolith pattern |
| `git` CLI | 2.34.1 | Commit/reset via agent shell commands | Already available, agent drives directly |

### Supporting

No additional libraries. The autoresearch philosophy is radical simplicity.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| metrics.json | stdout parsing (grep) | JSON is more reliable; stdout can be corrupted by logging noise. Decision D-02 locks this. |
| Agent writes results.tsv | train.py writes results.tsv | Agent writing is better -- agent knows the commit hash and description, train.py does not |

**Installation:** No new packages needed.

## Architecture Patterns

### Recommended Data Flow

```
Agent edits train.py
    |
    v
Agent: git add train.py && git commit -m "description"
    |
    v
Agent: python train.py > run.log 2>&1
    |
    v
train.py runs 10 epochs, writes metrics.json at exit
    |
    v
Agent: reads metrics.json (or run.log tail on crash)
    |
    v
Agent: appends row to results.tsv
    |
    v
If improved: keep commit
If regression or crash: git reset --hard HEAD~1
    |
    v
LOOP
```

### Pattern 1: metrics.json Output Contract

**What:** train.py writes a `metrics.json` file in the working directory after every run (success or crash).
**When:** Always -- at the end of main() on success, in except block on crash.
**Example:**

```python
# Success case -- end of main()
import json

metrics = {
    "status": "success",
    "combined_metric": combined_metric,
    "recall_at_1": recall_at_1,
    "recall_at_5": recall_at_5,
    "mean_cosine": stats.mean_cosine,
    "distill_loss": stats.distill_loss,
    "arc_loss": stats.arc_loss,
    "vat_loss": stats.vat_loss,
    "sep_loss": stats.sep_loss,
    "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    "epochs": EPOCHS,
    "elapsed_seconds": elapsed,
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Also print summary to stdout for human readability
print("---")
print(f"combined_metric:  {combined_metric:.6f}")
print(f"recall@1:         {recall_at_1:.6f}")
print(f"mean_cosine:      {stats.mean_cosine:.6f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
```

```python
# Crash case -- __main__ except block
import json
import sys

if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        metrics = {
            "status": "oom",
            "peak_vram_mb": peak,
            "error": "CUDA out of memory",
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("---")
        print("status: OOM")
        print(f"peak_vram_mb: {peak:.1f}")
        sys.exit(1)
    except Exception as e:
        metrics = {
            "status": "crash",
            "error": str(e),
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("---")
        print(f"status: crash")
        print(f"error: {e}")
        sys.exit(1)
```

**Key:** metrics.json is overwritten every run. It is ephemeral. results.tsv is the persistent log.

### Pattern 2: results.tsv Format

**What:** Tab-separated log of all experiments, written by the agent.
**Format:**

```
commit	combined_metric	recall_at_1	mean_cosine	peak_vram_mb	status	description
a1b2c3d	0.623400	0.482100	0.764700	18432.5	keep	baseline
b2c3d4e	0.635200	0.501000	0.769400	18440.2	keep	increase LR to 0.2
c3d4e5f	0.610000	0.460000	0.760000	18435.0	discard	switch to GeLU in projection
d4e5f6g	0.000000	0.000000	0.000000	22100.5	crash	double batch size (OOM)
```

**Recommendation (Claude's Discretion):** Agent writes results.tsv. Rationale:
1. Agent knows the commit hash (from `git rev-parse --short HEAD`)
2. Agent knows the description (it wrote the commit message)
3. Agent determines the status (keep/discard/crash) based on metric comparison
4. train.py should not need to know about git or experiment history

### Pattern 3: Git Workflow -- Commit Before Run

**What:** Agent commits train.py changes BEFORE running the experiment.
**When:** Every experiment iteration.
**Recommendation (Claude's Discretion):** Commit-before-run pattern (matches original autoresearch).

```
1. Agent edits train.py
2. Agent: git add train.py && git commit -m "experiment: <description>"
3. Agent: python train.py > run.log 2>&1
4. Agent: read metrics.json
5. If improvement: keep (commit stays)
6. If regression: git reset --hard HEAD~1 (removes the commit)
7. If crash: git reset --hard HEAD~1 (removes the commit)
```

**Why commit-before-run:**
- On crash, the commit is already made, so `git reset --hard HEAD~1` cleanly reverts
- Only successful improvements survive in git history
- results.tsv (untracked) records everything including discards and crashes

### Pattern 4: VRAM Tracking

**What:** `torch.cuda.max_memory_allocated()` recorded after training.
**When:** Always -- in both success and crash paths.
**Key detail:** Must call `torch.cuda.reset_peak_memory_stats()` at the START of training to get per-experiment peaks, not cumulative. But since each experiment is a fresh Python process, this happens automatically.

### Pattern 5: Fixed 10-Epoch Budget

**What:** EPOCHS = 10 is a constant in prepare.py (immutable). train.py imports it.
**When:** Every experiment.
**Key detail:** The early stopping logic from the monolith (patience-based) must be REMOVED. Every experiment runs exactly 10 epochs for fair comparison. The combined metric is computed on the final epoch.

### Anti-Patterns to Avoid

- **Anti-Pattern: stdout-only metrics.** Even though we print a summary to stdout, the authoritative source is metrics.json. Stdout can be corrupted by warnings, logger output, or tracebacks.
- **Anti-Pattern: train.py writing results.tsv.** train.py does not know the commit hash or experiment description. Mixing concerns.
- **Anti-Pattern: Catching exceptions too broadly in train.py.** The try/except should distinguish OOM (recoverable, VRAM info useful) from other crashes (might be syntax errors the agent introduced).
- **Anti-Pattern: Tracking metrics.json in git.** It is ephemeral, overwritten each run. Must be in .gitignore.
- **Anti-Pattern: Early stopping within 10 epochs.** Every experiment must run all 10 epochs for fair comparison.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Experiment tracking | Custom DB or tracking server | results.tsv + git log | Autoresearch philosophy: TSV IS the tracker |
| Crash recovery orchestration | Watchdog process or retry daemon | Agent try/except + git reset | Agent IS the orchestrator per D-01 |
| Metric serialization | Custom binary format | json.dump / json.load | Standard, human-readable, zero dependencies |
| Process isolation | Docker containers per experiment | Fresh `python train.py` invocation | Each process gets clean CUDA state automatically |

## Common Pitfalls

### Pitfall 1: metrics.json Not Written on Hard Crash
**What goes wrong:** If train.py is killed by the OS (e.g., Linux OOM killer sends SIGKILL), the except block never runs. metrics.json is not written or contains stale data from a previous run.
**Why it happens:** SIGKILL cannot be caught. Also, CUDA OOM sometimes triggers a segfault rather than a Python exception.
**How to avoid:** Agent must check metrics.json timestamp or existence. If metrics.json is missing or older than run.log, treat as crash. Agent reads `tail -n 50 run.log` for error info.
**Warning signs:** metrics.json has data from a previous run (stale commit hash or status="success" but process exited non-zero).

### Pitfall 2: results.tsv Tab Corruption
**What goes wrong:** Agent accidentally uses spaces or commas. Descriptions containing tabs break parsing.
**Why it happens:** LLM agents sometimes mix up delimiters.
**How to avoid:** Define exact format in program.md. Replace tabs in description strings with spaces. Always use `\t` explicitly.
**Warning signs:** `wc -l results.tsv` vs actual experiment count diverges.

### Pitfall 3: Git Reset After Failed Commit
**What goes wrong:** Agent runs `git reset --hard HEAD~1` but the commit was not actually made (e.g., nothing to commit). This resets to a PREVIOUS experiment's state, destroying work.
**Why it happens:** Agent forgets to check if the commit succeeded before running reset.
**How to avoid:** Agent must verify commit succeeded (`git rev-parse HEAD` before and after). Only reset if HEAD actually changed.
**Warning signs:** results.tsv shows fewer "keep" experiments than git log shows commits.

### Pitfall 4: Stale metrics.json from Previous Run
**What goes wrong:** If train.py crashes without writing metrics.json, the agent reads the PREVIOUS run's metrics.json and thinks the current run succeeded.
**Why it happens:** metrics.json persists between runs.
**How to avoid:** Delete metrics.json BEFORE each run. Agent runs: `rm -f metrics.json && python train.py > run.log 2>&1`. Then check if metrics.json exists after the run.
**Warning signs:** The same metric values appear for two consecutive experiments.

### Pitfall 5: CUDA Memory Not Freed Between Experiments
**What goes wrong:** Not applicable here because each experiment is a fresh Python process. But if someone tries to run multiple experiments in the same process, VRAM accumulates.
**How to avoid:** Always run experiments as separate processes: `python train.py > run.log 2>&1`. Never import and call main() from another Python script.

### Pitfall 6: Race Between metrics.json Write and Agent Read
**What goes wrong:** Agent reads metrics.json before train.py finishes writing it. Gets truncated JSON.
**Why it happens:** Should not happen with synchronous subprocess execution. But if agent uses background processes, it could.
**How to avoid:** Agent waits for the process to exit before reading. Use synchronous execution: the shell command blocks until train.py exits.

## Code Examples

### metrics.json Schema (Success)

```json
{
  "status": "success",
  "combined_metric": 0.6234,
  "recall_at_1": 0.4821,
  "recall_at_5": 0.7123,
  "mean_cosine": 0.7647,
  "distill_loss": 0.3210,
  "arc_loss": 0.1540,
  "vat_loss": 0.0200,
  "sep_loss": 0.0150,
  "peak_vram_mb": 18432.5,
  "epochs": 10,
  "elapsed_seconds": 245.3
}
```

### metrics.json Schema (OOM Crash)

```json
{
  "status": "oom",
  "peak_vram_mb": 23800.2,
  "error": "CUDA out of memory. Tried to allocate 512.00 MiB..."
}
```

### metrics.json Schema (Runtime Crash)

```json
{
  "status": "crash",
  "peak_vram_mb": 8200.0,
  "error": "RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x512 and 256x256)"
}
```

### Agent Git Workflow (Pseudocode)

```bash
# 1. Edit train.py (agent modifies code directly)

# 2. Commit
git add train.py
git commit -m "experiment: increase LR to 0.2"
COMMIT=$(git rev-parse --short HEAD)

# 3. Run
rm -f metrics.json
python train.py > run.log 2>&1
EXIT_CODE=$?

# 4. Read metrics
# Agent reads metrics.json with Read tool or cat

# 5. Log to results.tsv
# Agent appends: $COMMIT\t$metric\t$recall\t$cosine\t$vram\t$status\t$description

# 6. Keep or discard
# If combined_metric > best_metric: keep (do nothing)
# If regression or crash: git reset --hard HEAD~1
```

### train.py __main__ Block (Full Pattern)

```python
if __name__ == "__main__":
    import json
    import sys
    import traceback

    try:
        main()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        metrics = {
            "status": "oom",
            "peak_vram_mb": round(peak, 1),
            "error": "CUDA out of memory",
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("---")
        print("status: OOM")
        print(f"peak_vram_mb: {peak:.1f}")
        sys.exit(1)
    except Exception as e:
        peak = 0.0
        try:
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception:
            pass
        metrics = {
            "status": "crash",
            "peak_vram_mb": round(peak, 1),
            "error": str(e),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        traceback.print_exc()
        sys.exit(1)
```

### results.tsv Header Initialization

The agent creates results.tsv with the header on first use:

```bash
echo -e "commit\tcombined_metric\trecall_at_1\tmean_cosine\tpeak_vram_mb\tstatus\tdescription" > results.tsv
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| stdout grep for metrics | JSON file output (metrics.json) | Decision D-02 | More reliable parsing, structured data |
| Wall-clock time budget (5 min) | Fixed epoch budget (10 epochs) | ReID adaptation | Fair comparison across architecture changes that affect per-epoch time |
| val_bpb single metric | combined_metric (0.5*recall@1 + 0.5*mean_cosine) | ReID adaptation | Dual signal for more reliable hill-climbing |

## Open Questions

1. **Recall@5 in metrics.json vs results.tsv**
   - What we know: INFRA-06 requires recall@5 in greppable output. INFRA-02 does not list it in results.tsv columns.
   - What's unclear: Should recall@5 be a results.tsv column?
   - Recommendation: Include in metrics.json (for INFRA-06). Keep results.tsv columns as specified in INFRA-02 (combined, recall@1, mean_cosine, VRAM, status, description). Agent can always read metrics.json for full decomposition.

2. **3-Consecutive-Crash Logic Placement**
   - What we know: INFRA-04 requires skipping after 3 consecutive crashes on the same idea.
   - What's unclear: Is this infrastructure (code) or instructions (program.md)?
   - Recommendation: This is agent behavior, not code infrastructure. The mechanism is: agent reads results.tsv, counts consecutive crash rows, decides to skip. The INFRASTRUCTURE contribution from Phase 2 is ensuring results.tsv reliably records crash status. The LOGIC lives in program.md (Phase 3). Phase 2 should document the pattern but not implement agent decision logic.

3. **metrics.json Path**
   - What we know: Should be in the working directory.
   - What's unclear: Absolute or relative path?
   - Recommendation: Relative to CWD (`metrics.json` in the repo root). Add to .gitignore alongside results.tsv.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | train.py execution | Yes | 3.12.11 | -- |
| git | Commit/reset workflow | Yes | 2.34.1 | -- |
| NVIDIA GPU (RTX 4090) | Training + VRAM tracking | Yes | 24564 MiB | -- |
| torch.cuda | VRAM tracking | Yes | (installed) | -- |
| json (stdlib) | metrics.json I/O | Yes | builtin | -- |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (not yet installed) |
| Config file | none -- see Wave 0 |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | Git commit/reset cycle works correctly | manual-only | Manual: agent-driven git workflow | N/A -- agent behavior |
| INFRA-02 | results.tsv has correct columns and format | unit | `python -m pytest tests/test_infrastructure.py::test_results_tsv_format -x` | Wave 0 |
| INFRA-03 | OOM crash writes crash metrics to metrics.json | unit | `python -m pytest tests/test_infrastructure.py::test_crash_metrics_json -x` | Wave 0 |
| INFRA-04 | 3-consecutive-crash detection from results.tsv | unit | `python -m pytest tests/test_infrastructure.py::test_crash_streak_detection -x` | Wave 0 |
| INFRA-05 | peak_vram_mb present in metrics.json | unit | `python -m pytest tests/test_infrastructure.py::test_vram_in_metrics -x` | Wave 0 |
| INFRA-06 | All sub-metrics present in metrics.json | unit | `python -m pytest tests/test_infrastructure.py::test_submetics_in_output -x` | Wave 0 |
| INFRA-07 | 10-epoch budget enforced (EPOCHS imported from prepare.py) | unit | `python -m pytest tests/test_infrastructure.py::test_epoch_budget -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_infrastructure.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_infrastructure.py` -- covers INFRA-02 through INFRA-07
- [ ] `tests/conftest.py` -- shared fixtures (mock metrics.json, sample results.tsv)
- [ ] pytest install: `pip install pytest` (if not already in environment)

Note: INFRA-01 (git workflow) and the full agent loop are agent-behavioral and cannot be unit tested in isolation. They are validated in Phase 4 (VALD-01, VALD-02).

## Sources

### Primary (HIGH confidence)
- `program.md` -- Original autoresearch agent instructions (git workflow, results.tsv format, loop pattern)
- `finetune_trendyol_arcface3.py` -- Monolith with EpochStats dataclass, evaluate_retrieval, combined_metric computation, VRAM patterns
- `.planning/research/ARCHITECTURE.md` -- Verified architecture and data flow patterns
- `.planning/research/PITFALLS.md` -- Documented OOM cascade, metric gaming, git pollution pitfalls
- `.planning/research/FEATURES.md` -- Table stakes and anti-features list

### Secondary (MEDIUM confidence)
- `.planning/phases/02-experiment-infrastructure/02-CONTEXT.md` -- User decisions D-01, D-02, D-03
- Karpathy autoresearch GitHub pattern (train.py summary block, results.tsv format, git commit-before-run)

### Tertiary (LOW confidence)
- None -- all findings verified against existing codebase and project documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all patterns verified in existing code
- Architecture: HIGH -- directly adapted from Karpathy autoresearch pattern + monolith analysis + user decisions
- Pitfalls: HIGH -- drawn from PITFALLS.md research + code analysis of actual crash/VRAM/metric patterns
- metrics.json schema: HIGH -- derived from EpochStats dataclass fields in monolith (line 902)
- results.tsv format: HIGH -- specified in INFRA-02 requirements + program.md pattern

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable domain, no external API dependencies)
