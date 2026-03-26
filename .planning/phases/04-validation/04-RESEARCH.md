# Phase 4: Validation - Research

**Researched:** 2026-03-25
**Domain:** End-to-end system validation -- ML training pipeline, autonomous agent loop, crash recovery
**Confidence:** HIGH

## Summary

Phase 4 validates the complete autoresearch system built across Phases 1-3. The validation is not about writing new code -- it is about proving the system works by executing it. There are three distinct validation steps: (1) baseline run of unmodified train.py to establish the reference metric, (2) at least one full autonomous agent cycle (read history, modify train.py, run experiment, evaluate, keep/discard), and (3) intentional crash recovery via OOM trigger.

The critical insight is that this phase is primarily **execution and observation**, not development. The only "new" artifacts are the validation evidence (results.tsv rows, git history, log output) and a small OOM trigger modification to train.py for crash recovery testing. After validation passes, the system transitions directly to the first autonomous overnight run per D-02 -- no manual gate.

**Primary recommendation:** Execute validation as a sequential script: baseline first, then autonomous cycle, then OOM crash test, then launch overnight. Each step's success criteria are observable from results.tsv and git log. Keep the validation lightweight -- the system either works or it does not.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Claude's discretion on how many baseline runs to perform for stability verification. At least 1 required.
- **D-02:** After validation passes, automatically launch the first autonomous overnight run. No manual step needed -- validation flows directly into production use.

### Claude's Discretion
- Number of baseline runs for stability (1 vs 3)
- How to intentionally trigger OOM for crash recovery testing (increase batch size? allocate extra tensors?)
- Whether to set a maximum experiment count for the overnight run or let it run indefinitely

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VALD-01 | Baseline run of unmodified train.py produces a valid combined metric and logs to results.tsv | Baseline execution pattern documented; metric validation criteria defined; results.tsv format from INFRA-02 |
| VALD-02 | At least one full autonomous loop cycle completes: agent modifies train.py, runs, evaluates, keeps or discards | Autonomous cycle verification steps defined; success observable from results.tsv row count and git log |
| VALD-03 | Crash recovery verified: intentionally trigger OOM, confirm system logs crash and continues | OOM trigger technique documented; verification observable from results.tsv "crash" row + subsequent experiment |

</phase_requirements>

## Standard Stack

### Core (all pre-existing from Phases 1-3)

No new libraries needed for Phase 4. Validation uses the system as-is.

| Component | Source Phase | Purpose in Validation |
|-----------|-------------|----------------------|
| prepare.py | Phase 1 | Immutable evaluation -- produces combined metric |
| train.py | Phase 1 | Agent-editable training script -- baseline and experiments |
| results.tsv | Phase 2 | Experiment log -- validation evidence lives here |
| Git loop | Phase 2 | Commit/reset infrastructure -- observed during autonomous cycle |
| Crash recovery | Phase 2 | OOM catch/log/reset -- tested explicitly in VALD-03 |
| program.md | Phase 3 | Agent instructions -- drives the autonomous cycle |

### Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12.11 | Runtime |
| PyTorch | 2.9.1+cu128 | Training framework |
| Git | 2.34.1 | Experiment versioning |
| uv | 0.8.20 | Package management / script runner |
| NVIDIA RTX 4090 | 24GB VRAM | Training hardware |

No new installations required.

## Architecture Patterns

### Validation Sequence

The validation is a strict sequence. Each step depends on the previous:

```
Step 1: Baseline (VALD-01)
  |-- Run unmodified train.py for 10 epochs
  |-- Verify combined metric is valid (non-zero, non-NaN)
  |-- Verify results.tsv has baseline row with status "kept"
  |-- Verify git commit exists for baseline
  |
Step 2: Autonomous Cycle (VALD-02)
  |-- Agent reads results.tsv (sees baseline)
  |-- Agent modifies train.py (any small change)
  |-- Agent commits, runs experiment, evaluates
  |-- Agent keeps or discards based on metric comparison
  |-- Verify results.tsv has second row
  |-- Verify git state is correct (commit kept OR reset performed)
  |
Step 3: Crash Recovery (VALD-03)
  |-- Modify train.py to trigger OOM (allocate huge tensor)
  |-- Commit and run
  |-- Verify OOM is caught (not a hard crash)
  |-- Verify results.tsv logs "crash" status
  |-- Verify git reset reverts to pre-crash state
  |-- Verify loop continues (next experiment starts)
  |
Step 4: Launch Overnight Run (D-02)
  |-- All 3 validations passed
  |-- Launch autonomous agent with program.md instructions
  |-- No experiment count limit (run indefinitely until interrupted)
```

### Baseline Stability Assessment

**Recommendation: 1 baseline run.** Rationale:
- The combined metric (0.5 * recall@1 + 0.5 * mean_cosine) is computed over a fixed validation set
- With deterministic seeds set in prepare.py (Pitfall 7 prevention from Phase 1), a single run should be reproducible
- Multiple baselines would consume 30+ minutes of GPU time for marginal confidence gain
- If the single baseline produces a reasonable metric (non-zero, non-NaN, within expected range), that is sufficient evidence
- If the metric looks suspicious, a second run can be added as a fallback

### OOM Trigger Technique

**Recommendation: Allocate an oversized tensor at the start of training.**

This is the simplest, most reliable, and most reversible approach:

```python
# Add near the top of train.py's training function, before training loop:
# VALIDATION ONLY: intentional OOM trigger
_oom_trigger = torch.zeros(24_000, 1024, 1024, device="cuda")  # ~24GB, exceeds 4090
```

Why this over alternatives:
- **Increasing batch size** is unreliable -- may not trigger OOM on first forward pass, and the batch size that triggers OOM depends on model size and sequence length
- **Allocating a specific tensor** is deterministic and immediate -- it will always OOM if the allocation exceeds available VRAM
- **Easy to revert** -- delete the single line after validation, or git reset removes it automatically

After OOM is confirmed caught and logged, the agent (or validation script) should `git reset --hard` to remove the OOM trigger and continue.

### Overnight Run Launch

Per D-02, after validation passes, launch the autonomous overnight run immediately. Key considerations:

- **No experiment limit.** The autoresearch pattern is "NEVER STOP" -- the agent runs until manually interrupted. Setting an arbitrary limit defeats the purpose.
- **Branch naming.** The autonomous run should happen on a branch like `autoresearch/<tag>` following the pattern in program.md. The tag should be date-based (e.g., `autoresearch/mar25`).
- **results.tsv initialization.** The baseline from VALD-01 should already be in results.tsv. The overnight run continues from this state -- do not reset results.tsv.
- **Output redirection.** Each experiment: `python train.py > run.log 2>&1`. The run.log is overwritten per experiment (Pitfall 13).
- **Process isolation.** Each experiment must be a separate Python process to avoid VRAM leaks (Pitfall 11).

### Anti-Patterns to Avoid
- **Over-validating:** Running 10 baseline experiments "for statistical significance" wastes hours. One clean run is sufficient for v1.
- **Manual intervention between steps:** The validation should flow automatically. If baseline passes, proceed to autonomous cycle. If that passes, proceed to crash test. If that passes, launch overnight.
- **Validating infrastructure in isolation:** Phase 4 validates the *integrated system*, not individual components. Phase 1's tests already verify the split boundary. Phase 4 proves the whole chain works together.
- **Forgetting to remove OOM trigger:** After crash recovery test, the OOM trigger line must be removed (via git reset) before the autonomous run starts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Results validation | Custom metric parsing | grep results.tsv + check row count | results.tsv format is already defined by Phase 2; parsing it is a one-liner |
| Git state verification | Custom git wrappers | `git log --oneline -N` and `git diff` | Git CLI is sufficient; wrapping it adds complexity with no benefit |
| OOM triggering | Complex memory pressure simulation | `torch.zeros(big, device="cuda")` | Simple allocation is deterministic and immediate |
| Overnight launch | Custom scheduler or daemon | Direct agent invocation in the terminal | The agent IS the scheduler -- program.md says "NEVER STOP" |

## Common Pitfalls

### Pitfall 1: Non-Deterministic Baseline
**What goes wrong:** Baseline run produces different metrics each time. Agent keeps experiments that were only "better" due to random seed variance.
**Why it happens:** Random seeds not set, CUDA non-determinism, data shuffling variance.
**How to avoid:** Verify Phase 1 implemented deterministic seeds in prepare.py. Run baseline, note the metric. If a second run produces a significantly different metric (>1% relative), investigate seed handling before proceeding.
**Warning signs:** Combined metric varies by more than 0.01 between identical runs.

### Pitfall 2: OOM Not Actually Caught
**What goes wrong:** The intentional OOM crashes the entire process instead of being caught by the crash recovery handler.
**Why it happens:** Phase 2's crash recovery wraps `torch.cuda.OutOfMemoryError` but the OOM might manifest as a different exception (e.g., `RuntimeError: CUDA error: out of memory` in some PyTorch versions), or the allocation happens outside the try/except block.
**How to avoid:** The OOM trigger must occur inside the code path that Phase 2's crash handler wraps. Verify the exact exception type that PyTorch 2.9.1 raises for OOM -- it should be `torch.cuda.OutOfMemoryError` (subclass of RuntimeError, introduced in PyTorch 1.13+).
**Warning signs:** Process exits with non-zero code instead of logging "crash" and continuing.

### Pitfall 3: results.tsv Corruption During Validation
**What goes wrong:** Multiple validation steps append to results.tsv in unexpected ways. Tab/comma confusion, missing newlines, or duplicate headers corrupt the file.
**Why it happens:** Manual appending between automated steps, or the crash recovery handler not properly writing to the file.
**How to avoid:** After each validation step, `cat results.tsv` and visually verify the format. Check for exactly the expected number of rows.
**Warning signs:** results.tsv has non-tab separators, missing columns, or multiple header rows.

### Pitfall 4: Overnight Run Starts With Stale Git State
**What goes wrong:** The OOM trigger from VALD-03 is still in train.py when the overnight run launches. Every experiment immediately OOMs.
**Why it happens:** Forgot to git reset after crash recovery test, or the reset targeted the wrong commit.
**How to avoid:** After VALD-03, verify `git diff` shows clean train.py (no OOM trigger). Then verify train.py matches the post-VALD-02 state (baseline + any kept experiments).
**Warning signs:** First overnight experiment immediately crashes with OOM.

### Pitfall 5: Agent Cannot Import prepare.py at Runtime
**What goes wrong:** train.py fails to import from prepare.py because of missing data files, uncached teacher embeddings, or environment issues.
**Why it happens:** prepare.py may require certain cached data to exist (teacher embeddings, dataset shards). If these are not pre-built, the import or initialization fails.
**How to avoid:** Before starting validation, ensure `python prepare.py` has been run to build all caches. This should be a Phase 1 prerequisite but must be verified.
**Warning signs:** ImportError or FileNotFoundError when running train.py.

## Code Examples

### VALD-01: Baseline Run and Verification

```bash
# Step 1: Ensure we're on the correct branch
git checkout -b autoresearch/val-$(date +%b%d | tr '[:upper:]' '[:lower:]')

# Step 2: Initialize results.tsv with header
printf 'commit\tcombined_metric\trecall_at_1\tmean_cosine\tpeak_vram_gb\tstatus\tdescription\n' > results.tsv

# Step 3: Commit baseline state
git add train.py
git commit -m "baseline: unmodified train.py"

# Step 4: Run baseline
python train.py > run.log 2>&1

# Step 5: Extract metrics from run.log
grep "^combined_metric:\|^recall@1:\|^mean_cosine:\|^peak_vram" run.log

# Step 6: Log to results.tsv (agent does this programmatically)
# Verify: combined_metric > 0, not NaN
```

### VALD-03: OOM Trigger and Recovery

```python
# Insert at the top of train.py training function:
# This allocates ~24GB, guaranteed to exceed RTX 4090's available VRAM
# (some VRAM already consumed by PyTorch context and model loading)
_oom_trigger = torch.zeros(24_000, 1024, 1024, device="cuda", dtype=torch.float32)
```

```bash
# After inserting trigger:
git add train.py
git commit -m "test: intentional OOM trigger for crash recovery validation"
python train.py > run.log 2>&1

# Verify crash was caught (not a hard process crash):
# - results.tsv should have a new row with status "crash"
# - The process should still be running (if in a loop) or exit cleanly (if single-shot)
grep "crash" results.tsv

# Verify git reset happened:
git log --oneline -3  # should show reset back to pre-OOM commit

# Clean up: remove OOM trigger
git reset --hard HEAD~1  # or HEAD if the loop already reset
```

### Verification Assertions

```bash
# After all 3 validation steps, verify:

# 1. results.tsv has at least 3 data rows (baseline + experiment + crash)
awk 'NR>1' results.tsv | wc -l  # should be >= 3

# 2. At least one "kept" or "keep" status exists (baseline)
grep -c "keep" results.tsv  # should be >= 1

# 3. At least one "crash" status exists (VALD-03)
grep -c "crash" results.tsv  # should be == 1

# 4. Combined metric is non-zero for kept experiments
awk -F'\t' 'NR>1 && $6=="keep" { if ($2+0 > 0) print "PASS: "$7" metric="$2; else print "FAIL: zero metric" }' results.tsv

# 5. Git state is clean (no uncommitted OOM trigger)
git diff --stat  # should be empty or only results.tsv (untracked)
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (to be installed by Phase 1, Plan 3) |
| Config file | none -- see Wave 0 |
| Quick run command | `uv run pytest tests/ -x --timeout=60` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map

Phase 4 validation is fundamentally different from unit testing. The requirements are validated by **executing the system** and observing outcomes, not by running pytest. However, automated verification scripts can confirm the observable outcomes.

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VALD-01 | Baseline produces valid metric in results.tsv | integration (live GPU) | `python train.py > run.log 2>&1 && grep "combined_metric" run.log` | N/A -- manual execution |
| VALD-02 | Autonomous cycle completes keep/discard | integration (live GPU + agent) | Observe results.tsv row count >= 2 after agent cycle | N/A -- agent-driven |
| VALD-03 | OOM caught, logged as crash, loop continues | integration (live GPU) | Insert OOM trigger, run, verify "crash" in results.tsv | N/A -- manual execution |

### Sampling Rate
- **Per validation step:** Visual inspection of results.tsv + git log
- **Phase gate:** All 3 VALD requirements pass before overnight launch

### Wave 0 Gaps
- pytest should already be installed from Phase 1 Plan 3
- No new test files needed for Phase 4 -- validation is execution-based, not test-based
- A validation script (`validate.sh` or inline commands) could optionally automate the verification assertions, but this is Claude's discretion

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All training | Yes | 3.12.11 | -- |
| PyTorch 2.9.1 | Training/OOM | Yes | 2.9.1+cu128 | -- |
| NVIDIA RTX 4090 | GPU training | Yes | 24564 MiB | -- |
| Git | Experiment versioning | Yes | 2.34.1 | -- |
| uv | Script runner | Yes | 0.8.20 | -- |
| Dataset/cache | Evaluation | Unknown | -- | Must run prepare.py first |
| ONNX teacher model | Distillation | Unknown | -- | Must verify workspace/output/trendyol_teacher_cache2/ exists |

**Missing dependencies with no fallback:**
- None -- all core tools are available

**Missing dependencies with fallback:**
- Dataset cache and teacher embeddings: May not exist yet. If missing, `python prepare.py` must be run first (this is a prerequisite, not a fallback).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.cuda.OutOfMemoryError` as RuntimeError only | Dedicated `torch.cuda.OutOfMemoryError` exception class | PyTorch 1.13+ | Can catch OOM specifically without catching other RuntimeErrors |
| Wall-clock training budget (5 min) | Epoch-based budget (10 epochs) | Project decision | Makes experiments comparable regardless of batch size changes |
| results.tsv left untracked | results.tsv untracked by git | Autoresearch pattern | Git history only shows kept experiments; results.tsv records everything |

## Open Questions

1. **Dataset and teacher cache state at Phase 4 start**
   - What we know: Phase 1 will create prepare.py which builds these caches
   - What's unclear: Will the caches already exist from Phase 1 testing, or do they need to be rebuilt?
   - Recommendation: Plan should include a "verify caches exist, rebuild if needed" step before baseline

2. **Exact results.tsv column format**
   - What we know: INFRA-02 specifies columns (commit hash, combined metric, recall@1, mean_cosine, peak VRAM, status, description)
   - What's unclear: Exact header names (snake_case? camelCase? with @?)
   - Recommendation: Plan should read results.tsv format from Phase 2's implementation, not assume

3. **How the autonomous agent is invoked**
   - What we know: program.md (Phase 3) documents the loop. The agent reads program.md and drives the loop.
   - What's unclear: Is the agent invoked via Claude Code CLI? A custom wrapper? Direct terminal?
   - Recommendation: Plan should specify the exact invocation command for launching the overnight agent

## Sources

### Primary (HIGH confidence)
- `program.md` -- Karpathy autoresearch agent instructions (the pattern being adapted)
- `.planning/REQUIREMENTS.md` -- VALD-01, VALD-02, VALD-03 definitions
- `.planning/phases/04-validation/04-CONTEXT.md` -- User decisions D-01, D-02
- `.planning/research/SUMMARY.md` -- System architecture and phase rationale
- `.planning/research/PITFALLS.md` -- Pitfalls 3, 7, 11, 13 directly relevant

### Secondary (MEDIUM confidence)
- `.planning/ROADMAP.md` -- Phase 4 success criteria
- `.planning/phases/01-core-refactoring/01-RESEARCH.md` -- Phase 1 outputs that Phase 4 validates
- PyTorch 2.9.1 OOM exception handling -- verified `torch.cuda.OutOfMemoryError` exists since PyTorch 1.13

### Tertiary (LOW confidence)
- Exact metric ranges for ReID baseline (depends on dataset quality, teacher model, and Phase 1 implementation choices)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all tools verified as installed
- Architecture: HIGH -- validation pattern is straightforward execution + observation
- Pitfalls: HIGH -- pitfalls derived from direct analysis of the autoresearch pattern and PyTorch OOM behavior

**Research date:** 2026-03-25
**Valid until:** 2026-04-25 (stable -- validation pattern does not change)
