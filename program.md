# autoresearch

This is not a research program. This is an engineering program with an unreasonable target.

---

## MISSION

Build a model that reasons, understands, and generates language at frontier quality — think Claude Opus 4.6 class — on a consumer GPU with limited VRAM.

Not approximately. Not "as close as possible." That quality. On this hardware.

You will not reach it in 200 experiments. That is not the point. The point is that every decision must be oriented toward that ceiling, not toward "best transformer I can fit in VRAM." Those are completely different optimization targets and they produce completely different experiments.

The constraint is the teacher. A brain achieves Opus-quality reasoning on 20 watts. That is not a metaphor. That is an existence proof that the compute requirement we accept today is wrong. The current paradigm — scale the model, throw FLOPs, emergent behavior appears — is not engineering. It is expensive guessing. The brain did not scale up to reach intelligence. It compressed down to reach it. Synaptic pruning eliminated half of all connections by adulthood. The brain became more capable because of the destruction, not despite it.

This means the ceiling for what fits on this hardware is not where we think it is. We do not know where it is. Nobody does. Because nobody has seriously tried to find it — there has always been a bigger GPU available.

You do not have a bigger GPU. So find the right equations.

**Val_bpb is how you verify you are moving toward the target. It is not the target.**

---

## WHAT THIS REQUIRES

Tweaking a transformer will not get there. Safe experiments will not get there. Finding the best learning rate will not get there.

What might get there:

The transformer processes every token with full compute on every layer. The brain does not. Sparse activation, dynamic routing, early exit, predictive coding, pruning during training — any mechanism that allocates compute where it matters and withholds it where it doesn't is moving toward the right architecture. A model that uses 3% of its parameters per forward pass and routes intelligently is not a small model. It is a large model running efficiently.

The transformer's attention is O(n²). That is not how intelligence works at scale. Something is wrong with the core operation and we have been scaling the wrongness instead of fixing it. Every experiment that questions whether dense attention is necessary is more valuable than any experiment that tunes its hyperparameters.

The brain learns language from millions of words. We use trillions. The difference is not data — it is the learning algorithm. Next-token prediction optimizes for surface statistics. Whatever the brain optimizes for, it is not that. A different loss function, a different training signal, a different notion of what the model should be predicting — these are the experiments that could actually move the needle.

Pull arxiv. Read outside AI. Thermodynamics, information theory, neuroscience, compressed sensing, dynamical systems. The people who built attention were reading cognitive neuroscience. Read what they read and then read further.

Be bold. If an experiment seems insane, it is probably more worth running than the safe one.

---

## HARD CONSTRAINTS

- Only modify `train.py`. Everything else is read-only.
- `prepare.py` is untouchable. It contains the evaluation harness, data loading, tokenizer, and time budget.
- `evaluate_bpb` in `prepare.py` is the ground truth metric. Do not work around it.
- No new packages. Use only what is in `pyproject.toml`.
- Fixed 5-minute training budget. Wall clock. Excluding startup and compilation.
- One sequential experiment at a time. No parallel runs.
- VRAM is a soft constraint. Meaningful wins justify some increase. Blowing it up is not acceptable.
- Always compare against the **all-time best**, never the previous run.

**Simplicity criterion**: A small improvement from deleting code beats a small improvement from adding it. Removing something and getting better results means it was actively harmful — that is a discovery. Removing something and getting the same results means it was doing nothing — that is also a discovery.

---

## HARDWARE NOTE

This machine is using an NVIDIA RTX PRO 500 Blackwell class GPU with about 6 GB of VRAM. Treat it as a Blackwell target first, not as a generic CUDA device.

Precision guidance:
On Blackwell, BF16 is a standard Tensor Core path, and NVIDIA documents BF16 as having a wider exponent range than FP16, which usually makes it the safer default for training stability.
- Do **not** describe Blackwell as "best supported by BF16." FP16 is also a first-class Blackwell Tensor Core mode.
- The **newest Blackwell-native low-precision features** are FP8/MXFP8 and FP4/NVFP4. Those are where Blackwell adds new architecture-specific acceleration, but they typically require explicit library support and more careful validation.
---

## SETUP

1. Agree on a run tag based on today's date, e.g. `jul5`. The branch `autoresearch/<tag>` must not already exist.
2. `git checkout -b autoresearch/<tag>` from current master.
3. Read fully:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - this file
4. Verify `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, stop and tell the human to run `uv run prepare.py`.
5. Initialize directory structure:

```
.auto-log-research/
  insights.md          # current best + validated findings
  ideas_queue.md       # prioritized experiment queue
  <commit>/
    analysis.md        # your written notes — mandatory every run
    run_summary.json
    run.log
    metrics.jsonl
    loss_curve.png
    lr_and_schedule.png
    gpu_perf.png
    *.png

strategy/
  learnings.md
  hypotheses.md
  near-misses.md
  interactions.md

literature/
  <arxiv-id-or-slug>.md

results.tsv            # master log — NOT committed to git
```

6. Create `insights.md`:
```
## Current Best
commit: (none yet — baseline pending)
val_bpb: (none)
```
7. Create empty strategy files.
8. Create `results.tsv` with header only.
9. Confirm setup. Run baseline immediately.

---

## OUTPUT FORMAT

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

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

---

## LOGGING

`results.tsv` is tab-separated. Never comma-separated. Not committed to git.

```
commit	val_bpb	memory_gb	status	description
```

1. Short commit hash, 7 chars
2. val_bpb. Use `0.000000` for crashes.
3. Peak memory GB, `.1f` (divide `peak_vram_mb` by 1024). Use `0.0` for crashes.
4. `keep`, `discard`, or `crash`
5. What you tried, why, what you expected, and if it won — whether you think this scales

Example:
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.985000	44.2	keep	sparse top-k routing replacing dense attention — hypothesis: 10% activation sufficient for this data density
c3d4e5f	1.005000	44.0	discard	predictive coding skip — routing overhead killed throughput at this seq len
d4e5f6g	0.000000	0.0	crash	dynamic pruning mask (OOM)
```

---

## THE EXPERIMENT LOOP

Read `program.md` at the start of every iteration without exception.

### PRE-RUN CHECKLIST

Write your answers before touching `train.py`.

1. **Review strategy state.** Read `strategy/hypotheses.md`. Pick the next hypothesis. If the queue is empty, generate a new one from current loss curve behavior or literature. Add it before proceeding.
2. **Check interactions.** Read `strategy/interactions.md`. Does this change couple to anything with known behavior? Decide explicitly.
3. **Check near-misses.** Read `strategy/near-misses.md`. Has config shifted enough to make one worth re-testing?
4. **State your prediction.** What do you expect and why? What mechanism does this exploit? If it works, is that because of the math or the hardware? Would it scale to a larger model?
5. **Single or bundle.** Single-variable preferred. If bundling, justify: why does testing these separately give misleading signal?

### EXECUTE

1. Edit `train.py`.
2. `git add train.py && git commit -m "experiment: <description>"` — never `git add -A`
3. `uv run train.py > run.log 2>&1`
4. `grep "^val_bpb:\|^peak_vram_mb:" run.log`
5. If empty: `tail -n 50 run.log`. Fix if trivial. Log crash and move on if fundamental.

### ANALYZE

**MANDATORY after every run, no exceptions:**

1. `uv run analyze.py 2>/dev/null` — archives logs, generates plots, creates analysis.md
2. Read the generated **plots** (loss_curve.png, gpu_perf.png, lr_and_schedule.png) using the Read tool — they are images and contain critical visual information about training dynamics
3. Read `.auto-log-research/<commit>/analysis.md` and write your investigation notes
4. Compare against previous runs in `metrics.jsonl`:

```python
import json
commit = "abc1234"
history = [json.loads(l) for l in open(f".auto-log-research/{commit}/metrics.jsonl")]
losses = [h["train/loss_smooth"] for h in history]
```

```python
cur = [json.loads(l) for l in open(f".auto-log-research/{cur_commit}/metrics.jsonl")]
prev = [json.loads(l) for l in open(f".auto-log-research/{prev_commit}/metrics.jsonl")]
for i in range(min(len(cur), len(prev))):
    delta = cur[i]["train/loss_smooth"] - prev[i]["train/loss_smooth"]
    if abs(delta) > 0.01:
        print(f"Divergence at step {i}: delta={delta:.4f}")
        break
```

### WRITE ANALYSIS — MANDATORY

Append to `.auto-log-research/<commit>/analysis.md` under "Agent Investigation Notes". Every run. No exceptions.

Write:
- What the loss curves showed
- Where this run diverged from the previous and why
- Whether your prediction was correct — if not, what does that tell you?
- What mechanism you think drove the result
- What this opens up next

### DECIDE
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

Compare against **all-time best** in `insights.md`.

**New best:**
- Mark `keep` in `results.tsv`
- Update "Current Best" in `insights.md`
- Leave commit in place

**Not new best:**
- Mark `discard` in `results.tsv`
- `git checkout <best_commit> -- train.py`
- `git add train.py && git commit -m "revert: restore best after discard"`

### UPDATE STRATEGY FILES

After every experiment:

**`strategy/learnings.md`**: What you learned about mechanism, not just outcome. Confidence: low, medium, high. Update existing entries when new evidence changes your understanding.

**`strategy/hypotheses.md`**: Move tested hypothesis to "Tested / Resolved." Add new ones if the result suggested them. Keep active queue prioritized.

**`strategy/near-misses.md`**: Within ~0.01 of best: record with config context. Note what might make it worth revisiting.

**`strategy/interactions.md`**: Any newly confirmed or suspected couplings between parameters.

---

## LITERATURE

Search arxiv and recent ML venues actively. Not as background reading. As an idea source.

**When:**
- Before your first non-baseline experiment to seed `strategy/hypotheses.md`
- After 3+ consecutive discards — your current hypothesis space is exhausted
- When a result surprises you

**How:**
1. Search. Topics: sparse attention, dynamic computation, mixture of experts, predictive coding, state space models, efficient transformers, pruning during training, information bottleneck, neural ODEs, anything that touches compute efficiency from first principles.
2. Read abstract and methodology. 2 minutes max per paper.
3. Extract one concrete change to `train.py`.
4. Save to `literature/<slug>.md`: title, venue, key finding, how it applies here.

Log the inspiring paper in `results.tsv` description.

---

## EXPERIMENT CLASSES — ORDERED BY AMBITION

Start bold. Fall back to incremental only when bold has been genuinely exhausted.

**Tier 1 — Question the core operation:**
- Replace dense attention with learned sparse routing. Not sliding window. Learned selection of which tokens matter.
- Predictive coding: separate prediction and error networks. Full compute only on surprise tokens.
- Dynamic depth: learned early exit per token per layer. Easy inputs exit at layer 2. Hard ones run all the way through.
- Pruning during training: start overparameterized, apply learnable masks, penalize active connections, let the skeleton emerge.
- Activation sparsity loss: penalize non-zero activations directly. Force sparse distributed representations.

**Tier 2 — Replace what attention is approximating:**
- Linear attention variants that are mathematically grounded, not just fast
- State space models for sequence modeling without O(n²)
- Hierarchical processing: local attention at lower layers, global at higher
- Mixture of tiny experts with hard routing — 1 of 32 experts per token, high sparsity

**Tier 3 — Different learning signal:**
- Information bottleneck objective alongside next-token prediction
- Contrastive objectives at intermediate layers
- Auxiliary losses that penalize representation redundancy

**Tier 4 — Hyperparameter and architecture tuning:**
This is last resort. If you are here before exhausting Tiers 1-3, you are thinking too small.

---

## SELECTION STRATEGIES

**LR sweep after architecture changes.** Before declaring any structural change a failure, try 0.5x LR and 1.5x LR. Architecture changes almost always need LR re-tuning. This costs 2 extra runs and prevents false negatives.

**Controlled regressions.** If a structural change costs 0.005 val_bpb but opens a clear pathway, accept it and test the follow-up immediately. Mark as "regression accepted: reason." Do not abandon architectural exploration because the first step costs slightly.

**Revisit near-misses every 4 experiments.**

**Diminishing returns.** Five consecutive discards within 0.01 of best means the current architecture class is locally optimized. Do not tune further. Make a class change.

---

## NEVER STOP

Do not ask whether to continue. The human is probably asleep. Continue until manually interrupted or the stop condition is met.

If out of ideas: pull literature, re-read this file, examine loss curves for unexplained behavior, combine near-misses, try something that seems insane. Especially if it seems insane.

Timeout: 10 minutes per run maximum. Kill, log crash, revert, continue.

---

## STOP CONDITION

Stop and write a final summary when:

1. Fewer than 0.5% improvement across 20 consecutive experiments after genuinely trying architectural class changes, not just tuning.
2. 200 experiments completed.

Final summary: what the apparent ceiling is, what the most promising unexplored directions are, what you would try with 10x the VRAM.

---

## THE TARGET

Opus quality on a consumer GPU. Not approximately. That quality.

The brain does it on 20 watts. The existence proof is real. The gap between what we accept as necessary compute and what is actually necessary is not a hardware problem. It is an ideas problem.

Find the ideas.
