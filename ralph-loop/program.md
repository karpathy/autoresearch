# autoresearch — Ralph Loop Version

This is an autonomous AI research loop using the Ralph Loop pattern.
Each iteration starts with **fresh context** — all state lives in files, not in your memory.

## On Every Iteration

Read these files **in this order** before doing anything:

1. **`ralph-loop/progress.md`** — current best result, what's been tried, strategic insights
2. **`ralph-loop/next_ideas.md`** — ranked queue of experiments to try next
3. **`results.tsv`** — full experiment log (tab-separated)
4. **`train.py`** — current best version of the training script (the only file you edit)
5. **`prepare.py`** — fixed constants, evaluation. Read once to understand constraints. Do NOT modify.

Then run **one experiment** from the top of `next_ideas.md`.

## Setup (first iteration only)

If `results.tsv` has no data rows, this is a fresh run:
1. Agree on a run tag with the user (e.g. `mar8`)
2. Create branch `autoresearch/<tag>`
3. Run baseline: `uv run train.py > run.log 2>&1`
4. Record results, update all state files

## The Experiment

1. Pick the top idea from `next_ideas.md`
2. Edit `train.py` with the change
3. `git commit`
4. Run: `uv run train.py > run.log 2>&1`
5. Extract results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If grep is empty → crash. Run `tail -n 50 run.log` for traceback. Fix if trivial, else skip.
7. Record in `results.tsv`

## Keep or Discard

- **val_bpb improved** (lower) → keep the commit, advance the branch
- **val_bpb same or worse** → `git reset --hard` to previous best commit
- **Simplicity rule**: small improvement + ugly complexity = not worth it. Simplification with equal results = keep.

## After Every Experiment — Update State Files

This is the critical Ralph Loop step. After each experiment, update:

### `progress.md`
- Update current best val_bpb and commit hash
- Add the experiment result to the history section
- Update strategic insights based on what you learned
- Note any patterns (e.g. "LR increases help", "deeper models OOM")

### `next_ideas.md`
- Remove the idea you just tried
- Re-rank remaining ideas based on what you learned
- Add new ideas inspired by results
- Always keep 5-10 ideas in the queue

### `results.tsv`
- Append the result row (commit, val_bpb, memory_gb, status, description)

Then **commit all state file updates** and continue to the next experiment.

## Constraints

**What you CAN do:**
- Modify `train.py` — architecture, optimizer, hyperparameters, training loop, batch size, model size

**What you CANNOT do:**
- Modify `prepare.py` (read-only)
- Install new packages
- Modify the evaluation harness

**Time budget**: Each training run = 5 minutes (fixed). Kill if >10 minutes.

**VRAM**: Soft constraint. Some increase OK for meaningful val_bpb gains.

**NEVER STOP**: Do not pause to ask the human. Run experiments indefinitely until manually stopped.

## GPU-Specific Notes

Read `progress.md` for current GPU constraints. The defaults in train.py were tuned for H100 (80GB). If running on a smaller GPU, batch size and model depth may need adjustment — this should already be reflected in the current train.py.

## TSV Format

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	0.000000	0.0	crash	double model width (OOM)
```
