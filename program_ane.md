# autoresearch-ane

This is an experiment to have the LLM do its own research, running training on Apple Neural Engine (ANE).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch-ane/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-ane/<tag>` from current `ane-backend`.
3. **Read the in-scope files**: Read these files for full context:
   - `program_ane.md` — this file, your instructions.
   - `ane/experiment_config.h` — the file you modify. Architecture and optimizer hyperparameters.
   - `harness_ane.py` — the orchestrator. Do not modify.
4. **Verify data exists**: Check that `ane/tinystories_data00.bin` exists. If not, run `bash ane/download_data.sh`.
5. **Verify compilation**: Run `make -C ane train_ane` to confirm the binary compiles.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on the Apple Neural Engine. The training binary runs for a **fixed time budget of 5 minutes** (wall clock). You launch it simply as: `python harness_ane.py`.

**What you CAN do:**
- Modify `ane/experiment_config.h` — this is the ONLY file you edit. It contains architecture defines (DIM, HIDDEN, HEADS, SEQ, NLAYERS) and optimizer hyperparameters (LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ACCUM_STEPS).

**What you CANNOT do:**
- Modify any other file (`train_ane.m`, `stories_config.h`, `harness_ane.py`, etc.). They are read-only.
- Change VOCAB — it must stay 32000 (Llama2 BPE tokenizer).
- Install new packages or add dependencies.

**The goal is simple: get the lowest val_loss.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything in `experiment_config.h` is fair game.

**Architecture vs hyperparameter changes:**
- **Architecture changes** (DIM, HIDDEN, HEADS, SEQ, NLAYERS): These reset the checkpoint to random initialization. This is "expensive" — you lose all training progress.
- **Hyperparameter changes** (LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ACCUM_STEPS): These continue from the existing checkpoint. This is "cheap" — training progress is preserved.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement from a radical architecture change that loses checkpoint progress may not be worth it compared to a hyperparameter tweak.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_loss:         3.456789
train_loss:       3.234567
steps:            120
ms_per_step:      412.3
wall_time_s:      300.1
compile_time_s:   45.2
ane_util_pct:     12.5
```

You can extract the key metric from the log file:

```
grep "^val_loss:" run_ane.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_loss	ane_util_pct	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 3.456789) — use 0.000000 for crashes
3. ANE utilization % (e.g. 12.5) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-ane/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `ane/experiment_config.h` with an experimental idea by directly editing the defines.
3. git commit
4. Run the experiment: `python harness_ane.py > run_ane.log 2>&1` (redirect everything)
5. Read out the results: `grep "^val_loss:\|^ane_util_pct:" run_ane.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run_ane.log` to read the error and attempt a fix.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_loss improved (lower), you "advance" the branch, keeping the git commit
9. If val_loss is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes total (+ overhead for compilation and validation). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment. If it's a simple fix (e.g. DIM not divisible by HEADS), fix and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try different learning rates, different accumulation steps, different model sizes, combine previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
