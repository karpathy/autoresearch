# autoresearch

This experiment lets the LLM do its own research.

## Setup

To set up a new experiment, do these steps with the user:

1. **Agree on a run tag**: Propose a tag based on today's date (for example, `mar5`). Make sure the branch `autoresearch/<tag>` does not already exist. This is a new run.
2. **Create the branch**: Run `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repository is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Make sure `~/.cache/autoresearch/` contains data shards and a tokenizer. If it does not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with only the header row. Record the baseline after the first run.
6. **Confirm and go**: Confirm that the setup is correct. Then continue.

After the user confirms, start the experiments.

## Experimentation

Each experiment uses one GPU. The training script runs for a **fixed time budget of 5 minutes**. This is wall clock training time and excludes startup or compilation. Start it with: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. You can change the model architecture, optimizer, hyperparameters, training loop, batch size, model size, and related code.

**What you CANNOT do:**
- Do not modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants. These include the time budget and sequence length.
- Do not install new packages or add dependencies. Use only what is already in `pyproject.toml`.
- Do not modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** The fixed time budget removes training time as a variable. You do not need to optimize training time. Each run is always 5 minutes. You can change the architecture, optimizer, hyperparameters, batch size, and model size. The code must run without a crash. It must finish within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains. Do not let VRAM increase by a large amount.

**Simplicity criterion**: When all other results are equal, simple code is better. Do not keep a small improvement if it adds unnecessary complexity. Keep a change that removes code and gives equal or better results. When you evaluate a change, compare the complexity cost with the improvement size. For example, do not keep a 0.001 val_bpb improvement if it adds 20 lines of bad code. Keep a 0.001 val_bpb improvement if it deletes code. Keep a change with approximately no improvement if it makes the code much simpler.

**The first run**: For your first run, establish the baseline. Run the training script without changes.

## Output format

After the script finishes, it prints a summary like this:

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

The script always stops after 5 minutes. Thus, numbers can differ by computing platform. Use this command to extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv`. Use tabs, not commas. Commas break descriptions.

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb result (for example, 1.234567). Use 0.000000 for crashes.
3. peak memory in GB, rounded to .1f (for example, 12.3). Divide peak_vram_mb by 1024. Use 0.0 for crashes.
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

The experiment runs on a dedicated branch (for example, `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Tune `train.py` with an experimental idea. Edit the code directly.
3. Create a git commit.
4. Run the experiment: `uv run train.py > run.log 2>&1` (Redirect all output. Do not use tee. Do not let output fill your context.)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log`. Read the Python stack trace. Try to fix the problem. If you cannot fix it after more than a few attempts, stop that idea.
7. Record the results in the tsv. Do not commit the results.tsv file. Leave it untracked by git.
8. If val_bpb improved (lower), advance the branch. Keep the git commit.
9. If val_bpb is equal or worse, reset git to where you started.

Act as an autonomous researcher. Try changes. Keep changes that work. Discard changes that do not work. Advance the branch so that you can iterate. If you get stuck, you can rewind. Use this option very sparingly, if ever.

**Timeout**: Each experiment should take approximately 5 minutes total. Startup and eval overhead can add a few seconds. If a run exceeds 10 minutes, kill it. Treat it as a failure. Discard it and revert.

**Crashes**: If a run crashes because of OOM, a bug, or another issue, use your judgment. If the fix is simple, fix it. Then run it again. For example, fix a typo or add a missing import. If the idea itself is broken, skip it. Log `crash` as the status in the tsv. Then continue.

**NEVER STOP**: After the experiment loop starts, do not pause to ask the human if you should continue. Do not ask "should I keep going?" Do not ask "is this a good stopping point?". The human might be asleep or away from the computer. The human expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder. Read papers referenced in the code. Read the in-scope files again for new angles. Try combinations of previous near-misses. Try more radical architectural changes. The loop runs until the human interrupts you. No other stop condition applies.

For example, a user might let you run while they sleep. If each experiment takes approximately 5 minutes, you can run approximately 12 per hour. That is approximately 100 experiments during average human sleep. The user then wakes up to experimental results. You completed all experiments while they slept!
