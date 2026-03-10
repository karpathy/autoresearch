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

**Memory probing**: For architecture changes that might affect memory (model width, depth, batch size), run `uv run train.py --probe > probe.log 2>&1` first (~30s, runs 3 steps and reports peak VRAM). Check `grep probe_peak_vram_mb probe.log` before committing to a full run.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**Suggested experiment ordering** (rough priority):

1. **Learning rates** — try 2x/0.5x on matrix_lr, embedding_lr. Cheap, often high impact.
2. **Model size** — DEPTH ±2, ASPECT_RATIO changes.
3. **Batch size** — try 2x/0.5x TOTAL_BATCH_SIZE.
4. **Warmup/cooldown** — WARMUP_RATIO, WARMDOWN_RATIO.
5. **Architecture** — attention patterns, activations, MLP ratio. Higher risk, higher reward.
6. **Optimizer** — betas, weight decay, momentum. Usually small gains.

Start with cheap, high-probability wins before riskier architectural changes.

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
total_batch_size: 524288
matrix_lr:        0.04
loss_trajectory:  25%:3.2145 50%:2.8901 75%:2.6543 100%:2.5012
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^num_params_M:\|^depth:\|^total_batch_size:\|^matrix_lr:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 10 columns:

```
commit	val_bpb	memory_gb	mfu	num_params_M	depth	total_batch_size	matrix_lr	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. mfu_percent from run output (e.g. 39.80) — use 0.00 for crashes
5. num_params_M from run output (e.g. 50.3) — use 0.0 for crashes
6. depth from run output (e.g. 8) — use 0 for crashes
7. total_batch_size from run output (e.g. 524288) — use 0 for crashes
8. matrix_lr from run output (e.g. 0.04) — use 0.00 for crashes
9. status: `keep`, `discard`, or `crash`
10. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	mfu	num_params_M	depth	total_batch_size	matrix_lr	status	description
a1b2c3d	0.997900	44.0	39.80	50.3	8	524288	0.04	keep	baseline
b2c3d4e	0.993200	44.2	39.50	50.3	8	524288	0.06	keep	increase matrix_lr to 0.06
c3d4e5f	1.005000	44.0	39.80	50.3	8	524288	0.04	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	0.00	0.0	0	0	0.00	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

Before each experiment, review `results.tsv` to inform your next choice:
- Which experiment types (LR, architecture, size) yielded the biggest gains?
- What has already been tried and discarded?
- Are there patterns you can extrapolate (e.g., "LR 0.04→0.03 helped, try 0.025")?

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^num_params_M:\|^depth:\|^total_batch_size:\|^matrix_lr:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Combining improvements**: Every 10-15 experiments, review results.tsv for independently-tested improvements that could be combined. If "increase matrix_lr" and "increase depth" both helped separately, try both together.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Noise threshold**: Due to non-determinism in Flash Attention and limited eval data, val_bpb has measurement noise of roughly ±0.002. Only count improvements >0.003 as clearly real. For borderline results (0.001-0.003), consider re-running to confirm. When in doubt, prefer the simpler configuration.

**Loss trajectory**: The summary includes `loss_trajectory` showing smoothed training loss at 25/50/75/100% progress. If loss is still dropping steeply at 100%, the model may benefit from being smaller (more steps). If loss flattens early, try a larger model or higher LR.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Multi-Agent Mode

If you find a `CLAUDE.md` file in your working directory, you are running in multi-agent mode. **Read CLAUDE.md first** — it contains your specific role, agent ID, and tailored instructions that take precedence over the generic instructions above.

In multi-agent mode:

- **Use `run_experiment.py`** instead of raw `uv run train.py`:
  ```
  python run_experiment.py --scale quick --description "what you tried" --agent-id <your-id> --agent-role <your-role>
  ```
- **Check the research briefing** before each experiment:
  ```
  python run_experiment.py --briefing
  ```
- **Log lessons** when you discover something:
  ```
  python run_experiment.py --lesson <category> <confidence> "lesson text"
  ```
- **Respect the research agenda**: The director agent writes guidance to the journal. Check it.

### Variable Time Scales

Instead of a fixed 5-minute budget, experiments run at different scales:
- **probe** (30s): Memory/compilation check only
- **quick** (2min): Rough signal, enough to kill bad ideas fast
- **standard** (5min): Real evaluation, comparable to baseline
- **long** (15min): Confirmation of promising results
- **deep** (30min): Final validation, gold standard

Start with quick runs and escalate promising ideas to longer scales.

### Knowledge Base

Results are logged to a shared knowledge base (`results/` directory) instead of a local `results.tsv`. The knowledge base includes:
- **experiments.jsonl**: All experiment records from all agents
- **lessons.jsonl**: Discovered patterns and insights
- **journal.md**: Research summaries and the director's agenda

A backward-compatible `results.tsv` is auto-generated for the analysis notebook.
