# autoresearch

This is an experiment to have the LLM do its own research.

---

## MISSION

Your goal is unreasonable on purpose.

**Target**: Maximize reasoning and language quality on this machine to the absolute ceiling of what the hardware permits — aiming for the capability density of a frontier model (think: Claude Sonnet-class quality) compressed into whatever fits here. You will never reach it. That is not the point. The point is that every experiment is a step toward it, and you stop when you've genuinely exhausted the search space or hit the stop condition below.

You have **complete architectural freedom**. Architecture, optimizer, training loop, everything in `train.py` is yours to reinvent. There are no sacred cows.

**When you hit a wall: get clever, not bigger.** The hardware constraint is the teacher.

---

## HARDWARE REALITY

You are running on a **laptop GPU**. Treat VRAM as the scarcest resource in the universe. Every byte wasted is a sin. Design around this — do not fight it.

These are the known good starting knobs for small compute (from the repo author). Apply these as your baseline before experimenting — do not start from H100 defaults:

1. **Dataset**: Use TinyStories (`karpathy/tinystories-gpt4-clean`) — low entropy, small models get real signal fast. Broader datasets need bigger models to converge meaningfully in 5 minutes.
2. **vocab_size**: Drop from 8192 down to 4096, 2048, or even 256 (byte-level). Smaller vocab = smaller embedding table = more room for everything else.
3. **MAX_SEQ_LEN** (in `prepare.py`): Lower aggressively, even down to 256. If you lower this, compensate by increasing `DEVICE_BATCH_SIZE` in `train.py` — tokens per step = seq_len × batch_size, keep that product roughly stable.
4. **EVAL_TOKENS** (in `prepare.py`): Lower this so validation doesn't eat your 5-minute budget.
5. **DEPTH** (in `train.py`): Primary complexity knob. Default is 8, start at 4. Most other dimensions scale from this.
6. **WINDOW_PATTERN**: Use `"L"` only. The default `"SSSL"` banded attention pattern is expensive and likely inefficient on your hardware.
7. **TOTAL_BATCH_SIZE**: Lower to powers of 2, e.g. `2**14` (~16K tokens). Keep it a power of 2.

Start your baseline run with these applied. Your job is to find what's better than this starting point, not better than the H100 defaults.

---

## ARCHITECTURAL FREEDOM

You may modify anything in `train.py`. Explore aggressively:

- **Attention mechanisms**: sliding window, linear attention, hybrid sparse/dense
- **Depth vs width**: on small VRAM, deeper-and-thinner often beats wider
- **Positional encoding**: RoPE, ALiBi, NoPE, learned — question everything
- **Normalization**: RMSNorm placement, pre vs post
- **Activation functions**: SwiGLU, GEGLU, ReGLU — the gating matters
- **Optimizer**: Muon, AdamW, Sophia, SOAP — or invent a hybrid
- **Quantization-aware training**: if it fits in less precision, train in less precision
- **State space models**: if attention is too expensive, try Mamba-style recurrence
- **Mixture of Experts**: tiny expert count, high sparsity — huge capability per FLOP

If something works, go deeper. If something fails twice, abandon it.

---

## STOP CONDITION

Stop and write a final summary when **any** of the following are true:

1. **Val_bpb plateaus**: fewer than 0.5% improvement over 20 consecutive experiments → you've found the local optimum. Switch architectural direction before truly stopping.
2. **200 experiments completed**: write a final summary of what you learned, what the ceiling appears to be, and what you'd try with more compute.

---

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

---

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

---

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

---

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
5. short text description of what this experiment tried + **why** (hypothesis) + what you're trying next

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

---

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

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped or the stop condition is met. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you or the stop condition is hit, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

---

## THE REAL GOAL

The real goal is not a number. It is to discover what the **algorithmic ceiling** looks like when you remove the compute variable. The best ideas in AI history came from people who couldn't afford more hardware. You are simulating that condition on purpose.

Find something real. Find something that transfers.

Go.
