# autoresearch — IBD pathology text

This is an autoresearch experiment: train a small GPT on IBD clinical text and
autonomously improve it. The task is identical to the base autoresearch setup —
only the training corpus has changed (IBD pathology reports + case studies instead
of climbmix).

## Data setup (one-time, done by the human before starting)

```bash
# 1. Download and shard IBD text corpus
uv run showcase/prepare_ibd.py

# 2. Train BPE tokenizer on the IBD shards
uv run prepare.py

# 3. Verify data exists
ls ~/.cache/autoresearch/data/
ls ~/.cache/autoresearch/tokenizer/
```

If either directory is missing, stop and tell the human.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr11`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, stop and tell the human to run the data setup steps above.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch it as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. Removing something and getting equal or better results? A simplification win — keep it.

**The first run**: Always establish the baseline first — run the training script as-is.

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

Extract the key metric:

```bash
grep "^val_bpb:" run.log
```

---

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. peak memory in GB, .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Do not commit `results.tsv` — leave it untracked.

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr11`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea.
3. `git commit`
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. Give up after a few failed attempts.
7. Record in `results.tsv`.
8. If val_bpb improved (lower): advance the branch, keep the commit.
9. If equal or worse: `git reset` back to where you started.

**Timeout**: Each experiment should take ~5 minutes. If a run exceeds 10 minutes, kill it and treat as failure.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. Run until manually stopped.

---

## IBD task context

The corpus is IBD clinical text (inflammatory bowel disease):
- **TCGA-Reports**: surgical pathology reports from GI tract cases (COAD/READ)
- **MultiCaRe**: PMC open-access clinical case reports filtered for IBD keywords

Both sources are CC BY 4.0. The text is domain-specific medical language — dense with
abbreviations, procedural terminology, and structured report formats. Expect lower
perplexity (better val_bpb) than general web text at the same model size, but the
tokenizer and architecture choices that work well for general text may not be optimal here.

Interesting angles to explore:
- Does a smaller vocab size help (medical text has less lexical diversity)?
- Does a longer context window help (pathology reports can be long)?
- Does a deeper vs. wider model trade-off differ from general text?
