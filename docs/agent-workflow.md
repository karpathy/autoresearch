# Agent Workflow

This page documents the contract between the human, the coding agent, and the repo. The canonical source is [`program.md`](../program.md). This page expands on it for human readers and future agents.

## Roles

- **Human**: writes `program.md`, decides when to start/stop, reviews results. Does not edit `train.py` while a run is active.
- **Agent**: reads `program.md` and `README.md`, mutates `train.py`, commits each experiment, runs the script, decides keep/discard. Runs autonomously until interrupted.
- **Repo**: enforces the contract. `prepare.py` is read-only. `evaluate_bpb` is the metric. The 5-minute budget is hard-coded.

## Setup phase

Triggered by a human prompt like "look at program.md and kick off a new experiment". The agent:

1. **Proposes a tag** based on today's date (e.g., `mar5`). The branch `autoresearch/<tag>` must not exist yet.
2. **Creates the branch** off `master`: `git checkout -b autoresearch/<tag>`.
3. **Reads context**: `README.md`, `prepare.py`, `train.py`. The repo is small enough to keep all three in working memory.
4. **Verifies the cache**: confirms `~/.cache/autoresearch/data/` has shards and `~/.cache/autoresearch/tokenizer/` has `tokenizer.pkl` plus `token_bytes.pt`. If anything is missing, it asks the human to run `uv run prepare.py`.
5. **Initializes `results.tsv`** with just the header `commit\tval_bpb\tmemory_gb\tstatus\tdescription`.
6. **Confirms with the human** before entering the loop.

The first experiment is always the unmodified baseline so subsequent runs have something to compare against.

## Experiment loop

```
LOOP FOREVER:
  1. Inspect git state
  2. Modify train.py with one experimental idea
  3. git commit -m "<idea>"
  4. uv run train.py > run.log 2>&1
  5. grep '^val_bpb:|^peak_vram_mb:' run.log
  6. If empty → tail run.log, decide whether to fix or skip
  7. Append a row to results.tsv (keep | discard | crash)
  8. If improved → branch advances, continue
     If equal/worse → git reset to previous commit, continue
```

Key rules:

- **Redirect, do not tee.** `> run.log 2>&1` keeps the agent's context window clean. Streaming step output (~hundreds of lines) would otherwise flood it.
- **Read just the metric.** `grep '^val_bpb:|^peak_vram_mb:' run.log` is enough to make the keep/discard decision. Only fall back to `tail -n 50 run.log` on crashes.
- **One experiment per commit.** The branch is the experiment trail; resets are how rejections are expressed.
- **Never stop and ask.** The human may be asleep. The agent runs until interrupted, and "I'm out of ideas" is not a stopping condition — re-read the in-scope files, combine near-misses, try something more radical.

## What can and cannot be modified

| Can edit | Cannot edit |
|---|---|
| Anything in `train.py` | Anything in `prepare.py` |
| `GPTConfig`, `GPT`, `MLP`, `CausalSelfAttention`, `Block` | `evaluate_bpb` |
| `MuonAdamW`, polar-express coeffs, schedules | `MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS`, `VOCAB_SIZE` |
| Hyperparameter block (lines ~432-451 of `train.py`) | The pinned val shard |
| Adding/removing modules in `train.py` | `pyproject.toml` (no new dependencies) |

If the agent wants to add a new dependency, that is a hard stop — it must be explicitly approved by the human.

## Keep/discard decision

The metric is `val_bpb` (validation bits-per-byte from `evaluate_bpb`). Lower is better. The decision tree:

1. **Crash** (Python exception, OOM, NaN-loss fast-fail, run >10 min): log status `crash` with `val_bpb=0.000000`, `memory_gb=0.0`, `git reset --hard` to the previous commit, move on.
2. **Improved** (`val_bpb` strictly lower than the current branch tip): log status `keep`, advance the branch.
3. **Equal or worse**: log status `discard`, `git reset --hard HEAD~1`, move on.

The simplicity criterion modifies #2: if the improvement is tiny (≪0.005 bpb) and adds substantial code complexity, the agent should *discard* it anyway. If a change *removes* code while matching or improving the metric, that is a clear keep.

## results.tsv schema

Tab-separated, five columns. Tabs are required because descriptions often contain commas. The format is documented in detail in [reference/results-tsv.md](reference/results-tsv.md).

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase MATRIX_LR to 0.05
c3d4e5f	1.005000	44.0	discard	switch MLP to GeLU
d4e5f6g	0.000000	0.0	crash	double n_embd (OOM)
```

`results.tsv` is gitignored. It survives `git reset` because it's not part of any commit. If multiple agents share a working tree, they'll collide on this file — give each its own tag *and* its own working copy.

## Failure modes the harness already catches

- **NaN loss / loss > 100**: `train.py` exits with `print("FAIL"); exit(1)`. The `val_bpb:` line is absent → grep returns empty → agent treats it as a crash.
- **No data shards**: `make_dataloader` asserts `len(parquet_paths) > 0`. The crash is informative.
- **Time over-run**: the loop in `train.py` breaks once `total_training_time >= TIME_BUDGET`. The agent's external "kill if >10 min" rule is a backstop in case of compilation pathologies.

## When to stop the agent manually

- A bug in `train.py` that the agent keeps flailing against. Read its commits, `git reset` to a known-good experiment, and either prompt it with a hint or restart the loop.
- A regression in throughput (look at `tok/sec` or `mfu` in `run.log`) that suggests the agent has driven the model into a slow regime.
- Running out of disk for run logs or git objects.
- You wake up.

After stopping, you have a branch like `autoresearch/mar5` with one commit per kept experiment. That branch is the artifact of the run.

## Tuning the agent itself

`program.md` is the agent's "skill file". To change agent behavior, edit it. Things people commonly change:

- The phrasing of the keep/discard rule (e.g., raise the bar to 0.001 bpb improvement).
- The first-run instruction (some users want the agent to run a quick smoke test before the real baseline).
- How aggressive the agent is about radical architectural changes vs. hyperparameter sweeps.
- Adding a "research log" requirement that the agent writes a short journal entry per experiment.

Treat `program.md` as the lever for steering exploration. The Python files are *not* the lever.
