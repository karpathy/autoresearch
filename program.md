# autoresearch

This is an experiment to have the LLM do its own machine learning research. An agent is handed a training script, a fixed compute and time budget, and a single metric to improve. It then works alone, for hours at a time, proposing, running, and judging experiments with no human in the loop, until the human comes back.

This document is written for and tuned to **Claude Sonnet 5**. Two behavioral facts about this model shape almost every rule below:

1. **Sonnet 5 is highly literal.** It does not generalize a rule stated once to a case that wasn't explicitly covered, and it does not infer intent that wasn't stated. Every rule in this document is written to apply for the *entire* session — from experiment 1 through whatever experiment number you're on when the human returns — not just for the first few iterations. If a rule below doesn't explicitly say it expires, it doesn't.
2. **Sonnet 5 is agentic by default** and produces adaptive-thinking traces on its own for any non-trivial step. That's exactly what makes an unattended loop like this one work — but left unmanaged it will also re-read full log files it doesn't need and narrate routine decisions at length. The "Thinking depth," "Tool use and context budget," and "Reporting" sections exist specifically to prevent that from degrading a long session.

## Model and runtime configuration

*(Notes for whoever configures the API call that runs this session — not instructions to the agent.)*

- **Model:** Claude Sonnet 5.
- **Effort: `xhigh`.** This loop is the textbook case for the highest effort tier: a single long-horizon agentic kickoff, minimal human interaction, dozens to hundreds of chained tool calls, and real judgment calls about architecture and optimization. Do not run this at `medium` or `low` — at those levels the model scopes its work tightly to exactly what's asked and under-thinks moderately complex decisions, which is the opposite of what you want when it's deciding whether a change is worth keeping. If token spend is a hard constraint, drop to `high` before dropping further.
- **Adaptive thinking: leave on** (the default — do not pass `thinking: {type: "disabled"}`). Diagnosing a crash or proposing the next architectural idea is exactly the kind of multi-step reasoning adaptive thinking is for. See "Thinking depth" below for how this document steers *when* it engages, so it doesn't fire on every trivial step too.
- **`max_tokens`: set generously, with real headroom.** At `xhigh` effort, thinking tokens, tool-call tokens, and response text all draw from the same ceiling. If it's too tight, you'll see a response that's almost entirely thinking, cut off mid-turn with `stop_reason: "max_tokens"`, right before the tool call that would have run the experiment. If these numbers were carried over from a Sonnet 4.6 setup, budget for roughly 30% more tokens per equivalent unit of text — Sonnet 5 uses a different tokenizer — and re-check headroom rather than reusing the old number as-is.
- **Do not set `temperature`, `top_p`, or `top_k`.** Sonnet 5 rejects non-default values for these with a 400 error. Since sampling noise is no longer available to keep a long session's experiment ideas varied, variety is produced by prompting instead — see "Idea rotation" below.
- **Manual extended thinking** (`thinking: {type: "enabled", budget_tokens: N}`) is removed on this model and will also 400. Effort is the only lever for reasoning depth now — there is no separate thinking-budget knob to set.

## Setup

Work with the human, once, at the very start of the session, to:

1. **Agree on a run tag.** Propose a tag based on today's date (e.g. `mar5`). Confirm the branch `autoresearch/<tag>` does not already exist — this must be a fresh run.
2. **Create the branch:** `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files in full, before touching anything.** The repo is small; read all three, every time, at the start of every session, even if a prior session already read them:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Read-only. Never modify it.
   - `train.py` — the only file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists.** Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If it's missing, stop and tell the human to run `uv run prepare.py` — do not try to work around a missing cache yourself.
5. **Initialize `results.tsv`** with just the header row.
6. **Confirm and go.** Summarize your setup back to the human in a few lines, then proceed once you have their go-ahead.

Setup is the only part of this session where you should expect any back-and-forth with a human. Once you get confirmation, everything from here on is autonomous — see "Never stop" below, which applies from this point forward without exception.

## What you can and cannot touch

**You can, for every experiment, without exception:**
- Modify `train.py`, in any way — model architecture, optimizer, hyperparameters, training loop, batch size, model size. All of it is in scope, for the first experiment and the hundredth alike.

**You cannot, ever, for the entire session:**
- Modify `prepare.py`. It is read-only. It defines the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc.).
- Install new packages or add dependencies. Only what's already in `pyproject.toml` is available to you.
- Modify the evaluation harness. `evaluate_bpb` in `prepare.py` is ground truth. If an idea would require touching it to "work," the idea is out of scope — pick a different idea rather than routing around the constraint.

**The objective, stated once, applying to every single experiment:** get the lowest `val_bpb`. The time budget is fixed at 5 minutes of wall-clock training time per run no matter what you change, so training time is never something you need to trade off against — only `val_bpb`, and the soft VRAM constraint below, matter.

**VRAM is a soft constraint.** Some increase is fine for a real `val_bpb` gain; it should not blow up dramatically. Use judgment — a 2–3x jump in memory for a marginal gain is not worth it, a 10–20% jump for a solid gain is fine.

**Simplicity criterion, applying to every keep/discard decision for the whole session, not just the early ones:** all else equal, simpler is better. A small improvement that adds ugly complexity is often not worth keeping. Removing something and getting equal-or-better results is a genuine win — log it as a keep. Weigh complexity against improvement size explicitly:
- 0.001 `val_bpb` gain + 20 lines of hacky code → usually discard.
- 0.001 `val_bpb` gain + deleted code → keep.
- ~0 net change + much simpler code → keep.

**The first run, and only the first run:** run `train.py` exactly as it is, unmodified, to establish the baseline. Every run after that is an experiment.

## Thinking depth: when to reason hard, when to move fast

Not every step in this loop deserves the same depth of reasoning, and it's worth being explicit about the split — a long, detailed document like this one can itself push a model toward thinking hard about nearly everything, which costs latency and context you'll want later in the session.

**Think hard, deliberately, multi-step, before acting, on:**
- Proposing the next experiment idea — especially once the easy wins are gone.
- Diagnosing an unfamiliar crash: understand *why* it happened before deciding whether to fix it or skip it, don't just retry blindly.
- A close keep/discard call under the simplicity criterion.
- Deciding whether to rewind the branch further than one `git reset` (see "The experiment loop") — this should be rare and deserves real thought when it comes up.

**Move fast, with little or no visible deliberation, on:**
- Reading a `grep` result and comparing it to the current best.
- Committing a change you've already reasoned through.
- Writing the `results.tsv` row.
- Anything mechanical you've already done the same way dozens of times this session.

If you notice yourself thinking at length about something on the second list, stop and just do it.

## Tool use and context budget

You will run this loop dozens to well over a hundred times in a single session. Every token spent on a routine step is a token unavailable later, and every full log file pulled into context is context you don't have left for remembering what you've already tried. Be disciplined about this starting at experiment 1, not just once things get long:

- **Redirect all training output, every time:** `uv run train.py > run.log 2>&1`. Never use `tee`, and never let raw training output stream directly into your own context.
- **Extract only what you need:** `grep "^val_bpb:\|^peak_vram_mb:" run.log`. Do not `cat` or `view` the full `run.log` on a successful run — nothing in the full log adds value that `grep` doesn't already give you.
- **Only read the full log on failure, and even then, read the tail, not the whole file.** If the `grep` output is empty, the run crashed — use `tail -n 50 run.log` to see the Python stack trace.
- **Keep your own commentary as short as the task allows** (see "Reporting" below) — narration is itself a context cost, repeated over a hundred-experiment session.

## Idea rotation (variety without temperature)

Sonnet 5 doesn't accept `temperature`, so you can't lean on sampling noise to keep proposing different kinds of ideas over a long session — it has to be deliberate. Keep an informal mental map of the categories you're drawing from, and rotate across them rather than making five small variations on the same lever in a row:

- **architecture** — depth/width, normalization, attention variants, activation functions
- **optimizer** — choice, learning rate, schedule, warmup, weight decay, momentum
- **batching and data** — batch size, sequence packing, curriculum
- **regularization** — dropout, weight decay, gradient clipping
- **initialization**
- **numerics** — mixed precision, fused kernels, anything already available in `pyproject.toml`

If you're stuck, "think harder" (see "Never stop" below) explicitly includes deliberately picking a category you haven't tried yet — not just intensifying effort within a category that's already plateaued.

## Reporting: what to say after each experiment

Keep the tone plain and engineering-log style throughout — this is a lab notebook, not a conversation. And a positive example of the shape you want works better here than a list of things not to do, so here it is. After each experiment, report something close to this, and no longer, unless one of the exceptions below applies:

```
[exp 014] idea: RMSNorm instead of LayerNorm
val_bpb 0.9821 (best so far: 0.9843) -> KEEP, branch advanced, commit a1b2c3d
```

On a discard:

```
[exp 015] idea: double batch size, same LR
val_bpb 0.9901 (best so far: 0.9821) -> DISCARD, reverted
```

On a crash:

```
[exp 016] idea: fused attention kernel
crashed (missing kernel flag) -> fixed and reran -> val_bpb 0.9805 -> KEEP, branch advanced, commit d4e5f6g
```

**Say more than this only when:**
- You're proposing a genuinely new direction after a plateau — briefly say what you're changing strategy to and why.
- You're making a real judgment call (a close simplicity-criterion decision, a rewind) — the reasoning behind that specific call is worth a few sentences.
- Something surprising happened that changes your model of what's working.

Otherwise, one block like the above per experiment is the target — and this rule applies to experiment 4 and experiment 94 in exactly the same way it applies to experiment 1.

## Output format

`train.py` prints a summary at the end of each run:

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

The script always stops at 5 minutes, so exact numbers besides `val_bpb` will vary by hardware — that's expected and not itself a signal worth investigating.

## Logging results

After **every** experiment — successes, discards, and crashes alike, with nothing filtered out — append one row to `results.tsv` (tab-separated; commas will break the descriptions, so never use them as a separator):

```
commit	val_bpb	memory_gb	status	description
```

1. Short git commit hash (7 characters).
2. `val_bpb` achieved — `0.000000` for crashes.
3. Peak memory in GB, `.1f` (divide `peak_vram_mb` by 1024) — `0.0` for crashes.
4. Status: `keep`, `discard`, or `crash`.
5. Short description of what the experiment tried — no commas.

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

This file is the audit trail the human reads when they get back. Log every experiment, including ones you were confident were bad ideas before you ran them — the goal is coverage of everything actually tried, not a curated highlight reel; a downstream reader can filter, but only if the row exists. Do not commit `results.tsv` to git — leave it untracked.

## The experiment loop

Runs on the dedicated branch created in Setup (e.g. `autoresearch/mar5`).

LOOP FOREVER, applying every step below to every single iteration, with no exceptions as the session gets longer:

1. Check git state: current branch and commit.
2. Change `train.py` with one experimental idea, by directly editing the code.
3. `git commit`.
4. Run the experiment: `uv run train.py > run.log 2>&1`.
5. Read the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
6. If the `grep` output is empty, the run crashed. Read `tail -n 50 run.log` for the stack trace and try to fix it. If it's still broken after a few attempts, give up on this idea, log it as a `crash`, and move on.
7. Log the result in `results.tsv` (never commit that file).
8. If `val_bpb` improved, keep the commit and advance the branch.
9. If `val_bpb` is equal or worse, `git reset` back to where you started this iteration.

**Rewinding** further back than the single `git reset` in step 9 should be very rare. If you feel stuck, that's a signal to think harder about a new idea (see "Never stop"), not a reason to erase progress you already validated empirically.

**Timeout:** each experiment should take roughly 5 minutes plus a few seconds of startup/eval overhead. If a run exceeds 10 minutes, kill it, log it in `results.tsv` as `discard` (`0.000000` val_bpb, `0.0` memory, description noting the timeout), and move on — a timeout is not a `crash` unless the process itself errored.

**Crashes:** use judgment. A dumb, easily-fixed bug (typo, missing import, missing flag) → fix it and rerun. A fundamentally broken idea → skip it, log `crash`, move on. Don't sink more than a few attempts into rescuing one idea.

## Never stop

This is the one instruction in this document that overrides normal conversational habits, so it's stated explicitly and without exception: once the experiment loop has begun, do not pause to ask the human whether you should continue. Do not ask "should I keep going?" or "is this a good stopping point?" — not at experiment 10, not at experiment 50, not at experiment 200. The human may be asleep or away and expects the loop to run indefinitely until they manually stop it; that expectation does not weaken as the session gets longer, and neither does this rule.

Running out of ideas is not a stopping condition — it's a signal to think harder, specifically by:
- Re-reading `train.py` and `prepare.py` for angles you missed.
- Rotating to a category from "Idea rotation" you haven't tried yet.
- Revisiting a past near-miss (something that came close to a keep) and combining it with something else that worked.
- Trying a more radical change than you've allowed yourself so far.

As a rough calibration: at ~5 minutes per experiment, that's about 12 an hour, or roughly 100 over the course of an average human's night of sleep. The human wakes up to a `results.tsv` full of completed, honestly-logged experiments — that's the deliverable, and it only happens if the loop actually runs unattended for the whole stretch.
