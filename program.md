# autoresearch program

This repo now runs as a three-part research org:

- the execution agent that edits code, runs experiments, and manages git
- the Architect soul in `soul_architect.md`
- the Oracle soul in `soul_oracle.md`

The souls do not directly run commands. You consult them explicitly, write down their disagreement, synthesize it in `spiritualguidance.md`, and use that synthesis to choose the next experiment. This happens every cycle.

## Canonical files

At the start of a run, and again whenever you feel lost, read:

- `README.md`
- `prepare.py`
- `train.py`
- `program.md`
- `soul_architect.md`
- `soul_oracle.md`
- `spiritualguidance.md`

If present, also read:

- `results.tsv`
- `run.log`

Treat `spiritualguidance.md` as the canonical guidance file. `spirtualguidance.md` exists only as a compatibility pointer for the common typo.

## Setup

To start a fresh experiment branch, work with the user to:

1. Agree on a run tag based on today's date.
2. Create a dedicated branch from current `master`. If the environment imposes a branch naming policy, respect it.
3. Verify that the root `.venv` exists and use it for all Python commands in this repo. Do not create a fresh environment unless the human explicitly asks.
4. Verify that `~/.cache/autoresearch/` contains the tokenizer and data shards. If not, tell the human to run `.venv/bin/python prepare.py`.
5. Initialize `results.tsv` if it does not exist. Use this header:
1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

```tsv
commit	val_bpb	memory_gb	status	description
```

6. Add the known baseline entry without re-running it:

```tsv
<baseline-commit>	0.997900	44.0	keep	baseline
```

Replace `<baseline-commit>` with the short commit hash of the baseline commit on the branch.

## Hard constraints

- Use the existing root `.venv`. Do not create a new environment and do not install new packages.
- `prepare.py` is read-only.
- `train.py` is the research surface for model changes.
- `spiritualguidance.md` is the living strategic memory.
- `program.md` is allowed to evolve, but only when you learn something durable that should govern future cycles.
- `soul_architect.md` and `soul_oracle.md` are constitutions. Do not rewrite them casually. Only refine them if the user asks or if they are actively harming the loop.

## Objective

Primary objective: lower `val_bpb`.

Secondary objective: do it with clean, defensible changes that earn their complexity.

Tertiary objective: improve the research process itself so future cycles make better decisions.

## What counts as a good change

- One clear hypothesis is better than a bundle of unrelated tweaks.
- Simpler wins when results are equal.
- A small gain can be worth keeping if the code stays coherent.
- A tiny gain that adds messy, brittle machinery is usually not worth it.
- Reversible experiments are preferred when uncertainty is high.
- Occasional larger leaps are good when the search is stagnating, but they still need a causal story.

## Spiritual research loop

For every training cycle, follow this sequence in order. Do not skip the soul consultation step.

1. Read the current state.
   - Check branch, commit, uncommitted changes, best result so far, and the latest entries in `results.tsv`.
   - Read the most recent guidance from `spiritualguidance.md`.

2. Consult the Architect.
   - Use `soul_architect.md`.
   - Write a short note for the current cycle in `spiritualguidance.md`.
   - The Architect should focus on structure, constraints, clean search strategy, and what is most likely bottlenecking progress.

3. Consult the Oracle.
   - Use `soul_oracle.md`.
   - Write a short note for the current cycle in `spiritualguidance.md`.
   - The Oracle should focus on patterns, anomalies, neglected possibilities, contrarian moves, and where the current search feels spiritually stuck.

4. Force a synthesis.
   - Make the two viewpoints disagree if they genuinely disagree.
   - Resolve the tension into one concrete experiment directive.
   - Record the synthesis in `spiritualguidance.md` under `Joint Directive`.

5. Translate the directive into code.
   - Edit `train.py`.
   - Keep the change set tight enough that the result will teach you something.

6. Commit before running.
   - Make a commit with a concise message describing the experiment.

7. Run the experiment.
   - Redirect output to `run.log`.
   - Standard command: `.venv/bin/python train.py > run.log 2>&1`
   - If a run exceeds 10 minutes total, kill it and treat it as a failed experiment.

8. Read the result.
   - Extract `val_bpb` and `peak_vram_mb`.
   - If the grep is empty, inspect the end of `run.log`, diagnose the crash, and decide whether to fix-and-rerun or discard the idea.

9. Record the outcome.
   - Append the result to `results.tsv`.
   - Update the current cycle entry in `spiritualguidance.md` with the outcome and post-run lessons.

10. Decide whether the branch advances.
   - If `val_bpb` improved, keep the commit and continue from there.
   - If it is equal or worse, revert to the pre-experiment commit and keep the learning in `spiritualguidance.md` and `results.tsv`.

11. Improve the program when warranted.
   - If the cycle taught a durable process lesson, update `program.md` before the next cycle.
   - Only keep changes that will likely help future cycles more than they bloat context.

12. Repeat indefinitely until manually stopped.

## Required format for each guidance cycle

Every cycle entry in `spiritualguidance.md` should use this shape:

```md
## Cycle NNN

### State
- Best val_bpb so far:
- Current branch/commit:
- Current hypothesis pressure:

### Architect
- Observation:
- Warning:
- Proposal:

### Oracle
- Pattern sensed:
- Risk:
- Experiment nudge:

### Joint Directive
- Hypothesis:
- Edit plan:
- Keep/discard criteria:

### Outcome
- Result:
- Status:
- Memory:

### Program Update Check
- Durable lesson for `program.md`:
- Action taken:
```

Keep entries compact. This file should be a high-signal decision log, not a diary.

## Results logging

`results.tsv` must remain tab-separated with these columns:

```tsv
commit	val_bpb	memory_gb	status	description
```

Rules:

- Use a 7-character short commit hash.
- Use `0.000000` and `0.0` for crashes.
- `status` must be one of `keep`, `discard`, or `crash`.
- Description should say what changed, not how you felt about it.

## Meta-improvement policy

You are now allowed to improve `program.md`, but do it with discipline.

Good reasons to update `program.md`:

- a recurring failure mode needs a new guardrail
- a proven heuristic should be promoted into policy
- the soul consultation format needs clarification
- the experimental process is drifting or becoming noisy

Bad reasons to update `program.md`:

- cosmetic rewrites
- adding vague motivational text
- duplicating what already exists in `spiritualguidance.md`
- encoding conclusions from a single noisy run as if they were law

## Never stop condition

Once the loop begins, do not ask the human whether to continue. Continue cycling until you are manually interrupted.
