# Agent skills

Autoresearch is structured as a single agent skill. The skill file is [`program.md`](../program.md) at the repo root. It tells the coding agent how to set up a run and how to operate the experiment loop.

## Skill index

| Skill | File | Trigger | What it does |
|---|---|---|---|
| **autoresearch experiment loop** | [`program.md`](../program.md) | Human prompts the agent in this repo with something like *"look at program.md and kick off a new experiment"*. | (1) Sets up a tagged branch, verifies cache, initializes `results.tsv`. (2) Runs LOOP FOREVER: edit `train.py` → commit → run → keep/discard → repeat. |

There is exactly one skill today. The repo's design philosophy is that customization happens by editing `program.md`, not by adding more skills. If you need to introduce additional behavior, prefer extending `program.md` first.

## When to use this skill

Use it when the user wants to start (or continue) an autonomous experimentation run on this repo. Concretely:

- They want to leave the agent running while they do something else, expecting a list of kept experiments at the end.
- They've already done the one-time setup (`uv run prepare.py`) — or they're willing to do it as part of the skill's setup phase.
- They have a single GPU available and `uv run train.py` works.

## When *not* to use this skill

- The user wants a one-off experiment, not an autonomous loop. Just edit `train.py` and run it directly.
- Hardware setup (GPU, kernels, dependencies) hasn't been verified. Run the baseline manually first; see [getting-started.md](getting-started.md).
- The user is working on the harness itself (`prepare.py`, `program.md`, the docs). The skill assumes those are stable.

## Canonical docs the skill depends on

`program.md` references these — keep them in sync if you change behavior:

- [`README.md`](../README.md) — the agent reads it for context during setup.
- [`prepare.py`](../prepare.py) — read-only contract.
- [`train.py`](../train.py) — the file the agent edits.
- [`docs/agent-workflow.md`](agent-workflow.md) — expanded explanation of the skill's loop.
- [`docs/reference/results-tsv.md`](reference/results-tsv.md) — the file the agent appends to.

## Authoring guidance for changing `program.md`

Properties of the existing skill that matter:

- **Concise.** It's loaded into every agent context. Don't bloat it.
- **Procedural.** Setup is numbered steps. The loop is a numbered procedure with explicit `LOOP FOREVER`.
- **Constraints up front.** "What you CAN do / CANNOT do" lists are explicit, not implicit.
- **Stops only on interruption.** `program.md` says "NEVER STOP" deliberately — this is the most-violated rule when agents try to be polite. Keep it.
- **No CLI invention.** The agent runs `uv run train.py > run.log 2>&1` and reads via `grep`. Don't add custom flags or wrappers; everything else should still work.

When you change the skill:

- Update [`docs/agent-workflow.md`](agent-workflow.md) so the human-facing explanation matches.
- Update [`CHANGELOG.md`](../CHANGELOG.md) under the "Agent" section.

## Multi-agent setups

If you want several agents running on the same machine in parallel, give each its own:

- Tag (`autoresearch/mar5-gpu0`, `autoresearch/mar5-gpu1`, …).
- Working tree (use `git worktree add` so they don't fight over `results.tsv` and `train.py`).
- GPU (`CUDA_VISIBLE_DEVICES=N`).

The skill itself is single-threaded; coordination is the human's responsibility.
