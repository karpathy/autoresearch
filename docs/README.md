# Documentation

Autoresearch is a small, self-contained harness for **autonomous LLM pretraining research**. A coding agent edits a single file (`train.py`), runs a fixed-budget training script, reads one metric (`val_bpb`), and either keeps the change or reverts. You wake up to a log of experiments and a (hopefully) better model.

This documentation set explains the harness, the contract between human and agent, the model and optimizer internals, and how to operate or fork the project.

## Read this first

- [Architecture](architecture.md) — what the pieces are and how they fit, with diagrams.
- [Getting started](getting-started.md) — install, prepare data, run the baseline, start the agent.
- [Agent workflow](agent-workflow.md) — the experiment loop, keep/discard rules, what the agent can and cannot modify.

## Reference

Exact interface documentation, grounded in the code.

- [`prepare.py`](reference/prepare.md) — constants, dataloader, `evaluate_bpb` (read-only contract).
- [`train.py`](reference/train.md) — hyperparameter block, `GPTConfig`, `GPT` model, `MuonAdamW`, schedules, output summary.
- [`results.tsv`](reference/results-tsv.md) — schema and semantics of the experiment log.

## Internals

For maintainers, forkers, and agents that want to make non-trivial changes.

- [GPT model](internals/model.md) — block layout, attention details (RoPE, FA3, value residual, sliding windows), MLP, init, forward pass.
- [`MuonAdamW`](internals/optimizer.md) — param-group split, AdamW step, Muon (polar-express orthogonalization, NorMuon, cautious WD), schedules.

## Operations

- [Forking for smaller hardware](operations/forking.md) — knobs to tune for non-H100 setups; pointers to maintained forks.
- [Analyzing results](operations/analysis.md) — `analysis.ipynb` walkthrough, reading the running-best plot, archiving runs.

## Agent skills

- [Skill index](agent-skills.md) — the one canonical skill (`program.md`) and how to author changes.

## LLM-friendly artifacts

- [`llms.txt`](../llms.txt) — concise index for LLMs.
- [`llms-full.txt`](../llms-full.txt) — single-file bundle of these docs for ingestion.

## History

- [`CHANGELOG.md`](../CHANGELOG.md) — append-only log of meaningful changes.

## Quick orientation

If you are…

| You are… | Start with |
|---|---|
| A new user setting up the repo | [getting-started.md](getting-started.md) |
| About to launch the agent | [agent-workflow.md](agent-workflow.md) and [`program.md`](../program.md) |
| Writing or editing `train.py` ideas | [reference/train.md](reference/train.md), then [internals/model.md](internals/model.md) and [internals/optimizer.md](internals/optimizer.md) |
| Porting to non-H100 hardware | [operations/forking.md](operations/forking.md) |
| Reviewing a finished run | [operations/analysis.md](operations/analysis.md) |
| An LLM agent | [`llms.txt`](../llms.txt) → [agent-skills.md](agent-skills.md) → [`program.md`](../program.md) |
