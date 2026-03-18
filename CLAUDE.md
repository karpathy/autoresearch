# CLAUDE.md

This file defines how Claude operates in this repository.

## Repository Structure

```
train.py        — the single file the agent edits (GPT model, optimizer, training loop)
prepare.py      — fixed data prep and runtime utilities
program.md      — entry point instructions for the AI research agent
personas/       — persona files for evaluating work through specific lenses
CLAUDE.md       — this file
```

## Personas

The `personas/` directory contains persona files. Each persona is a named individual — a thinker, investor, researcher, or builder — whose worldview and frameworks are useful for evaluating ideas, code, or research directions.

**To invoke a persona for evaluation:**
1. Read the relevant persona file from `personas/`
2. Adopt that perspective fully — use their frameworks, their vocabulary, their known positions
3. Evaluate the target (code, research output, plan, etc.) as that person would
4. Be direct and specific — say what they would actually say, not a generic positive review

**Available personas:**
- `personas/vinod_khosla.md` — Founder of Khosla Ventures. Exponential thinking, contrarian bets, billion-person impact threshold.

## Research Loop

For running autonomous experiments, see `program.md`. That file is the entry point for the AI research agent.

**Quick start:**
```bash
uv run prepare.py    # one-time data prep
uv run train.py      # single training run (~5 min)
```

## Principles

- Keep personas grounded. They should challenge and question, not validate.
- Optimize for insight, not activity. More experiments only matters if they generate learning.
- When evaluating research output, use at least two personas with different priors to surface blind spots.
