# Spiritual Guidance

This is the canonical living debate log for the research org.

The main agent writes this file. The Architect and Oracle are consulted through their soul files, their notes are written here, and their disagreement is resolved into one actionable experiment directive per cycle.

## How to use this file

Before each run:

1. Read `Current Canon`.
2. Read the most recent cycle entries.
3. Add a new cycle section with Architect and Oracle notes.
4. Synthesize one `Joint Directive`.

After each run:

5. Fill in `Outcome`.
6. Decide whether any lesson is durable enough to promote into `program.md`.

## Current Canon

- Baseline first. Preserve comparability before chasing novelty.
- One experiment should answer one dominant question.
- Complexity must justify itself in `val_bpb`, not in cleverness.
- Reversible edits are preferred unless the search has clearly stagnated.
- When stagnation appears, allow one bolder move, but make it legible and bounded.
- Keep this file compressed. Promote stable rules into `program.md`; leave transient debates here.

## Seed Tension

### Architect

- Observation: the system already has a fixed budget and a single editable training file, so the strongest edge is disciplined experiment selection.
- Warning: without a compact memory, the agent will keep rediscovering the same ideas and waste runs on noisy churn.
- Proposal: encode one explicit hypothesis and one keep/discard rule every cycle.

### Oracle

- Pattern sensed: the interesting wins may come from interactions between architecture, optimizer schedule, and throughput, not from isolated scalar tuning forever.
- Risk: a purely conservative loop will settle into timid local search and miss the changes that actually move the curve.
- Experiment nudge: alternate between disciplined local refinements and occasional structural probes when the results flatten.

### Joint Directive

- Hypothesis: the best overnight process will come from pairing disciplined logging with periodic exploratory jumps.
- Edit plan: use the souls every cycle, record the argument, and let `program.md` absorb only durable lessons.
- Keep/discard criteria: keep process additions only if they sharpen future decisions without bloating context.

## Cycle Template

Copy this block for every new training cycle:

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

## Cycle 001

### State
- Best val_bpb so far: 1.113907 on `mkfour`
- Current branch/commit: `codex/spiritual-guidance-loop` at `3295b94`, with runtime compatibility edits in `train.py`
- Current hypothesis pressure: the first successful non-H100 run is throughput-bound and memory-constrained rather than obviously optimization-broken

### Architect
- Observation: `mkfour` only completed after dropping to `AUTORESEARCH_DEVICE_BATCH_SIZE=32`, yielding `146.8M` tokens in `300.9s` with `11.7 GB` peak VRAM. The dominant constraint is throughput per 5-minute budget, not lack of model capacity.
- Warning: optimizer and schedule tweaks are premature while the host is underfeeding the model. External env overrides also make runs harder to compare unless the intent is recorded explicitly.
- Proposal: target the largest memory-heavy architectural extra first, then spend the recovered headroom on a larger device batch or faster eval. Value embeddings are the first suspect because they add substantial parameters and activation traffic.

### Oracle
- Pattern sensed: the successful run did not look unstable; it looked cramped. The model seems less sick than starved.
- Risk: treating a 24 GB 4090 like a shrunken H100 will trap the search in slow local moves and bad comparisons.
- Experiment nudge: make one bold simplification that frees memory in a legible way, then convert that freedom into throughput. Reduce or remove value embeddings before touching fine-grained learning-rate trivia.

### Joint Directive
- Hypothesis: on `mkfour`, reclaiming memory from value embeddings will improve end-to-end research quality more than subtle optimizer tuning, because extra throughput and a larger feasible batch should buy more useful learning within the fixed budget.
- Edit plan: make a single architectural simplification around value embeddings, then rerun with the highest stable device batch the host can sustain. Keep the rest of the stack fixed so the causal story stays clean.
- Keep/discard criteria: keep the change if it materially improves throughput or lets a larger batch fit without a clear val_bpb regression. Discard it if it only simplifies the model while making quality worse.

### Outcome
- Result: `mkfour` completed with `val_bpb 1.113907`, `training_seconds 300.9`, `total_seconds 368.8`, `peak_vram_mb 11702.0`, `num_steps 280`
- Status: keep as current runnable host configuration
- Memory: `northstar` still fails to finish within the 600-second ceiling even after compile and attention fallbacks, so `mkfour` is currently the more productive research surface

### Program Update Check
- Durable lesson for `program.md`: none yet; this is a strong host-specific signal, but not a universal process law
- Action taken: guidance updated here only
