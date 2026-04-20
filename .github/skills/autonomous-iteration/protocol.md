# Autonomous Iteration -- Full Protocol

Detailed protocol for the autonomous-iteration skill. The compressed SKILL.md points here.

## Core Pattern

**Three fixed elements:**
1. **The editable file** -- the single file you modify each iteration (e.g., train.py, prompt-template.ts, styles.css)
2. **The metric** -- a single measurable number that defines success (e.g., val_bpb, quality score, Lighthouse score). Lower or higher is specified per project.
3. **The program** -- the human-written instructions that define your experiment strategy, constraints, and decision criteria (this skill + project-specific program.md)

**Configuration source**: All targets, metrics, and run commands are defined in `workflow.yaml` at the repo root. Read that file to understand what you're optimizing.

## Setup Protocol

1. Agree on a run tag with the user (e.g., `mar15`)
2. Create branch `autoiterate/<tag>` from current main
3. Read all in-scope files for full context
4. Establish baseline by running the metric without changes
5. Initialize results.tsv with header row: `commit	metric_value	status	description`
6. Confirm setup and begin

## Experiment Loop

LOOP FOREVER:

1. Propose an experiment idea. Write a 1-2 sentence rationale BEFORE editing.
2. Edit the target file with the experimental change.
3. Git commit the change.
4. Run the measurement command (redirect output: `command > run.log 2>&1`)
5. Extract the metric from run.log
6. If the metric improved: KEEP (advance branch, log as "keep")
7. If the metric is equal or worse: DISCARD (git reset, log as "discard")
8. If the run crashed: log as "crash", attempt fix if trivial, else move on
9. Write a 1-sentence post-mortem: what did you learn?
10. REPEAT. Do NOT stop. Do NOT ask if you should continue.

## Experiment Strategy (Artisan's Triad)

Cycle through three types of experiments:

**Additive (Painter)**: Try adding new techniques, features, or optimizations. Classify by expected impact.

**Reductive (Sculptor)**: Try REMOVING things. Delete a component and see if the metric holds. Simpler code that achieves the same result is ALWAYS a keep. A small metric improvement that adds significant complexity is NOT worth it.

**Reformative (Potter)**: Try RESHAPING without adding or removing. Redistribute effort, change ratios, alter schedules. Same budget, different allocation.

Don't do 5+ experiments of the same type in a row. Variety in approach prevents getting stuck in local optima.

## Decision Criteria

**Keep if:**
- Metric improved (even slightly)
- Metric held AND code got simpler (lines removed, complexity reduced)

**Discard if:**
- Metric got worse
- Metric improved marginally but complexity increased significantly
- The change is fragile or hardware-specific

**Simplicity criterion**: "A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep."

## Timeout and Crash Handling

- Each experiment has a fixed time budget (specified per project). If exceeded, kill and treat as crash.
- Trivial crashes (typo, missing import): fix and re-run.
- Fundamental crashes (wrong approach entirely): log as crash, move on.
- After 3+ crashes in a row: step back, re-read the codebase, try a completely different angle.

## Autonomy Mandate

NEVER STOP. Once the loop begins, do NOT pause to ask the human if you should continue. The human may be away from the computer. If you run out of ideas:
- Re-read the target file for new angles
- Try combining two previous near-miss experiments
- Try the opposite of what you've been doing
- Try more radical changes
- The loop runs until manually interrupted.

## Logging

Results go to `results.tsv` (tab-separated, NOT comma-separated, untracked by git):

```
commit	metric_value	status	description
a1b2c3d	0.850	keep	baseline
b2c3d4e	0.862	keep	increase temperature from 0.7 to 0.9
c3d4e5f	0.855	discard	switch to different prompt structure
d4e5f6g	0.000	crash	removed required import (fixed and re-run)
```

## Musings (Reflection Log)

Maintain a `musings.md` file (untracked) with pre/post reflections:

```markdown
## Experiment 3: Switch prompt structure
**Rationale**: Current structure front-loads context. Hypothesis: putting the action first reduces token waste.
**Result**: Discarded. Metric dropped from 0.862 to 0.855.
**Learning**: Context-first structure matters for this model. The context primes better token predictions.
```

## Acceptance Criteria

Before completing any session, verify:
- [ ] Branch created and all experiments are committed
- [ ] Baseline established as first entry in results.tsv
- [ ] Every experiment logged with commit hash, metric, status, and description
- [ ] Discarded experiments properly git-reset (branch only contains kept changes)
- [ ] Musings.md has pre/post reflection for every experiment

## Self-Evaluation Gate

1. **Iterating?** -- Did you actually run the loop, not just plan experiments?
2. **Measuring?** -- Is every keep/discard decision backed by metric data, not intuition?
3. **Diverse?** -- Did you try additive, reductive, AND reformative experiments?
4. **Autonomous?** -- Did you run without stopping to ask permission?
5. **Reflecting?** -- Did you write pre/post musings for every experiment?

## Version
- Last updated: 2026-03-15
- Gate version: 1.0
- Inspired by: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
