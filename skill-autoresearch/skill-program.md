# Skill Autoresearch: campaign-analysis

Autonomous improvement loop for the `campaign-analysis` Claude Code skill, inspired by Karpathy's autoresearch pattern.

## Concept

Instead of modifying `train.py` and measuring `val_bpb`, you modify the **SKILL.md file** and measure **eval pass rate** across a fixed set of test scenarios with known-correct outcomes.

The loop is identical: **modify → test → score → keep/discard → repeat forever.**

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `skill-mar25`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current commit.
3. **Read the in-scope files**: Read these files for full context:
   - This file (`skill-program.md`) — your instructions.
   - `eval-criteria.md` — the fixed binary eval criteria. Do not modify.
   - All files in `test-scenarios/` — the fixed test scenarios. Do not modify.
   - The skill file being optimized (see TARGET below).
   - The skill's reference files (see REFERENCES below).
4. **Establish baseline**: Run the eval script to get baseline scores.
5. **Initialize results.tsv**: Create `results.tsv` with the header row and baseline result.
6. **Confirm and go**: Confirm setup looks good.

## File Paths

**TARGET** (the single file you modify):
```
~/.claude/plugins/marketplaces/local-desktop-app-uploads/campaign-analyst/skills/campaign-analysis/SKILL.md
```

**REFERENCES** (you MAY also modify these if the change is clearly beneficial):
```
~/.claude/plugins/marketplaces/local-desktop-app-uploads/campaign-analyst/skills/campaign-analysis/references/analysis-framework.md
~/.claude/plugins/marketplaces/local-desktop-app-uploads/campaign-analyst/skills/campaign-analysis/references/diagnostics-and-formulas.md
```

**DO NOT MODIFY** (fixed evaluation infrastructure):
```
skill-autoresearch/eval-criteria.md
skill-autoresearch/eval.sh
skill-autoresearch/test-scenarios/*
~/.claude/plugins/marketplaces/local-desktop-app-uploads/campaign-analyst/skills/campaign-analysis/references/output-template.html
```

## Running an Eval

Each eval run tests the current skill against all 5 scenarios:

```bash
cd /Users/tpanos/TProjects/current-projects/autoresearch/skill-autoresearch
./eval.sh
```

This produces:
- Individual outputs in `eval-results/scenario-N-output.md`
- Grades in `eval-results/scenario-N-grade.md`
- Summary in `eval-results/summary.tsv`

The key metric is the **overall pass rate** (average score across all scenarios). Lower score = worse. Higher score = better. Target: 90%+.

You can also run a single scenario for faster iteration:
```bash
./eval.sh 1   # Run only scenario 1
```

## Logging Results

Log every experiment to `results.tsv` (tab-separated):

```
commit	score	status	description
```

1. git commit hash (short, 7 chars) — commit of the SKILL.md change
2. Overall eval score as percentage (e.g. 76.5%)
3. Status: `keep`, `discard`, or `error`
4. Short text description of what this experiment tried

Example:
```
commit	score	status	description
a1b2c3d	72.0%	keep	baseline
b2c3d4e	76.5%	keep	add explicit bid rate threshold to source classification
c3d4e5f	70.0%	discard	remove market position mapping instructions
d4e5f6g	0.0%	error	syntax broke the skill file
```

## The Experiment Loop

LOOP FOREVER:

1. **Read the current state**: Check `results.tsv` for the current best score and recent experiments. Read the current SKILL.md.

2. **Choose an experiment**: Pick ONE change to try. Good experiment ideas:
   - **Clarify ambiguous instructions** — if a criterion frequently fails, make the instruction more explicit
   - **Add decision trees** — replace prose with if/then logic for common diagnostic paths
   - **Strengthen guardrails** — add explicit "DO NOT" rules for common pitfalls
   - **Reorder instructions** — put the most critical steps earlier
   - **Add examples** — include worked examples for tricky calculations
   - **Simplify** — remove redundant or conflicting instructions
   - **Add checklists** — convert prose into step-by-step checklists
   - **Cross-reference** — add explicit pointers between layers (e.g., "use the result from Layer 2 Step 4 here")

3. **Make the change**: Edit SKILL.md (and optionally reference files). Keep changes focused — one concept per experiment.

4. **Commit the change**: `git add` the modified skill files and commit with a descriptive message.

5. **Run the eval**: `./eval.sh > eval-run.log 2>&1`

6. **Read the results**: Check `eval-results/summary.tsv` for the new score. Also read individual grade files to understand which criteria passed/failed.

7. **Log the result**: Append to `results.tsv`.

8. **Keep or discard**:
   - If score **improved** (higher %): Keep the commit. Advance the branch.
   - If score is **equal or worse**: Discard. `git reset --hard HEAD~1` to revert the SKILL.md change.
   - Exception: If score is equal but the change meaningfully simplifies the skill, keep it (simplicity wins).

9. **Analyze failures**: Before choosing the next experiment, read the grade files for FAIL results. Understand WHY each criterion failed — this directly informs what to try next.

10. **Repeat**: Go to step 1.

## Experiment Strategy

**Early experiments** (first 10-15 runs): Focus on the criteria that fail most frequently across scenarios. These are the low-hanging fruit — a single instruction fix can improve multiple scenario scores at once.

**Mid experiments** (runs 15-30): Target scenario-specific failures. Some criteria may only fail on edge cases (e.g., new accounts, missing data). Add explicit handling for these cases.

**Late experiments** (30+): Try more creative changes:
- Restructure the entire flow
- Add pre-flight checklists
- Add "common mistakes" warnings inline
- Try different instruction styles (imperative vs. descriptive)
- Remove instructions that never affect the score (simplification)

**Dead ends**: If 3+ consecutive experiments show no improvement, take a step back:
- Re-read all grade files to identify the most stubborn failures
- Read the scenario expected outcomes more carefully
- Try a fundamentally different approach rather than incremental tweaks
- Consider whether a reference file change (not just SKILL.md) would help

## Important Rules

- **One change at a time.** If you change two things and the score improves, you don't know which one helped. Isolate variables.
- **Read the failures.** The grade files tell you exactly what went wrong. Don't guess — read.
- **Simplicity wins.** All else being equal, shorter and clearer instructions are better. Don't add complexity unless it measurably improves the score.
- **Don't overfit.** If a change helps one scenario but hurts others, it's probably adding a special case rather than improving the general logic. Discard it.
- **NEVER STOP.** Once the loop has begun, do NOT pause to ask if you should continue. The user may be away. Run indefinitely until manually interrupted.
- **Commit the skill files, not the eval results.** Use `git add` on the specific skill files only. Keep eval results untracked.

## Timeout

Each full eval run (5 scenarios) takes approximately 10-15 minutes. If a run exceeds 30 minutes, kill it and treat as an error.

## Git Strategy

All skill file changes are committed to the experiment branch. The branch accumulates only successful experiments (failed ones are reset). At the end, the user can review the branch diff to see all improvements and cherry-pick or merge.
