# Harness Optimization

Optimize agent instructions, conventions, and configuration for maximum effectiveness across all benchmark tasks. The target files form the agent's entire instruction surface -- AGENTS.md, copilot-instructions.md, agent definitions, skill definitions, and harness.yaml. Every change is measured against the benchmark suite using a Pareto ratchet that protects higher-tier metrics from regression.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr19`).
2. **Create the branch**: `git checkout -b autoiterate/<tag>` from current main.
3. **Read the in-scope files**: Read all files in this workflow directory for full context. Also read each target file listed in workflow.yaml.
4. **Verify prerequisites**: Confirm `uv sync` succeeds and `uv run python ../../tests/benchmark.py --quick` runs without error.
5. **Initialize results**: Create `results/results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment runs the full benchmark suite: `uv run python ../../tests/benchmark.py --all > run.log 2>&1`. Timeout is 1800 seconds.

**What you CAN do:**
- Modify `../../AGENTS.md` -- restructure, reword, add or remove content
- Modify `../../.github/copilot-instructions.md` -- adjust platform integration
- Modify `../../agents/research-runner.md` -- refine agent behavioral contract
- Modify `../../skills/autonomous-iteration.md` -- improve skill protocol
- Modify `../../harness.yaml` -- adjust configuration values and conventions

**What you CANNOT do:**
- Modify tests/benchmark.py, tests/evaluator.py, or tests/tasks/
- Modify workflow.yaml or program.md
- Install new packages or add dependencies
- Add platform-specific instructions that only work on one agent platform

**The goal is simple: get the highest harness_score.**

**Simplicity criterion**: All else being equal, simpler is better. Fewer words that convey the same information is always preferred. Instruction bloat degrades agent performance because it consumes context window budget and dilutes signal.

## Anti-Overfitting

Changes must improve general agent effectiveness, not game specific benchmark tasks. Signs of overfitting to watch for:

- Hardcoding answers or patterns that match specific test inputs
- Adding instructions that only help with one benchmark task type
- Optimizing phrasing for the evaluator rather than for agent comprehension
- Increasing instruction volume without proportional clarity gain

If a change improves one metric but cannot be justified on general-effectiveness grounds, discard it.

## Output Format

The benchmark suite writes structured output to stdout. The composite score is extracted by the evaluator:

```bash
uv run python ../../tests/evaluator.py --extract-composite
```

## Logging Results

When an experiment is done, log it to `results/results.tsv` (tab-separated).

The TSV has a header row and columns:

```
commit	harness_score	task_success_rate	quality_gate_pass_rate	rework_rate	avg_token_consumption	time_per_turn	status	description
```

## The Experiment Loop

LOOP FOREVER:

1. Look at the git state and recent results
2. Analyze which tasks fail and cluster failures by root cause
3. Generate a hypothesis about what instruction change would address the root cause
4. Edit one or more target files to test the hypothesis
5. git commit
6. Run the benchmark: `uv run python ../../tests/benchmark.py --all > run.log 2>&1`
7. Extract the composite score: `uv run python ../../tests/evaluator.py --extract-composite`
8. If empty, the run crashed. Read `tail -n 50 run.log` for the error.
9. Record all metrics in the TSV (do NOT commit results.tsv)
10. If harness_score improved and no T1 metrics regressed below floor: keep the commit
11. If harness_score is equal or worse, or T1 floors are violated: git reset back

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human.

## Experiment Strategy (Artisan's Triad)

Cycle through three types. Don't do 5+ of the same type in a row.

**Additive (Painter)**: Add new instructions, conventions, guardrails, or navigation hints that help agents find information faster or avoid common mistakes. Examples: add a "common pitfalls" section, add cross-references between related files, add explicit decision criteria for ambiguous situations.

**Reductive (Sculptor)**: Remove redundant content, simplify language, eliminate dead conventions, collapse sections that repeat the same idea. Examples: merge two sections that say the same thing, remove a convention that no test exercises, shorten a verbose explanation to its core sentence.

**Reformative (Potter)**: Restructure instructions for faster agent comprehension without changing their content. Examples: reorder sections so the most-referenced content appears first, convert prose to tables for scanability, rename ambiguous terms for precision.

## Between-Session Reflection

Before starting a new experiment, review the musings log from prior sessions:

1. Which tasks consistently fail? What do they have in common?
2. Which instruction changes had the largest positive impact?
3. Are there diminishing returns on a particular type of change?
4. What hypotheses remain untested?

Use these observations to prioritize the next experiment rather than repeating approaches that have plateaued.

## Musings (Reflection Log)

Maintain a `results/musings.md` file (untracked) with pre/post reflections:

```markdown
## Experiment N: <short title>
**Hypothesis**: What instruction change might improve which metric and why.
**Rationale**: Why this might work, grounded in failure analysis.
**Result**: Keep/Discard/Crash. harness_score = X (delta: +/-X)
**Per-metric**: task_success=X, gate_pass=X, rework=X, tokens=X, time=X
**Learning**: What did this teach you about instruction design?
```
