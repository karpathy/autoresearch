# {workflow_name}

{One paragraph describing what this research workflow investigates and why.}

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr11`).
2. **Create the branch**: `git checkout -b autoiterate/<tag>` from current main.
3. **Read the in-scope files**: Read all files in this workflow directory for full context.
4. **Verify prerequisites**: {Describe any data, dependencies, or setup needed.}
5. **Initialize results**: Create `results/results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment runs for a **fixed time/resource budget of {budget}**. You launch it as: `{run_command}`.

**What you CAN do:**
- Modify `{target_file}` -- {describe what's fair game}.

**What you CANNOT do:**
- Modify fixed files listed in workflow.yaml.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**The goal is simple: get the {best_direction} {metric_name}.**

**Simplicity criterion**: All else being equal, simpler is better.

## Output format

{Describe what the run output looks like and how to extract the metric.}

## Logging results

When an experiment is done, log it to `results/results.tsv` (tab-separated).

The TSV has a header row and columns:

```
commit	{metric_name}	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state
2. Tune `{target_file}` with an experimental idea
3. git commit
4. Run the experiment: `{run_command} > run.log 2>&1`
5. Read out the results: `{extract_command}`
6. If empty, the run crashed. Read `tail -n 50 run.log` for the error.
7. Record results in the TSV (do NOT commit results.tsv)
8. If {metric_name} improved ({direction}): keep the commit
9. If {metric_name} is equal or worse: git reset back

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human.

## Experiment Strategy (Artisan's Triad)

Cycle through three types. Don't do 5+ of the same type in a row.

**Additive (Painter)**: {domain-specific additive experiments}
**Reductive (Sculptor)**: {domain-specific reductive experiments}
**Reformative (Potter)**: {domain-specific reformative experiments}

## Musings (Reflection Log)

Maintain a `results/musings.md` file (untracked) with pre/post reflections:

```markdown
## Experiment N: <short title>
**Rationale**: Why this might work.
**Result**: Keep/Discard/Crash. {metric_name} = X (delta: +/-X)
**Learning**: What did this teach you?
```
