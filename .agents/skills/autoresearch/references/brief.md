# Intake and passport

The user does not have to create a file before the first run. You scout the project, ask only for missing facts, show a passport, and save it after the user confirms.

## Where the brief comes from

1. A path the user named (`prepare --program PATH`).
2. `program.md` in the project root, if it is already there.
3. Otherwise, intake. Save the confirmed passport with `prepare --project <project> --brief-text -`. The runtime writes `AUTORESEARCH_HOME/briefs/<project-id>/brief.md`.

Never create `program.md` inside the measured project.

## Scout

Read only, before the first question:

- README and metric notes
- tests and evaluator scripts
- `git log` for a previous baseline
- large data files that must stay immutable

Anything you find becomes the proposed default. Do not ask a question whose answer is already in the brief or the code.

## Questions

One round, at most five questions. Each question includes a proposed default.

| Gap | Ask | Default when the code is silent |
| --- | --- | --- |
| Goal | What should improve, and what explicit outcome would count as success? | The user's own words; leave quality-goal status exploratory if no gate is specified |
| Metric and acceptance | Primary metric and direction, plus optional goal thresholds and confirmation repeats? Are there separate external gates? | Use the existing evaluator's primary metric; leave goal status exploratory if supported goal criteria are unspecified |
| Evaluation | Which script or reference file measures it? | Stop if neither exists. Do not switch to `fresh` |
| Constraints | What must not change? | Do not edit data files, schemas, or the evaluator |
| Limits | Any time cap, trial cap, or paid calls? | No paid requests. No `max_seconds` or `max_trials` |

## Passport

Show this and wait for one confirmation before `prepare`:

```text
## Project
<path>

## Goal
<what to improve>

## Metric and direction
<name>, max|min

## Goal criteria
<optional goal_criteria.thresholds and goal_criteria.confirmation_repeats, or "not specified, exploratory only">

## Evaluation
<evaluator path and args, or reference files>

## Constraints
<files and schemas that stay unchanged>

## Limits
<exact user wording, or "no paid requests; no time or trial cap">
```

## Spec

`preflight` and `prepare` read a JSON spec. A minimal one:

```json
{
  "project": "<project path>",
  "evaluator": ["{python}", "<evaluator.py>", "{candidate}", "{run_dir}"],
  "metrics": {"primary": "mean_error", "direction": "min",
              "guards": [{"name": "coverage", "direction": "max"}]},
  "repeats": 1,
  "timeout_seconds": 30,
  "budget": {"max_trials": 3},
  "budget_sources": {"max_trials": "at most 3 trials"},
  "baseline_validation": {"mode": "fresh",
                          "known_case": {"path": "<known_case.json>", "sha256": "<its sha256>"}}
}
```

The goal comes from the spec's optional `goal`, a `Goal: ...` line, or the passport's `## Goal` section. `{python}`, `{candidate}` and `{run_dir}` are filled in at run time.

`known_case` pins what a fresh baseline must reproduce. Write the record outside the project and the lab, then put its SHA-256 in the spec:

```json
{
  "schema": 1,
  "id": "my-baseline-v1",
  "reproducible": true,
  "evaluator_sha256": "<sha256 of the evaluator script>",
  "input_hashes": {},
  "expected_metrics": {"mean_error": 2.0, "coverage": 3.0}
}
```

Take `expected_metrics` from a real earlier run or from values you can derive from the evaluator, and tell the user which. If the baseline does not reproduce them, it is recorded as incomparable and trials cannot start.

## Paths

Write absolute paths in `spec.json` for the project, the evaluator script, `input_files`, and `known_case.path`. Relative paths are resolved from the brief file's directory. After `--brief-text`, that directory is `AUTORESEARCH_HOME/briefs/<project-id>/`.

## Exact quote

`max_seconds` and `max_trials` are enforced only when the spec's `budget_sources` value is an exact substring of the saved brief. Copy the user's limit into the Limits section word for word. A limit you invent is rejected.

Paid calls stay off unless the brief allows them. AutoResearch does not limit spending and never checks an API-key balance: the user sets a spending limit on each API key, and request and cost numbers are recorded as measurements only. See [the harness profile contract](harness-profile.md).

When criteria come from an artifact, record its path and hash as provenance. The lab goal_criteria block supports thresholds and confirmation repeats; assess any additional external gate separately. Keep framework trial acceptance separate from passing the goal.
