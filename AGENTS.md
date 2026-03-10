# Agent Runbook: Autoresearch Execution

Use `workflows/run_experiment.py` for all autoresearch execution. Do not use `workflow/dag.json` or task-graph-planner for running experiments.

## Core Rules

1. Keep all run artifacts under `workflows/runs/`.
2. Modify only `train.py` during experiments.
3. Never modify `prepare.py`.
4. Always run training via `uv run train.py` through the script.

## Natural-language to Command Mapping

- User says: "Start running the experiment, run 5 loops"
  - Run: `python workflows/run_experiment.py start --loops 5`

- User says: "Run another 5 iterations"
  - Run: `python workflows/run_experiment.py resume --loops 5`

- User says: "Resume run <run_id> and run 5 loops"
  - Run: `python workflows/run_experiment.py resume --run-id <run_id> --loops 5`

- User says: "Only run setup and baseline"
  - Run: `python workflows/run_experiment.py start --only setup,baseline`

- User says: "Only run training and decision parts in loops for 3 iterations"
  - Run: `python workflows/run_experiment.py resume --loops 3 --only loop --loop-only train,record,decide`

- User says: "Show run status"
  - Run: `python workflows/run_experiment.py status`

## Stage Controls

- Top-level stages: `setup`, `baseline`, `loop`
- Loop stages: `propose`, `apply`, `commit`, `train`, `triage`, `record`, `decide`

Supported control flags:

- `--only <comma-list>`: run only selected stages
- `--from-stage <setup|baseline|loop>` + `--to-stage <...>`: run a top-level stage range
- `--loop-only <comma-list>`: limit loop internals to selected stages
- `--loops N`: run `N` loop iterations

## Resume Behavior

- The script checkpoints state at `workflows/runs/<run_id>/state.json`.
- If a loop iteration is partially complete, `resume` continues that iteration from the next pending stage.
- "Run another N iterations" means execute N more loop iterations from current state.

## Logging and Observability

- Human-readable execution log: `workflows/runs/<run_id>/runner.log`
- Structured event log: `workflows/runs/<run_id>/history.jsonl`
- Checkpoint state: `workflows/runs/<run_id>/state.json`
- Per-iteration details (including opencode raw outputs): `workflows/runs/<run_id>/iterations/<NNNN>/`

## Run ID Policy

- Default run id: `<branch-slug>-rNNN`
- Example: branch `autoresearch/mar10` -> `autoresearch-mar10-r001`
- On `resume` without `--run-id`, script picks latest run for current branch.

## Notes

- Use `--no-stochastic` only when opencode stochastic execution is unavailable.
- Setup auto-runs `uv run prepare.py` if cache/tokenizer are missing (disable via `--no-auto-prepare`).
- `results.tsv` is maintained in repo root and should remain untracked.
