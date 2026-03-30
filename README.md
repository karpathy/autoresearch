# autoresearch

This checkout now includes a Codex-driven research runner for long-form web research jobs. The original `prepare.py` / `train.py` demo from the upstream repo is still here, but the added workflow in this repo is aimed at prompts like large equipment inventories, consular directories, industry datasets, and similar structured research tasks.

## What was added

- `program.md` - reusable research instructions that every run prepends to your spec
- `scripts/run_research.py` - `doctor`, one-shot `run`, and queued `queue` execution
- `specs/heavy_construction_industrial_equipment.md` - your heavy construction example as a reusable template
- `job_templates/*.json` - example queue jobs you can drop into `queue/pending/`

## Requirements

- Python 3.10+
- Codex CLI on `PATH`
- Working Codex auth
- Web search enabled at run time for research-heavy jobs

## First check

Run this once from the repo root:

```powershell
python scripts/run_research.py doctor --search
```

That verifies Python, `codex`, and a live web-enabled `codex exec` call.

## Run one job now

Example using the tracked heavy-equipment template:

```powershell
python scripts/run_research.py run `
  --spec specs/heavy_construction_industrial_equipment.md `
  --phase "1: Heavy Equipment" `
  --depth seed `
  --tag heavy-equipment-phase1-seed
```

For a deeper pass:

```powershell
python scripts/run_research.py run `
  --spec specs/heavy_construction_industrial_equipment.md `
  --phase "1: Heavy Equipment" `
  --depth exhaustive `
  --tag heavy-equipment-phase1-exhaustive
```

You can also inject any template value with repeated `--var KEY=VALUE`.

## Build a merged dataset

This runs the heavy-equipment template phase by phase, then exports merged CSV, JSON, manifest, and combined markdown files under `results\datasets\`.

The default path is intentionally fast and shallow: `--depth seed` plus the script default model (`gpt-5.4-mini`) is good for getting a structured inventory quickly, not for maximizing image coverage.

```powershell
python scripts/build_heavy_equipment_dataset.py --depth seed
```

If you want the builder to revisit only unresolved image rows after the first pass, enable the targeted recovery loop:

```powershell
python scripts/build_heavy_equipment_dataset.py `
  --depth seed `
  --image-followup `
  --image-followup-rounds 2 `
  --max-unconfirmed-images 150
```

Use `--max-unconfirmed-images` when you want the script to fail loudly instead of treating a partially covered dataset as success.

## Dashboard

This repo now includes a lightweight run-history dashboard that reads the existing `results\` artifacts directly. It does not require pi, a web framework, or extra build steps.

Print a terminal summary:

```powershell
python scripts/research_dashboard.py summary
```

Export the HTML dashboard:

```powershell
python scripts/research_dashboard.py export-html
```

Serve it locally from the repo root:

```powershell
python scripts/research_dashboard.py serve
```

By default that writes [results/dashboard/index.html](/C:/Users/uphol/Documents/Design/Coding/autoresearch/results/dashboard/index.html) and serves the repo at `http://127.0.0.1:8765/results/dashboard/index.html`.

## PR Planning

This repo also includes a commit-bundle planner for reviewable PRs. It looks at commits on the current branch that are not in the base ref, ignores metadata-heavy paths like `results/` and `.vscode/`, and groups selected commits into:

- `consolidated` - one PR with all selected commits plus required dependencies
- `stacked` - one PR per selected commit, subtracting commits already emitted in earlier PRs
- `individual` - one independent PR bundle per selected commit

List branch-only commits:

```powershell
python scripts/research_pr_plan.py list
```

Plan a consolidated PR from all branch-only commits:

```powershell
python scripts/research_pr_plan.py plan --all
```

Plan stacked PRs from specific commits:

```powershell
python scripts/research_pr_plan.py plan `
  --hash abc1234 `
  --hash def5678 `
  --mode stacked
```

## Snapshot Helper

To make the PR planner useful before you manually commit changes, there is also a worktree snapshot helper. It groups the current dirty tree into reviewable buckets such as `scripts`, `specs`, `job_templates`, and `docs`, while ignoring metadata-heavy paths like `results/` and `.history/` by default.

Preview the proposed groups:

```powershell
python scripts/research_snapshot.py plan
```

Preview the commits for a subset of groups:

```powershell
python scripts/research_snapshot.py commit `
  --group scripts `
  --group docs
```

Actually create the snapshot commits:

```powershell
python scripts/research_snapshot.py commit --yes
```

The helper refuses to run if you already have staged changes, so it does not trample a manual commit in progress.

## Queue mode

1. Create `queue\pending\` if it does not exist.
2. Drop JSON job files into it using the same shape as `job_templates\*.json`.
3. Start the worker:

```powershell
python scripts/run_research.py queue --watch
```

Processed jobs are moved to:

- `queue\completed\`
- `queue\failed\`

## Output layout

Each run creates a timestamped folder under `results\` containing:

- `prompt.txt` - full prompt sent to Codex
- `resolved_spec.md` - spec after `${var}` substitution
- `final_report.md` - last Codex message, intended as the deliverable
- `codex.log` - streamed CLI output for debugging
- `run.json` - metadata, paths, and exit code

`results/` and `queue/` are already ignored by git.

## How to customize

- Edit `program.md` to change the baseline research behavior.
- Add new templates under `specs/`.
- Create queue jobs that point to those specs and inject variables through the `vars` object.

## Notes

- The runner uses `codex --search exec` so the model can browse during research.
- The default model is `gpt-5.4`, but you can override it with `--model`.
- The repo's original ML training files were left untouched, so you can still use the upstream experiment loop if needed.
