# Copilot Instructions

Reference harness for agent-assisted development. See [AGENTS.md](../AGENTS.md) for full context.

## Session Setup (mandatory)
- On every new session start, immediately tell the user to run `/remote` to enable remote control from GitHub web and mobile. This is a required first-response action before any other work.

## Commands
- `uv sync` | install dependencies
- `python scaffold.py onboard` | guided harness.yaml creation
- `python scaffold.py generate harness.yaml` | generate workspace
- `python scaffold.py workflow <name>` | create workflow
- `python scaffold.py report <workflow> [--open]` | generate HTML research report
- `python scaffold.py harness-bg [--resume|--refresh]` | background harness optimizer
- `uv run python .github/tests/benchmark.py --all` | run benchmarks (agent mode)
- `uv run python .github/tests/benchmark.py --all --dry-run` | validate task structure only

## Design intelligence (ui-ux-pro-max)
- `python .github/prompts/ui-ux-pro-max/scripts/search.py "<query>" --design-system` | full design system
- `python .github/prompts/ui-ux-pro-max/scripts/search.py "<query>" --domain <d>` | domain search
- Domains: style, color, typography, chart, ux, landing, product, prompt
- `--design-system --persist -p "Name"` | saves MASTER.md for session reuse

## Architecture
- **Two-layer config**: `harness.yaml` declares the workspace (agent behavior, skills, hooks, guardrails, gates, metrics); each `workflows/<name>/workflow.yaml` declares one experiment (targets, fixed files, metric, run command, timeout).
- **Source of truth**: edit `harness.yaml` and re-run `python scaffold.py generate harness.yaml` to propagate to platform files (`.github/copilot-instructions.md`, `CLAUDE.md`, AGENTS.md tree). Don't hand-edit generated files when the change should persist across platforms.
- **Workflow anatomy**: `workflow.yaml` (manifest) + `program.md` (strategy/hypothesis) + optional `evaluate.py` + `results/results.tsv` (append-only log, gitignored) + `outputs/report/` (synthesis HTML).
- **Loop cycle** (see `.github/skills/autonomous-iteration/`): edit target > run cmd > extract metric > keep (commit) or discard (revert) > append row to results.tsv. Cycle Additive (add), Reductive (remove), Reformative (reshape) to avoid local optima.
- **AGENTS.md tree**: root `AGENTS.md` is the router; every subdir with agent-relevant context has its own local `AGENTS.md`. Don't duplicate content across them -- link from root, keep local files specific.
- **Pareto ratchet (tiered metrics)**: T1 gates (`task_success_rate`, `quality_gate_pass_rate`) must not regress; T2 (`rework_rate`) optimized subject to T1; T3 (`avg_token_consumption`, `time_per_turn`) subject to T1+T2. See `harness.yaml` `metrics:`.
- **Workflow manifest contract** (`workflow.yaml`): `targets` = agent-editable, `fixed` = read-only (enforced), `metric.extract` = shell command reading from `run.log`, `run.command` writes to `run.log`. Stages: `design | experiment | synthesis`.

## Key files
- `AGENTS.md` | root manifest/router | start here
- `harness.yaml` | machine-readable harness config (schema v3.0)
- `scaffold.py` | CLI: onboard, generate, workflow, report, harness-bg
- `.github/skills/` | SKILL.md per skill (autonomous-iteration, map-harness, report-generator)
- `.github/prompts/ui-ux-pro-max/` | design intelligence with search CLI
- `.github/hooks/` | lifecycle scripts (session_start, pre_tool_use, pre_commit, post_tool_use, session_end)
- `.github/tests/` | `benchmark.py` + `tasks/` (fixed, read-only) + `evaluator.py` (fixed)
- `workflows/_template/` | copy target for new workflows

## Workflows are domain-agnostic
Pattern is always: editable target + metric + run command. Existing workflows cover ML training (`examples/ml-training`), prompt engineering (`exec-summarizer`), report UX (`report-design`), visual design (`design-optimize`), and meta-optimization (`harness-optimize`). Create new with `python scaffold.py workflow <name>`.

## Benchmarks
- `uv run python .github/tests/benchmark.py --all` | full run (agent mode, requires `github-copilot-sdk`)
- `uv run python .github/tests/benchmark.py --quick` | subset (fastest iteration loop)
- `uv run python .github/tests/benchmark.py --all --dry-run` | structural validation only (no agent calls)
- `--json` for machine-readable output; `--model <id>` to override baseline
- No per-task filter flag exists -- use `--quick` to narrow scope, or invoke tasks directly via `tasks/<id>/`

## Guardrails (enforced by hooks)
- Read-only paths: `.github/tests/tasks/`, `.github/tests/evaluator.py` -- benchmark integrity
- Never commit: `results.tsv`, `musings.md`, `run.log`, `CLAUDE.md`, `*.key`, `*.pem`, `.env`, `local.settings.json`
- Blocked commands: `git push --force`, `git push -f`, `git reset --hard`, `git clean -f[d]`

## Conventions
- Conventional commits with Co-authored-by trailer
- AGENTS.md and SKILL.md under 1000 chars -- validate with `(Get-Content file.md -Raw).Length`
- Pipe-compressed index format for all instruction files
- Constraints include rationale: `No X -- reason`
- TSV for structured logs, gitignored
- Experiment status semantics: `keep` = committed, `discard` = reverted, `crash` = failed run
- Aggregate metrics (best, improvement%) use `keep` entries only, never crashes

## Windows caveats
- Always pass `encoding="utf-8"` to `subprocess.run()` and file I/O
- Avoid non-ASCII characters in print statements (cp1252 encoding errors)
- Use `Path` objects, not string concatenation for file paths
- Pre-commit hooks in harness.yaml are declared but not yet implemented as scripts
