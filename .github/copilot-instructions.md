# Copilot Instructions

Reference harness for agent-assisted development. See [AGENTS.md](../AGENTS.md) for full context.

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

## Key files
- `AGENTS.md` | root manifest/router | start here
- `harness.yaml` | machine-readable harness config
- `.github/skills/` | skill definitions (SKILL.md per skill)
- `.github/prompts/` | Copilot prompts (ui-ux-pro-max design intelligence)
- `.github/skills/report-generator/` | Jinja2+Chart.js report templates
- `.github/hooks/` | lifecycle enforcement scripts
- `.github/tests/` | benchmark suite with Pareto ratchet

## Workflows are domain-agnostic
This is a general-purpose optimization loop, not just ML training.
The pattern is always: editable file + metric + run command.
Workflows exist for ML training, prompt engineering, report design,
and harness config. Create new ones with `scaffold.py workflow`.

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
