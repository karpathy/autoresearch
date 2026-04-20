# Copilot Instructions

Reference harness for agent-assisted development. See [AGENTS.md](../AGENTS.md) for full context.

## Commands
- `uv sync` | install dependencies
- `python scaffold.py onboard` | guided harness.yaml creation
- `python scaffold.py generate harness.yaml` | generate workspace
- `python scaffold.py workflow <name>` | create workflow
- `uv run python .github/tests/benchmark.py --all` | run benchmarks

## Key files
- `AGENTS.md` | root manifest/router | start here
- `harness.yaml` | machine-readable harness config
- `.github/skills/` | skill definitions (SKILL.md per skill)
- `.github/hooks/` | lifecycle enforcement scripts
- `.github/tests/` | benchmark suite with Pareto ratchet

## Conventions
- Conventional commits with Co-authored-by trailer
- AGENTS.md and SKILL.md under 1000 chars -- context window budget
- Pipe-compressed index format for all instruction files
- Constraints include rationale: `No X -- reason`
- TSV for structured logs, gitignored
