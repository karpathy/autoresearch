# autoresearch

Canonical reference harness for agent-assisted development. This repo contains the optimized set of agent instructions, skills, hooks, guardrails, tests, and conventions that make AI agents maximally effective in any workspace.

**Core Invariant**: A harness declares agent instructions, hooks, skills, guardrails, quality gates, and conventions that maximize agent effectiveness. Everything else is optional.

## Discovery Index

Navigate to the sub-tree that matches your task.

| Directory | Purpose | Navigate when... |
|-----------|---------|------------------|
| [agents/](agents/AGENTS.md) | Agent behavioral contracts | Defining or modifying how an autonomous agent behaves |
| [skills/](skills/AGENTS.md) | Reusable agent capabilities | Adding or invoking a portable skill |
| [hooks/](hooks/AGENTS.md) | Deterministic lifecycle scripts | Enforcing guardrails that AI instructions alone cannot guarantee |
| [tests/](tests/AGENTS.md) | Benchmark suite for harness quality | Measuring whether a change improved or regressed agent effectiveness |
| [workflows/](workflows/) | Concrete experiment configurations | Running an autonomous iteration loop against a specific target |

Platform-specific integration (GitHub Copilot) lives in `.github/copilot-instructions.md` -- pointers only, never duplicated definitions.

## Cross-Cutting Conventions

| Convention | Detail |
|------------|--------|
| Commit style | Conventional commits (`feat:`, `fix:`, `docs:`, etc.) with `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer |
| Logging | TSV format for all structured logs (`results.tsv`). Untracked by git. |
| Gitignore | `results.tsv`, `musings.md`, `run.log`, `__pycache__/`, `.venv/` are local artifacts |
| Definitions | Agent/skill definitions live in top-level `agents/` and `skills/` -- single source of truth |
| Scaffolding | `python scaffold.py <name>` creates a new workflow from `workflows/_template/` |

## Version

- Last updated: 2026-04-19
- Harness version: 3.0
