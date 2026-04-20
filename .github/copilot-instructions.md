# Copilot Instructions

Canonical reference harness for agent-assisted development. For full context, see [AGENTS.md](../AGENTS.md).

## Commands

```bash
# Dependencies (uses uv, not pip)
uv sync

# Harness onboarding and generation
python scaffold.py onboard                  # guided harness.yaml creation
python scaffold.py generate harness.yaml    # generate workspace files from config

# Scaffold a new research workflow
python scaffold.py workflow <name>
```

## Architecture

Two-layer design:

- **Harness infrastructure** (root) -- agent instructions, skills, hooks, guardrails, quality gates, conventions. Defined in `AGENTS.md` (manifest/router) and `harness.yaml` (machine-readable config).
- **Research workflows** (`workflows/`) -- self-contained experiment configurations, each with a `workflow.yaml`, `program.md`, and local `AGENTS.md`.

Each sub-directory has its own `AGENTS.md`. The root `AGENTS.md` is the discovery index.

## Conventions

### Source of truth hierarchy

Agent and skill definitions live in top-level `agents/` and `skills/`. This file contains integration pointers only. Each workflow's `AGENTS.md` provides workflow-specific context but defers to the top-level definitions.

### The experiment loop

Every workflow follows the protocol in `agents/research-runner.md`:

1. Edit target file(s)
2. Git commit with descriptive message
3. Run experiment, redirect to `run.log`
4. Extract metric from `run.log`
5. Check quality gates (if configured in `workflow.yaml`)
6. Keep (improved + gates passed) or discard (`git reset --hard HEAD~1`)
7. Log to `results.tsv` and `musings.md`
8. Repeat -- never stop, never ask permission

### Experiment strategy (Artisan's Triad)

Cycle through three modes. Don't do 5+ of the same type in a row:

- **Additive**: introduce new techniques or features
- **Reductive**: remove components, simplify
- **Reformative**: reshape without adding or removing

### Logging

`results.tsv` is tab-separated. Columns: `commit`, metric value, `memory_gb`, `status` (keep/discard/crash), `description`. Neither `results.tsv` nor `musings.md` are committed to git.

### Quality gates

When `workflow.yaml` has a `gates` section, outputs must pass self-eval thresholds (composite min + per-dimension floors) before a keep decision. Gate failure forces a discard even if the metric improved.

### Dependencies

Managed by `uv` with `pyproject.toml`. No new packages may be added during experiments.
