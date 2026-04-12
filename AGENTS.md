# autoresearch

## Overview

General-purpose autonomous research framework. An AI agent modifies code or configuration, measures a metric, keeps improvements, discards regressions, and repeats. Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and generalized beyond ML training.

**Core Invariant**: A workflow declares editable targets, an evaluator, a score direction, and a keep/discard policy. Everything else is optional.

## Repo Structure

```
autoresearch/
├── .github/                  # Copilot integration pointers
│   └── copilot-instructions.md
├── agents/                   # Portable agent definitions
│   └── research-runner.md
├── skills/                   # Portable skill definitions
│   └── autonomous-iteration.md
├── workflows/
│   ├── _template/            # Template for new research workflows
│   │   ├── AGENTS.md
│   │   ├── program.md
│   │   ├── workflow.yaml     # Machine-readable workflow manifest
│   │   ├── results/
│   │   └── outputs/
│   └── examples/
│       └── ml-training/      # Reference implementation (original autoresearch)
├── scaffold.py               # CLI to create new workflows
├── AGENTS.md                 # This file
└── README.md
```

## How to Create a Workflow

```bash
python scaffold.py <workflow-name>
```

This creates a new directory under `workflows/<workflow-name>/` from the template. Then:

1. Edit `workflow.yaml` to declare:
   - Editable targets (files the agent can modify)
   - Evaluator command (how to measure success)
   - Score direction (higher_is_better or lower_is_better)
   - Keep/discard policy (threshold, improvement ratio, etc.)

2. Edit `program.md` with domain-specific instructions for the agent.

3. Point an AI agent at `workflows/<workflow-name>/program.md` and let it iterate.

## How to Run

Point an AI agent (GitHub Copilot, Claude, etc.) at the workflow's `program.md` file. The agent will:

1. Read the workflow manifest
2. Propose changes to editable targets
3. Run the evaluator
4. Keep improvements or discard regressions
5. Log results and repeat

## Conventions

| Convention | Rationale |
|------------|-----------|
| Agent/skill definitions live in top-level `agents/` and `skills/` | Single source of truth |
| `.github/` contains pointers only | Copilot integration without duplication |
| `workflow.yaml` is the machine-readable contract | Parseable by both agents and tooling |
| `results.tsv` and `musings.md` are gitignored | Experiment logs are local artifacts |
| Artifact extraction is human-curated | Agent proposes candidates in `outputs/`, human reviews |

## Workflow Manifest Schema

A minimal `workflow.yaml`:

```yaml
name: "example-workflow"
description: "Optimize script.py for best accuracy"

targets:
  - path: "script.py"
    description: "The file being optimized"

fixed:
  - path: "evaluate.py"
    description: "Evaluation harness (read-only)"

metric:
  name: "accuracy"
  direction: higher
  extract: "grep '^accuracy:' run.log | awk '{print $2}'"

run:
  command: "python script.py > run.log 2>&1"
  timeout: 600
```

## Skills

This framework uses the `autonomous-iteration` skill. See `skills/autonomous-iteration.md` for details.

Agents consume this skill via:
- Direct reference: `skills/autonomous-iteration.md`
- Copilot pointer: `.github/copilot-instructions.md`

## Reference Implementations

| Workflow | Domain | Description |
|----------|--------|-------------|
| `workflows/examples/ml-training/` | ML research | Original karpathy/autoresearch (PyTorch training loop optimization) |

## Version

- Last updated: 2026-04-12
- Framework version: 2.0 (generalized from ML-specific 1.0)
