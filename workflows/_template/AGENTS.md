# {workflow_name}

## Overview

{Brief description of this research workflow and its goal.}

## Workflow Structure

```
{workflow_name}/
├── AGENTS.md          -- this file (workflow-specific agent context)
├── program.md         -- human-written research program and constraints
├── workflow.yaml      -- machine-readable manifest (targets, metric, run command)
├── results/
│   ├── results.tsv    -- experiment log (untracked)
│   └── musings.md     -- experiment reflections (untracked)
└── outputs/           -- reports and candidate artifacts (untracked until promoted)
```

## Key Files

| File | Role | Who Edits |
|------|------|-----------|
| workflow.yaml | Declares targets, metric, run command | Human (once) |
| program.md | Research strategy, constraints, domain knowledge | Human (iteratively) |
| {target_file} | The file being optimized | Agent (each experiment) |

## Constraints

{List any hardware, time, or domain-specific constraints here.}

## Running

1. Read `program.md` for research strategy
2. Read `workflow.yaml` for targets, metric, and run command
3. Follow the `autonomous-iteration` skill in `../../skills/autonomous-iteration.md`
