# skills/

Skills are reusable capabilities that agents can invoke. Each skill defines a self-contained instruction set -- a pattern an agent follows to accomplish a specific type of work. Skills are portable across repos and agents.

Skills use the convention: filename is the skill identifier, content is the full instruction set including setup, loop, decision criteria, logging, and self-evaluation gates.

## Available Skills

| Skill | File | Description | Trigger |
|-------|------|-------------|---------|
| autonomous-iteration | [autonomous-iteration.md](autonomous-iteration.md) | Experiment loop -- edit target, measure metric, keep improvements, discard regressions | Agent is pointed at a `workflow.yaml` with targets and a metric |

## Conventions

- One markdown file per skill
- Each skill includes: Core Pattern, Setup Protocol, Loop, Decision Criteria, Logging, Acceptance Criteria, Self-Evaluation Gate
- Skills reference `workflow.yaml` for configuration but do not embed workflow-specific values
- Quality gates (self-eval, rubber-duck judge) are optional extensions defined in `workflow.yaml`

## Version

- Last updated: 2026-07-15
