# agents/

Agent definitions are behavioral contracts for autonomous agents. Each file describes a role, inputs, loop behavior, constraints, and outputs -- everything an AI agent needs to operate without human intervention.

Agents are platform-agnostic markdown. Platform-specific integration (e.g., GitHub Copilot) is handled by `.github/copilot-instructions.md` or equivalent pointers -- never by duplicating the definition.

## Available Agents

| Agent | File | Purpose | Used when... |
|-------|------|---------|--------------|
| research-runner | [research-runner.md](research-runner.md) | Autonomous experiment loop -- propose, run, measure, keep/discard, repeat | Running any `workflows/` iteration against a measurable metric |

## Conventions

- One markdown file per agent
- Each definition includes: Role, Inputs, Behavior (startup + loop), Constraints (hard + soft), Outputs
- Agents read `workflow.yaml` for targets and metrics, `program.md` for strategy, `AGENTS.md` for context
- Agents never modify their own definition files

## Version

- Last updated: 2026-07-15
