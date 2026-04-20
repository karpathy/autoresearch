# hooks/

Hooks are deterministic scripts that run at lifecycle events. They enforce guardrails that AI instructions alone cannot guarantee -- security checks, lint-after-edit, audit logging, environment validation. Hooks are the "hard floor" beneath agent behavior.

## Lifecycle Events

| Event | Directory | Fires when... | Typical use |
|-------|-----------|---------------|-------------|
| session-start | `session-start/` | Agent session begins | Environment health checks, AGENTS.md freshness, dependency verification |
| pre-tool-use | `pre-tool-use/` | Before a tool executes | Block dangerous commands, prevent secret exposure, validate file targets |
| post-tool-use | `post-tool-use/` | After a tool completes | Lint/typecheck edited files, log tool invocations, enforce conventions |
| session-end | `session-end/` | Agent session ends | Git status summary, cleanup temporary artifacts, emit session metrics |

## Conventions

- Hook scripts are shell scripts or simple executables -- one per file, placed in the matching event directory
- Platform-specific integration maps hooks to `.copilot/hooks.json` or equivalent configuration
- Hooks must be fast (under 5 seconds) and deterministic -- no LLM calls, no network dependencies
- Exit code 0 allows the action to proceed; non-zero blocks it (for pre-tool-use) or logs a warning (for post-tool-use)
- Hook scripts will be added in a subsequent step; the directory structure is established

## Version

- Last updated: 2026-07-15
