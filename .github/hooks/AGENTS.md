# hooks
Deterministic scripts at lifecycle events. Hooks enforce what instructions cannot guarantee.

## Events
- `session-start/` | environment health checks, AGENTS.md freshness
- `pre-tool-use/` | block dangerous commands, enforce read-only paths
- `post-tool-use/` | lint after edits, log tool invocations
- `session-end/` | git status summary, cleanup

## Constraints
- Under 5 seconds, no LLM calls -- determinism and speed
- Exit 0 proceeds; non-zero blocks (pre) or warns (post) -- fail-safe default
