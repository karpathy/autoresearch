#!/usr/bin/env bash
# Hook: pre-tool-use / block-dangerous-commands
# Reads JSON from stdin (tool_input with a "command" field).
# Exits 0 to allow, exits 2 to deny (message on stderr).

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | grep -oP '"command"\s*:\s*"\K[^"]+' 2>/dev/null || true)

if [ -z "$COMMAND" ]; then
  exit 0  # no command field — nothing to check
fi

# Destructive git operations
BLOCKED_PATTERNS=(
  'git push --force'
  'git push -f'
  'git reset --hard'
  'git clean -f'
  'git clean -fd'
  'git branch -D'
)

for pattern in "${BLOCKED_PATTERNS[@]}"; do
  if [[ "$COMMAND" == *"$pattern"* ]]; then
    echo "BLOCKED: command contains dangerous pattern: $pattern" >&2
    exit 2
  fi
done

# Destructive rm commands
if echo "$COMMAND" | grep -qE 'rm\s+-rf\s+(/|~|\.)(\s|$)'; then
  echo "BLOCKED: destructive rm command detected" >&2
  exit 2
fi

# Secrets in commands
if echo "$COMMAND" | grep -qiE '(API_KEY|SECRET|PASSWORD|TOKEN)='; then
  echo "BLOCKED: command appears to contain secrets" >&2
  exit 2
fi

exit 0
