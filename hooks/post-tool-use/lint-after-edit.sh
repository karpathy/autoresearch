#!/usr/bin/env bash
# Hook: post-tool-use / lint-after-edit
# Runs linting after file-edit tools (Write, Edit, Create).
# Reads JSON from stdin with tool name and file path info.
# Always exits 0 — post-tool hooks are informational only.

set -uo pipefail

INPUT=$(cat)
TOOL=$(echo "$INPUT" | grep -oP '"tool"\s*:\s*"\K[^"]+' 2>/dev/null || true)
FILE=$(echo "$INPUT" | grep -oP '"path"\s*:\s*"\K[^"]+' 2>/dev/null || true)

# Only act on file-edit tools
case "$TOOL" in
  Write|Edit|Create|write|edit|create) ;;
  *) exit 0 ;;
esac

if [ -z "$FILE" ]; then
  exit 0
fi

# Lint Python files with ruff if available
if [[ "$FILE" == *.py ]] && command -v ruff &>/dev/null; then
  echo "Running ruff on $FILE"
  ruff check --fix "$FILE" 2>&1 || true
fi

exit 0
