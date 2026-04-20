#!/usr/bin/env bash
# Block writes to read-only paths listed in harness.yaml.
# Reads tool input JSON from stdin, checks if target path matches read-only patterns.
# Exit 0 = allow, Exit 2 = block.

set -euo pipefail

READONLY_PATTERNS=(
  "tests/tasks/"
  "tests/evaluator.py"
)

INPUT=$(cat)

# Extract path-like fields from tool input JSON.
TARGET=$(echo "$INPUT" \
  | grep -oP '"(?:path|file|target)"\s*:\s*"([^"]*)"' \
  | head -1 \
  | sed 's/.*: *"//;s/"//')

if [ -z "$TARGET" ]; then
  exit 0
fi

for pattern in "${READONLY_PATTERNS[@]}"; do
  case "$TARGET" in
    ${pattern}*)
      echo "BLOCKED: $TARGET is read-only per harness.yaml guardrails"
      exit 2
      ;;
  esac
done

exit 0
