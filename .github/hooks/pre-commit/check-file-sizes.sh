#!/usr/bin/env bash
# Enforce 1000-character ceiling on AGENTS.md and SKILL.md files.
# Scans staged files. Exit 1 if any exceed ceiling.

set -euo pipefail

CEILING=1000
FAILED=0

FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '(AGENTS|SKILL)\.md$' || true)

if [ -z "$FILES" ]; then
  echo "All instruction files within ceiling"
  exit 0
fi

while IFS= read -r f; do
  CHARS=$(git show ":$f" | wc -m)
  CHARS=$(echo "$CHARS" | tr -d '[:space:]')
  if [ "$CHARS" -gt "$CEILING" ]; then
    echo "FAIL: $f is $CHARS chars (ceiling: $CEILING)"
    FAILED=1
  fi
done <<< "$FILES"

if [ "$FAILED" -eq 1 ]; then
  exit 1
fi

echo "All instruction files within ceiling"
exit 0
