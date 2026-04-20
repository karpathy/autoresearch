#!/usr/bin/env bash
# Hook: session-end / git-status-summary
# Prints a summary of git working-tree status at session close.
# Always exits 0 — informational only.

set -uo pipefail

BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
CHANGED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
STAGED=$(git diff --cached --name-only 2>/dev/null | wc -l | tr -d ' ')
UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')

echo "--- Git Status Summary ---"
echo "Branch:    $BRANCH"
echo "Staged:    $STAGED file(s)"
echo "Modified:  $CHANGED file(s)"
echo "Untracked: $UNTRACKED file(s)"

exit 0
