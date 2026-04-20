#!/usr/bin/env bash
# Hook: session-start / check-agents-freshness
# Finds all AGENTS.md files in the repo and warns if any are 90+ days stale.
# Always exits 0 — informational only.

set -uo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
STALE_DAYS=90
NOW=$(date +%s)
FOUND_STALE=false

while IFS= read -r file; do
  MOD_TIME=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null || echo 0)
  AGE_DAYS=$(( (NOW - MOD_TIME) / 86400 ))

  if [ "$AGE_DAYS" -ge "$STALE_DAYS" ]; then
    echo "WARNING: $file is ${AGE_DAYS} days old (threshold: ${STALE_DAYS})"
    FOUND_STALE=true
  fi
done < <(find "$REPO_ROOT" -name "AGENTS.md" -type f 2>/dev/null)

if [ "$FOUND_STALE" = false ]; then
  echo "All AGENTS.md files are fresh."
fi

exit 0
