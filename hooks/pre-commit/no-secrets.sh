#!/usr/bin/env bash
# Scan staged diff for potential secrets. Exit 1 if found.

set -euo pipefail

PATTERNS='(API_KEY=|SECRET=|PASSWORD=|TOKEN=|PRIVATE_KEY|sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|-----BEGIN)'

DIFF=$(git diff --cached --diff-filter=ACM -U0 || true)

if [ -z "$DIFF" ]; then
  exit 0
fi

MATCHES=$(echo "$DIFF" | grep -nP "$PATTERNS" || true)

if [ -n "$MATCHES" ]; then
  echo "BLOCKED: potential secret detected in staged diff"
  echo "$MATCHES" | while IFS= read -r line; do
    REDACTED=$(echo "$line" | sed -E 's/(=|: *).+/=***REDACTED***/')
    echo "  $REDACTED"
  done
  exit 1
fi

exit 0
