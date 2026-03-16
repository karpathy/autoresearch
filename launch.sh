#!/usr/bin/env bash
# Launch autoresearch with the ML researcher persona active.
#
# Usage:
#   ./launch.sh                    # interactive mode with persona
#   ./launch.sh "kick off mar16"   # one-shot prompt with persona
#
# The persona is injected via --append-system-prompt-file so it layers
# on top of CLAUDE.md (project context) + the default Claude Code system
# prompt (tool usage, safety, etc). This is the recommended approach per
# Anthropic's persona selection research: shaping *who* the agent is,
# not just *what* it does.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PERSONA_FILE="$SCRIPT_DIR/persona.md"

if [[ ! -f "$PERSONA_FILE" ]]; then
    echo "Error: persona.md not found at $PERSONA_FILE"
    exit 1
fi

if [[ $# -gt 0 ]]; then
    # Non-interactive: pass the user's message as -p
    exec claude --append-system-prompt-file "$PERSONA_FILE" -p "$*" --allow-dangerously-skip-permissions 
else
    # Interactive mode
    exec claude --append-system-prompt-file "$PERSONA_FILE" --allow-dangerously-skip-permissions 
fi
