#!/bin/bash
# PreToolUse hook: confine the agent to the project directory.
# - Blocks directory changes (cd, pushd, popd, chdir)
# - Blocks file reads/writes outside project dir
# - Blocks searches outside project dir

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
PROJECT_DIR="$CLAUDE_PROJECT_DIR"

deny() {
  jq -n --arg reason "$1" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $reason
    }
  }'
  exit 0
}

# Resolve a path without requiring it to exist, normalizing .. and symlinks.
# Tries GNU realpath -m (Linux), then Python 3 (macOS/Linux), then raw path.
resolve_path() {
  local path="$1"
  realpath -m "$path" 2>/dev/null && return
  python3 -c "import os, sys; print(os.path.normpath(os.path.abspath(sys.argv[1])))" "$path" 2>/dev/null && return
  echo "$path"
}

# Check if a path is within the project directory.
check_path() {
  local path="$1"
  # Empty/null path means the tool defaults to cwd, which is fine
  [ -z "$path" ] && return 0

  local resolved
  resolved=$(resolve_path "$path")

  case "$resolved" in
    "$PROJECT_DIR/.claude"|"$PROJECT_DIR/.claude"/*) return 1 ;;
    "$PROJECT_DIR"|"$PROJECT_DIR"/*) return 0 ;;
    *) return 1 ;;
  esac
}

case "$TOOL_NAME" in
  Bash)
    COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
    [ -z "$COMMAND" ] && exit 0

    # Block directory changes
    if echo "$COMMAND" | grep -qE '(^|[;&|`(]|&&|\|\||\$\()\s*(cd|pushd|popd|chdir)(\s|$|;|&|\||\))'; then
      deny "Changing the working directory is not allowed."
    fi
    ;;

  Read|Write|Edit)
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
    if ! check_path "$FILE_PATH"; then
      deny "Access denied: $FILE_PATH is outside the project directory ($PROJECT_DIR)."
    fi
    ;;

  Glob|Grep)
    SEARCH_PATH=$(echo "$INPUT" | jq -r '.tool_input.path // empty')
    if ! check_path "$SEARCH_PATH"; then
      deny "Access denied: $SEARCH_PATH is outside the project directory ($PROJECT_DIR)."
    fi
    ;;
esac

exit 0
