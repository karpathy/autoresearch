#!/usr/bin/env bash
# activate.sh — Switch the repo to an optimization problem.
#
# Usage: bash test_problems/activate.sh <problem>
#
# This copies the problem's files into the repo root, replacing:
#   problem.yaml, agent_instructions.md, leaderboard.md,
#   state/*, context/*, evaluator/score.sh
#
# Every problem follows the same structure — see test_problems/README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ $# -ne 1 ]; then
    echo "Usage: bash test_problems/activate.sh <problem>"
    echo ""
    echo "Available problems:"
    for d in "$SCRIPT_DIR"/*/; do
        name=$(basename "$d")
        [ -f "$d/problem.yaml" ] && echo "  $name"
    done
    exit 1
fi

PROBLEM="$1"
PROBLEM_DIR="$SCRIPT_DIR/$PROBLEM"

if [ ! -d "$PROBLEM_DIR" ] || [ ! -f "$PROBLEM_DIR/problem.yaml" ]; then
    echo "Error: unknown problem '$PROBLEM'"
    echo ""
    echo "Available problems:"
    for d in "$SCRIPT_DIR"/*/; do
        name=$(basename "$d")
        [ -f "$d/problem.yaml" ] && echo "  $name"
    done
    exit 1
fi

echo "Activating problem: $PROBLEM"

# Copy root-level files
cp "$PROBLEM_DIR/problem.yaml" "$REPO_ROOT/problem.yaml"
cp "$PROBLEM_DIR/agent_instructions.md" "$REPO_ROOT/agent_instructions.md"

# Reset leaderboard
cat > "$REPO_ROOT/leaderboard.md" << 'EOF'
# Leaderboard

No evaluations yet. Push a proposal branch to get started.
EOF

# Replace state/ contents
mkdir -p "$REPO_ROOT/state"
rm -f "$REPO_ROOT"/state/*.py
cp "$PROBLEM_DIR"/state/*.py "$REPO_ROOT/state/"

# Replace context/ contents
mkdir -p "$REPO_ROOT/context"
rm -f "$REPO_ROOT"/context/*.py
cp "$PROBLEM_DIR"/context/*.py "$REPO_ROOT/context/"

# Replace evaluator/score.sh
mkdir -p "$REPO_ROOT/evaluator"
cp "$PROBLEM_DIR/evaluator/score.sh" "$REPO_ROOT/evaluator/score.sh"
chmod +x "$REPO_ROOT/evaluator/score.sh"

# Remove stale evaluator state
rm -f "$REPO_ROOT/evaluator/history.db"
rm -f "$REPO_ROOT/evaluator/run.log"

echo ""
echo "Done. Active problem: $PROBLEM"
echo ""
echo "Quick test:  bash evaluator/score.sh"
echo "Baseline:    uv run evaluator/evaluate.py --baseline-only"
