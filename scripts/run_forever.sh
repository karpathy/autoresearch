#!/usr/bin/env bash
# Auto-restart wrapper for autoresearch agent on Blackwell GPUs.
# Handles Triton sm_120 segfaults by restarting the agent automatically.
# Usage: bash scripts/run_forever.sh --dataset pubmed

set -o pipefail
# Source environment (API keys, PATH)
if [[ -f /etc/profile.d/autoresearch.sh ]]; then
    source /etc/profile.d/autoresearch.sh
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Pass all args through to the agent
AGENT_ARGS="$@"

# Clear stale Triton caches
rm -rf ~/.triton ~/.cache/triton ~/.cache/torch_extensions /tmp/torch* /tmp/triton* /tmp/torchinductor* 2>/dev/null

CRASHES=0
MAX_CRASHES=999

echo "=== autoresearch auto-restart wrapper ==="
echo "Agent args: $AGENT_ARGS"
echo "Max restarts: $MAX_CRASHES"
echo ""

while [ $CRASHES -lt $MAX_CRASHES ]; do
    echo "[$(date)] Starting agent (crash count: $CRASHES)..."

    # Clear Triton cache before each restart to prevent stale kernel segfaults
    rm -rf ~/.triton ~/.cache/triton /tmp/torchinductor* 2>/dev/null

    uv run scripts/agent.py --resume $AGENT_ARGS
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Agent completed normally."
        break
    fi

    CRASHES=$((CRASHES + 1))
    echo "[$(date)] Agent crashed (exit $EXIT_CODE). Restart $CRASHES/$MAX_CRASHES..."

    # Brief pause to let GPU memory free
    sleep 5
done

if [ $CRASHES -ge $MAX_CRASHES ]; then
    echo "[$(date)] Max crashes ($MAX_CRASHES) reached. Stopping."
fi

echo "Total crashes: $CRASHES"
