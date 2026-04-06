#!/usr/bin/env bash
# Loop 1 - Daily Report (9pm CET, weekdays)
# Runs the training experiment, captures results, and logs them.
# Designed to be invoked by cron on a schedule.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/cron/logs"
RESULTS_FILE="$REPO_DIR/results.tsv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"
REPORT_LOG="$LOG_DIR/report_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

exec > "$REPORT_LOG" 2>&1
echo "=== Daily Report - $(date) ==="

cd "$REPO_DIR"

# Initialize results.tsv if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > "$RESULTS_FILE"
    echo "Initialized results.tsv"
fi

# Record the current commit before the run
BASELINE_COMMIT="$(git rev-parse --short=7 HEAD)"
echo "Current commit: $BASELINE_COMMIT"

# Run the training experiment (5-minute budget)
echo "Starting training run..."
TIMEOUT_SECONDS=600  # 10 min hard timeout (5 min budget + overhead)

if timeout "$TIMEOUT_SECONDS" uv run train.py > "$RUN_LOG" 2>&1; then
    echo "Training completed successfully."

    # Extract metrics
    VAL_BPB="$(grep '^val_bpb:' "$RUN_LOG" | awk '{print $2}' || echo '0.000000')"
    PEAK_VRAM="$(grep '^peak_vram_mb:' "$RUN_LOG" | awk '{print $2}' || echo '0.0')"

    if [ -z "$VAL_BPB" ] || [ "$VAL_BPB" = "0.000000" ]; then
        echo "WARNING: Could not extract val_bpb from run log"
        MEMORY_GB="0.0"
        STATUS="crash"
        DESCRIPTION="daily run - failed to extract metrics"
    else
        # Convert peak VRAM from MB to GB
        MEMORY_GB="$(echo "$PEAK_VRAM" | awk '{printf "%.1f", $1/1024}')"
        STATUS="keep"
        DESCRIPTION="daily scheduled run"
    fi
else
    EXIT_CODE=$?
    echo "Training failed or timed out (exit code: $EXIT_CODE)"

    # Check last 50 lines for error context
    if [ -f "$RUN_LOG" ]; then
        echo "--- Last 50 lines of run log ---"
        tail -n 50 "$RUN_LOG"
        echo "--- End of error context ---"
    fi

    VAL_BPB="0.000000"
    MEMORY_GB="0.0"
    if [ "$EXIT_CODE" -eq 124 ]; then
        STATUS="crash"
        DESCRIPTION="daily run - timed out after ${TIMEOUT_SECONDS}s"
    else
        STATUS="crash"
        DESCRIPTION="daily run - training crashed"
    fi
fi

COMMIT="$(git rev-parse --short=7 HEAD)"

# Log to results.tsv
printf '%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "$VAL_BPB" "$MEMORY_GB" "$STATUS" "$DESCRIPTION" >> "$RESULTS_FILE"

echo ""
echo "=== Results ==="
echo "Commit:    $COMMIT"
echo "val_bpb:   $VAL_BPB"
echo "Memory:    ${MEMORY_GB} GB"
echo "Status:    $STATUS"
echo "=== Daily Report Complete - $(date) ==="
