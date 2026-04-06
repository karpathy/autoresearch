#!/usr/bin/env bash
# Loop 2 - Auto-Research (continuous experiment loop)
# Runs multiple experiments in sequence, applying the keep/discard methodology.
# Each experiment: run train.py, evaluate, keep if improved, discard if not.
#
# Usage:
#   ./auto_research.sh              # Run 12 experiments (~1 hour)
#   ./auto_research.sh 100          # Run 100 experiments (~8 hours, overnight)
#   ./auto_research.sh unlimited    # Run until manually stopped

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/cron/logs"
RESULTS_FILE="$REPO_DIR/results.tsv"
TIMEOUT_SECONDS=600  # 10 min hard timeout per experiment

MAX_EXPERIMENTS="${1:-12}"
UNLIMITED=false
if [ "$MAX_EXPERIMENTS" = "unlimited" ]; then
    UNLIMITED=true
    MAX_EXPERIMENTS=999999
fi

mkdir -p "$LOG_DIR"

SESSION_LOG="$LOG_DIR/session_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$SESSION_LOG") 2>&1

cd "$REPO_DIR"

# Initialize results.tsv if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > "$RESULTS_FILE"
fi

# Read the best val_bpb from results so far
get_best_bpb() {
    if [ -f "$RESULTS_FILE" ] && [ "$(wc -l < "$RESULTS_FILE")" -gt 1 ]; then
        tail -n +2 "$RESULTS_FILE" | awk -F'\t' '$4=="keep" && $2+0>0 {print $2}' | sort -n | head -1
    else
        echo ""
    fi
}

echo "============================================="
echo "  Auto-Research Loop"
echo "  Started: $(date)"
if [ "$UNLIMITED" = true ]; then
    echo "  Mode: Unlimited (run until stopped)"
else
    echo "  Experiments: $MAX_EXPERIMENTS"
fi
echo "============================================="
echo ""

EXPERIMENT_NUM=0
KEPT=0
DISCARDED=0
CRASHED=0

while [ "$EXPERIMENT_NUM" -lt "$MAX_EXPERIMENTS" ]; do
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

    echo "--- Experiment $EXPERIMENT_NUM / $([ "$UNLIMITED" = true ] && echo 'unlimited' || echo "$MAX_EXPERIMENTS") ---"
    echo "Time: $(date)"

    BEST_BEFORE="$(get_best_bpb)"
    COMMIT_BEFORE="$(git rev-parse --short=7 HEAD)"

    echo "Current best val_bpb: ${BEST_BEFORE:-N/A (no baseline)}"
    echo "Running training..."

    if timeout "$TIMEOUT_SECONDS" uv run train.py > "$RUN_LOG" 2>&1; then
        VAL_BPB="$(grep '^val_bpb:' "$RUN_LOG" | awk '{print $2}' || echo '')"
        PEAK_VRAM="$(grep '^peak_vram_mb:' "$RUN_LOG" | awk '{print $2}' || echo '0.0')"

        if [ -z "$VAL_BPB" ]; then
            echo "ERROR: Could not extract val_bpb"
            MEMORY_GB="0.0"
            STATUS="crash"
            DESCRIPTION="experiment $EXPERIMENT_NUM - no metrics extracted"
            CRASHED=$((CRASHED + 1))
        else
            MEMORY_GB="$(echo "$PEAK_VRAM" | awk '{printf "%.1f", $1/1024}')"

            # Decide: keep or discard
            if [ -z "$BEST_BEFORE" ]; then
                # No baseline yet - this is the baseline
                STATUS="keep"
                DESCRIPTION="baseline (experiment $EXPERIMENT_NUM)"
                KEPT=$((KEPT + 1))
                echo "BASELINE: val_bpb=$VAL_BPB memory=${MEMORY_GB}GB"
            else
                # Compare against best
                IMPROVED="$(echo "$VAL_BPB $BEST_BEFORE" | awk '{print ($1 < $2) ? "yes" : "no"}')"
                if [ "$IMPROVED" = "yes" ]; then
                    STATUS="keep"
                    DESCRIPTION="experiment $EXPERIMENT_NUM - improved from $BEST_BEFORE to $VAL_BPB"
                    KEPT=$((KEPT + 1))
                    echo "KEEP: val_bpb=$VAL_BPB (improved from $BEST_BEFORE) memory=${MEMORY_GB}GB"
                else
                    STATUS="discard"
                    DESCRIPTION="experiment $EXPERIMENT_NUM - no improvement ($VAL_BPB >= $BEST_BEFORE)"
                    DISCARDED=$((DISCARDED + 1))
                    echo "DISCARD: val_bpb=$VAL_BPB (not better than $BEST_BEFORE) memory=${MEMORY_GB}GB"
                    # Reset to the commit before this experiment
                    git checkout -- train.py 2>/dev/null || true
                fi
            fi
        fi
    else
        EXIT_CODE=$?
        echo "CRASH: Training failed (exit code: $EXIT_CODE)"
        if [ -f "$RUN_LOG" ]; then
            tail -n 10 "$RUN_LOG" >&2
        fi
        VAL_BPB="0.000000"
        MEMORY_GB="0.0"
        STATUS="crash"
        if [ "$EXIT_CODE" -eq 124 ]; then
            DESCRIPTION="experiment $EXPERIMENT_NUM - timed out"
        else
            DESCRIPTION="experiment $EXPERIMENT_NUM - crashed"
        fi
        CRASHED=$((CRASHED + 1))
        git checkout -- train.py 2>/dev/null || true
    fi

    COMMIT="$(git rev-parse --short=7 HEAD)"
    printf '%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "$VAL_BPB" "$MEMORY_GB" "$STATUS" "$DESCRIPTION" >> "$RESULTS_FILE"

    echo ""
done

echo "============================================="
echo "  Auto-Research Complete"
echo "  Finished: $(date)"
echo "  Total experiments: $EXPERIMENT_NUM"
echo "  Kept: $KEPT"
echo "  Discarded: $DISCARDED"
echo "  Crashed: $CRASHED"
echo "  Best val_bpb: $(get_best_bpb)"
echo "============================================="
