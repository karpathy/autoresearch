#!/bin/bash
# Run a single autoresearch experiment and extract results.
# Usage: ./run_experiment.sh
# Output: Prints val_bpb and peak_vram_mb, or "CRASH" if the run failed.

set -e

echo "Starting experiment..."
uv run train.py > run.log 2>&1 || true

# Extract results
VAL_BPB=$(grep "^val_bpb:" run.log 2>/dev/null | awk '{print $2}')
PEAK_VRAM=$(grep "^peak_vram_mb:" run.log 2>/dev/null | awk '{print $2}')

if [ -z "$VAL_BPB" ]; then
    echo "CRASH"
    echo "--- Last 30 lines of run.log ---"
    tail -n 30 run.log
    exit 1
else
    echo "val_bpb: $VAL_BPB"
    echo "peak_vram_mb: $PEAK_VRAM"
    MEMORY_GB=$(echo "scale=1; $PEAK_VRAM / 1024" | bc)
    echo "memory_gb: $MEMORY_GB"
fi
