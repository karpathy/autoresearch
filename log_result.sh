#!/bin/bash
# Log an experiment result to results.tsv
# Usage: ./log_result.sh <val_bpb> <memory_gb> <status> <description>
# Example: ./log_result.sh 0.997900 44.0 keep "baseline"
# Status must be: keep, discard, or crash

VAL_BPB="${1:?Usage: ./log_result.sh <val_bpb> <memory_gb> <status> <description>}"
MEMORY_GB="${2:?Missing memory_gb}"
STATUS="${3:?Missing status (keep/discard/crash)}"
DESCRIPTION="${4:?Missing description}"

COMMIT=$(git rev-parse --short=7 HEAD)

# Create header if results.tsv doesn't exist
if [ ! -f results.tsv ]; then
    printf "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n" > results.tsv
fi

printf "%s\t%s\t%s\t%s\t%s\n" "$COMMIT" "$VAL_BPB" "$MEMORY_GB" "$STATUS" "$DESCRIPTION" >> results.tsv
echo "Logged: $COMMIT | $VAL_BPB | $MEMORY_GB | $STATUS | $DESCRIPTION"
