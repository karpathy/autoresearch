#!/usr/bin/env bash
# Spike (pi-autoresearch-serverless, plan 17a, Trial 4): the Cloud Run job's command for the
# diagnostic. Is FlexAttention why this GPU's 953 steps reach val_bpb 1.027, not the H100's 0.995?
# spike/grad_check.py on weights with noise; train.py with SDPA for AR_STEPS steps (953 by default),
# its weights saved; spike/grad_check.py again on those trained weights. A failed stage is logged,
# and the next one runs.
set -uo pipefail
cd /app

stage() { echo "=== $1 $(date -u +%FT%TZ)"; }
WEIGHTS=/tmp/sdpa.pt

stage nvidia-smi
nvidia-smi || echo "nvidia-smi failed: $?"

stage "grad_check noise"
python spike/grad_check.py || echo "grad_check failed: $?"

stage "train sdpa"
start=$(date +%s)
# train.py redraws its progress line with \r: one line a step in the log instead.
AR_ATTN=sdpa AR_STEPS=${AR_STEPS:-953} AR_WEIGHTS=$WEIGHTS python train.py 2>&1 | tr '\r' '\n'
echo "train sdpa: exit ${PIPESTATUS[0]} after $(( $(date +%s) - start )) s"

stage "grad_check trained"
if [ -f "$WEIGHTS" ]; then
    python spike/grad_check.py "$WEIGHTS" || echo "grad_check failed: $?"
else
    echo "no trained weights to check"
fi

stage done
