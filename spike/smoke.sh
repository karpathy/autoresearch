#!/usr/bin/env bash
# Spike (pi-autoresearch-serverless, plan 17a): the Cloud Run job's command. The device as torch sees
# it; the val rows upstream's eval scores (spike/val_rows.py, prepare.py's code only); train.py with
# FlexAttention for AR_STEPS steps (953 by default: the H100's 499.6M tokens) and upstream's eval,
# exporting its model; then spike/onnx_check.py on the export and the val rows, isolated by -I. A
# failed stage is logged, and the next one runs.
set -uo pipefail
cd /app

stage() { echo "=== $1 $(date -u +%FT%TZ)"; }
EXPORT=/tmp/export
VAL=/tmp/val

stage nvidia-smi
nvidia-smi || echo "nvidia-smi failed: $?"

stage torch
python - <<'EOF'
import json, torch
p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print("SPIKE_DEVICE " + json.dumps({
    "torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available(),
    "name": p.name if p else None, "capability": list(torch.cuda.get_device_capability()) if p else None,
    "bf16": torch.cuda.is_bf16_supported() if p else None, "memory_gb": p.total_memory / 1e9 if p else None,
    "sms": p.multi_processor_count if p else None, "arch_list": torch.cuda.get_arch_list(),
}), flush=True)
EOF

stage val_rows
python spike/val_rows.py "$VAL" || echo "val_rows failed: $?"

stage "train flex"
start=$(date +%s)
# train.py redraws its progress line with \r: one line a step in the log instead.
AR_ATTN=flex AR_STEPS=${AR_STEPS:-953} AR_EXPORT_DIR=$EXPORT python train.py 2>&1 | tr '\r' '\n'
echo "train flex: exit ${PIPESTATUS[0]} after $(( $(date +%s) - start )) s"

stage onnx_check
if [ -f "$EXPORT/model.onnx" ]; then
    if [ -f "$VAL/x.npy" ]; then val=$VAL; else val=""; fi
    python -I spike/onnx_check.py "$EXPORT" $val || echo "onnx_check failed: $?"
else
    echo "no export to check"
fi

stage done
