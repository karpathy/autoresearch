#!/usr/bin/env bash
# Spike (pi-autoresearch-serverless, plan 17a): the Cloud Run job's command. The device as torch sees
# it; train.py with FlexAttention, then with SDPA, for AR_TIME_BUDGET seconds (60 by default) and
# upstream's full eval, the first that finishes exporting its model; then spike/onnx_check.py on the
# export, in the scorer's environment. A failed stage is logged, and the next one runs.
set -uo pipefail
cd /app

stage() { echo "=== $1 $(date -u +%FT%TZ)"; }
EXPORT=/tmp/export

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

for attn in flex sdpa; do
    stage "train $attn"
    if [ -f "$EXPORT/model.onnx" ]; then export_dir=""; else export_dir="$EXPORT"; fi
    start=$(date +%s)
    # train.py redraws its progress line with \r: one line a step in the log instead.
    AR_ATTN=$attn AR_TIME_BUDGET=${AR_TIME_BUDGET:-60} AR_EXPORT_DIR=$export_dir python train.py 2>&1 | tr '\r' '\n'
    echo "train $attn: exit ${PIPESTATUS[0]} after $(( $(date +%s) - start )) s"
done

stage onnx_check
if [ -f "$EXPORT/model.onnx" ]; then
    /opt/scorer/bin/python -I spike/onnx_check.py "$EXPORT" || echo "onnx_check failed: $?"
else
    echo "no export to check"
fi

stage done
