#!/bin/bash
# =============================================================================
# startup.sh — Lives in the repo at /working/autoresearch/startup.sh
# Called by entrypoint.sh after clone/pull.
# =============================================================================
set -euo pipefail

REPO_NAME="${REPO_NAME:-autoresearch}"
REPO_DIR="/working/$REPO_NAME"
ENV_FILE="$REPO_DIR/.env"
BASHRC="/root/.bashrc"

echo "[startup] $(date)"

# -----------------------------------------------------------------------------
# 1. Source .env for this session
# -----------------------------------------------------------------------------
if [ -f "$ENV_FILE" ]; then
    echo "[startup] Sourcing $ENV_FILE"
    set -a; source "$ENV_FILE"; set +a
else
    echo "[startup] NOTE: $ENV_FILE not found — create it with HF_TOKEN, WANDB_KEY, etc."
fi

# -----------------------------------------------------------------------------
# 2. Patch .bashrc so .env and UV_PROJECT_ENVIRONMENT are set in every shell
# -----------------------------------------------------------------------------
BASHRC_MARKER="# cloud-gpu: auto-source project .env"
if ! grep -qF "$BASHRC_MARKER" "$BASHRC" 2>/dev/null; then
    echo "[startup] Patching $BASHRC"
    cat >> "$BASHRC" << BASHEOF

$BASHRC_MARKER
export UV_PROJECT_ENVIRONMENT=/opt/venv
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi
BASHEOF
fi

# -----------------------------------------------------------------------------
# 3. GPU check
# -----------------------------------------------------------------------------
echo "[startup] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || echo "[startup] WARNING: nvidia-smi failed"

# -----------------------------------------------------------------------------
# 4. Volume / disk check
# -----------------------------------------------------------------------------
for vol in /working /data /scratch; do
    if [ -d "$vol" ]; then
        echo "[startup] $vol: $(df -h "$vol" | tail -1 | awk '{print $4}') free"
    else
        echo "[startup] WARNING: $vol not available"
    fi
done

# -----------------------------------------------------------------------------
# 5. Standard directories
# -----------------------------------------------------------------------------
mkdir -p /data/datasets /data/checkpoints /data/logs /data/wandb /data/.cache/huggingface
mkdir -p /scratch/tmp /scratch/compile

# -----------------------------------------------------------------------------
# 6. Export standard paths
# -----------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-/data/wandb}"
export TMPDIR="${TMPDIR:-/scratch/tmp}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

# -----------------------------------------------------------------------------
# 7. DATA SETUP — download data if not already present
# -----------------------------------------------------------------------------
DATA_READY_FLAG="/data/datasets/.ready"
if [ ! -f "$DATA_READY_FLAG" ]; then
    echo "[startup] Downloading data (prepare.py)..."
    cd "$REPO_DIR"
    uv run prepare.py
    touch "$DATA_READY_FLAG"
else
    echo "[startup] Data already present, skipping download"
fi

# -----------------------------------------------------------------------------
# 8. Convenience symlinks into repo dir
# -----------------------------------------------------------------------------
ln -sfn /data/checkpoints "$REPO_DIR/checkpoints" 2>/dev/null || true
ln -sfn /data/datasets    "$REPO_DIR/datasets"    2>/dev/null || true
ln -sfn /data/logs        "$REPO_DIR/logs"        2>/dev/null || true

# -----------------------------------------------------------------------------
# 9. Ready — keep alive for interactive SSH use
#    To auto-start training: replace with: exec uv run train.py
# -----------------------------------------------------------------------------
echo "[startup] Ready — repo at $REPO_DIR"
cd "$REPO_DIR"
tail -f /dev/null
