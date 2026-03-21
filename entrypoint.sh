#!/bin/bash
# =============================================================================
# entrypoint.sh — Baked into Docker image.
# Bootstraps /working volume: clones/pulls repo, sets up /data, hands off
# to /working/REPO_NAME/startup.sh.
#
# Environment variables (set in pod config):
#   REPO_NAME        - repo directory name under /working (default: autoresearch)
#   GITHUB_REPO      - GitHub repo path (default: rstager/autoresearch)
#   DATA_PERSISTENT  - "true" = /data → /working/data | "false" = /data on root fs
# =============================================================================
set -euo pipefail

export UV_PROJECT_ENVIRONMENT=/opt/venv

REPO_NAME="${REPO_NAME:-autoresearch}"
GITHUB_REPO="${GITHUB_REPO:-rstager/autoresearch}"
DATA_PERSISTENT="${DATA_PERSISTENT:-false}"
WORKING_DIR="/working"
REPO_DIR="$WORKING_DIR/$REPO_NAME"

echo "[entrypoint] Starting at $(date)"

# --- Verify /working is mounted ---
if [ ! -d "$WORKING_DIR" ]; then
    echo "[entrypoint] ERROR: /working volume not mounted — keeping alive for debug"
    tail -f /dev/null
    exit 1
fi

# --- Clone or pull repo ---
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[entrypoint] Cloning https://github.com/$GITHUB_REPO → $REPO_DIR"
    git clone "https://github.com/$GITHUB_REPO.git" "$REPO_DIR"
else
    echo "[entrypoint] Updating $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only || echo "[entrypoint] WARNING: git pull failed, using existing code"
fi

# --- Set up /data ---
if [ "$DATA_PERSISTENT" = "true" ]; then
    echo "[entrypoint] /data → /working/data (persistent)"
    mkdir -p "$WORKING_DIR/data"
    ln -sfn "$WORKING_DIR/data" /data
else
    echo "[entrypoint] /data on container root (ephemeral)"
    mkdir -p /data
fi

# --- Set up /scratch ---
mkdir -p /scratch/tmp /scratch/compile
echo "[entrypoint] /scratch ready (ephemeral)"

# --- Hand off to repo's startup.sh ---
STARTUP="$REPO_DIR/startup.sh"
if [ -f "$STARTUP" ]; then
    chmod +x "$STARTUP"
    echo "[entrypoint] Handing off to $STARTUP"
    exec bash "$STARTUP"
else
    echo "[entrypoint] WARNING: $STARTUP not found — keeping alive for debug"
    tail -f /dev/null
fi
