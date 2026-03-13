#!/usr/bin/env bash
# ============================================================================
# autoresearch setup script
# One-command setup for WSL + NVIDIA GPU environments
#
# Usage (from inside WSL):
#   bash scripts/setup.sh
#   bash scripts/setup.sh --api-key sk-ant-api03-YOUR-KEY-HERE
#   bash scripts/setup.sh --api-key sk-ant-... --data-dir /mnt/g/autoresearch-data
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
API_KEY=""
DATA_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key) API_KEY="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) error "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
info "Running pre-flight checks..."

# Must be Linux/WSL
if [[ "$(uname -s)" != "Linux" ]]; then
    error "This script must run inside WSL (Linux). Run: wsl bash scripts/setup.sh"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Make sure NVIDIA drivers are installed on Windows."
    error "Download from: https://www.nvidia.com/drivers"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
info "GPU detected: ${GPU_NAME} (${GPU_VRAM} MB VRAM)"

# ---------------------------------------------------------------------------
# Install system deps
# ---------------------------------------------------------------------------
info "Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3 python3-pip python3-venv build-essential git curl > /dev/null 2>&1

# ---------------------------------------------------------------------------
# Install uv
# ---------------------------------------------------------------------------
if command -v uv &>/dev/null; then
    info "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# Set up environment variables (persistent)
# ---------------------------------------------------------------------------
PROFILE_SCRIPT="/etc/profile.d/autoresearch.sh"
info "Setting up environment..."

# Build profile script content
PROFILE_LINES="export PATH=\"/home/$(whoami)/.local/bin:\$PATH\""

# API key
if [[ -n "$API_KEY" ]]; then
    PROFILE_LINES="${PROFILE_LINES}\nexport ANTHROPIC_API_KEY=\"$API_KEY\""
    export ANTHROPIC_API_KEY="$API_KEY"
    info "API key configured."
elif [[ -n "$ANTHROPIC_API_KEY" ]]; then
    PROFILE_LINES="${PROFILE_LINES}\nexport ANTHROPIC_API_KEY=\"$ANTHROPIC_API_KEY\""
    info "Using existing ANTHROPIC_API_KEY from environment."
else
    warn "No API key provided. Set it later with:"
    warn "  echo 'export ANTHROPIC_API_KEY=\"sk-ant-...\"' | sudo tee -a /etc/profile.d/autoresearch.sh"
fi

# Data directory (for large datasets on a different drive)
if [[ -n "$DATA_DIR" ]]; then
    mkdir -p "$DATA_DIR"
    PROFILE_LINES="${PROFILE_LINES}\nexport AUTORESEARCH_DATA_DIR=\"$DATA_DIR\""
    export AUTORESEARCH_DATA_DIR="$DATA_DIR"
    info "Data directory: $DATA_DIR"
fi

echo -e "$PROFILE_LINES" | sudo tee "$PROFILE_SCRIPT" > /dev/null

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
info "Installing Python dependencies (this may take a few minutes)..."
cd "$(dirname "$0")/.."
uv sync

# ---------------------------------------------------------------------------
# Prepare data (tokenizer + dataset)
# ---------------------------------------------------------------------------
if [[ -f "data/fineweb10B/fineweb_train_000001.bin" ]]; then
    info "Training data already prepared."
else
    info "Downloading data and training tokenizer (~2 min)..."
    uv run prepare.py
fi

# ---------------------------------------------------------------------------
# Git credential helper (for auto-push)
# ---------------------------------------------------------------------------
if [[ -f "/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe" ]]; then
    git config --global credential.helper '/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe'
    info "Git credential manager configured (uses Windows credentials)."
else
    warn "Windows Git not found. Auto-push won't work until you set up git credentials."
fi

# ---------------------------------------------------------------------------
# Verify everything works
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
info "Setup complete! Verifying..."
echo "============================================"

echo ""
echo "  GPU:        ${GPU_NAME} (${GPU_VRAM} MB)"
echo "  Python:     $(python3 --version 2>&1)"
echo "  uv:         $(uv --version 2>&1)"
echo "  PyTorch:    $(uv run python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not yet')"
echo "  CUDA:       $(uv run python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'checking...')"

if [[ -n "$ANTHROPIC_API_KEY" ]]; then
    echo "  API Key:    ...${ANTHROPIC_API_KEY: -8}"
else
    echo "  API Key:    NOT SET (provide with --api-key)"
fi
echo ""

# ---------------------------------------------------------------------------
# Print next steps
# ---------------------------------------------------------------------------
echo "============================================"
echo "  NEXT STEPS"
echo "============================================"
echo ""
echo "  # Run the autonomous agent:"
echo "  uv run scripts/agent.py"
echo ""
echo "  # Train on PubMed medical abstracts:"
echo "  uv run prepare.py --dataset pubmed"
echo "  uv run scripts/agent.py --dataset pubmed"
echo ""
echo "  # Resume a previous run:"
echo "  uv run scripts/agent.py --resume"
echo ""
echo "  # Just run a single training test:"
echo "  uv run train.py"
echo ""
