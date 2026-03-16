#!/usr/bin/env bash
# ============================================================================
# autoresearch setup script
# Interactive one-command setup — prompts for everything it needs.
#
# Usage:
#   bash scripts/setup.sh              # interactive (prompts for everything)
#   bash scripts/setup.sh --auto       # skip prompts, use defaults + env vars
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; }

# ---------------------------------------------------------------------------
# Parse args (flags still work for scripted/CI use)
# ---------------------------------------------------------------------------
API_KEY=""
DATA_DIR=""
DATASET=""
AUTO_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key) API_KEY="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --auto) AUTO_MODE=true; shift ;;
        *) error "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         autoresearch setup               ║${NC}"
echo -e "${BOLD}║   Autonomous AI research agent           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
info "Running pre-flight checks..."

# Must be Linux/WSL
if [[ "$(uname -s)" != "Linux" ]]; then
    error "This script must run inside WSL (Linux). Run: wsl bash scripts/setup.sh"
    exit 1
fi

# Check for NVIDIA GPU (handle WSL lib path not on PATH)
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_SMI="nvidia-smi"
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    export PATH="/usr/lib/wsl/lib:$PATH"
    NVIDIA_SMI="nvidia-smi"
    warn "Added /usr/lib/wsl/lib to PATH (nvidia-smi was not on PATH)."
    warn "To make this permanent: echo 'export PATH=/usr/lib/wsl/lib:\$PATH' >> ~/.bashrc"
else
    error "nvidia-smi not found. Make sure NVIDIA drivers are installed on Windows."
    error "Download from: https://www.nvidia.com/drivers"
    error ""
    error "If using an older WSL distro (e.g. Ubuntu 20.04), try upgrading:"
    error "  wsl --install Ubuntu-22.04    (from PowerShell)"
    exit 1
fi

GPU_NAME=$($NVIDIA_SMI --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_VRAM=$($NVIDIA_SMI --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
info "GPU detected: ${GPU_NAME} (${GPU_VRAM} MB VRAM)"

# Detect GPU architecture for flash-attn3 compatibility
GPU_ARCH=$($NVIDIA_SMI --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
if [[ -n "$GPU_ARCH" ]]; then
    GPU_MAJOR=$(echo "$GPU_ARCH" | cut -d. -f1)
    if (( GPU_MAJOR >= 10 )); then
        info "Blackwell GPU detected (SM ${GPU_ARCH}) — will use PyTorch SDPA attention backend."
    fi
fi

# ---------------------------------------------------------------------------
# Interactive prompts (skipped if --auto or flags provided)
# ---------------------------------------------------------------------------
if [[ "$AUTO_MODE" == false ]]; then
    # API key
    if [[ -z "$API_KEY" && -z "$ANTHROPIC_API_KEY" ]]; then
        echo ""
        echo -e "${CYAN}Anthropic API key${NC} (get one at console.anthropic.com)"
        echo -n "  Paste key: "
        read -r API_KEY
        if [[ -z "$API_KEY" ]]; then
            warn "No key entered. You can set it later in /etc/profile.d/autoresearch.sh"
        fi
    fi

    # Dataset selection
    if [[ -z "$DATASET" ]]; then
        echo ""
        echo -e "${CYAN}Which dataset?${NC}"
        echo "  1) default  — FineWeb 10B (small, fast, ~2 min download)"
        echo "  2) pubmed   — PubMed medical abstracts (27M abstracts, ~14.6GB download)"
        echo -n "  Choice [1]: "
        read -r ds_choice
        case "$ds_choice" in
            2|pubmed) DATASET="pubmed" ;;
            *) DATASET="default" ;;
        esac
    fi

    # Data directory (only ask for large datasets)
    if [[ "$DATASET" != "default" && -z "$DATA_DIR" ]]; then
        echo ""
        echo -e "${CYAN}Where to store dataset?${NC} (${DATASET} is ~14.6GB)"
        echo "  Default: ~/.cache/autoresearch/"
        echo "  Or enter a path like /mnt/k/autoresearch-data"
        echo -n "  Path [default]: "
        read -r DATA_DIR
    fi
fi

echo ""

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
fi

# Data directory (for large datasets on a different drive)
if [[ -n "$DATA_DIR" ]]; then
    mkdir -p "$DATA_DIR"
    PROFILE_LINES="${PROFILE_LINES}\nexport AUTORESEARCH_DATA_DIR=\"$DATA_DIR\""
    export AUTORESEARCH_DATA_DIR="$DATA_DIR"
    info "Data directory: $DATA_DIR"
fi

# Dataset preference
if [[ -n "$DATASET" && "$DATASET" != "default" ]]; then
    PROFILE_LINES="${PROFILE_LINES}\nexport AUTORESEARCH_DATASET=\"$DATASET\""
    export AUTORESEARCH_DATASET="$DATASET"
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
    info "Default training data already prepared."
else
    info "Downloading default data and training tokenizer (~2 min)..."
    uv run prepare.py
fi

if [[ -n "$DATASET" && "$DATASET" != "default" ]]; then
    info "Downloading ${DATASET} dataset (this may take a while for large datasets)..."
    uv run prepare.py --dataset "$DATASET"
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
# Done!
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║           Setup complete!                ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  GPU:        ${GPU_NAME} (${GPU_VRAM} MB)"
echo "  Python:     $(python3 --version 2>&1)"
echo "  uv:         $(uv --version 2>&1)"
echo "  Dataset:    ${DATASET:-default}"
if [[ -n "$ANTHROPIC_API_KEY" ]]; then
    echo "  API Key:    ...${ANTHROPIC_API_KEY: -8}"
else
    echo "  API Key:    NOT SET"
fi
echo ""
echo -e "  ${GREEN}Run the agent:${NC}"
if [[ -n "$DATASET" && "$DATASET" != "default" ]]; then
    echo "  uv run scripts/agent.py --dataset ${DATASET}"
else
    echo "  uv run scripts/agent.py"
fi
echo ""
