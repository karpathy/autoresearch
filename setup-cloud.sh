#!/bin/bash
# setup-cloud.sh — Run autoresearch on root-only cloud GPUs (RunPod, Lambda, Vast.ai, etc.)
#
# Usage: ./setup-cloud.sh [workspace_dir]
# Default workspace: /workspace/autoresearch

set -e

WORK_DIR="${1:-$(dirname "$(realpath "$0")")}"
USER="researcher"

echo "Setting up non-root user for autonomous Claude Code..."
echo "Workspace: $WORK_DIR"

# Create user if needed
if ! id "$USER" &>/dev/null; then
    echo "Creating user: $USER"
    useradd -m -s /bin/bash "$USER"
else
    echo "User $USER already exists"
fi

# Copy workspace to user's home
echo "Copying workspace to /home/$USER/autoresearch..."
rm -rf "/home/$USER/autoresearch"
cp -r "$WORK_DIR" "/home/$USER/autoresearch"
chown -R "$USER:$USER" "/home/$USER/autoresearch"

# Fix git safe directory
echo "Configuring git safe directory..."
su - "$USER" -c "git config --global --add safe.directory /home/$USER/autoresearch"

# Check for data directory
DATA_DIR="/home/$USER/.cache/autoresearch"
if [ -d "$DATA_DIR" ]; then
    echo "Data directory found at $DATA_DIR"
    chown -R "$USER:$USER" "$DATA_DIR"
else
    echo ""
    echo "WARNING: Data not found at $DATA_DIR"
    echo "Run 'uv run prepare.py' as root first, then re-run this script."
    echo ""
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To launch autonomous research, run:"
echo ""
echo "  su - $USER -c 'cd ~/autoresearch && claude --dangerously-skip-permissions'"
echo ""
echo "Or with a custom prompt:"
echo ""
echo "  su - $USER -c 'cd ~/autoresearch && claude --dangerously-skip-permissions -p \"Hi, have a look at program.md and let'"'"'s kick off a new experiment!\"'"
echo ""