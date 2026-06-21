#!/bin/bash
# setup-cloud.sh — Run autoresearch on root-only cloud GPUs (RunPod, Lambda, etc.)

set -e
WORK_DIR="${1:-$(pwd)}"
USER="researcher"

echo "Setting up non-root user for autonomous Claude Code..."

# Create user if needed
id $USER &>/dev/null || useradd -m -s /bin/bash $USER

# Copy workspace to the researcher user's home directory
echo "Copying workspace to /home/$USER/autoresearch..."
mkdir -p /home/$USER/autoresearch
cp -r "$WORK_DIR"/. /home/$USER/autoresearch/
chown -R $USER:$USER /home/$USER/autoresearch

# Configure git safe directory for the new user
echo "Configuring git safe directory..."
su - $USER -c "git config --global --add safe.directory /home/$USER/autoresearch"

# Copy cached dataset/tokenizer artifacts if present
if [ -d "/root/.cache/autoresearch" ]; then
    echo "Copying cached data artifacts to /home/$USER/.cache/autoresearch..."
    mkdir -p /home/$USER/.cache
    cp -r /root/.cache/autoresearch /home/$USER/.cache/
    chown -R $USER:$USER /home/$USER/.cache/autoresearch
fi

echo ""
echo "Setup complete. Launch Claude Code with:"
echo "  su - $USER -c 'cd ~/autoresearch && claude --dangerously-skip-permissions'"
