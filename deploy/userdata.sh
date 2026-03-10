#!/bin/bash
set -euxo pipefail

# Update system packages
apt-get update -y

# Install uv for the ubuntu user
curl -LsSf https://astral.sh/uv/install.sh | sudo -u ubuntu sh
echo 'source /home/ubuntu/.local/bin/env' >> /home/ubuntu/.bashrc

# Clone the repo and set up the project
sudo -u ubuntu git clone https://github.com/jackrieck/autoresearch.git /home/ubuntu/autoresearch
cd /home/ubuntu/autoresearch
sudo -u ubuntu /home/ubuntu/.local/bin/uv sync

# Ensure all files are owned by ubuntu
chown -R ubuntu:ubuntu /home/ubuntu/autoresearch
