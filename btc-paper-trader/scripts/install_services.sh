#!/bin/bash
# Setup cron jobs and systemd service for the paper trader.
#
# Usage: bash scripts/install_services.sh
#
# Creates:
#   1. Hourly cron at :05 for main inference pipeline
#   2. Daily cron at 00:15 UTC for report generation
#   3. systemd service for liquidation websocket aggregator

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python venv not found at $PYTHON"
    echo "Create it first: python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

echo "Setting up paper trader services..."
echo "  Project dir: $SCRIPT_DIR"
echo "  Python: $PYTHON"

# --- 1. Cron jobs ---
echo ""
echo "Installing cron jobs..."

# Remove any existing paper trader cron entries
EXISTING_CRON=$(crontab -l 2>/dev/null | grep -v "btc-paper-trader" || true)

# Add hourly inference (minute :05) and daily report (00:15 UTC)
NEW_CRON="$EXISTING_CRON
# BTC Paper Trader — hourly inference
5 * * * * cd $SCRIPT_DIR && $PYTHON -m src.main >> logs/cron.log 2>&1
# BTC Paper Trader — daily report (00:15 UTC)
15 0 * * * cd $SCRIPT_DIR && $PYTHON -m src.main --report >> logs/cron.log 2>&1"

echo "$NEW_CRON" | crontab -
echo "  Cron jobs installed"

# --- 2. Liquidation websocket systemd service ---
echo ""
echo "Installing liquidation aggregator service..."

SERVICE_FILE="/etc/systemd/system/btc-liquidations.service"

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=BTC Paper Trader — Liquidation Websocket Aggregator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON -m src.liquidations --config config.yaml
Restart=always
RestartSec=10
User=$(whoami)

# Logging
StandardOutput=append:$SCRIPT_DIR/logs/liquidations.log
StandardError=append:$SCRIPT_DIR/logs/liquidations.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable btc-liquidations
sudo systemctl start btc-liquidations

echo "  Liquidation service installed and started"

echo ""
echo "Setup complete. Verify with:"
echo "  crontab -l                           # Check cron jobs"
echo "  sudo systemctl status btc-liquidations  # Check websocket service"
echo "  tail -f logs/system.log              # Monitor hourly runs"
