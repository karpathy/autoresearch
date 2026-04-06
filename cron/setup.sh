#!/usr/bin/env bash
# Setup script for autoresearch cron jobs
# Usage: ./cron/setup.sh [install|uninstall|status]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CRON_MARKER="# autoresearch cron jobs"

usage() {
    echo "Usage: $0 [install|uninstall|status]"
    echo ""
    echo "Commands:"
    echo "  install    Install cron jobs (merges with existing crontab)"
    echo "  uninstall  Remove autoresearch cron jobs from crontab"
    echo "  status     Show current autoresearch cron job status"
    echo ""
    echo "The cron jobs will:"
    echo "  - Run a daily training report at 9pm CET (weekdays)"
    echo "  - Run overnight auto-research (~100 experiments) at 11pm CET (weekdays)"
    echo "  - Clean up old logs weekly"
}

install_crons() {
    # Make scripts executable
    chmod +x "$SCRIPT_DIR/daily_report.sh"
    chmod +x "$SCRIPT_DIR/auto_research.sh"

    # Create log directory
    mkdir -p "$SCRIPT_DIR/logs"

    # Update REPO_DIR in crontab.conf to actual path
    CRONTAB_CONTENT="$(sed "s|REPO_DIR=.*|REPO_DIR=$REPO_DIR|" "$SCRIPT_DIR/crontab.conf")"

    # Check if autoresearch crons already installed
    EXISTING="$(crontab -l 2>/dev/null || true)"
    if echo "$EXISTING" | grep -q "$CRON_MARKER"; then
        echo "Autoresearch cron jobs already installed. Updating..."
        # Remove old entries and add new ones
        CLEANED="$(echo "$EXISTING" | sed "/$CRON_MARKER/,/^$/d")"
        echo "$CLEANED"$'\n'"$CRONTAB_CONTENT" | crontab -
    else
        # Append to existing crontab
        if [ -n "$EXISTING" ]; then
            echo "$EXISTING"$'\n'"$CRONTAB_CONTENT" | crontab -
        else
            echo "$CRONTAB_CONTENT" | crontab -
        fi
    fi

    echo "Cron jobs installed successfully."
    echo ""
    echo "Schedule:"
    echo "  Daily report:   9pm CET (19:00 UTC) Mon-Fri"
    echo "  Auto-research: 11pm CET (21:00 UTC) Mon-Fri (~100 experiments)"
    echo "  Log cleanup:    3am UTC Sundays"
    echo ""
    echo "Logs: $SCRIPT_DIR/logs/"
    echo ""
    echo "To verify: crontab -l"
}

uninstall_crons() {
    EXISTING="$(crontab -l 2>/dev/null || true)"
    if [ -z "$EXISTING" ]; then
        echo "No crontab found."
        return
    fi

    if echo "$EXISTING" | grep -q "$CRON_MARKER"; then
        # Remove autoresearch block
        CLEANED="$(echo "$EXISTING" | sed "/$CRON_MARKER/,/^$/d" | sed '/^$/N;/^\n$/d')"
        if [ -z "$CLEANED" ]; then
            crontab -r 2>/dev/null || true
        else
            echo "$CLEANED" | crontab -
        fi
        echo "Autoresearch cron jobs removed."
    else
        echo "No autoresearch cron jobs found in crontab."
    fi
}

show_status() {
    echo "=== Autoresearch Cron Status ==="
    echo ""

    # Check if crons are installed
    EXISTING="$(crontab -l 2>/dev/null || true)"
    if echo "$EXISTING" | grep -q "$CRON_MARKER"; then
        echo "Status: INSTALLED"
        echo ""
        echo "Active cron entries:"
        echo "$EXISTING" | grep -A1 "$CRON_MARKER" | grep -v "^#" | grep -v "^$" | grep -v "REPO_DIR\|SHELL\|PATH" || true
    else
        echo "Status: NOT INSTALLED"
        echo "Run '$0 install' to set up cron jobs."
    fi

    echo ""

    # Show recent logs if any
    if [ -d "$SCRIPT_DIR/logs" ]; then
        LOG_COUNT="$(find "$SCRIPT_DIR/logs" -name '*.log' 2>/dev/null | wc -l)"
        echo "Logs: $LOG_COUNT files in $SCRIPT_DIR/logs/"
        if [ "$LOG_COUNT" -gt 0 ]; then
            echo "Latest:"
            ls -lt "$SCRIPT_DIR/logs/"*.log 2>/dev/null | head -3 | awk '{print "  " $NF " (" $6, $7, $8 ")"}'
        fi
    else
        echo "Logs: No log directory yet"
    fi

    echo ""

    # Show results summary if available
    RESULTS_FILE="$REPO_DIR/results.tsv"
    if [ -f "$RESULTS_FILE" ] && [ "$(wc -l < "$RESULTS_FILE")" -gt 1 ]; then
        TOTAL="$(tail -n +2 "$RESULTS_FILE" | wc -l)"
        KEEPS="$(tail -n +2 "$RESULTS_FILE" | awk -F'\t' '$4=="keep"' | wc -l)"
        BEST="$(tail -n +2 "$RESULTS_FILE" | awk -F'\t' '$4=="keep" && $2+0>0 {print $2}' | sort -n | head -1)"
        echo "Results: $TOTAL experiments ($KEEPS kept)"
        echo "Best val_bpb: ${BEST:-N/A}"
    else
        echo "Results: No experiments recorded yet"
    fi
}

case "${1:-}" in
    install)   install_crons ;;
    uninstall) uninstall_crons ;;
    status)    show_status ;;
    *)         usage ;;
esac
