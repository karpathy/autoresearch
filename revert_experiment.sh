#!/bin/bash
# Revert the last commit (used when an experiment didn't improve val_bpb).
# Usage: ./revert_experiment.sh

echo "Reverting last commit..."
git reset --hard HEAD~1
echo "Reverted. Current HEAD:"
git log --oneline -1
