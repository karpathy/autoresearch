#!/usr/bin/env bash
set -euo pipefail

# Scoring script — outputs JSON on the last line.
# The metric key must match score.name in problem.yaml.
#
# Example: if problem.yaml has score.name: cost, output:
#   {"cost": 42.0}
#
# You can include additional metrics — they will be recorded
# but only the named metric is used for accept/reject decisions.
#   {"cost": 42.0, "iterations": 100, "time_s": 1.5}

echo '{"{{metric}}": 0.0}'
