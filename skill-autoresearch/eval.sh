#!/bin/bash
# eval.sh — Run all test scenarios through the campaign-analysis skill and grade them
#
# Usage: ./eval.sh [scenario_number]
#   No args: run all scenarios
#   With arg: run only that scenario (e.g., ./eval.sh 3)
#
# Prerequisites:
#   - Claude CLI (claude) must be installed and authenticated
#   - The campaign-analysis skill must be installed
#
# Output:
#   - Individual reports in eval-results/scenario-N-output.md
#   - Grading results in eval-results/scenario-N-grade.md
#   - Summary in eval-results/summary.tsv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$SCRIPT_DIR/test-scenarios"
EVAL_CRITERIA="$SCRIPT_DIR/eval-criteria.md"
RESULTS_DIR="$SCRIPT_DIR/eval-results"
SKILL_PATH="$HOME/.claude/plugins/marketplaces/local-desktop-app-uploads/campaign-analyst/skills/campaign-analysis/SKILL.md"

mkdir -p "$RESULTS_DIR"

# Determine which scenarios to run
if [ $# -gt 0 ]; then
    SCENARIOS=("$SCENARIOS_DIR/scenario-$1-"*.md)
else
    SCENARIOS=("$SCENARIOS_DIR"/scenario-*.md)
fi

echo "=== Campaign Analysis Skill Eval ==="
echo "Scenarios: ${#SCENARIOS[@]}"
echo "Results dir: $RESULTS_DIR"
echo ""

# Initialize summary
SUMMARY_FILE="$RESULTS_DIR/summary.tsv"
echo -e "scenario\tpass\tfail\tna\ttotal_applicable\tscore\ttimestamp" > "$SUMMARY_FILE"

for SCENARIO_FILE in "${SCENARIOS[@]}"; do
    SCENARIO_NAME=$(basename "$SCENARIO_FILE" .md)
    echo "--- Running: $SCENARIO_NAME ---"

    OUTPUT_FILE="$RESULTS_DIR/${SCENARIO_NAME}-output.md"
    GRADE_FILE="$RESULTS_DIR/${SCENARIO_NAME}-grade.md"

    # Step 1: Run the scenario through Claude with the skill context
    # Feed the skill instructions + scenario data, ask it to produce the analysis
    echo "  [1/2] Generating analysis output..."
    claude -p "$(cat <<PROMPT
You are testing a campaign analysis skill. Below are the skill instructions, followed by test scenario data.

Execute the skill instructions against the provided data. Produce:
1. Your analytical reasoning (show your work for each layer)
2. The final HTML report

=== SKILL INSTRUCTIONS ===
$(cat "$SKILL_PATH")

=== SCENARIO DATA ===
$(cat "$SCENARIO_FILE" | sed '/^## Expected Analysis Outcomes/,$ d')

Produce the full analysis now. Show your reasoning for each layer, then output the complete HTML report.
PROMPT
)" > "$OUTPUT_FILE" 2>/dev/null

    # Step 2: Grade the output against eval criteria
    echo "  [2/2] Grading output..."
    claude -p "$(cat <<PROMPT
You are an eval grader. Grade the following skill output against the eval criteria and expected outcomes.

For each criterion, output exactly one line in this format:
CRITERION_ID: PASS|FAIL|N/A — brief reason

After all criteria, output a summary line:
SUMMARY: X pass, Y fail, Z n/a, SCORE%

=== EVAL CRITERIA ===
$(cat "$EVAL_CRITERIA")

=== EXPECTED OUTCOMES (ground truth) ===
$(grep -A 100 "## Expected Analysis Outcomes" "$SCENARIO_FILE")

=== SKILL OUTPUT TO GRADE ===
$(cat "$OUTPUT_FILE")

Grade every criterion now. Be strict — if the output is ambiguous or partially correct, grade FAIL.
PROMPT
)" > "$GRADE_FILE" 2>/dev/null

    # Extract score from grade file
    SCORE_LINE=$(grep "SUMMARY:.*pass.*fail" "$GRADE_FILE" 2>/dev/null | tail -1 || echo "SUMMARY: 0 pass, 0 fail, 0 n/a")
    PASS_COUNT=$(echo "$SCORE_LINE" | awk -F'pass' '{print $1}' | grep -oE '[0-9]+' | tail -1)
    PASS_COUNT=${PASS_COUNT:-0}
    FAIL_COUNT=$(echo "$SCORE_LINE" | awk -F'fail' '{print $1}' | awk -F'pass' '{print $2}' | grep -oE '[0-9]+' | tail -1)
    FAIL_COUNT=${FAIL_COUNT:-0}
    NA_COUNT=$(echo "$SCORE_LINE" | awk -F'n/a' '{print $1}' | awk -F'fail' '{print $2}' | grep -oE '[0-9]+' | tail -1)
    NA_COUNT=${NA_COUNT:-0}
    TOTAL=$((PASS_COUNT + FAIL_COUNT))
    if [ "$TOTAL" -gt 0 ]; then
        SCORE=$(echo "scale=1; $PASS_COUNT * 100 / $TOTAL" | bc)
    else
        SCORE="0.0"
    fi

    echo -e "  Score: ${SCORE}% ($PASS_COUNT pass, $FAIL_COUNT fail, $NA_COUNT n/a)"
    echo -e "${SCENARIO_NAME}\t${PASS_COUNT}\t${FAIL_COUNT}\t${NA_COUNT}\t${TOTAL}\t${SCORE}%\t$(date -Iseconds)" >> "$SUMMARY_FILE"
    echo ""
done

echo "=== Eval Complete ==="
echo "Summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE" | column -t -s $'\t'
