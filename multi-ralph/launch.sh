#!/bin/bash
# Multi-Ralph: Launch N agents sharing a single GPU
# All agents run concurrently on the same GPU (each ~7-13GB VRAM).
#
# Usage: ./multi-ralph/launch.sh [num_agents]
#   num_agents defaults to 3
#
# Prerequisites:
#   1. Repo cloned: git clone <repo> ~/autoresearch && cd ~/autoresearch
#   2. Dependencies: uv sync
#   3. Data: uv run prepare.py
#   4. Claude Code installed and authenticated

set -e

NUM_AGENTS="${1:-3}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

SHARED_DIR="$REPO_DIR/multi-ralph"
WORKTREE_DIR="$REPO_DIR/worktrees"

echo "=== Multi-Ralph Launch (single GPU, $NUM_AGENTS agents) ==="
echo "Repo:      $REPO_DIR"
echo "Shared:    $SHARED_DIR"
echo "Worktrees: $WORKTREE_DIR"
echo "Agents:    $NUM_AGENTS (all sharing GPU 0)"
echo ""

# --- Initialize shared state ---

mkdir -p "$SHARED_DIR/queue" "$SHARED_DIR/active" "$SHARED_DIR/done" "$SHARED_DIR/best"
rm -f "$SHARED_DIR/queue/"*.md "$SHARED_DIR/active/"*.md

# Snapshot current train.py as starting point
cp "$REPO_DIR/train.py" "$SHARED_DIR/best/train.py"

# Initialize results.tsv if needed
if [ ! -s "$SHARED_DIR/results.tsv" ] || [ "$(wc -l < "$SHARED_DIR/results.tsv")" -le 1 ]; then
    printf 'commit\tval_bpb\tmemory_gb\tstatus\tdescription\n' > "$SHARED_DIR/results.tsv"
fi

# --- Create worktrees ---

mkdir -p "$WORKTREE_DIR"

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    BRANCH="autoresearch/multi/agent${AGENT}"
    TREE="$WORKTREE_DIR/agent${AGENT}"

    # Clean up existing
    if [ -d "$TREE" ]; then
        git worktree remove --force "$TREE" 2>/dev/null || rm -rf "$TREE"
    fi
    git branch -D "$BRANCH" 2>/dev/null || true

    echo "Creating worktree agent${AGENT}..."
    git worktree add -b "$BRANCH" "$TREE" HEAD

    # Symlink shared dir into worktree
    ln -sfn "$SHARED_DIR" "$TREE/multi-ralph"
    touch "$TREE/run.log"
done

echo ""
echo "=== Launching $NUM_AGENTS agents ==="
echo ""

# --- Write agent runner scripts ---

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    TREE="$WORKTREE_DIR/agent${AGENT}"

    if [ "$AGENT" -eq 0 ]; then
        PROMPT="You are agent 0 in a multi-ralph experiment. Your working directory is $TREE.

You are the FIRST agent. Your job:
1. Read multi-ralph/program-multi.md for the full protocol
2. Read train.py and prepare.py to understand the codebase
3. Run the baseline: uv run train.py > run.log 2>&1
4. Record baseline in multi-ralph/results.tsv
5. Write multi-ralph/strategy.md with baseline result
6. Copy train.py to multi-ralph/best/train.py
7. Generate $((NUM_AGENTS - 1)) diverse experiment files in multi-ralph/queue/ (001.md through $(printf '%03d' $((NUM_AGENTS - 1))).md)
   - Diverse categories: LR scaling, schedule, architecture, RoPE, window, optimizer, wild card
   - Each file has: experiment title, rationale, specific code changes, expected outcome
8. Then pick your own experiment and enter the main loop

IMPORTANT:
- CUDA_VISIBLE_DEVICES=0 is set. All $NUM_AGENTS agents share this GPU concurrently.
- A100 SXM4 40GB — BF16 is native, no dtype hacks needed.
- ***CRITICAL***: DEVICE_BATCH_SIZE MUST be 32. NEVER change it. 3 processes run concurrently
  at ~12GB each = ~36GB. Batch 64 = 25GB each = OOM with 3 processes. This is non-negotiable.
  The batch=32 baseline IS your real baseline. Optimize everything else (LR, schedule, arch, etc).
- Do NOT test depth > 10 (VRAM). Do NOT change DEVICE_BATCH_SIZE or TOTAL_BATCH_SIZE.
- Always start from: cp multi-ralph/best/train.py train.py
- After each experiment, check if multi-ralph/queue/ is empty. If so, become coordinator.
- Append results to multi-ralph/results.tsv (tab-separated, use >>)
- Run as many experiments as you can. Do not stop. Do not ask questions."
    else
        PROMPT="You are agent $AGENT in a multi-ralph experiment. Your working directory is $TREE.

Read multi-ralph/program-multi.md for the full protocol. Then:
1. Pick the lowest-numbered .md file from multi-ralph/queue/
2. Move it to multi-ralph/active/agent${AGENT}.md
3. cp multi-ralph/best/train.py train.py
4. Apply the changes described in the task file
5. Run: uv run train.py > run.log 2>&1
6. Record results in multi-ralph/results.tsv and multi-ralph/done/
7. If you beat the best in strategy.md, update multi-ralph/best/train.py and multi-ralph/strategy.md
8. If not, cp multi-ralph/best/train.py train.py
9. rm multi-ralph/active/agent${AGENT}.md
10. If queue is empty, become coordinator: read ALL results, generate next batch of 2-4 tasks in queue/
11. Loop — run as many experiments as you can

IMPORTANT:
- CUDA_VISIBLE_DEVICES=0 is set. All $NUM_AGENTS agents share this GPU concurrently.
- A100 SXM4 40GB — BF16 is native, no dtype hacks needed.
- ***CRITICAL***: DEVICE_BATCH_SIZE MUST be 32. NEVER change it. 3 processes run concurrently
  at ~12GB each = ~36GB. Batch 64 = 25GB each = OOM with 3 processes. This is non-negotiable.
  Optimize LR, schedule, architecture, RoPE, etc — but NOT batch size.
- Do NOT test depth > 10 (VRAM). Do NOT change DEVICE_BATCH_SIZE or TOTAL_BATCH_SIZE.
- Always start from: cp multi-ralph/best/train.py train.py
- To claim a task: mv multi-ralph/queue/NNN.md multi-ralph/active/agent${AGENT}.md
- Append results with >> (never overwrite results.tsv)
- Do not stop. Do not ask questions."
    fi

    # Write the prompt to a file so we can pass it cleanly
    echo "$PROMPT" > "$TREE/.agent-prompt.txt"

    # Write the runner script that loops forever
    cat > "$TREE/.run-agent.sh" << 'RUNNER_EOF'
#!/bin/bash
AGENT_ID=$1
TREE_DIR=$2
cd "$TREE_DIR"
export CUDA_VISIBLE_DEVICES=0

ROUND=0
while true; do
    ROUND=$((ROUND + 1))
    echo "$(date): agent $AGENT_ID starting round $ROUND" >> agent.log

    if [ "$ROUND" -eq 1 ] && [ "$AGENT_ID" -ne 0 ]; then
        # Non-zero agents wait for initial tasks on first round
        echo "Agent $AGENT_ID waiting for initial tasks..."
        while [ -z "$(ls multi-ralph/queue/*.md 2>/dev/null)" ]; do
            sleep 10
        done
    fi

    # Run claude with the prompt
    claude -p "$(cat .agent-prompt.txt)

This is round $ROUND. Check multi-ralph/results.tsv and multi-ralph/strategy.md for latest state from all agents. Continue the experiment loop." \
        --dangerously-skip-permissions \
        --max-turns 200 \
        2>> agent.log || true

    echo "$(date): agent $AGENT_ID claude exited round $ROUND, restarting in 5s..." >> agent.log
    sleep 5
done
RUNNER_EOF
    chmod +x "$TREE/.run-agent.sh"
done

# --- Launch screen sessions ---

for AGENT in $(seq 0 $((NUM_AGENTS - 1))); do
    TREE="$WORKTREE_DIR/agent${AGENT}"
    SESSION="ralph-agent${AGENT}"

    screen -S "$SESSION" -X quit 2>/dev/null || true
    screen -dmS "$SESSION" "$TREE/.run-agent.sh" "$AGENT" "$TREE"

    echo "  -> Agent $AGENT: screen=$SESSION worktree=$TREE"
done

echo ""
echo "=== All $NUM_AGENTS agents launched (sharing GPU 0) ==="
echo ""
echo "Monitor:"
echo "  screen -ls                             # list sessions"
echo "  screen -r ralph-agent0                 # attach (Ctrl+A D to detach)"
echo "  tail -f worktrees/agent*/run.log       # training logs"
echo "  tail -f worktrees/agent*/agent.log     # agent restart logs"
echo "  cat multi-ralph/results.tsv            # all results"
echo "  cat multi-ralph/strategy.md            # search strategy"
echo "  ls multi-ralph/queue/                  # pending tasks"
echo "  ls multi-ralph/active/                 # running now"
echo "  ls multi-ralph/done/                   # completed"
echo ""
echo "Dashboard:"
echo "  watch -n 30 'echo \"=== BEST ===\"; head -5 multi-ralph/strategy.md; echo; echo \"=== RESULTS ===\"; cat multi-ralph/results.tsv; echo; echo \"=== QUEUE ===\"; ls multi-ralph/queue/ 2>/dev/null; echo; echo \"=== ACTIVE ===\"; ls multi-ralph/active/ 2>/dev/null'"
echo ""
echo "GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Stop:"
echo "  for i in \$(seq 0 $((NUM_AGENTS - 1))); do screen -S ralph-agent\$i -X quit; done"
echo ""
echo "Cleanup:"
echo "  for i in \$(seq 0 $((NUM_AGENTS - 1))); do git worktree remove --force worktrees/agent\$i; done"
