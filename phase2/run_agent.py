"""
Phase 1: Agent-driven autonomous research (Karpathy method)
Agent makes creative decisions, modifies train.py, runs experiments
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Note: This is a placeholder/launcher script
# The actual agent logic will be executed by Claude following program.md

def setup_agent_experiment(output_dir, n_trials):
    """Set up directory structure for agent experiments."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create results.tsv with header
    results_file = output_dir / "results.tsv"
    if not results_file.exists():
        with open(results_file, 'w') as f:
            f.write("trial\tval_bpb\tmemory_gb\twall_time_s\tdescription\n")

    # Create metadata
    metadata = {
        'method': 'agent',
        'n_trials_target': n_trials,
        'start_time': datetime.now().isoformat(),
        'output_dir': str(output_dir),
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create instructions for agent
    instructions = f"""
# Agent Research Instructions (Phase 1A)

You are running the **Agent** method for Phase 1 of the optimization comparison.

## Goal
Run {n_trials} autonomous research trials, making creative decisions to improve val_bpb.

## How to Proceed

1. **Read the context**:
   - `train.py` - the file you'll modify
   - `EXPERIMENT_PLAN.md` - overall experiment design
   - Baseline val_bpb: 1.451316

2. **For each trial**:
   - Think creatively about what to try
   - Modify `train.py` with your idea
   - Run: `uv run train.py > {output_dir}/trial_{{N}}.log 2>&1`
   - Extract val_bpb from log: `grep "^val_bpb:" trial_{{N}}.log`
   - Log to `{output_dir}/results.tsv`:
     ```
     {{trial}}\\t{{val_bpb}}\\t{{memory_gb}}\\t{{wall_time_s}}\\t{{description}}
     ```
   - If improved: keep changes
   - If worse: revert changes (git restore train.py or manual undo)

3. **Ideas to try** (be creative!):
   - Hyperparameters: LRs, batch sizes, warmup/warmdown
   - Architecture: depth, width, attention patterns
   - Optimizer: momentum, weight decay schedules
   - Novel ideas: skip connections, different activations, etc.

4. **Track your reasoning**:
   - Save your thoughts in `{output_dir}/agent_log.md`
   - What did you try and why?
   - What worked and what didn't?

5. **Run {n_trials} trials total** (not necessarily {n_trials} improvements - try things even if they might fail!)

## Important Notes

- You have full freedom to modify any part of train.py
- Be systematic but creative
- Document your reasoning
- Some trials will fail - that's okay and expected!
- Time budget is fixed at 5 minutes per trial

## Current Status

- Trials completed: 0/{n_trials}
- Best val_bpb so far: 1.451316 (baseline)
- Output directory: {output_dir}

Good luck! Start with trial 1.
"""

    with open(output_dir / "agent_instructions.md", 'w') as f:
        f.write(instructions)

    print(f"✅ Agent experiment setup complete: {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Read: {output_dir}/agent_instructions.md")
    print(f"2. Start experimenting!")
    print(f"3. Log results to: {output_dir}/results.tsv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Agent-driven research")
    parser.add_argument(
        '--n_trials',
        type=int,
        default=10,
        help='Number of trials to run'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/phase1/agent',
        help='Output directory'
    )
    args = parser.parse_args()

    setup_agent_experiment(args.output_dir, args.n_trials)
