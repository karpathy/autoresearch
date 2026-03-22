"""
Entry point: import strategy, run backtest, print results.

DO NOT MODIFY — the agent only edits strategy.py.

Usage:
    uv run backtest.py > run.log 2>&1
"""

import sys
import time

from prepare import evaluate, TIME_BUDGET

t0 = time.time()

import strategy

metrics = evaluate(strategy.Strategy)

elapsed = time.time() - t0
print(f"total_seconds: {elapsed:.1f}")

if elapsed > TIME_BUDGET * 2:
    print("TIMEOUT: exceeded time budget")
    sys.exit(1)
