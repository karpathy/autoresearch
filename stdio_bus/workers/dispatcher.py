#!/usr/bin/env python3
"""
GPU dispatcher worker for stdio_bus. JSON-RPC 2.0 over NDJSON.
Distributes train.py experiments across GPUs.

Methods:
    status  - GPU pool status
    sync    - Full state sync
    history - Experiment history
    run     - Run experiment
"""
# Worker implementation: ../dispatcher.py (legacy location)
# Config: configs/dispatcher.json

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    exec(open(Path(__file__).parent.parent / "dispatcher.py").read())
