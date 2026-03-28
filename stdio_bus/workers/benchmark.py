#!/usr/bin/env python3
"""
Benchmark worker for stdio_bus. JSON-RPC 2.0 over NDJSON.
Runs MPS vs MLX benchmarks on Apple Silicon.

Methods:
    benchmark.status  - Current state
    benchmark.run     - Run benchmark (MPS + MLX)
    benchmark.history - Get history
    benchmark.analyze - AI analysis
    benchmark.plot    - Generate plots
"""
# Worker implementation in project root: benchmark.py
# This file documents the worker interface.
# Config: configs/benchmark.json

from pathlib import Path
import sys

# Import from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from benchmark import main
    main()
