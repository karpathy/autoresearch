"""
Autoresearch pretraining entrypoint.

Usage:
    uv run train.py
    uv run train.py --benchmark-steps 5
    uv run train.py --experiment-profile baseline
    uv run train.py --experiment-profile mfu50 --benchmark-steps 20
"""

import os
import sys

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from autoresearch_trainer import main


if __name__ == "__main__":
    raise SystemExit(main())
