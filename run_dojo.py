"""
DOJO mode runner: adversarial testing of trained models.
Loads a checkpoint from train.py, runs test_protocol.py against it,
and reports the robustness gap.

Usage: uv run run_dojo.py
"""

import gc
import json
import math
import os
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from prepare import MAX_SEQ_LEN, Tokenizer, get_token_bytes, make_dataloader
from train import GPT, GPTConfig

CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "checkpoints")
DOJO_TIME_BUDGET = 300  # 5 minutes for adversarial testing


def load_checkpoint():
    """Load trained model from checkpoint directory."""
    config_path = os.path.join(CHECKPOINT_DIR, "config.json")
    weights_path = os.path.join(CHECKPOINT_DIR, "latest.npz")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print("No checkpoint found. Train a model first: uv run train.py")
        raise SystemExit(1)

    with open(config_path) as f:
        config_dict = json.load(f)

    baseline_bpb = config_dict.pop("val_bpb", None)

    config = GPTConfig(**config_dict)
    model = GPT(config)

    # Load weights
    saved = mx.load(weights_path)
    # Build nested structure from flat keys
    model_params = dict(tree_flatten(model.parameters()))
    for key in model_params:
        if key in saved:
            _set_param(model, key, saved[key])
    mx.eval(model.parameters())

    return model, config, baseline_bpb


def _set_param(model, path, value):
    """Set a parameter on the model by dot-separated path."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def get_peak_memory_mb():
    return mx.get_peak_memory() / 1024 / 1024


def main():
    t_start = time.time()

    print("Loading checkpoint...")
    model, config, baseline_bpb = load_checkpoint()
    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    t_load = time.time()
    print(f"Model loaded in {t_load - t_start:.1f}s ({num_params / 1e6:.1f}M params)")

    if baseline_bpb is not None:
        print(f"Baseline val_bpb: {baseline_bpb:.6f}")
    else:
        print("No baseline val_bpb in checkpoint, computing...")
        tokenizer = Tokenizer.from_directory()
        baseline_bpb = _compute_baseline_bpb(model, tokenizer)
        print(f"Baseline val_bpb: {baseline_bpb:.6f}")

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_directory()

    print("Running adversarial tests...")
    import test_protocol
    results = test_protocol.run_all_tests(model, tokenizer, baseline_bpb, DOJO_TIME_BUDGET)

    t_done = time.time()
    peak_vram_mb = get_peak_memory_mb()

    # Find worst result
    if results:
        worst = max(results, key=lambda r: r.robustness_gap)
        worst_case_bpb = worst.adversarial_bpb
        robustness_gap = worst.robustness_gap
        worst_test = worst.test_name
    else:
        worst_case_bpb = baseline_bpb
        robustness_gap = 0.0
        worst_test = "none"

    # Print results
    print()
    print("---")
    print(f"baseline_bpb:     {baseline_bpb:.6f}")
    print(f"worst_case_bpb:   {worst_case_bpb:.6f}")
    print(f"robustness_gap:   {robustness_gap:.6f}")
    print(f"worst_test:       {worst_test}")
    print(f"num_tests:        {len(results)}")
    print("tests_summary:")
    for r in results:
        print(f"  {r.test_name:30s} gap={r.robustness_gap:.3f}  ({r.description})")
    print(f"total_seconds:    {t_done - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")


def _compute_baseline_bpb(model, tokenizer):
    """Fallback: compute baseline BPB from validation data."""
    from prepare import evaluate_bpb
    return evaluate_bpb(model, tokenizer, 256)


if __name__ == "__main__":
    main()
