"""
Model checkpoint save/load for autoresearch.
Allows promising runs to be resumed at longer time scales.
"""

import json
import os
import torch

from config import get_checkpoint_dir


def save_checkpoint(model, optimizer, step, config_dict, metadata, path=None):
    """Save model + optimizer state for later resumption.

    Args:
        model: The compiled or raw GPT model.
        optimizer: The MuonAdamW optimizer.
        step: Current training step.
        config_dict: Model config as dict (from dataclasses.asdict).
        metadata: Dict with val_bpb, scale, branch, commit, etc.
        path: Override save path. Default: checkpoints/<commit>_<scale>.pt
    """
    ckpt_dir = get_checkpoint_dir()
    os.makedirs(ckpt_dir, exist_ok=True)

    if path is None:
        commit = metadata.get("commit", "unknown")
        scale = metadata.get("scale", "unknown")
        path = os.path.join(ckpt_dir, f"{commit}_{scale}.pt")

    # Handle torch.compile'd models
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    checkpoint = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": config_dict,
        "metadata": metadata,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
    return path


def load_checkpoint(path, device="cuda"):
    """Load a checkpoint. Returns dict with model_state, optimizer_state, step, config, metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    print(f"Checkpoint loaded from {path}")
    return checkpoint


def list_checkpoints(sort_by="val_bpb"):
    """List available checkpoints sorted by metric."""
    ckpt_dir = get_checkpoint_dir()
    if not os.path.exists(ckpt_dir):
        return []

    checkpoints = []
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(ckpt_dir, fname)
        try:
            # Load only metadata (not full weights) for listing
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            meta["path"] = path
            meta["filename"] = fname
            checkpoints.append(meta)
        except Exception:
            continue

    if sort_by == "val_bpb":
        checkpoints.sort(key=lambda c: c.get("val_bpb", float("inf")))
    elif sort_by == "time":
        checkpoints.sort(key=lambda c: os.path.getmtime(c["path"]), reverse=True)

    return checkpoints


def find_best_checkpoint():
    """Return path to checkpoint with lowest val_bpb, or None."""
    ckpts = list_checkpoints(sort_by="val_bpb")
    if ckpts:
        return ckpts[0]["path"]
    return None


if __name__ == "__main__":
    ckpts = list_checkpoints()
    if not ckpts:
        print("No checkpoints found.")
    else:
        print(f"{'val_bpb':>10}  {'scale':>8}  {'commit':>8}  path")
        print("-" * 60)
        for c in ckpts:
            print(f"{c.get('val_bpb', '?'):>10.6f}  {c.get('scale', '?'):>8}  {c.get('commit', '?'):>8}  {c['path']}")
