#!/usr/bin/env python3
# Requirements: torch, safetensors
#
# Generates initial weights matching the Python reference (original/train.py)
# exactly, including RNG call order. Saves as safetensors with tensor names
# matching the Rust engine's load_checkpoint().
#
# Usage:
#   python3 generate_init_weights.py                     # depth=14 (default)
#   python3 generate_init_weights.py --depth 8           # depth=8
#   python3 generate_init_weights.py --depth 14 -o init_weights.safetensors

import argparse
import math
from collections import OrderedDict

import torch
from safetensors.torch import save_file


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Matches original/train.py: layer_idx % 2 == (n_layer - 1) % 2"""
    return layer_idx % 2 == (n_layer - 1) % 2


def generate_weights(depth: int, vocab_size: int, seq_len: int,
                     aspect_ratio: int, head_dim: int,
                     window_pattern: str, seed: int) -> OrderedDict:
    """Generate initial weights matching the Python reference RNG order exactly."""

    # Compute model dimensions (same as build_model_config in train.py)
    base_dim = depth * aspect_ratio
    d_model = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = d_model // head_dim
    n_kv_head = n_head
    mlp_dim = 4 * d_model
    kv_dim = n_kv_head * head_dim  # == d_model when n_kv_head == n_head
    ve_gate_ch = 32

    print(f"depth={depth}, d_model={d_model}, n_head={n_head}, n_kv_head={n_kv_head}")
    print(f"mlp_dim={mlp_dim}, kv_dim={kv_dim}, vocab_size={vocab_size}")

    # Identify VE layers (insertion order matches Python's ModuleDict)
    ve_layers = [i for i in range(depth) if has_ve(i, depth)]
    print(f"VE layers: {ve_layers}")

    # Set seed to match Python reference
    torch.manual_seed(seed)

    # The Python reference does:
    #   with torch.device("meta"):
    #       model = GPT(config)
    #   model.to_empty(device=device)
    #   model.init_weights()
    #
    # The "meta" construction + to_empty does NOT consume RNG.
    # All RNG calls happen inside init_weights() in this exact order:

    s = math.sqrt(3) * d_model ** -0.5

    # 1. wte: Normal(0, 1), shape [vocab, d_model]
    wte = torch.nn.init.normal_(torch.empty(vocab_size, d_model), mean=0.0, std=1.0)

    # 2. lm_head: Normal(0, 0.001), shape [vocab, d_model]
    lm_head = torch.nn.init.normal_(torch.empty(vocab_size, d_model), mean=0.0, std=0.001)

    # 3. Per-block weights (layer 0 .. depth-1)
    layer_weights = {}
    for i in range(depth):
        lw = {}
        lw["c_q"] = torch.nn.init.uniform_(torch.empty(d_model, d_model), -s, s)
        lw["c_k"] = torch.nn.init.uniform_(torch.empty(kv_dim, d_model), -s, s)
        lw["c_v"] = torch.nn.init.uniform_(torch.empty(kv_dim, d_model), -s, s)
        lw["c_proj"] = torch.zeros(d_model, d_model)  # no RNG
        lw["c_fc"] = torch.nn.init.uniform_(torch.empty(mlp_dim, d_model), -s, s)
        lw["mlp_proj"] = torch.zeros(d_model, mlp_dim)  # no RNG
        layer_weights[i] = lw

    # 4. resid_lambdas: fill_(1.0) — no RNG
    resid_lambdas = torch.ones(depth)

    # 5. x0_lambdas: fill_(0.1) — no RNG
    x0_lambdas = torch.full((depth,), 0.1)

    # 6. Value embeddings: uniform_(-s, s) in ModuleDict insertion order
    ve_weights = {}
    for i in ve_layers:
        ve_weights[i] = torch.nn.init.uniform_(torch.empty(vocab_size, kv_dim), -s, s)

    # 7. VE gate weights: zeros — no RNG
    ve_gates = {}
    for i in range(depth):
        if has_ve(i, depth):
            ve_gates[i] = torch.zeros(n_kv_head, ve_gate_ch)

    # 8. Rotary embeddings recomputed (no RNG, deterministic)
    # 9. Cast embeddings to bf16 (wte and ve)

    # Now cast to bf16 matching Python's autocast behavior:
    # The Python code casts wte and value_embeds to bf16 at the end of init_weights.
    # All other weights stay f32 on the Python side but run under autocast (bf16).
    # The Rust engine stores everything as bf16, so we cast all to bf16.

    # Build safetensors dict with engine tensor names
    tensors = OrderedDict()

    # wte — Python casts to bf16
    tensors["wte.weight"] = wte.to(torch.bfloat16)

    # lm_head — stays f32 in Python, but engine stores bf16
    tensors["lm_head.weight"] = lm_head.to(torch.bfloat16)

    # resid_lambdas — f32 scalars in Python, engine stores bf16
    tensors["resid_lambdas"] = resid_lambdas.to(torch.bfloat16)

    # x0_lambdas — f32 scalars in Python, engine stores bf16
    tensors["x0_lambdas"] = x0_lambdas.to(torch.bfloat16)

    # Per-layer weights
    for i in range(depth):
        prefix = f"h.{i}"
        lw = layer_weights[i]
        tensors[f"{prefix}.attn.c_q.weight"] = lw["c_q"].to(torch.bfloat16)
        tensors[f"{prefix}.attn.c_k.weight"] = lw["c_k"].to(torch.bfloat16)
        tensors[f"{prefix}.attn.c_v.weight"] = lw["c_v"].to(torch.bfloat16)
        tensors[f"{prefix}.attn.c_proj.weight"] = lw["c_proj"].to(torch.bfloat16)
        tensors[f"{prefix}.mlp.c_fc.weight"] = lw["c_fc"].to(torch.bfloat16)
        tensors[f"{prefix}.mlp.c_proj.weight"] = lw["mlp_proj"].to(torch.bfloat16)

    # Value embeddings
    for i in ve_layers:
        tensors[f"ve.{i}.weight"] = ve_weights[i].to(torch.bfloat16)

    # VE gate weights
    for i in ve_layers:
        tensors[f"h.{i}.attn.ve_gate.weight"] = ve_gates[i].to(torch.bfloat16)

    # Summary
    total_params = sum(t.numel() for t in tensors.values())
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"Total parameters: {total_params:,} ({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"Tensors: {len(tensors)}")

    return tensors


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial weights matching Python reference exactly"
    )
    parser.add_argument("--depth", type=int, default=14,
                        help="Number of transformer layers (default: 14)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Vocabulary size (default: auto from tokenizer or 32768)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--aspect-ratio", type=int, default=64,
                        help="d_model = depth * aspect_ratio, rounded up to head_dim (default: 64)")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension (default: 128)")
    parser.add_argument("--window-pattern", type=str, default="SSSL",
                        help="Sliding window pattern (default: SSSL)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (default: init_weights_d{depth}.safetensors)")
    args = parser.parse_args()

    # Try to read vocab size from tokenizer if not specified
    vocab_size = args.vocab_size
    if vocab_size is None:
        try:
            import sys
            sys.path.insert(0, "original")
            from prepare import Tokenizer
            tok = Tokenizer.from_directory()
            vocab_size = tok.get_vocab_size()
            print(f"Loaded vocab_size={vocab_size} from tokenizer")
        except Exception:
            vocab_size = 32768
            print(f"Could not load tokenizer, using default vocab_size={vocab_size}")

    tensors = generate_weights(
        depth=args.depth,
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        window_pattern=args.window_pattern,
        seed=args.seed,
    )

    output_path = args.output or f"init_weights_d{args.depth}.safetensors"
    save_file(tensors, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
