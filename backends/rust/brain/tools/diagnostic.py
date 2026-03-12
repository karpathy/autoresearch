"""
Oracle diagnostic script: runs Python reference for N steps and dumps
intermediate values for comparison with the Rust engine.

Outputs:
  - oracle_weights.safetensors: initialized weights in engine naming convention
  - oracle_batches.bin: first N batches of (input_ids, targets) as raw u16
  - oracle_diagnostics.jsonl: per-step norms for every layer

Usage: python3 oracle/diagnostic.py [--steps 3]
"""

import os
import sys
import json
import struct
import time

import torch
import torch.nn.functional as F
import numpy as np

# Add original/ to path so we can import prepare
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'original'))
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, get_token_bytes

# ---------------------------------------------------------------------------
# Model (copy from original/train.py, no torch.compile, no FA3 dependency)
# We use pure-PyTorch attention for the oracle to avoid FA version issues.
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512
    window_pattern: str = "SSSL"

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Store norms for diagnostics (set by forward hooks externally)
        self._diag_q_norm = q.float().norm().item()
        self._diag_k_norm = k.float().norm().item()
        self._diag_v_norm = v.float().norm().item()

        # Pure PyTorch scaled dot-product attention (matches FA semantics)
        # Reshape: [B, T, H, D] -> [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's SDPA with causal mask + sliding window
        # For sliding window: create custom mask
        ws_left = window_size[0]
        if ws_left >= T:
            # Full causal
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Sliding window causal: build mask
            # mask[i,j] = True where j <= i and j >= i - ws_left
            row_idx = torch.arange(T, device=q.device).unsqueeze(1)  # [T, 1]
            col_idx = torch.arange(T, device=q.device).unsqueeze(0)  # [1, T]
            mask = (col_idx <= row_idx) & (col_idx >= row_idx - ws_left)
            # Expand for batch and heads: [1, 1, T, T]
            mask = mask.unsqueeze(0).unsqueeze(0)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        self._diag_attn_out_norm = y.float().norm().item()

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        self._diag_h_pre_act_norm = x.float().norm().item()
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.softcap = 15.0

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, x, y):
        B, T = x.size()
        x = self.transformer.wte(x)
        x = norm(x)
        cos_sin = (self.cos[:, :T, :, :], self.sin[:, :T, :, :])
        x0 = x.clone()
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_emb = self.value_embeds[str(i)](y if False else x.new_zeros(1).long()) if str(i) in self.value_embeds else None
            # Actually need input_ids for VE, not y. Store input_ids.
            x = block(x, ve_emb, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)
        logits = logits.float()
        logits = self.softcap * torch.tanh(logits / self.softcap)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss


def forward_with_diagnostics(model, input_ids, targets):
    """Run forward pass collecting per-layer diagnostics."""
    B, T = input_ids.size()
    config = model.config

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = model.transformer.wte(input_ids)
        x = norm(x)
        cos_sin = (model.cos[:, :T, :, :], model.sin[:, :T, :, :])
        x0 = x.clone()

        diag = {"per_layer": []}

        for i, block in enumerate(model.transformer.h):
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0

            # VE embedding
            ve_emb = None
            if str(i) in model.value_embeds:
                ve_emb = model.value_embeds[str(i)](input_ids)

            x = block(x, ve_emb, cos_sin, model.window_sizes[i])

            layer_diag = {
                "q_norm": block.attn._diag_q_norm,
                "k_norm": block.attn._diag_k_norm,
                "v_norm": block.attn._diag_v_norm,
                "attn_out_norm": block.attn._diag_attn_out_norm,
                "mlp_h_pre_act_norm": block.mlp._diag_h_pre_act_norm,
            }
            diag["per_layer"].append(layer_diag)

        x = norm(x)
        logits = model.lm_head(x)

    # Softcap + CE in f32
    logits = logits.float()
    logits = model.softcap * torch.tanh(logits / model.softcap)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    diag["loss"] = loss.item()
    return loss, diag


def export_weights_safetensors(model, path):
    """Export model weights in the engine's naming convention."""
    from safetensors.torch import save_file

    tensors = {}
    # All weights stored as bf16 to match engine
    tensors["wte.weight"] = model.transformer.wte.weight.data.bfloat16()
    tensors["lm_head.weight"] = model.lm_head.weight.data.bfloat16()
    tensors["resid_lambdas"] = model.resid_lambdas.data.bfloat16()
    tensors["x0_lambdas"] = model.x0_lambdas.data.bfloat16()

    for i in range(model.config.n_layer):
        block = model.transformer.h[i]
        prefix = f"h.{i}"
        tensors[f"{prefix}.attn.c_q.weight"] = block.attn.c_q.weight.data.bfloat16()
        tensors[f"{prefix}.attn.c_k.weight"] = block.attn.c_k.weight.data.bfloat16()
        tensors[f"{prefix}.attn.c_v.weight"] = block.attn.c_v.weight.data.bfloat16()
        tensors[f"{prefix}.attn.c_proj.weight"] = block.attn.c_proj.weight.data.bfloat16()
        tensors[f"{prefix}.mlp.c_fc.weight"] = block.mlp.c_fc.weight.data.bfloat16()
        tensors[f"{prefix}.mlp.c_proj.weight"] = block.mlp.c_proj.weight.data.bfloat16()
        if block.attn.ve_gate is not None:
            tensors[f"{prefix}.attn.ve_gate.weight"] = block.attn.ve_gate.weight.data.bfloat16()
        if str(i) in model.value_embeds:
            tensors[f"ve.{i}.weight"] = model.value_embeds[str(i)].weight.data.bfloat16()

    save_file(tensors, path)
    print(f"[oracle] weights saved to {path} ({len(tensors)} tensors)")


def export_batches(dataloader, n_batches, path):
    """Export first n_batches of (input_ids, targets) as raw binary.
    Format: [n_batches: u32][B: u32][T: u32] then n_batches * 2 * B * T u16 values."""
    batches = []
    for _ in range(n_batches):
        x, y, _epoch = next(dataloader)
        batches.append((x.cpu(), y.cpu()))

    B, T = batches[0][0].shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', n_batches, B, T))
        for inp, tgt in batches:
            f.write(inp.numpy().astype(np.uint16).tobytes())
            f.write(tgt.numpy().astype(np.uint16).tobytes())

    print(f"[oracle] {n_batches} batches saved to {path} (B={B}, T={T})")
    return batches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="/root/.cache/autoresearch/oracle")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Build model (matching train.py exactly)
    DEPTH = 8
    ASPECT_RATIO = 64
    HEAD_DIM = 128
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    base_dim = DEPTH * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM

    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=DEPTH, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    )
    print(f"Config: n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}, vocab={config.vocab_size}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    # Export initial weights
    weights_path = os.path.join(args.output_dir, "oracle_weights.safetensors")
    export_weights_safetensors(model, weights_path)

    # Set up dataloader and export batches
    DEVICE_BATCH_SIZE = 128
    dataloader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

    # Need (grad_accum_steps * steps) + extra batches
    TOTAL_BATCH_SIZE = 524288
    tokens_per_micro = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_micro
    n_batches = grad_accum_steps * (args.steps + 1)  # +1 for warmup
    batches_path = os.path.join(args.output_dir, "oracle_batches.bin")
    all_batches = export_batches(dataloader, n_batches, batches_path)

    # Run diagnostic forward passes
    diagnostics_path = os.path.join(args.output_dir, "oracle_diagnostics.jsonl")
    batch_idx = 0

    with open(diagnostics_path, 'w') as diag_f:
        for step in range(args.steps):
            model.zero_grad(set_to_none=True)

            # Gradient accumulation
            total_loss = 0.0
            for micro in range(grad_accum_steps):
                inp, tgt = all_batches[batch_idx]
                inp, tgt = inp.to(device), tgt.to(device)
                batch_idx += 1

                loss, diag = forward_with_diagnostics(model, inp, tgt)
                total_loss += loss.item()
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

            avg_loss = total_loss / grad_accum_steps

            # Collect gradient norms
            grad_info = {"per_layer": []}
            for i in range(config.n_layer):
                block = model.transformer.h[i]
                layer_grads = {
                    "grad_wq_norm": block.attn.c_q.weight.grad.float().norm().item() if block.attn.c_q.weight.grad is not None else 0,
                    "grad_wk_norm": block.attn.c_k.weight.grad.float().norm().item() if block.attn.c_k.weight.grad is not None else 0,
                    "grad_wv_norm": block.attn.c_v.weight.grad.float().norm().item() if block.attn.c_v.weight.grad is not None else 0,
                    "grad_wo_norm": block.attn.c_proj.weight.grad.float().norm().item() if block.attn.c_proj.weight.grad is not None else 0,
                    "grad_wfc_norm": block.mlp.c_fc.weight.grad.float().norm().item() if block.mlp.c_fc.weight.grad is not None else 0,
                    "grad_wdn_norm": block.mlp.c_proj.weight.grad.float().norm().item() if block.mlp.c_proj.weight.grad is not None else 0,
                }
                grad_info["per_layer"].append(layer_grads)

            grad_info["grad_lm_head_norm"] = model.lm_head.weight.grad.float().norm().item() if model.lm_head.weight.grad is not None else 0
            grad_info["grad_wte_norm"] = model.transformer.wte.weight.grad.float().norm().item() if model.transformer.wte.weight.grad is not None else 0

            # Merge forward diag (last micro-step) with gradient info
            record = {
                "step": step,
                "loss": avg_loss,
                "forward": diag,  # from last micro-step
                "gradients": grad_info,
                "resid_lambdas": model.resid_lambdas.data.tolist(),
                "x0_lambdas": model.x0_lambdas.data.tolist(),
            }
            diag_f.write(json.dumps(record) + '\n')
            print(f"[oracle] step {step} | loss {avg_loss:.6f}")

            # No optimizer step in oracle -- just compare forward/backward
            # (Add optimizer if needed for multi-step comparison)

    print(f"[oracle] diagnostics saved to {diagnostics_path}")
    print(f"[oracle] done. Files in {args.output_dir}/")


if __name__ == "__main__":
    main()
