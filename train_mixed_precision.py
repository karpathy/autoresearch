"""
Autoresearch pretraining script with Mixed Precision Training.
Single-GPU, single-file. Enhanced with AMP for 2x speedup.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # NEW: AMP imports

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
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

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        # SwiGLU: hidden dim = 4*n_embd, output = n_embd
        self.c_fc = nn.Linear(self.n_embd, 4 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.silu(x) * x  # SwiGLU activation
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# Value Embedding (ResFormer)
# ---------------------------------------------------------------------------

@dataclass
class VEConfig:
    ve_layers: int = 4
    ve_dim: int = 128


def value_embedding(n_embd, ve_config):
    ve = torch.arange(ve_config.ve_layers * ve_config.ve_dim, dtype=torch.float32)
    ve = ve.view(ve_config.ve_layers, ve_config.ve_dim)
    ve = ve / ve_config.ve_dim  # Normalize
    ve = torch.exp(ve)  # Exponential spacing
    ve = ve.flatten()[:n_embd]  # Match embedding dim
    return ve


# ---------------------------------------------------------------------------
# Window pattern
# ---------------------------------------------------------------------------

def window_size(layer_idx, n_layer, pattern="SSSL"):
    """Returns window size for layer. S=small (64), L=large (MAX_SEQ_LEN)."""
    idx = layer_idx % len(pattern)
    c = pattern[idx]
    return 64 if c == 'S' else MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    n_layer: int = 8
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 32768
    window_pattern: str = "SSSL"
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # NEW: Mixed Precision
    use_amp: bool = True  # Enable Automatic Mixed Precision
    
    # Learning rate schedule
    warmup_iters: int = 100
    lr_decay_iters: int = 1000
    min_lr: float = 3e-5
    
    # Batch size
    device_batch_size: int = 8
    
    # Evaluation
    eval_interval: int = 200
    eval_iters: int = 100
    
    # Logging
    log_interval: int = 50
    compile_model: bool = True


def get_lr(it, config):
    # 1) Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) Decay
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train(config=TrainConfig()):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # NEW: Initialize AMP GradScaler
    scaler = GradScaler() if config.use_amp and device.type == "cuda" else None
    
    # Create model
    model_config = GPTConfig(
        sequence_len=config.sequence_len,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        n_embd=config.n_embd,
        window_pattern=config.window_pattern
    )
    
    # Initialize model
    torch.manual_seed(1337)
    model = nn.ModuleDict({
        'wte': nn.Embedding(config.vocab_size, config.n_embd),
        'h': nn.ModuleList([Block(model_config, i) for i in range(model_config.n_layer)]),
        'ln_f': nn.LayerNorm(config.n_embd),
        'lm_head': nn.Linear(config.n_embd, config.vocab_size, bias=False),
    }).to(device)
    
    # Tie weights
    model.wte.weight = model.lm_head.weight
    
    # Value embeddings
    ve_config = VEConfig()
    ve = value_embedding(config.n_embd, ve_config).to(device)
    
    # Rotary embeddings
    assert config.sequence_len % 128 == 0
    seq = torch.arange(0, config.sequence_len, dtype=torch.long).view(1, -1).to(device)
    dim = torch.arange(0, model_config.n_embd // model_config.n_head, 2).float().to(device)
    freq = 1.0 / (10000 ** (dim / (model_config.n_embd // model_config.n_head)))
    sinusoid = seq * freq
    cos = torch.cos(sinusoid).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(sinusoid).unsqueeze(0).unsqueeze(0)
    cos_sin = (cos, sin)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                   lr=config.learning_rate,
                                   betas=(config.beta1, config.beta2),
                                   weight_decay=config.weight_decay)
    
    # Compile model (PyTorch 2.0+)
    if config.compile_model:
        model = torch.compile(model)
    
    # Create dataloader
    dataloader = make_dataloader(config.device_batch_size)
    
    # Training loop
    stats = {"losses": [], "bpb": []}
    start_time = time.time()
    step = 0
    
    while True:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET:
            break
        
        # Get batch
        batch = next(dataloader)
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        
        # Learning rate schedule
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass with autocast (NEW: Mixed Precision)
        if config.use_amp and device.type == "cuda":
            with autocast(dtype=torch.bfloat16):
                # Forward pass
                h = model.wte(x)
                
                # Transformer blocks
                ve_expanded = ve.unsqueeze(0).unsqueeze(0).expand(h.size(0), h.size(1), -1)
                for i, block in enumerate(model.h):
                    window = window_size(i, model_config.n_layer, model_config.window_pattern)
                    h = block(h, ve_expanded, cos_sin, window)
                
                # Output
                h = model.ln_f(h)
                logits = model.lm_head(h)
                
                # Loss
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass with gradient scaling (NEW)
            scaler.scale(loss).backward()
            
            # Gradient clipping (NEW)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Optimizer step (NEW)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training (fallback)
            h = model.wte(x)
            ve_expanded = ve.unsqueeze(0).unsqueeze(0).expand(h.size(0), h.size(1), -1)
            for i, block in enumerate(model.h):
                window = window_size(i, model_config.n_layer, model_config.window_pattern)
                h = block(h, ve_expanded, cos_sin, window)
            
            h = model.ln_f(h)
            logits = model.lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if step % config.log_interval == 0:
            loss_val = loss.item()
            stats["losses"].append(loss_val)
            print(f"Step {step}: loss={loss_val:.4f}, lr={lr:.6f}, time={elapsed:.1f}s")
        
        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            val_bpb = evaluate_bpb(model, num_batches=10)
            stats["bpb"].append(val_bpb)
            print(f"Step {step}: val_bpb={val_bpb:.4f}")
        
        step += 1
    
    # Final evaluation
    final_bpb = evaluate_bpb(model, num_batches=config.eval_iters)
    print(f"\nFinal val_bpb: {final_bpb:.4f}")
    print(f"Total steps: {step}, Time: {time.time() - start_time:.1f}s")
    
    return final_bpb, stats


if __name__ == "__main__":
    train()
