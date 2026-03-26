"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import argparse
import gc
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from prepare import (
    DATASET_CHOICES,
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    evaluate_bpb,
    make_dataloader,
)

# ---------------------------------------------------------------------------
# User configuration (all tunable values)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PRE_EVAL_PATH = PROJECT_ROOT / "checkpoint_pre_eval.pt"
AUTOTUNE_CACHE_PATH = PROJECT_ROOT / "artifacts" / "autotune" / "gpu-profile-v2.json"

# Model architecture
ASPECT_RATIO = 64         # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128            # target head dimension for attention
WINDOW_PATTERN = "SSSL"   # sliding window pattern: L=full, S=half context
DEPTH = 8

# Optimization
TOTAL_BATCH_SIZE = 2 ** 19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Batch defaults
DEVICE_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8

# Runtime/profile heuristics
SUPPORTED_CONSUMER_CAPABILITIES = {
    (7, 5): "turing",
    (8, 6): "ampere",
    (8, 9): "ada",
    (12, 0): "blackwell",
}
MIN_SUPPORTED_VRAM_GB_BY_ARCH = {
    "turing": 8.0,
    "ampere": 10.0,
    "ada": 10.0,
    "blackwell": 10.0,
}
VRAM_FLOOR_TOLERANCE_GB = 0.05
AUTOTUNE_WARMUP_STEPS = 2
AUTOTUNE_MEASURE_STEPS = 3
AUTOTUNE_MAX_MEMORY_FRACTION = 0.90


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    device: torch.device
    device_type: str
    amp_dtype: torch.dtype
    use_compile: bool
    use_activation_checkpointing: bool
    attention_backend: str
    gpu_name: str
    gpu_vram_gb: float
    gpu_peak_flops: float | None
    gpu_cc: tuple[int, int]
    gpu_total_memory_bytes: int
    tf32_enabled: bool
    gpu_profile: "GpuProfile"


@dataclass(frozen=True)
class GpuProfile:
    name: str
    is_supported_consumer: bool
    is_compatibility_only: bool
    train_batch_candidates: tuple[int, ...]
    checkpoint_modes: tuple[bool, ...]
    default_checkpointing: bool
    eval_batch_cap: int = 16


AUTOTUNE_CACHE_VERSION = "gpu-profile-v2"


def _get_gpu_peak_flops(gpu_name):
    name = gpu_name.lower()
    lookup = (
        ("5090", 360.0e12),
        ("4090 d", 280.0e12),
        ("4090d", 280.0e12),
        ("4090", 330.3e12),
        ("5080", 280.0e12),
        ("4080 super", 260.0e12),
        ("4070 ti super", 176.4e12),
        ("4070 ti", 160.4e12),
        ("4070 super", 142.2e12),
        ("4070", 116.8e12),
        ("4080", 242.5e12),
        ("5070 ti", 190.0e12),
        ("5070", 150.0e12),
        ("5060 ti", 120.0e12),
        ("4060 ti", 88.4e12),
        ("2080 ti", 107.5e12),
        ("2080 super", 89.6e12),
        ("2080", 80.3e12),
        ("2070 super", 72.6e12),
        ("2070", 59.7e12),
        ("2060 super", 57.4e12),
        ("2060", 52.4e12),
        ("3090 ti", 160.0e12),
        ("3090", 142.6e12),
        ("3080 ti", 136.0e12),
        ("3080", 119.5e12),
        ("3060", 51.0e12),
        ("3070", 81.1e12),
    )
    for key, flops in lookup:
        if key in name:
            return flops
    return None


def _resolve_gpu_profile(gpu_name, capability, gpu_vram_gb, is_windows):
    name = gpu_name.lower()
    arch = SUPPORTED_CONSUMER_CAPABILITIES.get(capability)
    min_vram_gb = MIN_SUPPORTED_VRAM_GB_BY_ARCH.get(arch, float("inf"))
    is_rtx = "rtx" in name
    is_laptop = "laptop" in name
    supported_consumer = (
        is_rtx
        and not is_laptop
        and arch is not None
        and gpu_vram_gb >= (min_vram_gb - VRAM_FLOOR_TOLERANCE_GB)
    )

    if supported_consumer:
        if arch == "turing" and gpu_vram_gb < 12.0:
            return GpuProfile(
                name=f"{arch}-8-11gb",
                is_supported_consumer=True,
                is_compatibility_only=False,
                train_batch_candidates=(8, 4, 2, 1),
                checkpoint_modes=(True,),
                default_checkpointing=True,
                eval_batch_cap=4,
            )
        if gpu_vram_gb < 16.0:
            mid_tier_name = f"{arch}-12-15gb" if arch == "turing" else f"{arch}-10-15gb"
            return GpuProfile(
                name=mid_tier_name,
                is_supported_consumer=True,
                is_compatibility_only=False,
                train_batch_candidates=(16, 8, 4),
                checkpoint_modes=(True,),
                default_checkpointing=True,
            )
        if gpu_vram_gb < 24.0:
            return GpuProfile(
                name=f"{arch}-16gb",
                is_supported_consumer=True,
                is_compatibility_only=False,
                train_batch_candidates=(32, 16, 8, 4),
                checkpoint_modes=(False, True),
                default_checkpointing=False,
            )
        return GpuProfile(
            name=f"{arch}-24gb-plus",
            is_supported_consumer=True,
            is_compatibility_only=False,
            train_batch_candidates=(64, 32, 16, 8, 4),
            checkpoint_modes=(False, True),
            default_checkpointing=False,
        )

    default_checkpointing = is_windows or gpu_vram_gb <= 16.0
    return GpuProfile(
        name="compatibility",
        is_supported_consumer=False,
        is_compatibility_only=True,
        train_batch_candidates=(DEVICE_BATCH_SIZE, 16, 8, 4),
        checkpoint_modes=(default_checkpointing,),
        default_checkpointing=default_checkpointing,
    )


def _compatibility_warning(gpu_name, capability, gpu_vram_gb):
    name = gpu_name.lower()
    arch = SUPPORTED_CONSUMER_CAPABILITIES.get(capability)
    if "rtx" not in name:
        return None
    if "laptop" in name:
        return "laptop GPUs are outside the supported desktop matrix"
    if arch is None:
        return f"compute capability {capability[0]}.{capability[1]} is outside supported consumer tiers"
    min_vram_gb = MIN_SUPPORTED_VRAM_GB_BY_ARCH.get(arch, float("inf"))
    if gpu_vram_gb < (min_vram_gb - VRAM_FLOOR_TOLERANCE_GB):
        return f"{gpu_vram_gb:.1f} GB VRAM is below the {min_vram_gb:g} GB floor for {arch}"
    return None


def _get_autotune_cache_path():
    return AUTOTUNE_CACHE_PATH


def _load_autotune_entries(path):
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"Warning: could not read autotune cache ({exc}); ignoring cache.")
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def _save_autotune_entries(path, entries):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        payload = {"entries": entries}
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(path)
    except Exception as exc:
        print(f"Warning: could not write autotune cache ({exc}).")


def _make_autotune_cache_key(runtime):
    cc = f"{runtime.gpu_cc[0]}.{runtime.gpu_cc[1]}"
    return "|".join(
        [
            runtime.gpu_name,
            cc,
            str(runtime.gpu_total_memory_bytes),
            torch.__version__,
            platform.system(),
            str(MAX_SEQ_LEN),
        ]
    )


def _select_amp_dtype(gpu_cc):
    if gpu_cc >= (8, 0) and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float16


def detect_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No CUDA device detected.")

    is_windows = platform.system().lower().startswith("win")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    gpu_name = torch.cuda.get_device_name()
    gpu_total_memory_bytes = int(props.total_memory)
    gpu_vram_gb = gpu_total_memory_bytes / (1024 ** 3)
    gpu_cc = torch.cuda.get_device_capability()
    gpu_profile = _resolve_gpu_profile(gpu_name, gpu_cc, gpu_vram_gb, is_windows)
    warning = _compatibility_warning(gpu_name, gpu_cc, gpu_vram_gb)
    if warning is not None:
        print(f"Warning: {warning}; running compatibility runtime path.")

    amp_dtype = _select_amp_dtype(gpu_cc)
    tf32_enabled = bool(getattr(torch.cuda, "is_tf32_supported", lambda: False)())
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    use_compile = False
    print("torch.compile disabled in this fork runtime path.")
    attention_backend = "sdpa"
    print("Using PyTorch SDPA attention backend.")
    force_checkpointing = os.environ.get("AUTORESEARCH_FORCE_CHECKPOINTING")
    if force_checkpointing == "1":
        use_activation_checkpointing = True
    elif force_checkpointing == "0":
        use_activation_checkpointing = False
    else:
        use_activation_checkpointing = gpu_profile.default_checkpointing

    return RuntimeConfig(
        device=device,
        device_type=device.type,
        amp_dtype=amp_dtype,
        use_compile=use_compile,
        use_activation_checkpointing=use_activation_checkpointing,
        attention_backend=attention_backend,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_peak_flops=_get_gpu_peak_flops(gpu_name),
        gpu_cc=gpu_cc,
        gpu_total_memory_bytes=gpu_total_memory_bytes,
        tf32_enabled=tf32_enabled,
        gpu_profile=gpu_profile,
    )


USE_COMPILE = False
MUON_COMPUTE_DTYPE = torch.bfloat16


def _maybe_compile(obj, **kwargs):
    return obj


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
    attention_backend: str = "sdpa"
    use_activation_checkpointing: bool = False
    compute_dtype: torch.dtype = torch.bfloat16


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
        self.attention_backend = config.attention_backend
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self._mask_cache = {}

    def _get_sdpa_mask(self, seq_len, window_size, device):
        window = window_size[0] if isinstance(window_size, tuple) else window_size
        cache_key = (seq_len, int(window), device.type, device.index)
        mask = self._mask_cache.get(cache_key)
        if mask is not None:
            return mask

        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = col <= row  # causal
        if window is not None and window >= 0 and window < seq_len:
            mask = mask & (col >= (row - window))
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()
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

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, KVH, T, D)
        v = v.transpose(1, 2)  # (B, KVH, T, D)
        attn_mask = self._get_sdpa_mask(T, window_size, q.device)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=self.n_kv_head < self.n_head,
        )
        y = y.transpose(1, 2)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
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
        self.rotary_seq_len = config.sequence_len
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, dtype=config.compute_dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, embed_dtype=torch.bfloat16):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
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
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len,
            head_dim,
            dtype=self.config.compute_dtype,
        )
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=embed_dtype)
        for ve in self.value_embeds.values():
            ve.to(dtype=embed_dtype)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None, dtype=torch.bfloat16):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
        )
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        muon_group_chunk = 8
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            for ci in range(0, len(group_params), muon_group_chunk):
                chunk = group_params[ci:ci + muon_group_chunk]
                param_groups.append(
                    dict(
                        kind="muon",
                        params=chunk,
                        lr=matrix_lr,
                        momentum=0.95,
                        ns_steps=5,
                        beta2=0.95,
                        weight_decay=weight_decay,
                    )
                )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            window_size = self.window_sizes[i]
            if self.config.use_activation_checkpointing:
                x = torch_checkpoint(block, x, ve, cos_sin, window_size, use_reentrant=False)
            else:
                x = block(x, ve, cos_sin, window_size)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    # Keep moments in their own dtype (float32 for fp16 params) to avoid grad^2 underflow.
    g = grad.to(exp_avg.dtype)
    exp_avg.lerp_(g, 1 - beta1_t)
    exp_avg_sq.lerp_(g.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_((exp_avg / denom * (-step_size)).to(p.dtype))


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.to(dtype=MUON_COMPUTE_DTYPE)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


ADAMW_STEP_IMPL = adamw_step_fused
MUON_STEP_IMPL = muon_step_fused


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                moment_dtype = torch.float32 if p.dtype == torch.float16 else p.dtype
                state["exp_avg"] = torch.zeros_like(p, dtype=moment_dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=moment_dtype)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            ADAMW_STEP_IMPL(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        MUON_STEP_IMPL(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


def build_model_config(depth, vocab_size, runtime, use_activation_checkpointing=None):
    if use_activation_checkpointing is None:
        use_activation_checkpointing = runtime.use_activation_checkpointing
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        attention_backend=runtime.attention_backend,
        use_activation_checkpointing=use_activation_checkpointing,
        compute_dtype=runtime.amp_dtype,
    )


def _filter_train_batch_sizes(candidates):
    deduped = []
    for batch_size in list(candidates):
        if batch_size <= 0:
            continue
        tokens_per_fwdbwd = batch_size * MAX_SEQ_LEN
        if TOTAL_BATCH_SIZE % tokens_per_fwdbwd != 0:
            continue
        if batch_size not in deduped:
            deduped.append(batch_size)
    if not deduped:
        raise RuntimeError("No valid device batch sizes satisfy TOTAL_BATCH_SIZE divisibility.")
    return deduped


def _build_train_candidates(runtime):
    batch_sizes = _filter_train_batch_sizes(runtime.gpu_profile.train_batch_candidates)
    candidates = []
    for checkpointing in runtime.gpu_profile.checkpoint_modes:
        for batch_size in batch_sizes:
            candidate = (batch_size, checkpointing)
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No train candidates available for this runtime profile.")
    return candidates


def _build_eval_batch_candidates(train_batch_size, initial_eval_batch):
    candidates = [min(initial_eval_batch, train_batch_size), 8, 4, 2, 1]
    deduped = []
    for batch_size in candidates:
        if batch_size > 0 and batch_size not in deduped:
            deduped.append(batch_size)
    return deduped


def _benchmark_train_candidate(runtime, tokenizer, vocab_size, train_batch_size, use_checkpointing):
    config = build_model_config(
        DEPTH,
        vocab_size,
        runtime,
        use_activation_checkpointing=use_checkpointing,
    )
    tokens_per_fwdbwd = train_batch_size * MAX_SEQ_LEN
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    autocast_ctx = torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype)

    model = None
    optimizer = None
    train_loader = None
    x = y = None
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=runtime.device)
        model.init_weights(embed_dtype=runtime.amp_dtype)
        optimizer = model.setup_optimizer(
            unembedding_lr=UNEMBEDDING_LR,
            embedding_lr=EMBEDDING_LR,
            scalar_lr=SCALAR_LR,
            adam_betas=ADAM_BETAS,
            matrix_lr=MATRIX_LR,
            weight_decay=WEIGHT_DECAY,
        )
        train_loader = make_dataloader(
            tokenizer,
            train_batch_size,
            MAX_SEQ_LEN,
            "train",
            device=runtime.device,
            dataset=tokenizer.dataset,
        )
        x, y, _ = next(train_loader)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        total_steps = AUTOTUNE_WARMUP_STEPS + AUTOTUNE_MEASURE_STEPS
        measured_time = 0.0
        for step_idx in range(total_steps):
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(grad_accum_steps):
                with autocast_ctx:
                    loss = model(x, y)
                (loss / grad_accum_steps).backward()
                x, y, _ = next(train_loader)
            optimizer.step()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            if step_idx >= AUTOTUNE_WARMUP_STEPS:
                measured_time += dt

        peak_memory = torch.cuda.max_memory_allocated()
        peak_limit = runtime.gpu_total_memory_bytes * AUTOTUNE_MAX_MEMORY_FRACTION
        if peak_memory > peak_limit:
            return None
        tokens_measured = TOTAL_BATCH_SIZE * AUTOTUNE_MEASURE_STEPS
        tok_per_sec = tokens_measured / max(measured_time, 1e-6)
        return tok_per_sec, peak_memory
    except torch.cuda.OutOfMemoryError:
        return None
    except RuntimeError as exc:
        print(
            "Autotune candidate rejected "
            f"(batch_size={train_batch_size}, checkpointing={'on' if use_checkpointing else 'off'}): {exc}"
        )
        return None
    finally:
        del x, y, train_loader, optimizer, model
        torch.cuda.empty_cache()
        _restore_gc_after_attempt()


def _autotune_train_candidate(runtime, tokenizer, vocab_size, train_candidates):
    if not runtime.gpu_profile.is_supported_consumer:
        return None
    if os.environ.get("AUTORESEARCH_DISABLE_AUTOTUNE", "0") == "1":
        print("Autotune disabled by AUTORESEARCH_DISABLE_AUTOTUNE=1.")
        return None

    cache_path = _get_autotune_cache_path()
    cache_key = _make_autotune_cache_key(runtime)
    refresh_cache = os.environ.get("AUTORESEARCH_AUTOTUNE_REFRESH", "0") == "1"
    cache_entries = _load_autotune_entries(cache_path)
    if refresh_cache:
        print("Autotune cache refresh requested by AUTORESEARCH_AUTOTUNE_REFRESH=1.")
    else:
        cached = cache_entries.get(cache_key)
        if isinstance(cached, dict):
            cached_batch_size = cached.get("train_batch_size")
            cached_checkpointing = cached.get("use_activation_checkpointing")
            if isinstance(cached_batch_size, int) and isinstance(cached_checkpointing, bool):
                cached_candidate = (cached_batch_size, cached_checkpointing)
                if cached_candidate in train_candidates:
                    print(
                        "Using cached autotune candidate: "
                        f"batch_size={cached_batch_size}, checkpointing={'on' if cached_checkpointing else 'off'}."
                    )
                    return cached_candidate

    print("Running consumer GPU autotune in eager mode...")
    best_candidate = None
    best_tok_per_sec = -1.0
    best_peak_memory = 0
    for train_batch_size, use_checkpointing in train_candidates:
        ckpt_label = "on" if use_checkpointing else "off"
        print(f"Autotune probe: train_batch_size={train_batch_size}, checkpointing={ckpt_label}")
        result = _benchmark_train_candidate(
            runtime=runtime,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            train_batch_size=train_batch_size,
            use_checkpointing=use_checkpointing,
        )
        if result is None:
            print("  rejected (OOM, runtime error, or >90% VRAM use)")
            continue
        tok_per_sec, peak_memory = result
        print(f"  accepted: tok/sec={tok_per_sec:,.0f}, peak_vram_mb={peak_memory / 1024 / 1024:.1f}")
        if tok_per_sec > best_tok_per_sec:
            best_tok_per_sec = tok_per_sec
            best_candidate = (train_batch_size, use_checkpointing)
            best_peak_memory = peak_memory

    if best_candidate is None:
        print("Autotune could not find a viable candidate; using default fallback ordering.")
        return None

    cache_entries[cache_key] = {
        "train_batch_size": best_candidate[0],
        "use_activation_checkpointing": best_candidate[1],
        "tok_per_sec": round(best_tok_per_sec, 3),
        "peak_memory_bytes": int(best_peak_memory),
        "updated_unix": int(time.time()),
    }
    _save_autotune_entries(cache_path, cache_entries)
    print(
        "Autotune selected candidate: "
        f"batch_size={best_candidate[0]}, checkpointing={'on' if best_candidate[1] else 'off'}."
    )
    return best_candidate


def _prioritize_autotuned_candidate(train_candidates, autotuned_candidate):
    if autotuned_candidate is None or autotuned_candidate not in train_candidates:
        return train_candidates
    return [autotuned_candidate] + [c for c in train_candidates if c != autotuned_candidate]


def _configure_step_kernels(runtime):
    global ADAMW_STEP_IMPL, MUON_STEP_IMPL, USE_COMPILE, MUON_COMPUTE_DTYPE
    ADAMW_STEP_IMPL = adamw_step_fused
    MUON_STEP_IMPL = muon_step_fused
    if runtime.amp_dtype != torch.float16:
        MUON_COMPUTE_DTYPE = runtime.amp_dtype
        muon_reason = "matching AMP dtype"
    elif torch.cuda.is_bf16_supported(including_emulation=True):
        # Use bf16 for Muon orthogonalization when training runs in fp16 for better numeric headroom.
        MUON_COMPUTE_DTYPE = torch.bfloat16
        muon_reason = "fp16 AMP with bf16 support (native or emulated)"
    else:
        # Safety fallback when fp16 AMP is selected but bf16 isn't available in this runtime.
        MUON_COMPUTE_DTYPE = torch.float32
        muon_reason = "fp16 AMP without bf16 support; using fp32 fallback"
    print(f"Muon compute dtype: {MUON_COMPUTE_DTYPE} ({muon_reason})")
    USE_COMPILE = False


def _run_training_once(runtime, tokenizer, config, device_batch_size, smoke_test):
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    autocast_ctx = torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=runtime.device)
    model.init_weights(embed_dtype=runtime.amp_dtype)

    param_counts = model.num_scaling_params()
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()

    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    model = _maybe_compile(model, dynamic=False)

    train_loader = make_dataloader(
        tokenizer,
        device_batch_size,
        MAX_SEQ_LEN,
        "train",
        device=runtime.device,
        dataset=tokenizer.dataset,
    )
    x, y, epoch = next(train_loader)
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        if progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)

    target_training_seconds = 10 if smoke_test else TIME_BUDGET
    max_steps = 3 if smoke_test else None

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / max(target_training_seconds, 1e-6), 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        if train_loss_f > 100:
            raise RuntimeError("FAIL: training loss exploded")

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        if runtime.gpu_peak_flops:
            mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / runtime.gpu_peak_flops
            mfu_text = f"{mfu:.1f}%"
        else:
            mfu_text = "n/a"
        remaining = max(0, target_training_seconds - total_training_time)
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu_text} | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1
        if max_steps is not None and step >= max_steps:
            break
        if step > 10 and total_training_time >= target_training_seconds:
            break
        if smoke_test and total_training_time >= target_training_seconds:
            break

    print()
    return {
        "model": model,
        "num_params": num_params,
        "num_flops_per_token": num_flops_per_token,
        "total_training_time": total_training_time,
        "step": step,
        "t_start": t_start,
        "t_start_training": t_start_training,
    }


def _save_pre_eval_checkpoint(model):
    try:
        state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save(state_dict, CHECKPOINT_PRE_EVAL_PATH)
        print(f"Saved {CHECKPOINT_PRE_EVAL_PATH}")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not save pre-eval checkpoint: {exc}")


def _restore_gc_after_attempt():
    if hasattr(gc, "unfreeze"):
        gc.unfreeze()
    gc.enable()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Autoresearch training script")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short train/eval pass for validation.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None, help="Optional dataset override.")
    args = parser.parse_args()

    runtime = detect_runtime()
    print(f"GPU: {runtime.gpu_name}")
    print(f"GPU VRAM: {runtime.gpu_vram_gb:.1f} GB")
    print(f"GPU CC: {runtime.gpu_cc[0]}.{runtime.gpu_cc[1]}")
    print(f"GPU profile: {runtime.gpu_profile.name}")
    print(f"Consumer matrix support: {'yes' if runtime.gpu_profile.is_supported_consumer else 'compatibility path'}")
    print(f"TF32: {'enabled' if runtime.tf32_enabled else 'disabled'}")
    print(f"AMP dtype: {runtime.amp_dtype}")

    tokenizer = Tokenizer.from_directory(dataset=args.dataset)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")
    print(f"Dataset: {tokenizer.dataset}")

    # Configure optimizer kernels/dtypes before autotune so probes match real training runtime.
    _configure_step_kernels(runtime)

    train_candidates = _build_train_candidates(runtime)
    autotuned_candidate = _autotune_train_candidate(runtime, tokenizer, vocab_size, train_candidates)
    train_candidates = _prioritize_autotuned_candidate(train_candidates, autotuned_candidate)

    print(f"Attention backend: {runtime.attention_backend}")
    print(f"torch.compile: {'enabled' if USE_COMPILE else 'disabled'}")

    result = None
    chosen_train_batch = None
    chosen_checkpointing = None
    for train_batch_size, use_checkpointing in train_candidates:
        config = build_model_config(
            DEPTH,
            vocab_size,
            runtime,
            use_activation_checkpointing=use_checkpointing,
        )
        print(
            "Trying train candidate: "
            f"batch_size={train_batch_size}, "
            f"activation_checkpointing={'enabled' if use_checkpointing else 'disabled'}"
        )
        print(f"Model config: {asdict(config)}")
        try:
            result = _run_training_once(
                runtime=runtime,
                tokenizer=tokenizer,
                config=config,
                device_batch_size=train_batch_size,
                smoke_test=args.smoke_test,
            )
            chosen_train_batch = train_batch_size
            chosen_checkpointing = use_checkpointing
            break
        except torch.cuda.OutOfMemoryError:
            print(
                "Train OOM at "
                f"batch_size={train_batch_size}, checkpointing={'on' if use_checkpointing else 'off'}; "
                "trying next candidate."
            )
            torch.cuda.empty_cache()
            _restore_gc_after_attempt()
        except RuntimeError as exc:
            _restore_gc_after_attempt()
            print(str(exc))
            return 1

    if result is None:
        print("FAIL: training failed for all batch size candidates.")
        return 1

    model = result["model"]
    _save_pre_eval_checkpoint(model)
    model.eval()

    eval_tokens = max(MAX_SEQ_LEN * chosen_train_batch * 2, 8192) if args.smoke_test else EVAL_TOKENS
    val_bpb = None
    chosen_eval_batch = None
    initial_eval_batch = min(chosen_train_batch, runtime.gpu_profile.eval_batch_cap)
    eval_candidates = _build_eval_batch_candidates(chosen_train_batch, initial_eval_batch)
    for eval_batch_size in eval_candidates:
        try:
            torch.cuda.empty_cache()
            with torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype):
                val_bpb = evaluate_bpb(
                    model,
                    tokenizer,
                    eval_batch_size,
                    device=runtime.device,
                    dataset=tokenizer.dataset,
                    eval_tokens=eval_tokens,
                )
            chosen_eval_batch = eval_batch_size
            print(f"Eval completed with batch_size={eval_batch_size}")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"Eval OOM at batch_size={eval_batch_size}; trying smaller batch.")
            torch.cuda.empty_cache()

    if val_bpb is None:
        print("FAIL: eval failed for all batch sizes.")
        return 1

    t_end = time.time()
    step = result["step"]
    total_training_time = result["total_training_time"]
    num_flops_per_token = result["num_flops_per_token"]
    num_params = result["num_params"]
    steady_state_steps = max(step - 10, 0)
    if runtime.gpu_peak_flops and total_training_time > 0 and steady_state_steps > 0:
        steady_state_mfu = (
            100
            * num_flops_per_token
            * TOTAL_BATCH_SIZE
            * steady_state_steps
            / total_training_time
            / runtime.gpu_peak_flops
        )
    else:
        steady_state_mfu = None
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    total_tokens = step * TOTAL_BATCH_SIZE

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - result['t_start']:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    if steady_state_mfu is None:
        print("mfu_percent:      n/a")
    else:
        print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
    print(f"dataset:          {tokenizer.dataset}")
    print(f"train_batch_size: {chosen_train_batch}")
    print(f"eval_batch_size:  {chosen_eval_batch}")
    print(f"activation_checkpointing: {'enabled' if chosen_checkpointing else 'disabled'}")
    if args.smoke_test:
        print("smoke_test:       true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
