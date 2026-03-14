"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage:
    uv run train.py
    uv run train.py --benchmark-steps 1
    uv run train.py --benchmark-steps 5
    uv run train.py --benchmark-steps 5 --compile-backend inductor --compile-mode max-autotune
    uv run train.py --benchmark-steps 5 --compile-backend inductor --compile-scope microstep
    uv run train.py --benchmark-steps 5 --compile-backend inductor --compile-scope trunk
    uv run train.py --benchmark-steps 5 --compile-backend inductor --optimizer-compile-backend off
"""

import argparse
import glob
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import importlib.util
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor as inductor
import torch._inductor.cpp_builder as cpp_builder

HAS_TRITON = importlib.util.find_spec("triton") is not None
AVAILABLE_COMPILE_BACKENDS = set(torch.compiler.list_backends())
EXTRA_COMPILE_BACKENDS = {"aot_eager"}
AVAILABLE_INDUCTOR_MODES = set(inductor.list_mode_options().keys())


def resolve_compile_backend(requested_backend):
    if requested_backend == "auto":
        return "inductor" if HAS_TRITON else "off"
    return requested_backend


def validate_compile_backend(compile_backend):
    if compile_backend == "off":
        return
    if compile_backend not in AVAILABLE_COMPILE_BACKENDS and compile_backend not in EXTRA_COMPILE_BACKENDS:
        available = ", ".join(sorted(AVAILABLE_COMPILE_BACKENDS | EXTRA_COMPILE_BACKENDS))
        raise RuntimeError(f"Compile backend '{compile_backend}' is unavailable. Available backends: {available}")
    if compile_backend == "inductor" and not HAS_TRITON:
        raise RuntimeError("Compile backend 'inductor' requires Triton in this environment")


def maybe_compile_optimizer(**kwargs):
    if OPTIMIZER_COMPILE_BACKEND == "off":
        def decorator(fn):
            return fn
        return decorator
    return torch.compile(backend=OPTIMIZER_COMPILE_BACKEND, mode=INDUCTOR_COMPILE_MODE, **kwargs)


def validate_compile_mode(compile_backend, compile_mode):
    if compile_backend != "inductor":
        return
    if compile_mode not in AVAILABLE_INDUCTOR_MODES:
        available = ", ".join(sorted(AVAILABLE_INDUCTOR_MODES))
        raise RuntimeError(f"Compile mode '{compile_mode}' is unavailable. Available modes: {available}")


def find_vcvars64():
    if os.name != "nt":
        return None

    vswhere_candidates = []
    env_vswhere = os.environ.get("VSWHERE")
    if env_vswhere:
        vswhere_candidates.append(env_vswhere)

    which_vswhere = shutil.which("vswhere.exe")
    if which_vswhere and which_vswhere not in vswhere_candidates:
        vswhere_candidates.append(which_vswhere)

    default_vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.isfile(default_vswhere) and default_vswhere not in vswhere_candidates:
        vswhere_candidates.append(default_vswhere)

    for vswhere in vswhere_candidates:
        try:
            result = subprocess.run(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                check=True,
                capture_output=True,
                text=True,
                errors="ignore",
            )
        except (OSError, subprocess.CalledProcessError):
            continue

        install_path = result.stdout.strip()
        if not install_path:
            continue

        vcvars64 = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if os.path.isfile(vcvars64):
            return vcvars64

    fallback_matches = []
    for pattern in (
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ):
        fallback_matches.extend(glob.glob(pattern))
    if fallback_matches:
        return sorted(fallback_matches, reverse=True)[0]
    return None


def load_windows_msvc_env(vcvars64_path):
    command = f'call "{vcvars64_path}" >nul && set'
    result = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        errors="ignore",
    )
    for line in result.stdout.splitlines():
        if "=" not in line or line.startswith("="):
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def ensure_windows_msvc_compiler():
    if os.name != "nt":
        return None

    compiler = shutil.which("cl.exe")
    if compiler is not None:
        os.environ.setdefault("CC", compiler)
        os.environ.setdefault("CXX", compiler)
        return compiler

    vcvars64_path = find_vcvars64()
    if vcvars64_path is None:
        raise RuntimeError(
            "Optimizer compile on Windows requires MSVC cl.exe, but no Visual Studio C++ toolchain was found. "
            "Install Visual Studio Build Tools with Desktop development with C++."
        )

    try:
        load_windows_msvc_env(vcvars64_path)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            detail = f" Details: {detail}"
        raise RuntimeError(f"Failed to load MSVC environment from '{vcvars64_path}'.{detail}") from exc

    compiler = shutil.which("cl.exe")
    if compiler is None:
        raise RuntimeError(
            f"Loaded MSVC environment from '{vcvars64_path}', but cl.exe is still unavailable in PATH."
        )

    os.environ.setdefault("CC", compiler)
    os.environ.setdefault("CXX", compiler)
    return compiler


def maybe_configure_msvc_utf8_help(compiler):
    if os.name != "nt" or compiler is None:
        return None

    help_output = subprocess.check_output([compiler, "/help"], stderr=subprocess.STDOUT)
    try:
        help_output.decode(*cpp_builder.SUBPROCESS_DECODE_ARGS)
        return None
    except UnicodeDecodeError:
        try:
            help_output.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"cl.exe help output from '{compiler}' could not be decoded using "
                f"{cpp_builder.SUBPROCESS_DECODE_ARGS[0]!r} or 'utf-8'."
            ) from exc

    cpp_builder.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
    cpp_builder._is_msvc_cl.cache_clear()
    return "utf-8"


def parse_args():
    parser = argparse.ArgumentParser(description="Autoresearch pretraining script")
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help="Run N optimizer steps, report throughput/MFU, and skip final eval.",
    )
    parser.add_argument(
        "--compile-backend",
        choices=["auto", "off", "inductor", "cudagraphs", "aot_eager"],
        default="inductor",
        help="Model compile backend. Default is the validated high-throughput inductor path.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=sorted(AVAILABLE_INDUCTOR_MODES),
        default="default",
        help="TorchInductor compile mode. Used only when --compile-backend is inductor.",
    )
    parser.add_argument(
        "--compile-scope",
        choices=["model", "microstep", "trunk"],
        default="model",
        help="Compile only the model forward, or compile a whole microstep (forward + backward).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Default 1 is the validated high-throughput setting.",
    )
    parser.add_argument(
        "--optimizer-compile-backend",
        choices=["auto", "off", "inductor"],
        default="inductor",
        help="Control optimizer fused-step compilation separately from model compilation.",
    )
    args = parser.parse_args()
    if args.benchmark_steps < 0:
        parser.error("--benchmark-steps must be >= 0")
    if args.grad_accum_steps < 0:
        parser.error("--grad-accum-steps must be >= 0")
    return args


def estimate_device_peak_flops(device_props):
    # Approximate dense BF16/FP16 tensor-core peak FLOPs using SM count and SM clock.
    flops_per_cycle_by_capability = {
        (9, 0): 4096,  # Hopper
        (8, 0): 2048,  # A100 / A800
        (8, 6): 1024,  # RTX 30-series
        (8, 9): 1024,  # Ada Lovelace
        (7, 0): 1024,  # Volta
        (7, 5): 1024,  # Turing
    }
    capability = (device_props.major, device_props.minor)
    flops_per_cycle = flops_per_cycle_by_capability.get(capability)
    if flops_per_cycle is None:
        return None
    return device_props.multi_processor_count * flops_per_cycle * device_props.clock_rate * 1000


def compute_mfu(peak_flops, num_flops_per_token, tok_per_sec):
    if peak_flops is None or peak_flops <= 0 or tok_per_sec <= 0:
        return None
    return 100 * num_flops_per_token * tok_per_sec / peak_flops


ARGS = parse_args()
BENCHMARK_STEPS = ARGS.benchmark_steps
BENCHMARK_MODE = BENCHMARK_STEPS > 0
REQUESTED_COMPILE_BACKEND = ARGS.compile_backend
MODEL_COMPILE_BACKEND = resolve_compile_backend(REQUESTED_COMPILE_BACKEND)
validate_compile_backend(MODEL_COMPILE_BACKEND)
INDUCTOR_COMPILE_MODE = ARGS.compile_mode
validate_compile_mode(MODEL_COMPILE_BACKEND, INDUCTOR_COMPILE_MODE)
COMPILE_SCOPE = ARGS.compile_scope
GRAD_ACCUM_STEPS_OVERRIDE = ARGS.grad_accum_steps
REQUESTED_OPTIMIZER_COMPILE_BACKEND = ARGS.optimizer_compile_backend
if REQUESTED_OPTIMIZER_COMPILE_BACKEND == "auto":
    OPTIMIZER_COMPILE_BACKEND = "inductor" if MODEL_COMPILE_BACKEND == "inductor" else "off"
else:
    OPTIMIZER_COMPILE_BACKEND = REQUESTED_OPTIMIZER_COMPILE_BACKEND
USE_COMPILED_EXECUTION = MODEL_COMPILE_BACKEND != "off"
USE_COMPILED_MODEL = USE_COMPILED_EXECUTION and COMPILE_SCOPE == "model"
USE_COMPILED_MICROSTEP = USE_COMPILED_EXECUTION and COMPILE_SCOPE == "microstep"
USE_COMPILED_TRUNK = USE_COMPILED_EXECUTION and COMPILE_SCOPE == "trunk"
MSVC_CL_PATH = None
MSVC_HELP_ENCODING = None
if MODEL_COMPILE_BACKEND == "inductor" or OPTIMIZER_COMPILE_BACKEND == "inductor":
    MSVC_CL_PATH = ensure_windows_msvc_compiler()
    MSVC_HELP_ENCODING = maybe_configure_msvc_utf8_help(MSVC_CL_PATH)


ATTN_BACKEND = None

if os.name == "nt":
    try:
        from flash_attn import flash_attn_func
        ATTN_BACKEND = "flash_attn"
    except ImportError:
        pass

if ATTN_BACKEND is None:
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    flash_attn_func = get_kernel(repo).flash_attn_interface.flash_attn_func
    ATTN_BACKEND = repo

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

        y = flash_attn_func(q, k, v, causal=True, window_size=window_size)
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
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
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
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
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
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
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
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward_trunk(self, x, idx, cos, sin):
        cos_sin = cos, sin
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        return x

    def compute_loss(self, x, targets=None, reduction='mean'):
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x = self.forward_trunk(x, idx, cos, sin)
        return self.compute_loss(x, targets=targets, reduction=reduction)

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

@maybe_compile_optimizer(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@maybe_compile_optimizer(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
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
    # NorMuon variance reduction
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
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
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
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
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
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEFAULT_DEVICE_BATCH_SIZE = 12 # validated best-throughput local default


def pick_device_batch_size(default_batch_size):
    override = os.environ.get("DEVICE_BATCH_SIZE")
    if override is not None:
        return int(override)
    return default_batch_size


DEVICE_BATCH_SIZE = pick_device_batch_size(DEFAULT_DEVICE_BATCH_SIZE)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
device_props = torch.cuda.get_device_properties(0)
device_peak_flops = estimate_device_peak_flops(device_props)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")
print(f"Attention backend: {ATTN_BACKEND}")
print(f"Device batch size: {DEVICE_BATCH_SIZE}")
print(f"Mode: {'benchmark' if BENCHMARK_MODE else 'train'}")
print(f"Compile backend: {MODEL_COMPILE_BACKEND}")
if MODEL_COMPILE_BACKEND == "inductor":
    print(f"Compile mode: {INDUCTOR_COMPILE_MODE}")
print(f"Compile scope: {COMPILE_SCOPE}")
print(f"Optimizer compile backend: {OPTIMIZER_COMPILE_BACKEND}")
if MSVC_CL_PATH is not None:
    print(f"MSVC cl.exe: {MSVC_CL_PATH}")
if MSVC_HELP_ENCODING is not None:
    print(f"MSVC help decode override: {MSVC_HELP_ENCODING}")
if BENCHMARK_MODE:
    print(f"Benchmark steps: {BENCHMARK_STEPS}")
if device_peak_flops is not None:
    print(f"Device peak BF16/FP16 TFLOPS est: {device_peak_flops / 1e12:.2f}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
if GRAD_ACCUM_STEPS_OVERRIDE > 0:
    grad_accum_steps = GRAD_ACCUM_STEPS_OVERRIDE
    TOTAL_BATCH_SIZE = tokens_per_fwdbwd * grad_accum_steps
else:
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

compile_kwargs = None
if USE_COMPILED_EXECUTION:
    compile_kwargs = {"backend": MODEL_COMPILE_BACKEND, "dynamic": False}
    if MODEL_COMPILE_BACKEND == "inductor":
        compile_kwargs["mode"] = INDUCTOR_COMPILE_MODE

def run_microstep(x, y):
    with autocast_ctx:
        loss = model(x, y)
    (loss / grad_accum_steps).backward()
    return loss.detach()

def run_trunk_forward(x, idx, cos, sin):
    return model.forward_trunk(x, idx, cos, sin)

microstep_fn = run_microstep
trunk_fn = run_trunk_forward

if USE_COMPILED_MODEL:
    model = torch.compile(model, **compile_kwargs)
    if MODEL_COMPILE_BACKEND == "inductor":
        print(f"torch.compile: enabled ({MODEL_COMPILE_BACKEND}, mode={INDUCTOR_COMPILE_MODE}, scope=model)")
    else:
        print(f"torch.compile: enabled ({MODEL_COMPILE_BACKEND}, scope=model)")
elif USE_COMPILED_MICROSTEP:
    microstep_fn = torch.compile(microstep_fn, **compile_kwargs)
    if MODEL_COMPILE_BACKEND == "inductor":
        print(f"torch.compile: enabled ({MODEL_COMPILE_BACKEND}, mode={INDUCTOR_COMPILE_MODE}, scope=microstep)")
    else:
        print(f"torch.compile: enabled ({MODEL_COMPILE_BACKEND}, scope=microstep)")
elif USE_COMPILED_TRUNK:
    trunk_fn = torch.compile(trunk_fn, **compile_kwargs)
    if MODEL_COMPILE_BACKEND == "inductor":
        print(f"torch.compile: enabled ({MODEL_COMPILE_BACKEND}, mode={INDUCTOR_COMPILE_MODE}, scope=trunk)")
    else:
        print(f"torch.compile: enabled ({MODEL_COMPILE_BACKEND}, scope=trunk)")
else:
    print("torch.compile: disabled")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Total batch size: {TOTAL_BATCH_SIZE}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
measured_steps = 0
BENCHMARK_WARMUP_STEPS = 1 if BENCHMARK_MODE else 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    train_loss_f = 0.0
    for micro_step in range(grad_accum_steps):
        if USE_COMPILED_EXECUTION:
            torch.compiler.cudagraph_mark_step_begin()
        if COMPILE_SCOPE == "microstep":
            train_loss = microstep_fn(x, y)
            train_loss_f = train_loss.float().item()
        elif COMPILE_SCOPE == "trunk":
            with autocast_ctx:
                cos = model.cos[:, :x.size(1)]
                sin = model.sin[:, :x.size(1)]
                hidden = model.transformer.wte(x)
                hidden = norm(hidden)
                hidden = trunk_fn(hidden, x, cos, sin)
                loss = model.compute_loss(hidden, targets=y)
            train_loss_f = loss.detach().float().item()
            (loss / grad_accum_steps).backward()
        else:
            with autocast_ctx:
                loss = model(x, y)
            train_loss_f = loss.detach().float().item()
            loss = loss / grad_accum_steps
            loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    is_benchmark_warmup = BENCHMARK_MODE and step < BENCHMARK_WARMUP_STEPS

    if (BENCHMARK_MODE and not is_benchmark_warmup) or (not BENCHMARK_MODE and step > 10):
        total_training_time += dt
        measured_steps += 1

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = TOTAL_BATCH_SIZE / dt
    if is_benchmark_warmup:
        tok_per_sec_str = "warmup"
        device_mfu_str = "warmup"
        h100_mfu_str = "warmup"
    else:
        device_mfu = compute_mfu(device_peak_flops, num_flops_per_token, tok_per_sec)
        h100_mfu = compute_mfu(H100_BF16_PEAK_FLOPS, num_flops_per_token, tok_per_sec)
        tok_per_sec_str = f"{tok_per_sec:,.0f}"
        device_mfu_str = f"{device_mfu:.1f}%" if device_mfu is not None else "n/a"
        h100_mfu_str = f"{h100_mfu:.1f}%" if h100_mfu is not None else "n/a"

    if BENCHMARK_MODE:
        remaining_str = f"{max(BENCHMARK_STEPS - (step + 1), 0)} steps"
        progress_str = f"{100 * (step + 1) / BENCHMARK_STEPS:.1f}%"
    else:
        remaining_str = f"{max(0, TIME_BUDGET - total_training_time):.0f}s"
        progress_str = f"{pct_done:.1f}%"

    print(f"\rstep {step:05d} ({progress_str}) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec_str} | mfu: {device_mfu_str} | h100_mfu: {h100_mfu_str} | epoch: {epoch} | remaining: {remaining_str}    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if BENCHMARK_MODE and step >= BENCHMARK_STEPS:
        break

    # Time's up — but only stop after warmup steps so we don't count compilation
    if not BENCHMARK_MODE and step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

val_bpb = None
if not BENCHMARK_MODE:
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_tok_per_sec = (TOTAL_BATCH_SIZE * measured_steps / total_training_time) if total_training_time > 0 and measured_steps > 0 else None
steady_train_flops = num_flops_per_token * steady_tok_per_sec if steady_tok_per_sec is not None else None
steady_state_mfu = compute_mfu(device_peak_flops, num_flops_per_token, steady_tok_per_sec)
steady_state_h100_mfu = compute_mfu(H100_BF16_PEAK_FLOPS, num_flops_per_token, steady_tok_per_sec)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"mode:             {'benchmark' if BENCHMARK_MODE else 'train'}")
print(f"compile_backend:  {MODEL_COMPILE_BACKEND}")
print(f"compile_mode:     {INDUCTOR_COMPILE_MODE}" if MODEL_COMPILE_BACKEND == "inductor" else "compile_mode:     n/a")
print(f"compile_scope:    {COMPILE_SCOPE}")
print(f"optimizer_compile:{OPTIMIZER_COMPILE_BACKEND}")
if val_bpb is not None:
    print(f"val_bpb:          {val_bpb:.6f}")
else:
    print("val_bpb:          skipped")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"startup_seconds:  {startup_time:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"tok_per_sec:      {steady_tok_per_sec:,.0f}" if steady_tok_per_sec is not None else "tok_per_sec:      n/a")
print(f"train_tflops:     {steady_train_flops / 1e12:.2f}" if steady_train_flops is not None else "train_tflops:     n/a")
print(f"mfu_percent:      {steady_state_mfu:.2f}" if steady_state_mfu is not None else "mfu_percent:      n/a")
print(f"h100_mfu_percent: {steady_state_h100_mfu:.2f}" if steady_state_h100_mfu is not None else "h100_mfu_percent: n/a")
print(f"peak_tflops_est:  {device_peak_flops / 1e12:.2f}" if device_peak_flops is not None else "peak_tflops_est:  n/a")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"measured_steps:   {measured_steps}")
if BENCHMARK_MODE and measured_steps == 0:
    print("benchmark_note:   use --benchmark-steps >= 2 for steady-state throughput")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
