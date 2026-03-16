"""
platform.py — device detection and platform-specific defaults.

This module is imported by prepare.py and train.py to detect the
correct device and apply safe defaults. It does NOT change the
training loop, metric, or agent contract.

Design rule: prepare.py and train.py import from here but their
external interface (val_bpb output, 5-min budget, git workflow)
is completely unchanged.

Supported platforms:
  - Linux + NVIDIA CUDA  (original autoresearch target)
  - Linux + CPU          (fallback for CI / no-GPU Linux)
  - macOS + Apple MPS    (Apple Silicon M1/M2/M3/M4)
  - macOS + CPU          (Intel Mac fallback)
  - Docker CUDA          (Linux inside container, same as Linux CUDA)
"""

from __future__ import annotations

import os
import sys
import platform as _platform
from dataclasses import dataclass
from typing import Literal

import torch


PlatformKind = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class PlatformInfo:
    kind: PlatformKind
    device: torch.device
    is_mac: bool
    is_linux: bool
    supports_compile: bool        # torch.compile / Triton
    supports_flash_attn: bool     # Flash Attention 3 (Hopper+)
    supports_bf16: bool
    supports_fp16: bool
    recommended_depth: int        # default GPT depth for this platform
    recommended_batch: int        # default TOTAL_BATCH_SIZE
    recommended_seq_len: int      # default MAX_SEQ_LEN
    recommended_eval_tokens: int  # default EVAL_TOKENS
    description: str


def detect() -> PlatformInfo:
    """Detect current platform and return safe defaults."""
    is_mac   = sys.platform == "darwin"
    is_linux = sys.platform == "linux"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cap = torch.cuda.get_device_capability()
        # Flash Attention 3 requires Hopper (sm_90+)
        supports_fa3 = cap[0] >= 9
        # sm_80+ supports BF16 natively
        supports_bf16 = cap[0] >= 8
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram_gb >= 70:        # H100 80GB / A100 80GB
            depth, batch, seq, eval_tok = 8, 524288, 1024, 40 * 524288
            desc = f"CUDA {cap} ({vram_gb:.0f}GB) — H100/A100 class"
        elif vram_gb >= 20:      # A5000, RTX 3090/4090, etc.
            depth, batch, seq, eval_tok = 6, 262144, 512, 20 * 524288
            desc = f"CUDA {cap} ({vram_gb:.0f}GB) — consumer high-end"
        elif vram_gb >= 8:       # RTX 3070/4060 etc.
            depth, batch, seq, eval_tok = 4, 131072, 256, 10 * 524288
            desc = f"CUDA {cap} ({vram_gb:.0f}GB) — consumer mid"
        else:                    # 4–8 GB
            depth, batch, seq, eval_tok = 3, 65536, 128, 5 * 524288
            desc = f"CUDA {cap} ({vram_gb:.0f}GB) — limited VRAM"

        return PlatformInfo(
            kind="cuda", device=device,
            is_mac=False, is_linux=is_linux,
            supports_compile=True,
            supports_flash_attn=supports_fa3,
            supports_bf16=supports_bf16,
            supports_fp16=True,
            recommended_depth=depth,
            recommended_batch=batch,
            recommended_seq_len=seq,
            recommended_eval_tokens=eval_tok,
            description=desc,
        )

    if torch.backends.mps.is_available() and is_mac:
        # Apple Silicon — unified memory, no Triton
        # Detect chip generation via sysctl
        mem_gb = _get_mac_unified_memory_gb()
        if mem_gb >= 64:
            depth, batch, seq, eval_tok = 6, 131072, 512, 10 * 524288
        elif mem_gb >= 32:
            depth, batch, seq, eval_tok = 5, 65536, 512, 8 * 524288
        elif mem_gb >= 16:
            depth, batch, seq, eval_tok = 4, 32768, 256, 5 * 524288
        else:
            depth, batch, seq, eval_tok = 3, 16384, 128, 3 * 524288

        return PlatformInfo(
            kind="mps", device=torch.device("mps"),
            is_mac=True, is_linux=False,
            supports_compile=False,      # Triton not available on MPS
            supports_flash_attn=False,   # kernels package targets CUDA
            supports_bf16=False,         # MPS has limited BF16 support
            supports_fp16=True,
            recommended_depth=depth,
            recommended_batch=batch,
            recommended_seq_len=seq,
            recommended_eval_tokens=eval_tok,
            description=f"Apple MPS ({mem_gb:.0f}GB unified memory)",
        )

    # CPU fallback — CI, Intel Mac, no-GPU Linux
    return PlatformInfo(
        kind="cpu", device=torch.device("cpu"),
        is_mac=is_mac, is_linux=is_linux,
        supports_compile=False,
        supports_flash_attn=False,
        supports_bf16=False,
        supports_fp16=False,
        recommended_depth=2,
        recommended_batch=4096,
        recommended_seq_len=64,
        recommended_eval_tokens=64 * 1024,
        description="CPU (no GPU detected — for CI / testing only)",
    )


def _get_mac_unified_memory_gb() -> float:
    """Read total unified memory on macOS via sysctl."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2
        )
        return int(result.stdout.strip()) / 1e9
    except Exception:
        return 8.0  # safe fallback


# Singleton — import once
PLATFORM = detect()


def print_platform_info() -> None:
    p = PLATFORM
    print(f"Platform: {p.description}")
    print(f"  device={p.device}  compile={p.supports_compile}  "
          f"flash_attn={p.supports_flash_attn}  bf16={p.supports_bf16}")
    print(f"  defaults: depth={p.recommended_depth}  "
          f"batch={p.recommended_batch}  seq={p.recommended_seq_len}")


if __name__ == "__main__":
    print_platform_info()
