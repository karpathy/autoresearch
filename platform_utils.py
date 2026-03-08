"""
Platform abstraction utilities for multi-platform training support.
Provides unified interfaces for CUDA, MPS (Apple Silicon), and CPU backends
so that train.py can be portable across hardware.

Do NOT rename this file to platform.py — that shadows the builtin.
"""

import os
import subprocess
import threading
from contextlib import nullcontext

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Platform Detection
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    """Detect the available hardware platform.

    Returns:
        'cuda' if NVIDIA GPU is available,
        'mps' if Apple Silicon GPU is available,
        'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device() -> torch.device:
    """Return a torch.device for the best available platform."""
    return torch.device(detect_platform())


# ---------------------------------------------------------------------------
# Device Info
# ---------------------------------------------------------------------------

def get_device_info() -> dict:
    """Return a dict describing the current device.

    Keys:
        gpu: str — device name (e.g. "Apple M2 Max", "NVIDIA H100", "cpu")
        ram_gb: int — system RAM in GB (or GPU VRAM on CUDA)
        framework: str — e.g. "pytorch-cuda", "pytorch-mps", "pytorch-cpu"
    """
    plat = detect_platform()

    if plat == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        # Report GPU VRAM in GB
        total_mem = torch.cuda.get_device_properties(0).total_mem
        ram_gb = round(total_mem / (1024 ** 3))
        framework = "pytorch-cuda"
    elif plat == "mps":
        gpu_name = _get_macos_chip_name()
        ram_gb = _get_macos_ram_gb()
        framework = "pytorch-mps"
    else:
        gpu_name = "cpu"
        ram_gb = _get_system_ram_gb()
        framework = "pytorch-cpu"

    return {"gpu": gpu_name, "ram_gb": ram_gb, "framework": framework}


def _get_macos_chip_name() -> str:
    """Get the Apple Silicon chip name via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        name = result.stdout.strip()
        return name if name else "Apple Silicon (unknown)"
    except Exception:
        return "Apple Silicon (unknown)"


def _get_macos_ram_gb() -> int:
    """Get system RAM in GB on macOS using os.sysconf."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        ram_bytes = page_size * page_count
        return round(ram_bytes / (1024 ** 3))
    except (ValueError, OSError):
        return 0


def _get_system_ram_gb() -> int:
    """Get system RAM in GB (cross-platform fallback)."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        ram_bytes = page_size * page_count
        return round(ram_bytes / (1024 ** 3))
    except (ValueError, OSError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def get_attention_forward():
    """Return an attention function with the signature:

        attn_fn(q, k, v, causal: bool, window_size: tuple[int, int]) -> Tensor

    On CUDA: tries Flash Attention 3 from the `kernels` package, falls back
    to PyTorch's scaled_dot_product_attention (SDPA).

    On MPS/CPU: uses SDPA with a manually constructed sliding-window causal mask.

    Input shapes: q is (B, T, H_q, D), k/v are (B, T, H_kv, D).
    Output shape: (B, T, H_q, D).
    """
    plat = detect_platform()

    if plat == "cuda":
        return _get_cuda_attention()

    # MPS and CPU both use the SDPA path
    return _sdpa_attention


def _get_cuda_attention():
    """Try loading Flash Attention 3 from the kernels package; fall back to SDPA."""
    try:
        from kernels import get_kernel
        cap = torch.cuda.get_device_capability()
        repo = (
            "varunneal/flash-attention-3"
            if cap == (9, 0)
            else "kernels-community/flash-attn3"
        )
        fa3 = get_kernel(repo).flash_attn_interface

        def _fa3_attention(q, k, v, causal=True, window_size=(-1, 0)):
            return fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

        return _fa3_attention
    except Exception:
        return _sdpa_attention


def _build_sliding_window_causal_mask(T: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Build a boolean attention mask that is both causal and sliding-window.

    Returns a (T, T) bool tensor where True means "attend".
    """
    # Start with a causal mask (lower-triangular)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    if window_size > 0 and window_size < T:
        # Remove positions that are too far back: keep only the last `window_size` positions
        # triu with offset -(window_size - 1) zeroes out positions older than the window
        window_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device),
            diagonal=-(window_size - 1),
        )
        mask = mask & window_mask

    return mask


def _sdpa_attention(q, k, v, causal=True, window_size=(-1, 0)):
    """SDPA-based attention with GQA expansion and optional sliding window.

    Args:
        q: (B, T, H_q, D)
        k: (B, T, H_kv, D)
        v: (B, T, H_kv, D)
        causal: whether to apply causal masking
        window_size: tuple (left, right). left > 0 activates sliding window.
                     Matches FA3 convention: (-1, 0) means full causal.

    Returns:
        (B, T, H_q, D)
    """
    B, T, H_q, D = q.shape
    H_kv = k.shape[2]

    # GQA: expand KV heads to match query heads via repeat_interleave
    if H_kv < H_q:
        repeat_factor = H_q // H_kv
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

    # Transpose to (B, H, T, D) for SDPA
    q = q.transpose(1, 2)  # (B, H_q, T, D)
    k = k.transpose(1, 2)  # (B, H_q, T, D)
    v = v.transpose(1, 2)  # (B, H_q, T, D)

    # Determine whether we need a sliding window mask
    left_window = window_size[0] if isinstance(window_size, (tuple, list)) else window_size
    need_sliding_window = causal and left_window > 0 and left_window < T

    if need_sliding_window:
        # Build a sliding-window causal mask
        attn_mask = _build_sliding_window_causal_mask(T, left_window, q.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
    else:
        # Pure causal or no masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Transpose back to (B, T, H, D)
    y = y.transpose(1, 2)
    return y


def validate_window_pattern(pattern: str) -> bool:
    """Validate that a window pattern contains only 'S' and 'L' characters.

    Raises ValueError if invalid. Returns True if valid.
    """
    if not pattern:
        raise ValueError("WINDOW_PATTERN must be a non-empty string of 'S' and 'L' characters")
    if not all(c in "SL" for c in pattern.upper()):
        raise ValueError(
            f"WINDOW_PATTERN must contain only 'S' and 'L' characters, got '{pattern}'"
        )
    return True


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def get_compile_fn():
    """Return torch.compile on CUDA, identity function on MPS/CPU.

    MPS and CPU do not benefit from (or support well) torch.compile,
    so we return a no-op pass-through.
    """
    if detect_platform() == "cuda":
        return torch.compile
    return lambda model, **kwargs: model


def should_compile() -> bool:
    """Return True if torch.compile should be used on the current platform."""
    return detect_platform() == "cuda"


# ---------------------------------------------------------------------------
# Autocast
# ---------------------------------------------------------------------------

def get_autocast_ctx():
    """Return an appropriate autocast context manager for the current platform.

    - CUDA: bfloat16 autocast
    - CPU: bfloat16 autocast
    - MPS: nullcontext (autocast not well supported on MPS)
    """
    plat = detect_platform()

    if plat == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if plat == "cpu":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    # MPS: autocast not supported
    return nullcontext()


# ---------------------------------------------------------------------------
# Synchronization
# ---------------------------------------------------------------------------

def sync_device():
    """Synchronize the current device (wait for all kernels to complete).

    - CUDA: torch.cuda.synchronize()
    - MPS: torch.mps.synchronize()
    - CPU: no-op
    """
    plat = detect_platform()
    if plat == "cuda":
        torch.cuda.synchronize()
    elif plat == "mps":
        torch.mps.synchronize()
    # CPU: nothing to synchronize


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def get_peak_memory_mb() -> float:
    """Return peak memory usage in MB for the current device.

    - CUDA: peak GPU memory allocated
    - MPS: peak MPS memory allocated (requires PyTorch 2.1+)
    - CPU: returns 0.0 (no GPU memory tracking)
    """
    plat = detect_platform()

    if plat == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if plat == "mps":
        # Use tracked peak if available (more accurate), fall back to current
        if _tracking and _peak_memory_mb > 0:
            return _peak_memory_mb
        if hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# MPS Peak Memory Tracking
# ---------------------------------------------------------------------------

_peak_memory_mb = 0.0
_tracking = False
_tracker_stop = threading.Event()


def start_memory_tracking(interval_seconds: float = 0.1):
    """Start a background thread that polls MPS memory to track peak usage.

    On CUDA this is a no-op (CUDA tracks peak natively via max_memory_allocated).
    On CPU this is a no-op.
    """
    global _peak_memory_mb, _tracking

    plat = detect_platform()
    if plat != "mps" or not hasattr(torch.mps, "current_allocated_memory"):
        return

    _peak_memory_mb = 0.0
    _tracking = True
    _tracker_stop.clear()

    def _sampler():
        global _peak_memory_mb
        while not _tracker_stop.is_set():
            try:
                current = torch.mps.current_allocated_memory() / (1024 * 1024)
                if current > _peak_memory_mb:
                    _peak_memory_mb = current
            except Exception:
                pass
            _tracker_stop.wait(interval_seconds)

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()


def get_tracked_peak_memory_mb() -> float:
    """Return the peak memory observed since start_memory_tracking().

    On CUDA, delegates to torch.cuda.max_memory_allocated().
    On other platforms, returns the tracked peak or 0.0.
    """
    plat = detect_platform()
    if plat == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return _peak_memory_mb


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_device(seed: int = 42):
    """Seed the random number generator for the current device.

    Sets torch.manual_seed (covers CPU) and the device-specific RNG.
    """
    torch.manual_seed(seed)
    plat = detect_platform()

    if plat == "cuda":
        torch.cuda.manual_seed(seed)
    elif plat == "mps":
        # torch.manual_seed already covers MPS on recent PyTorch versions.
        # torch.mps.manual_seed is available since PyTorch 2.3+.
        if hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)
