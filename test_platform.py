"""
Tests for platform_utils.py — verifies return types and basic contracts
for all exported functions.
"""

import torch
import pytest

from platform_utils import (
    detect_platform,
    get_device,
    get_device_info,
    get_attention_forward,
    get_compile_fn,
    should_compile,
    get_autocast_ctx,
    sync_device,
    get_peak_memory_mb,
    seed_device,
    validate_window_pattern,
    start_memory_tracking,
    get_tracked_peak_memory_mb,
)


# ---------------------------------------------------------------------------
# detect_platform
# ---------------------------------------------------------------------------

class TestDetectPlatform:
    def test_returns_string(self):
        result = detect_platform()
        assert isinstance(result, str)

    def test_returns_valid_platform(self):
        result = detect_platform()
        assert result in ("cuda", "mps", "cpu")

    def test_deterministic(self):
        """Calling detect_platform twice should give the same result."""
        assert detect_platform() == detect_platform()


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type_matches_platform(self):
        plat = detect_platform()
        device = get_device()
        assert device.type == plat


# ---------------------------------------------------------------------------
# get_device_info
# ---------------------------------------------------------------------------

class TestGetDeviceInfo:
    def test_returns_dict(self):
        info = get_device_info()
        assert isinstance(info, dict)

    def test_has_required_keys(self):
        info = get_device_info()
        assert "gpu" in info
        assert "ram_gb" in info
        assert "framework" in info

    def test_gpu_is_string(self):
        info = get_device_info()
        assert isinstance(info["gpu"], str)
        assert len(info["gpu"]) > 0

    def test_ram_gb_is_numeric(self):
        info = get_device_info()
        assert isinstance(info["ram_gb"], (int, float))
        assert info["ram_gb"] >= 0

    def test_framework_contains_pytorch(self):
        info = get_device_info()
        assert "pytorch" in info["framework"]

    def test_framework_matches_platform(self):
        plat = detect_platform()
        info = get_device_info()
        assert plat in info["framework"]


# ---------------------------------------------------------------------------
# get_attention_forward
# ---------------------------------------------------------------------------

class TestGetAttentionForward:
    def test_returns_callable(self):
        attn_fn = get_attention_forward()
        assert callable(attn_fn)

    def test_basic_attention_shape(self):
        """Verify the attention function returns the correct output shape."""
        attn_fn = get_attention_forward()
        device = get_device()
        B, T, H_q, H_kv, D = 1, 16, 4, 4, 32
        q = torch.randn(B, T, H_q, D, device=device, dtype=torch.float32)
        k = torch.randn(B, T, H_kv, D, device=device, dtype=torch.float32)
        v = torch.randn(B, T, H_kv, D, device=device, dtype=torch.float32)
        out = attn_fn(q, k, v, causal=True, window_size=(-1, 0))
        assert out.shape == (B, T, H_q, D)

    def test_gqa_attention_shape(self):
        """Verify GQA works (fewer KV heads than query heads)."""
        attn_fn = get_attention_forward()
        device = get_device()
        plat = detect_platform()
        # FA3 on CUDA handles GQA natively; SDPA path expands heads
        B, T, H_q, H_kv, D = 1, 16, 8, 2, 32
        q = torch.randn(B, T, H_q, D, device=device, dtype=torch.float32)
        k = torch.randn(B, T, H_kv, D, device=device, dtype=torch.float32)
        v = torch.randn(B, T, H_kv, D, device=device, dtype=torch.float32)
        out = attn_fn(q, k, v, causal=True, window_size=(-1, 0))
        assert out.shape == (B, T, H_q, D)

    def test_sliding_window_attention(self):
        """Verify sliding window masking produces valid output."""
        attn_fn = get_attention_forward()
        device = get_device()
        B, T, H, D = 1, 32, 2, 16
        q = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
        k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
        v = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
        out = attn_fn(q, k, v, causal=True, window_size=(8, 0))
        assert out.shape == (B, T, H, D)
        # Output should be finite
        assert torch.isfinite(out).all()

    def test_causal_masking(self):
        """Verify that causal masking prevents attending to future tokens."""
        attn_fn = get_attention_forward()
        device = get_device()
        B, T, H, D = 1, 8, 1, 16
        torch.manual_seed(42)
        q = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
        k = torch.randn(B, T, H, D, device=device, dtype=torch.float32)
        # Make v values distinctive per position
        v = torch.zeros(B, T, H, D, device=device, dtype=torch.float32)
        v[0, 0, 0, :] = 1.0  # only first position has non-zero value
        out = attn_fn(q, k, v, causal=True, window_size=(-1, 0))
        # The first token can only attend to itself, so its output must be v[0]
        # (after softmax it gets weight 1.0 for position 0)
        assert torch.allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-4)


# ---------------------------------------------------------------------------
# get_compile_fn
# ---------------------------------------------------------------------------

class TestGetCompileFn:
    def test_returns_callable(self):
        compile_fn = get_compile_fn()
        assert callable(compile_fn)

    def test_identity_on_non_cuda(self):
        """On MPS/CPU, compile_fn should be identity."""
        plat = detect_platform()
        if plat != "cuda":
            compile_fn = get_compile_fn()
            model = torch.nn.Linear(4, 4)
            result = compile_fn(model)
            assert result is model


# ---------------------------------------------------------------------------
# should_compile
# ---------------------------------------------------------------------------

class TestShouldCompile:
    def test_returns_bool(self):
        result = should_compile()
        assert isinstance(result, bool)

    def test_matches_platform(self):
        plat = detect_platform()
        result = should_compile()
        if plat == "cuda":
            assert result is True
        else:
            assert result is False


# ---------------------------------------------------------------------------
# get_autocast_ctx
# ---------------------------------------------------------------------------

class TestGetAutocastCtx:
    def test_returns_context_manager(self):
        ctx = get_autocast_ctx()
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")

    def test_can_enter_and_exit(self):
        ctx = get_autocast_ctx()
        with ctx:
            x = torch.ones(2, 2)
            _ = x + x


# ---------------------------------------------------------------------------
# sync_device
# ---------------------------------------------------------------------------

class TestSyncDevice:
    def test_does_not_raise(self):
        """sync_device should execute without errors on any platform."""
        sync_device()

    def test_callable(self):
        assert callable(sync_device)


# ---------------------------------------------------------------------------
# get_peak_memory_mb
# ---------------------------------------------------------------------------

class TestGetPeakMemoryMb:
    def test_returns_float(self):
        result = get_peak_memory_mb()
        assert isinstance(result, float)

    def test_non_negative(self):
        result = get_peak_memory_mb()
        assert result >= 0.0


# ---------------------------------------------------------------------------
# seed_device
# ---------------------------------------------------------------------------

class TestSeedDevice:
    def test_does_not_raise(self):
        seed_device(42)

    def test_custom_seed(self):
        seed_device(123)

    def test_deterministic_output(self):
        """Seeding should produce deterministic random tensors."""
        seed_device(42)
        a = torch.randn(10)
        seed_device(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# validate_window_pattern
# ---------------------------------------------------------------------------

class TestWindowPatternValidation:
    def test_valid_patterns(self):
        assert validate_window_pattern("SSSL") is True
        assert validate_window_pattern("SL") is True
        assert validate_window_pattern("L") is True
        assert validate_window_pattern("S") is True

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="WINDOW_PATTERN"):
            validate_window_pattern("SSLX")
        with pytest.raises(ValueError, match="WINDOW_PATTERN"):
            validate_window_pattern("")

    def test_case_insensitive_valid(self):
        assert validate_window_pattern("sssl") is True


# ---------------------------------------------------------------------------
# Memory Tracking
# ---------------------------------------------------------------------------

class TestMemoryTracking:
    def test_tracked_peak_returns_float(self):
        start_memory_tracking()
        peak = get_tracked_peak_memory_mb()
        assert isinstance(peak, float)
        assert peak >= 0.0

    def test_start_tracking_is_idempotent(self):
        """Calling start twice should not crash."""
        start_memory_tracking()
        start_memory_tracking()
        peak = get_tracked_peak_memory_mb()
        assert isinstance(peak, float)
