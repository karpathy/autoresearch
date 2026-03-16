"""
tests/test_platform.py — unit tests for platform detection.

All tests run on any platform (Linux, Mac, CI without GPU).
No GPU required — torch.cuda.is_available() is mocked.
"""
from __future__ import annotations
import sys
from unittest.mock import MagicMock, patch
import pytest


def _mock_cuda(vram_gb: float, capability: tuple = (9, 0)):
    props = MagicMock()
    props.total_memory = int(vram_gb * 1e9)
    return {
        "torch.cuda.is_available": lambda: True,
        "torch.cuda.get_device_capability": lambda: capability,
        "torch.cuda.get_device_properties": lambda _: props,
        "torch.backends.mps.is_available": lambda: False,
    }


def _load_platform_config():
    """Load platform_config.py via importlib to avoid import side-effects."""
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "platform_config",
        pathlib.Path(__file__).parent.parent / "platform_config.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_platform_info_fields():
    """PlatformInfo dataclass accepts all fields and exposes them correctly."""
    mod = pytest.importorskip(
        "platform_config",
        reason="platform_config.py not yet created (Phase 2)",
    )
    import torch
    info = mod.PlatformInfo(
        kind="cpu", device=torch.device("cpu"),
        is_mac=False, is_linux=True,
        supports_compile=False, supports_flash_attn=False,
        supports_bf16=False, supports_fp16=False,
        recommended_depth=2, recommended_batch=4096,
        recommended_seq_len=64, recommended_eval_tokens=65536,
        description="test",
    )
    assert info.kind == "cpu"
    assert info.recommended_depth >= 1


@pytest.mark.unit
@pytest.mark.cpu
def test_detect_cpu_fallback(monkeypatch):
    """detect() returns cpu kind when no GPU is available."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    mod = _load_platform_config()
    info = mod.detect()
    assert info.kind == "cpu"
    assert str(info.device) == "cpu"
    assert info.supports_compile is False
    assert info.supports_flash_attn is False
    assert "CPU" in info.description


@pytest.mark.unit
@pytest.mark.cpu
def test_detect_cuda_hopper(monkeypatch):
    """detect() returns cuda kind with FA3 support on H100 (compute 9.0)."""
    mocks = _mock_cuda(vram_gb=80.0, capability=(9, 0))
    monkeypatch.setattr("torch.cuda.is_available", mocks["torch.cuda.is_available"])
    monkeypatch.setattr("torch.cuda.get_device_capability", mocks["torch.cuda.get_device_capability"])
    monkeypatch.setattr("torch.cuda.get_device_properties", mocks["torch.cuda.get_device_properties"])
    monkeypatch.setattr("torch.backends.mps.is_available", mocks["torch.backends.mps.is_available"])
    mod = _load_platform_config()
    info = mod.detect()
    assert info.kind == "cuda"
    assert info.supports_flash_attn is True
    assert info.supports_compile is True
    assert info.supports_bf16 is True


@pytest.mark.unit
@pytest.mark.cpu
def test_detect_cuda_mid_range(monkeypatch):
    """detect() returns cuda kind with FA3 support on non-Hopper GPU (compute 8.6)."""
    mocks = _mock_cuda(vram_gb=24.0, capability=(8, 6))
    monkeypatch.setattr("torch.cuda.is_available", mocks["torch.cuda.is_available"])
    monkeypatch.setattr("torch.cuda.get_device_capability", mocks["torch.cuda.get_device_capability"])
    monkeypatch.setattr("torch.cuda.get_device_properties", mocks["torch.cuda.get_device_properties"])
    monkeypatch.setattr("torch.backends.mps.is_available", mocks["torch.backends.mps.is_available"])
    mod = _load_platform_config()
    info = mod.detect()
    assert info.kind == "cuda"
    assert info.supports_compile is True


@pytest.mark.unit
@pytest.mark.cpu
def test_detect_cuda_limited_vram(monkeypatch):
    """detect() adjusts recommended_batch for GPUs with limited VRAM."""
    mocks = _mock_cuda(vram_gb=8.0, capability=(8, 6))
    monkeypatch.setattr("torch.cuda.is_available", mocks["torch.cuda.is_available"])
    monkeypatch.setattr("torch.cuda.get_device_capability", mocks["torch.cuda.get_device_capability"])
    monkeypatch.setattr("torch.cuda.get_device_properties", mocks["torch.cuda.get_device_properties"])
    monkeypatch.setattr("torch.backends.mps.is_available", mocks["torch.backends.mps.is_available"])
    mod = _load_platform_config()
    info = mod.detect()
    assert info.kind == "cuda"
    # Limited VRAM should yield a smaller recommended batch than H100
    h100_mocks = _mock_cuda(vram_gb=80.0, capability=(9, 0))
    monkeypatch.setattr("torch.cuda.is_available", h100_mocks["torch.cuda.is_available"])
    monkeypatch.setattr("torch.cuda.get_device_capability", h100_mocks["torch.cuda.get_device_capability"])
    monkeypatch.setattr("torch.cuda.get_device_properties", h100_mocks["torch.cuda.get_device_properties"])
    h100_info = mod.detect()
    assert info.recommended_batch <= h100_info.recommended_batch


@pytest.mark.unit
@pytest.mark.cpu
def test_detect_mps(monkeypatch):
    """detect() returns mps kind on Apple Silicon when no CUDA is present."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr("sys.platform", "darwin")
    mod = _load_platform_config()
    # Mock _get_mac_unified_memory_gb inside the freshly loaded module
    monkeypatch.setattr(mod, "_get_mac_unified_memory_gb", lambda: 36.0)
    info = mod.detect()
    assert info.kind == "mps"
    assert info.is_mac is True
    assert info.supports_flash_attn is False
    assert info.supports_bf16 is False
    assert info.supports_fp16 is True
    assert "MPS" in info.description or "Apple" in info.description
