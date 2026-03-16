# conftest.py
"""
Shared pytest configuration.

Markers:
  unit        — fast, no I/O, no GPU, no API keys
  integration — requires external services
  cuda        — requires NVIDIA GPU
  mps         — requires Apple Silicon
  cpu         — runs on any platform (CPU fallback)
"""
import sys
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: requires live external services")
    config.addinivalue_line("markers", "cuda: requires NVIDIA GPU")
    config.addinivalue_line("markers", "mps: requires Apple Silicon MPS")
    config.addinivalue_line("markers", "cpu: runs on any platform")


def pytest_collection_modifyitems(config, items):
    """Skip cuda/mps tests when hardware not available."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps  = torch.backends.mps.is_available()
    except ImportError:
        has_cuda = has_mps = False

    skip_cuda = pytest.mark.skip(reason="NVIDIA GPU not available")
    skip_mps  = pytest.mark.skip(reason="Apple Silicon MPS not available")

    for item in items:
        if "cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not has_mps:
            item.add_marker(skip_mps)
