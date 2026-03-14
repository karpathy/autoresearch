# Flash Attention 2.8.3 Verification Script

import sys

import torch

try:
    import flash_attn
    from flash_attn import flash_attn_func
except ImportError as exc:
    flash_attn = None
    flash_attn_func = None
    FLASH_ATTN_IMPORT_ERROR = exc
else:
    FLASH_ATTN_IMPORT_ERROR = None

def verify_installation():
    """Verify Flash Attention installation"""
    print("=" * 70)
    print("Flash Attention 2.8.3 Environment Verification")
    print("=" * 70)

    if FLASH_ATTN_IMPORT_ERROR is not None:
        print("\n[FAILED] Flash Attention is not installed in this environment.")
        print(f"  - Import error: {FLASH_ATTN_IMPORT_ERROR}")
        print("  - Note: the bundled Windows wheel is currently pinned to CPython 3.12 AMD64.")
        return False

    # Version info
    print(f"\nVersion Information:")
    print(f"  - PyTorch:         {torch.__version__}")
    print(f"  - Flash Attention: {flash_attn.__version__}")
    print(f"  - CUDA Available:  {torch.cuda.is_available()}")
    print(f"  - CUDA Version:    {torch.version.cuda}")

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Functional test
    print(f"\nFunctional Test:")
    try:
        # Create test tensors
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        dtype = torch.float16

        q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device='cuda', dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device='cuda', dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device='cuda', dtype=dtype)

        # Match the causal attention path used by train.py.
        output = flash_attn_func(q, k, v, causal=True)

        print(f"  [OK] Forward pass: {q.shape} -> {output.shape}")

        # Performance test
        import time

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(100):
            _ = flash_attn_func(q, k, v, causal=True)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"  [OK] Performance:  {100/elapsed:.2f} iterations/sec")
        print(f"\n[SUCCESS] All tests passed! Flash Attention 2.8.3 is ready!")

    except Exception as e:
        print(f"  [FAILED] Test failed: {e}")
        return False

    print("=" * 70)
    return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
