"""
Fix Triton sm_120 codegen for Blackwell GPUs (RTX 5070 Ti, 5080, 5090).

Triton 3.5.x adds an "a" suffix to sm_120, generating sm_120a which causes
invalid PTX instructions and random segfaults during torch.compile.

This script patches the installed Triton to use sm_120 (no suffix) for
consumer Blackwell GPUs. Hopper (sm_90a) is unaffected.

Usage: python scripts/fix_triton_blackwell.py
"""
import torch

cap = torch.cuda.get_device_capability()
if cap[0] < 10:
    print(f"GPU is SM {cap[0]}.{cap[1]} (not Blackwell). No fix needed.")
    exit(0)

import triton
import os

compiler_path = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "compiler.py")

with open(compiler_path) as f:
    code = f.read()

old = 'suffix = "a" if capability >= 90 else ""'
new = 'suffix = "a" if 90 <= capability < 120 else ""'

if new in code:
    print("Triton Blackwell fix already applied.")
    exit(0)

if old not in code:
    print(f"WARNING: Could not find expected code in {compiler_path}")
    print("Triton version may have changed. Manual fix may be needed.")
    exit(1)

# Backup
backup = compiler_path + ".bak"
if not os.path.exists(backup):
    with open(backup, "w") as f:
        f.write(code)

code = code.replace(old, new)
with open(compiler_path, "w") as f:
    f.write(code)

# Verify
from importlib import reload
import triton.backends.nvidia.compiler as c
reload(c)
result = c.sm_arch_from_capability(120)
print(f"Fixed: sm_arch_from_capability(120) = {result}")
assert result == "sm_120", f"Fix failed: got {result}"
print("Triton Blackwell fix applied successfully.")
