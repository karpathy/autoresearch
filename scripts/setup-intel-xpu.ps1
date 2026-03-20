$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

Write-Host "Creating project virtual environment..."
uv venv --python 3.10 $repoRoot\.venv

Write-Host "Installing Intel Arc / PyTorch XPU runtime..."
uv pip install --python $pythonExe --index-url https://download.pytorch.org/whl/xpu torch==2.10.0+xpu

Write-Host "Installing project dependencies..."
uv pip install --python $pythonExe `
    "matplotlib>=3.10.8" `
    "numpy>=2.2.6" `
    "pandas>=2.3.3" `
    "pyarrow>=21.0.0" `
    "requests>=2.32.0" `
    "rustbpe>=0.1.0" `
    "tiktoken>=0.11.0"

Write-Host "Verifying Intel XPU runtime..."
& $pythonExe -c "import torch; print('torch', torch.__version__); print('xpu_available', torch.xpu.is_available()); print('device', torch.xpu.get_device_name(0))"
