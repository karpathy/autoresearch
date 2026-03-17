#Requires -Version 5.1
<#
.SYNOPSIS
AutoResearch Installation Script for Windows 11

.DESCRIPTION
Unified installer for Windows 11 that handles Python, dependencies, and configuration

.PARAMETER Mode
Installation mode: 'interactive' or 'auto'

.PARAMETER SkipBaseline
Skip baseline training experiment

.PARAMETER SkipData
Skip data preparation

.EXAMPLE
.\install.ps1 -Mode interactive
.\install.ps1 -SkipBaseline

#>

param(
    [ValidateSet('interactive', 'auto')]
    [string]$Mode = 'interactive',
    [switch]$SkipBaseline,
    [switch]$SkipData,
    [switch]$Help
)

# Color functions for output
function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Show-Help {
    $help = @"
AutoResearch Installation Script for Windows 11

Usage: .\install.ps1 [OPTIONS]

Options:
    -Mode <interactive|auto>    Installation mode (default: interactive)
    -SkipBaseline              Skip baseline training experiment
    -SkipData                  Skip data preparation
    -Help                      Show this help message

Examples:
    .\install.ps1 -Mode interactive
    .\install.ps1 -SkipBaseline
    .\install.ps1 -Mode interactive -SkipData

For more information, visit: https://github.com/SuperInstance/autoclaw
"@
    Write-Host $help
    exit 0
}

function Check-Python {
    Write-Header "Checking Python Installation"

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Error "Python 3 is not installed or not in PATH"
        Write-Host ""
        Write-Host "Please install Python from: https://www.python.org/downloads/"
        Write-Host "Make sure to check 'Add Python to PATH' during installation"
        Write-Host ""
        Write-Host "Or use Windows Package Manager:"
        Write-Host "  winget install Python.Python.3.11"
        exit 1
    }

    $version = & python --version 2>&1
    Write-Success $version
}

function Check-Git {
    Write-Header "Checking Git Installation"

    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if (-not $git) {
        Write-Error "Git is not installed or not in PATH"
        Write-Host ""
        Write-Host "Please install Git from: https://git-scm.com/download/win"
        Write-Host ""
        Write-Host "Or use Windows Package Manager:"
        Write-Host "  winget install Git.Git"
        exit 1
    }

    $version = & git --version
    Write-Success $version
}

function Detect-PackageManager {
    Write-Header "Detecting Package Manager"

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Success "Using uv (preferred)"
        return "uv"
    }
    elseif (Get-Command conda -ErrorAction SilentlyContinue) {
        Write-Success "Using conda"
        return "conda"
    }
    elseif (Get-Command pip -ErrorAction SilentlyContinue) {
        Write-Success "Using pip"
        return "pip"
    }
    else {
        Write-Error "No package manager found (uv, pip, or conda)"
        Write-Host ""
        Write-Host "Install one of:"
        Write-Host "  pip: comes with Python (default)"
        Write-Host "  uv: pip install uv"
        Write-Host "  conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html"
        exit 1
    }
}

function Check-ExecutionPolicy {
    $policy = Get-ExecutionPolicy
    if ($policy -eq "Restricted") {
        Write-Warning "PowerShell Execution Policy is Restricted"
        Write-Host "Setting to RemoteSigned for this session..."
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
        Write-Success "Execution policy updated for this session"
    }
}

# Main installation
function Main {
    if ($Help) {
        Show-Help
    }

    Write-Header "AutoResearch Installation for Windows 11"
    Write-Host "OS: $(Get-WmiObject -Class Win32_OperatingSystem | Select-Object -ExpandProperty Caption)"
    Write-Host "PowerShell: $($PSVersionTable.PSVersion.Major).$($PSVersionTable.PSVersion.Minor)"
    Write-Host ""

    # Check execution policy
    Check-ExecutionPolicy

    # Check prerequisites
    Check-Python
    Write-Host ""
    Check-Git
    Write-Host ""

    # Detect package manager
    $pkgManager = Detect-PackageManager
    Write-Host ""

    # Change to repo root
    $repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
    Set-Location $repoRoot

    # Run Python installer
    Write-Header "Launching Installer"
    Write-Host "Running: python install\install.py --mode $Mode"
    Write-Host ""

    $args = @("install\install.py", "--mode", $Mode)
    if ($SkipBaseline) { $args += "--skip-baseline" }
    if ($SkipData) { $args += "--skip-data" }

    & python @args

    if ($LASTEXITCODE -eq 0) {
        Write-Header "Installation Complete!"
        Write-Host ""
        Write-Host "Next steps:"
        Write-Host "  1. Review configuration files in: config\"
        Write-Host "  2. Add your API keys to: config\services\"
        Write-Host "  3. Start agents: ar start --agents 1"
        Write-Host "  4. Check status: ar status"
        Write-Host ""
    }
    else {
        Write-Error "Installation failed"
        exit 1
    }
}

# Run main
Main
