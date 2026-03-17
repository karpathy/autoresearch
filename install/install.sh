#!/bin/bash
#
# AutoResearch Unified Installer for Linux/macOS
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}=======================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=======================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if Python is available
check_python() {
    print_header "Checking Python Installation"

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        echo "Please install Python 3.10 or later:"
        echo "  Ubuntu/Debian: sudo apt-get install python3 python3-venv python3-pip"
        echo "  macOS: brew install python3"
        echo "  Or download from: https://www.python.org/downloads/"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_success "Python $PYTHON_VERSION found"
}

# Check if Git is available
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed"
        echo "Please install Git:"
        echo "  Ubuntu/Debian: sudo apt-get install git"
        echo "  macOS: brew install git"
        exit 1
    fi

    GIT_VERSION=$(git --version | awk '{print $3}')
    print_success "Git $GIT_VERSION found"
}

# Detect package manager
detect_package_manager() {
    print_header "Detecting Package Manager"

    if command -v uv &> /dev/null; then
        print_success "Using uv (preferred)"
        echo "uv"
    elif command -v conda &> /dev/null; then
        print_success "Using conda"
        echo "conda"
    elif command -v pip &> /dev/null; then
        print_success "Using pip"
        echo "pip"
    else
        print_error "No package manager found (pip, conda, or uv)"
        exit 1
    fi
}

# Show usage
usage() {
    cat << EOF
AutoResearch Installation Script

Usage: $0 [OPTIONS]

Options:
    --interactive         Run interactive configuration wizard (default)
    --skip-baseline       Skip baseline training experiment
    --skip-data           Skip data preparation
    --help               Show this help message

Examples:
    $0 --interactive
    $0 --skip-baseline
    $0 --interactive --skip-data

For more information, visit: https://github.com/SuperInstance/autoclaw
EOF
    exit 0
}

# Main installation
main() {
    print_header "AutoResearch Installation"
    echo "OS: $(uname -s)"
    echo "Architecture: $(uname -m)"
    echo "Python: $(python3 --version)"
    echo ""

    # Check prerequisites
    check_python
    echo ""
    check_git
    echo ""

    # Detect package manager
    PKG_MANAGER=$(detect_package_manager)
    echo ""

    # Change to repo root
    cd "$REPO_ROOT"

    # Run Python installer
    print_header "Launching Installer"
    echo "Running: python3 install/install.py --mode interactive"
    echo ""

    python3 install/install.py --mode interactive "$@"

    if [ $? -eq 0 ]; then
        print_header "Installation Complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Review configuration files in: config/"
        echo "  2. Add your API keys to: config/services/"
        echo "  3. Start agents: ar start --agents 1"
        echo "  4. Check status: ar status"
        echo ""
    else
        print_error "Installation failed"
        exit 1
    fi
}

# Parse arguments
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
fi

main "$@"
