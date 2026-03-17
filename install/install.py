#!/usr/bin/env python3
"""
Unified AutoResearch Installer
Handles cross-platform installation for Linux, macOS, and Windows 11
"""

import os
import sys
import platform
import subprocess
import shutil
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/installation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnvironmentValidator:
    """Validate system environment and dependencies"""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.system_info = {}

    def detect_os(self) -> str:
        """Detect operating system"""
        system = platform.system()
        self.system_info['os'] = system
        self.system_info['platform'] = platform.platform()
        return system

    def check_python(self) -> bool:
        """Check Python version >= 3.10"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            self.issues.append(f"Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        self.system_info['python_version'] = f"{version.major}.{version.minor}.{version.micro}"
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True

    def check_git(self) -> bool:
        """Check if git is installed"""
        if shutil.which('git') is None:
            self.issues.append("Git not found. Please install git first.")
            return False
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            version = result.stdout.strip()
            self.system_info['git'] = version
            logger.info(f"✓ {version}")
            return True
        except Exception as e:
            self.issues.append(f"Git check failed: {e}")
            return False

    def detect_gpu(self) -> Optional[Dict]:
        """Detect NVIDIA GPU"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                   '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        gpus.append({
                            'index': parts[0].strip(),
                            'name': parts[1].strip(),
                            'memory_gb': int(parts[2].strip().split()[0]) / 1024
                        })
                self.system_info['gpus'] = gpus
                logger.info(f"✓ Found {len(gpus)} GPU(s)")
                for gpu in gpus:
                    logger.info(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.0f}GB)")
                return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.warnings.append("CUDA not detected. CPU-only mode available.")
        return None

    def detect_package_manager(self) -> str:
        """Detect available package manager (uv > pip > conda)"""
        if shutil.which('uv'):
            logger.info("✓ Package manager: uv (preferred)")
            return 'uv'
        elif shutil.which('conda'):
            logger.info("⊙ Package manager: conda")
            return 'conda'
        elif shutil.which('pip'):
            logger.info("⊙ Package manager: pip")
            return 'pip'
        else:
            self.issues.append("No package manager found (uv, pip, or conda)")
            return None

    def check_disk_space(self, required_gb: int = 250) -> bool:
        """Check available disk space"""
        try:
            import shutil
            stat = shutil.disk_usage(Path.home())
            available_gb = stat.free / (1024**3)
            if available_gb < required_gb:
                self.warnings.append(f"Only {available_gb:.0f}GB available (need {required_gb}GB)")
            else:
                logger.info(f"✓ Available disk: {available_gb:.0f}GB")
            return True
        except Exception as e:
            self.warnings.append(f"Disk check failed: {e}")
            return False

    def validate_all(self) -> bool:
        """Run all validation checks"""
        logger.info("\n" + "="*60)
        logger.info("ENVIRONMENT VALIDATION")
        logger.info("="*60)

        self.detect_os()
        success = True

        if not self.check_python():
            success = False
        if not self.check_git():
            success = False

        self.detect_gpu()
        self.detect_package_manager()
        self.check_disk_space()

        logger.info("\n" + "="*60)
        if self.issues:
            logger.error("BLOCKING ISSUES:")
            for issue in self.issues:
                logger.error(f"✗ {issue}")
            success = False

        if self.warnings:
            logger.warning("WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"⚠ {warning}")

        if success:
            logger.info("✓ All checks passed!")

        logger.info("="*60 + "\n")
        return success


class Installer:
    """Main installation orchestrator"""

    def __init__(self, repo_root: Path, mode: str = 'interactive'):
        self.repo_root = repo_root
        self.mode = mode
        self.config_wizard = None
        self.validator = EnvironmentValidator()
        self.package_manager = None

    def validate_environment(self) -> bool:
        """Run environment validation"""
        if not self.validator.validate_all():
            logger.error("Environment validation failed. Cannot proceed.")
            return False
        self.package_manager = self.validator.detect_package_manager()
        return True

    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("\n" + "="*60)
        logger.info("INSTALLING DEPENDENCIES")
        logger.info("="*60)

        requirements = self.repo_root / 'pyproject.toml'
        if not requirements.exists():
            logger.error(f"Dependencies file not found: {requirements}")
            return False

        try:
            if self.package_manager == 'uv':
                logger.info("Installing with uv...")
                subprocess.run(['uv', 'sync'], cwd=self.repo_root, check=True)
            elif self.package_manager == 'conda':
                logger.info("Installing with conda...")
                subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'],
                             cwd=self.repo_root, check=True)
            else:
                logger.info("Installing with pip...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                             cwd=self.repo_root, check=True)

            logger.info("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Dependency installation failed: {e}")
            return False

    def prepare_data(self) -> bool:
        """Run data preparation (one-time setup)"""
        logger.info("\n" + "="*60)
        logger.info("PREPARING DATA")
        logger.info("="*60)

        try:
            prepare_script = self.repo_root / 'prepare.py'
            if prepare_script.exists():
                logger.info("Running prepare.py (this may take 2-5 minutes)...")
                subprocess.run([sys.executable, 'prepare.py'],
                             cwd=self.repo_root, check=True)
                logger.info("✓ Data preparation complete")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Data preparation failed: {e}")
            return False

    def run_config_wizard(self) -> bool:
        """Run interactive configuration wizard"""
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION WIZARD")
        logger.info("="*60)

        # Import config wizard (will create it in next step)
        try:
            from config_wizard import ConfigWizard
            wizard = ConfigWizard(self.repo_root)
            config = wizard.run()
            logger.info("✓ Configuration created")
            return True
        except ImportError:
            logger.warning("Config wizard not yet available, skipping...")
            return True

    def run_baseline(self) -> bool:
        """Run baseline training experiment"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING BASELINE EXPERIMENT")
        logger.info("="*60)

        try:
            logger.info("Starting baseline training (5 minutes)...")
            subprocess.run([sys.executable, 'train.py'],
                         cwd=self.repo_root, check=True)
            logger.info("✓ Baseline experiment completed")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Baseline failed (not critical): {e}")
            return False

    def create_directories(self) -> bool:
        """Create necessary directories"""
        directories = [
            'logs',
            'results/experiments',
            'results/knowledge-graph',
            'results/podcast-content',
            'results/cold-storage',
            'config/agents',
            'config/services',
            'agents/prompts',
            'agents/tools'
        ]

        for dir_path in directories:
            full_path = self.repo_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        logger.info("✓ Directory structure created")
        return True

    def install(self) -> bool:
        """Execute complete installation"""
        logger.info("\n" + "="*80)
        logger.info("AUTORESEARCH INSTALLATION")
        logger.info(f"OS: {platform.system()}")
        logger.info(f"Mode: {self.mode}")
        logger.info("="*80)

        steps = [
            ("Environment Validation", self.validate_environment),
            ("Directory Setup", self.create_directories),
            ("Dependency Installation", self.install_dependencies),
            ("Data Preparation", self.prepare_data),
            ("Configuration Setup", self.run_config_wizard),
            ("Baseline Experiment", self.run_baseline),
        ]

        for step_name, step_func in steps:
            logger.info(f"\n[{'='*60}]")
            logger.info(f"Step: {step_name}")
            logger.info(f"[{'='*60}]")

            try:
                if not step_func():
                    if step_name in ["Baseline Experiment"]:
                        logger.warning(f"⚠ {step_name} failed (continuing anyway...)")
                        continue
                    logger.error(f"✗ {step_name} failed")
                    return False
            except Exception as e:
                logger.error(f"✗ {step_name} error: {e}")
                return False

        logger.info("\n" + "="*80)
        logger.info("✓ INSTALLATION COMPLETE!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("1. Run: ar status")
        logger.info("2. Run: ar start --agents 1")
        logger.info("3. Run: ar chat default")
        logger.info("\nFor more info: ar help")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='AutoResearch Installation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py --mode interactive
  python install.py --skip-baseline
  python install.py --docker
        """
    )

    parser.add_argument('--mode', choices=['interactive', 'auto', 'docker'],
                       default='interactive',
                       help='Installation mode (default: interactive)')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline experiment')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data preparation')
    parser.add_argument('--docker', action='store_true',
                       help='Setup for Docker deployment')
    parser.add_argument('--repo-root', type=Path, default=Path.cwd(),
                       help='Root directory of autoresearch repo')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create logs directory
    logs_dir = args.repo_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and run installer
    installer = Installer(args.repo_root, mode=args.mode)

    if installer.install():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
