"""
Universal AutoResearch Framework (UARF) - Core Package
Ein universelles Framework für autonomes ML-Training auf allen Plattformen.
"""

__version__ = "0.1.0-mvp"
__author__ = "UARF Team"

from .core.hardware_detector import HardwareDetector
from .core.model_selector import ModelSelector
from .core.trainer import UniversalTrainer
from .core.config import UARFConfig

__all__ = [
    "HardwareDetector",
    "ModelSelector", 
    "UniversalTrainer",
    "UARFConfig",
]
