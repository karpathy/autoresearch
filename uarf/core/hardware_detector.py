"""
Hardware Detector - Erkennt automatisch verfügbare Hardware-Ressourcen
und passt Konfigurationen entsprechend an.
"""

import platform
import psutil
import subprocess
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class HardwareSpecs:
    """Hardware-Spezifikationen des aktuellen Systems"""
    platform: str
    platform_release: str
    platform_version: str
    architecture: str
    processor: str
    cpu_count: int
    cpu_freq_min: float  # GHz
    cpu_freq_max: float  # GHz
    ram_total: float  # GB
    ram_available: float  # GB
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram: float  # GB
    gpu_compute_capability: Optional[tuple]
    storage_available: float  # GB
    is_mobile: bool
    is_colab: bool
    is_cluster: bool


class HardwareDetector:
    """Erkennt Hardware und schlägt optimale Konfigurationen vor"""
    
    def __init__(self):
        self.specs = self.detect()
    
    def detect(self) -> HardwareSpecs:
        """Führt vollständige Hardware-Erkennung durch"""
        return HardwareSpecs(
            platform=platform.system(),
            platform_release=platform.release(),
            platform_version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor(),
            cpu_count=psutil.cpu_count(logical=True),
            cpu_freq_min=self._get_cpu_freq_min(),
            cpu_freq_max=self._get_cpu_freq_max(),
            ram_total=psutil.virtual_memory().total / (1024**3),
            ram_available=psutil.virtual_memory().available / (1024**3),
            gpu_available=self._is_gpu_available(),
            gpu_name=self._get_gpu_name(),
            gpu_vram=self._get_gpu_vram(),
            gpu_compute_capability=self._get_compute_capability(),
            storage_available=self._get_storage_available(),
            is_mobile=self._is_mobile_platform(),
            is_colab=self._is_google_colab(),
            is_cluster=self._is_cluster_environment()
        )
    
    def _get_cpu_freq_min(self) -> float:
        """Minimale CPU-Frequenz in GHz ermitteln"""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return freq.min / 1000.0
        except:
            pass
        return 0.0
    
    def _get_cpu_freq_max(self) -> float:
        """Maximale CPU-Frequenz in GHz ermitteln"""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return freq.max / 1000.0
        except:
            pass
        return 0.0
    
    def _is_gpu_available(self) -> bool:
        """Prüft ob GPU verfügbar ist"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_gpu_name(self) -> Optional[str]:
        """GPU-Name ermitteln"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except:
            pass
        return None
    
    def _get_gpu_vram(self) -> float:
        """GPU VRAM in GB ermitteln"""
        try:
            import torch
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                return total_mem / (1024**3)
        except:
            pass
        return 0.0
    
    def _get_compute_capability(self) -> Optional[tuple]:
        """CUDA Compute Capability ermitteln"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_capability(0)
        except:
            pass
        return None
    
    def _get_storage_available(self) -> float:
        """Verfügbarer Speicherplatz in GB"""
        try:
            usage = psutil.disk_usage('/')
            return usage.free / (1024**3)
        except:
            return 0.0
    
    def _is_mobile_platform(self) -> bool:
        """Prüft ob es sich um eine mobile Plattform handelt"""
        mobile_indicators = ['android', 'termux', 'aarch64']
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Termux-Umgebung erkennen
        if 'TERMUX_VERSION' in os.environ:
            return True
            
        return any(ind in system or ind in machine for ind in mobile_indicators)
    
    def _is_google_colab(self) -> bool:
        """Google Colab Umgebung erkennen"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _is_cluster_environment(self) -> bool:
        """Cluster-Umgebung erkennen (SLURM, MPI, etc.)"""
        cluster_indicators = ['SLURM_JOB_ID', 'PBS_NODEFILE', 'OMPI_COMM_WORLD_SIZE']
        return any(ind in os.environ for ind in cluster_indicators)
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Berechnet optimale Konfiguration basierend auf Hardware"""
        config = {}
        
        # RAM-basierte Empfehlungen
        if self.specs.ram_total < 4:
            config['max_model_size'] = '125M'
            config['batch_size'] = 8
            config['precision'] = 'int8'
        elif self.specs.ram_total < 8:
            config['max_model_size'] = '350M'
            config['batch_size'] = 16
            config['precision'] = 'int8'
        elif self.specs.ram_total < 16:
            config['max_model_size'] = '1B'
            config['batch_size'] = 32
            config['precision'] = 'fp16'
        elif self.specs.ram_total < 32:
            config['max_model_size'] = '3B'
            config['batch_size'] = 64
            config['precision'] = 'fp16'
        else:
            config['max_model_size'] = '7B+'
            config['batch_size'] = 128
            config['precision'] = 'bf16'
        
        # GPU-basierte Optimierungen
        if self.specs.gpu_available:
            if self.specs.gpu_vram >= 80:
                config['gpu_mode'] = 'full'
                config['enable_flash_attn'] = True
            elif self.specs.gpu_vram >= 24:
                config['gpu_mode'] = 'large'
                config['enable_flash_attn'] = True
            elif self.specs.gpu_vram >= 8:
                config['gpu_mode'] = 'medium'
                config['enable_flash_attn'] = False
            else:
                config['gpu_mode'] = 'small'
                config['enable_flash_attn'] = False
        else:
            config['gpu_mode'] = 'cpu'
            config['enable_flash_attn'] = False
        
        # Plattformspezifische Optimierungen
        if self.specs.is_mobile:
            config['use_gradient_checkpointing'] = True
            config['save_every_n_steps'] = 50
            config['max_seq_len'] = 512
        elif self.specs.is_colab:
            config['use_gradient_checkpointing'] = False
            config['save_every_n_steps'] = 100
            config['max_seq_len'] = 1024
        else:
            config['use_gradient_checkpointing'] = False
            config['save_every_n_steps'] = 500
            config['max_seq_len'] = 2048
        
        return config
    
    def print_summary(self):
        """Druckt Hardware-Zusammenfassung"""
        print("=" * 60)
        print("UARF HARDWARE DETECTION")
        print("=" * 60)
        print(f"Plattform: {self.specs.platform} ({self.specs.architecture})")
        print(f"CPU: {self.specs.cpu_count} Kerne, {self.specs.cpu_freq_max:.2f} GHz max")
        print(f"RAM: {self.specs.ram_total:.1f} GB gesamt, {self.specs.ram_available:.1f} GB frei")
        
        if self.specs.gpu_available:
            print(f"GPU: {self.specs.gpu_name} ({self.specs.gpu_vram:.1f} GB VRAM)")
            if self.specs.gpu_compute_capability:
                print(f"     Compute Capability: {self.specs.gpu_compute_capability}")
        else:
            print("GPU: Nicht verfügbar")
        
        print(f"Speicher: {self.specs.storage_available:.1f} GB frei")
        print(f"Mobile: {'Ja' if self.specs.is_mobile else 'Nein'}")
        print(f"Colab: {'Ja' if self.specs.is_colab else 'Nein'}")
        print(f"Cluster: {'Ja' if self.specs.is_cluster else 'Nein'}")
        print("=" * 60)
        
        optimal = self.get_optimal_config()
        print("\nEMPFOHLENE KONFIGURATION:")
        print(f"  Maximale Modellgröße: {optimal['max_model_size']}")
        print(f"  Batch Size: {optimal['batch_size']}")
        print(f"  Präzision: {optimal['precision']}")
        print(f"  GPU Modus: {optimal['gpu_mode']}")
        print(f"  Max Sequenzlänge: {optimal['max_seq_len']}")
        print("=" * 60)


# Import für mobile Erkennung
import os
