"""
UARF Configuration - Zentrale Konfigurationsverwaltung
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import os


@dataclass
class UARFConfig:
    """
    Zentrale Konfiguration für UARF
    
    Alle Parameter können via CLI, Config-File oder Code überschrieben werden.
    """
    
    # Modell-Konfiguration
    model_id: str = "Qwen/Qwen2.5-0.5B"
    model_name: Optional[str] = None
    trust_remote_code: bool = True
    
    # Hardware & Performance
    device: str = "auto"  # auto, cuda, cpu, mps
    precision: str = "auto"  # auto, fp32, fp16, bf16, int8
    batch_size: int = 32
    max_seq_len: int = 1024
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = False
    
    # Training
    time_budget_seconds: int = 300  # 5 Minuten Standard
    max_steps: Optional[int] = None
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    
    # Dataset
    dataset_name: str = "karpathy/tinyshakespeare"
    dataset_split: str = "train"
    val_split_ratio: float = 0.1
    num_workers: int = 0  # 0 = main process
    
    # Evaluation
    eval_every_n_steps: int = 100
    eval_tokens: int = 524288  # ~0.5M Tokens
    save_every_n_steps: int = 500
    
    # Logging
    log_every_n_steps: int = 10
    project_name: str = "uarf-experiment"
    run_name: Optional[str] = None
    output_dir: str = "./outputs"
    
    # Advanced
    seed: int = 42
    deterministic: bool = False
    compile_model: bool = True
    flash_attention: bool = False
    
    # Distributed Training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Export
    export_format: Optional[str] = None  # onnx, gguf, tflite
    export_path: Optional[str] = None
    
    # Platform-spezifisch
    is_mobile: bool = False
    is_colab: bool = False
    is_cluster: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UARFConfig':
        """Erstellt Config aus Dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_path: str) -> 'UARFConfig':
        """Lädt Config aus JSON-Datei"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Config zu Dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    def to_json(self, json_path: str):
        """Speichert Config als JSON"""
        os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def update_from_hardware(self, hardware_config: Dict[str, Any]):
        """Aktualisiert Config basierend auf Hardware-Erkennung"""
        if 'batch_size' in hardware_config:
            self.batch_size = hardware_config['batch_size']
        if 'max_seq_len' in hardware_config:
            self.max_seq_len = hardware_config['max_seq_len']
        if 'precision' in hardware_config and self.precision == 'auto':
            self.precision = hardware_config['precision']
        if 'use_gradient_checkpointing' in hardware_config:
            self.use_gradient_checkpointing = hardware_config['use_gradient_checkpointing']
        if 'save_every_n_steps' in hardware_config:
            self.save_every_n_steps = hardware_config['save_every_n_steps']
        if 'flash_attention' in hardware_config:
            self.flash_attention = hardware_config['enable_flash_attn']
        
        # Platform flags setzen
        self.is_mobile = hardware_config.get('is_mobile', False)
        self.is_colab = hardware_config.get('is_colab', False)
        self.is_cluster = hardware_config.get('is_cluster', False)
    
    def validate(self) -> List[str]:
        """Validiert die Konfiguration und gibt Fehler zurück"""
        errors = []
        
        if self.batch_size < 1:
            errors.append("batch_size muss >= 1 sein")
        
        if self.max_seq_len < 64:
            errors.append("max_seq_len muss >= 64 sein")
        
        if self.learning_rate <= 0:
            errors.append("learning_rate muss > 0 sein")
        
        if self.time_budget_seconds < 60:
            errors.append("time_budget_seconds muss >= 60 sein (1 Minute)")
        
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            errors.append("warmup_ratio muss zwischen 0 und 1 liegen")
        
        valid_precisions = ['auto', 'fp32', 'fp16', 'bf16', 'int8']
        if self.precision not in valid_precisions:
            errors.append(f"precision muss einer von {valid_precisions} sein")
        
        valid_schedulers = ['cosine', 'linear', 'constant']
        if self.lr_scheduler not in valid_schedulers:
            errors.append(f"lr_scheduler muss einer von {valid_schedulers} sein")
        
        return errors
    
    def print_summary(self):
        """Druckt Konfigurations-Zusammenfassung"""
        print("\n" + "=" * 70)
        print("UARF KONFIGURATION")
        print("=" * 70)
        
        print(f"\n📦 MODELL:")
        print(f"   ID: {self.model_id}")
        print(f"   Trust Remote Code: {self.trust_remote_code}")
        
        print(f"\n🖥️  HARDWARE:")
        print(f"   Device: {self.device}")
        print(f"   Precision: {self.precision}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Max Seq Length: {self.max_seq_len}")
        print(f"   Gradient Checkpointing: {self.use_gradient_checkpointing}")
        
        print(f"\n🎯 TRAINING:")
        print(f"   Time Budget: {self.time_budget_seconds}s ({self.time_budget_seconds/60:.1f} min)")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Weight Decay: {self.weight_decay}")
        print(f"   Warmup Ratio: {self.warmup_ratio}")
        print(f"   LR Scheduler: {self.lr_scheduler}")
        
        print(f"\n📊 DATASET:")
        print(f"   Name: {self.dataset_name}")
        print(f"   Validation Split: {self.val_split_ratio*100:.1f}%")
        
        print(f"\n💾 OUTPUT:")
        print(f"   Directory: {self.output_dir}")
        print(f"   Save Every: {self.save_every_n_steps} steps")
        print(f"   Eval Every: {self.eval_every_n_steps} steps")
        
        print(f"\n⚙️  ADVANCED:")
        print(f"   Seed: {self.seed}")
        print(f"   Compile Model: {self.compile_model}")
        print(f"   Flash Attention: {self.flash_attention}")
        print(f"   Distributed: {self.distributed}")
        
        if self.export_format:
            print(f"\n📤 EXPORT:")
            print(f"   Format: {self.export_format}")
            print(f"   Path: {self.export_path}")
        
        print("\n" + "=" * 70)
