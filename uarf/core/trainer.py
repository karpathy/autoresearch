"""
Universal Trainer - Kernstück des UARF Frameworks
Unterstützt Training auf allen Plattformen mit automatischer Anpassung.
"""

import os
import time
import math
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

from .config import UARFConfig
from .hardware_detector import HardwareDetector


@dataclass
class TrainingMetrics:
    """Trainingsmetriken"""
    steps_completed: int = 0
    total_tokens: int = 0
    best_val_loss: float = float('inf')
    training_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    mfu_percent: float = 0.0


class UniversalTrainer:
    """
    Universeller Trainer für alle Plattformen
    
    Automatische Anpassung an:
    - Mobile Geräte (Android/Termux)
    - Desktop (Windows/Linux/Mac)
    - Cloud (Google Colab)
    - Cluster (Multi-GPU)
    """
    
    def __init__(self, config: UARFConfig):
        self.config = config
        self.hardware = HardwareDetector()
        self.metrics = TrainingMetrics()
        
        # Device setup
        self.device = self._setup_device()
        self.dtype = self._setup_dtype()
        
        # Model und Tokenizer
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.start_time = None
        
    def _setup_device(self) -> torch.device:
        """Richtet das richtige Device ein"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _setup_dtype(self) -> torch.dtype:
        """Richtet die richtige Präzision ein"""
        precision = self.config.precision
        
        if precision == "auto":
            if self.device.type == "cuda":
                cap = torch.cuda.get_device_capability(0)
                if cap[0] >= 8:  # Ampere oder neuer
                    return torch.bfloat16
                else:
                    return torch.float16
            elif self.device.type == "mps":
                return torch.float16
            else:
                return torch.float32
        elif precision == "fp32":
            return torch.float32
        elif precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        elif precision == "int8":
            # INT8 wird später via Quantisierung behandelt
            return torch.float16
        else:
            return torch.float32
    
    def load_model(self):
        """Lädt Modell und Tokenizer"""
        print(f"\n📦 Lade Modell: {self.config.model_id}")
        
        # Tokenizer laden
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"
        )
        
        # Pad Token setzen falls nicht vorhanden
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Modell-Konfiguration
        model_config = AutoConfig.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Modell auf Meta-Device erstellen (speichereffizient)
        with torch.device('meta'):
            self.model = AutoModelForCausalLM.from_config(model_config)
        
        # Auf richtiges Device bewegen und initialisieren
        self.model = self.model.to_empty(device=self.device)
        self.model.init_weights()
        
        # Gradient Checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("✅ Gradient Checkpointing aktiviert")
        
        # Torch Compile (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            print("🔧 Compiliere Modell mit torch.compile...")
            self.model = torch.compile(self.model)
        
        print(f"✅ Modell geladen: {sum(p.numel() for p in self.model.parameters()):,} Parameter")
    
    def prepare_data(self):
        """Bereitet Datensätze vor"""
        from datasets import load_dataset
        
        print(f"\n📊 Lade Dataset: {self.config.dataset_name}")
        
        # Dataset laden
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split
        )
        
        # Train/Validation Split
        if self.config.val_split_ratio > 0:
            splits = dataset.train_test_split(test_size=self.config.val_split_ratio)
            train_dataset = splits['train']
            val_dataset = splits['test']
        else:
            train_dataset = dataset
            val_dataset = None
        
        # Tokenization
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'] if 'text' in examples else examples[list(examples.keys())[0]],
                truncation=True,
                max_length=self.config.max_seq_len,
                padding='max_length'
            )
        
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # DataLoader erstellen
        self.train_loader = DataLoader(
            tokenized_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        if val_dataset:
            tokenized_val = val_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )
            tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            
            self.val_loader = DataLoader(
                tokenized_val,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        print(f"✅ Daten vorbereitet: {len(tokenized_train)} Trainings-Samples")
        if val_dataset:
            print(f"   Validierung: {len(tokenized_val)} Samples")
    
    def setup_optimizer(self):
        """Richtet Optimizer und Scheduler ein"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        if self.config.max_steps:
            total_steps = self.config.max_steps
        else:
            # Schätzung basierend auf Time Budget
            total_steps = 1000  # Default
        
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps - warmup_steps
            )
        else:
            self.scheduler = None
        
        print(f"✅ Optimizer eingerichtet: AdamW (LR={self.config.learning_rate})")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Führt einen Trainingsschritt durch"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = input_ids.clone()
        
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=(self.device.type != 'cpu')
        ):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        # Gradient Accumulation
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluierung auf Validierungsdaten"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = input_ids.clone()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def save_checkpoint(self, path: str):
        """Speichert Checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        # Modell speichern
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Optimizer State
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'metrics': self.metrics,
            'config': self.config.to_dict()
        }, os.path.join(path, 'training_state.pt'))
        
        print(f"💾 Checkpoint gespeichert: {path}")
    
    def train(self):
        """Haupt-Trainingsschleife"""
        print("\n" + "=" * 70)
        print("🚀 STARTING TRAINING")
        print("=" * 70)
        
        # Validierung
        errors = self.config.validate()
        if errors:
            print("❌ Konfigurationsfehler:")
            for error in errors:
                print(f"   - {error}")
            return
        
        # Setup
        self.load_model()
        self.prepare_data()
        self.setup_optimizer()
        
        # Training Loop
        self.start_time = time.time()
        self.model.train()
        
        progress_bar = tqdm(
            total=self.config.max_steps or 1000,
            desc="Training",
            unit="step"
        )
        
        accumulated_loss = 0.0
        
        try:
            for epoch in range(100):  # Maximal 100 Epochen
                for batch in self.train_loader:
                    step_start = time.time()
                    
                    # Training Step
                    loss = self.train_step(batch)
                    accumulated_loss += loss
                    
                    # Optimizer Step
                    if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            max_norm=1.0
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        if self.scheduler:
                            self.scheduler.step()
                    
                    self.global_step += 1
                    self.metrics.steps_completed = self.global_step
                    
                    # Tokens zählen
                    tokens_in_batch = batch['input_ids'].numel()
                    self.metrics.total_tokens += tokens_in_batch
                    
                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        elapsed_time = time.time() - self.start_time
                        tokens_per_sec = self.metrics.total_tokens / elapsed_time
                        
                        avg_loss = accumulated_loss / self.config.log_every_n_steps
                        accumulated_loss = 0.0
                        
                        progress_bar.set_postfix({
                            'loss': f"{avg_loss:.4f}",
                            'tps': f"{tokens_per_sec:.0f}"
                        })
                    
                    # Evaluation
                    if self.global_step % self.config.eval_every_n_steps == 0:
                        val_loss = self.evaluate()
                        self.metrics.best_val_loss = min(self.metrics.best_val_loss, val_loss)
                        print(f"\n📊 Step {self.global_step}: Val Loss = {val_loss:.4f}")
                    
                    # Checkpoint
                    if self.global_step % self.config.save_every_n_steps == 0:
                        checkpoint_path = os.path.join(
                            self.config.output_dir,
                            f"checkpoint-{self.global_step}"
                        )
                        self.save_checkpoint(checkpoint_path)
                    
                    # Time Budget prüfen
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time >= self.config.time_budget_seconds:
                        print(f"\n⏰ Zeitbudget erreicht ({elapsed_time:.1f}s)")
                        break
                    
                    # Max Steps prüfen
                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        print(f"\n🎯 Maximale Schritte erreicht ({self.global_step})")
                        break
                    
                    progress_bar.update(1)
                
                # Äußere Schleife (Time Budget)
                if time.time() - self.start_time >= self.config.time_budget_seconds:
                    break
            
        except KeyboardInterrupt:
            print("\n⚠️  Training unterbrochen durch Benutzer")
        
        finally:
            progress_bar.close()
            
            # Finale Metriken
            self.metrics.training_time_seconds = time.time() - self.start_time
            if self.device.type == 'cuda':
                self.metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Finales Checkpoint
            final_path = os.path.join(self.config.output_dir, "final")
            self.save_checkpoint(final_path)
            
            # Zusammenfassung
            self.print_training_summary()
    
    def print_training_summary(self):
        """Druckt Trainings-Zusammenfassung"""
        print("\n" + "=" * 70)
        print("📈 TRAININGS-ZUSAMMENFASSUNG")
        print("=" * 70)
        print(f"Steps:           {self.metrics.steps_completed:,}")
        print(f"Total Tokens:    {self.metrics.total_tokens:,} ({self.metrics.total_tokens/1e6:.2f}M)")
        print(f"Best Val Loss:   {self.metrics.best_val_loss:.4f}")
        print(f"Training Time:   {self.metrics.training_time_seconds:.1f}s ({self.metrics.training_time_seconds/60:.1f} min)")
        print(f"Peak Memory:     {self.metrics.peak_memory_mb:.1f} MB")
        print(f"Tokens/sec:      {self.metrics.total_tokens / max(self.metrics.training_time_seconds, 1):.0f}")
        print("=" * 70)
