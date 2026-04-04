"""
Model Selector - Intelligente Modellauswahl basierend auf Hardware und Aufgabe
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .hardware_detector import HardwareSpecs


@dataclass
class ModelInfo:
    """Informationen über ein Modell"""
    model_id: str
    name: str
    size_mb: float
    params_millions: int
    min_ram_gb: float
    min_vram_gb: float
    recommended_for: List[str]
    license: str
    description: str


class ModelSelector:
    """Wählt passende Modelle basierend auf Hardware-Spezifikationen aus"""
    
    # Vordefinierte Modelle für verschiedene Use-Cases
    AVAILABLE_MODELS = {
        "text-generation": [
            ModelInfo(
                model_id="Qwen/Qwen2.5-0.5B",
                name="Qwen 2.5 0.5B",
                size_mb=950,
                params_millions=494,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["mobile", "edge", "low-resource"],
                license="Apache-2.0",
                description="Kleines, effizientes Modell für mobile Geräte"
            ),
            ModelInfo(
                model_id="Qwen/Qwen2.5-1.5B",
                name="Qwen 2.5 1.5B",
                size_mb=2800,
                params_millions=1540,
                min_ram_gb=4.0,
                min_vram_gb=2.0,
                recommended_for=["desktop", "colab-free"],
                license="Apache-2.0",
                description="Ausgewogenes Modell für Desktop-Nutzung"
            ),
            ModelInfo(
                model_id="Qwen/Qwen2.5-3B",
                name="Qwen 2.5 3B",
                size_mb=5600,
                params_millions=3200,
                min_ram_gb=8.0,
                min_vram_gb=4.0,
                recommended_for=["desktop-gpu", "colab-pro"],
                license="Apache-2.0",
                description="Leistungsstarkes Modell für GPUs mit mittlerem VRAM"
            ),
            ModelInfo(
                model_id="microsoft/phi-2",
                name="Phi-2",
                size_mb=5200,
                params_millions=2700,
                min_ram_gb=6.0,
                min_vram_gb=3.0,
                recommended_for=["desktop", "research"],
                license="MIT",
                description="Microsofts kompaktes高性能 Modell"
            ),
            ModelInfo(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                name="TinyLlama 1.1B",
                size_mb=2100,
                params_millions=1100,
                min_ram_gb=3.0,
                min_vram_gb=1.5,
                recommended_for=["mobile", "edge", "quick-experiments"],
                license="Apache-2.0",
                description="Sehr schnelles Modell für Experimente"
            ),
        ],
        "text-classification": [
            ModelInfo(
                model_id="distilbert-base-uncased",
                name="DistilBERT Base",
                size_mb=250,
                params_millions=66,
                min_ram_gb=1.0,
                min_vram_gb=0.5,
                recommended_for=["all"],
                license="Apache-2.0",
                description="Leichte Klassifikations-Baseline"
            ),
        ],
        "fill-mask": [
            ModelInfo(
                model_id="bert-base-uncased",
                name="BERT Base",
                size_mb=420,
                params_millions=110,
                min_ram_gb=2.0,
                min_vram_gb=1.0,
                recommended_for=["all"],
                license="Apache-2.0",
                description="Standard Mask-Filling Modell"
            ),
        ]
    }
    
    def __init__(self, hardware_specs: HardwareSpecs):
        self.specs = hardware_specs
    
    def suggest_models(self, task_type: str = "text-generation", 
                       limit: int = 5) -> List[ModelInfo]:
        """
        Schlägt passende Modelle basierend auf Hardware vor
        
        Args:
            task_type: Art der Aufgabe (text-generation, classification, etc.)
            limit: Maximale Anzahl zurückgegebener Vorschläge
            
        Returns:
            Liste von empfohlenen Modellen
        """
        if task_type not in self.AVAILABLE_MODELS:
            task_type = "text-generation"
        
        all_models = self.AVAILABLE_MODELS[task_type]
        compatible_models = []
        
        for model in all_models:
            if self._is_model_compatible(model):
                score = self._calculate_compatibility_score(model)
                compatible_models.append((score, model))
        
        # Nach Kompatibilität sortieren (höchster Score zuerst)
        compatible_models.sort(key=lambda x: x[0], reverse=True)
        
        return [model for score, model in compatible_models[:limit]]
    
    def _is_model_compatible(self, model: ModelInfo) -> bool:
        """Prüft ob ein Modell auf der aktuellen Hardware laufen kann"""
        # RAM Check
        if self.specs.ram_available < model.min_ram_gb:
            return False
        
        # GPU VRAM Check (wenn GPU verfügbar)
        if self.specs.gpu_available:
            if self.specs.gpu_vram < model.min_vram_gb:
                # CPU Fallback möglich?
                if self.specs.ram_available < model.min_ram_gb * 1.5:
                    return False
        else:
            # Nur CPU - benötigt mehr RAM
            if self.specs.ram_available < model.min_ram_gb * 1.5:
                return False
        
        return True
    
    def _calculate_compatibility_score(self, model: ModelInfo) -> float:
        """Berechnet einen Kompatibilitäts-Score (höher = besser)"""
        score = 0.0
        
        # RAM Headroom
        ram_ratio = self.specs.ram_available / model.min_ram_gb
        score += min(ram_ratio, 3.0) * 10
        
        # GPU Bonus
        if self.specs.gpu_available:
            if self.specs.gpu_vram >= model.min_vram_gb:
                vram_ratio = self.specs.gpu_vram / model.min_vram_gb
                score += min(vram_ratio, 2.0) * 15
        
        # Platform recommendations
        if self.specs.is_mobile and "mobile" in model.recommended_for:
            score += 20
        elif self.specs.is_colab and "colab-free" in model.recommended_for:
            score += 20
        elif not self.specs.is_mobile and "desktop" in model.recommended_for:
            score += 15
        
        # Size preference (kleinere Modelle bevorzugt für schnellere Experimente)
        size_penalty = model.params_millions / 1000
        score -= size_penalty
        
        return score
    
    def get_best_model(self, task_type: str = "text-generation") -> Optional[ModelInfo]:
        """Gibt das beste kompatible Modell zurück"""
        suggestions = self.suggest_models(task_type, limit=1)
        return suggestions[0] if suggestions else None
    
    def print_suggestions(self, task_type: str = "text-generation"):
        """Druckt Modellauswahl-Vorschläge"""
        print("\n" + "=" * 70)
        print(f"UARF MODEL SUGGESTIONS für {task_type}")
        print("=" * 70)
        
        suggestions = self.suggest_models(task_type)
        
        if not suggestions:
            print("Keine kompatiblen Modelle gefunden!")
            print("Versuchen Sie:")
            print("  - Mehr RAM freigeben")
            print("  - Kleinere Modelle")
            print("  - Cloud-Ressourcen (Colab, Cluster)")
            return
        
        for i, model in enumerate(suggestions, 1):
            print(f"\n{i}. {model.name}")
            print(f"   ID: {model.model_id}")
            print(f"   Größe: {model.size_mb:.0f} MB ({model.params_millions}M Parameter)")
            print(f"   Mindestanforderungen: {model.min_ram_gb:.1f} GB RAM, {model.min_vram_gb:.1f} GB VRAM")
            print(f"   Lizenz: {model.license}")
            print(f"   Beschreibung: {model.description}")
            print(f"   Empfohlen für: {', '.join(model.recommended_for)}")
        
        print("\n" + "=" * 70)
        print(f"Tipp: Verwenden Sie 'uarf run --model {suggestions[0].model_id}'")
        print("=" * 70)
