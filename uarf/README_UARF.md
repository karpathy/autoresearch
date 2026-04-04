# Universal AutoResearch Framework (UARF)

**Ein Framework für alle - Von Edge Devices bis zu Server-Clustern**

[![Version](https://img.shields.io/badge/version-0.1.0--mvp-blue)](https://github.com/yourusername/uarf)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## 🚀 Was ist UARF?

UARF ist ein universelles Machine-Learning-Framework, das **automatisch** deine Hardware erkennt und sich optimal daran anpasst. Trainiere Modelle auf:

- 📱 **Android/Termux** (Mobile Geräte)
- 🖥️ **Windows/Linux/Mac** (Desktop)
- ☁️ **Google Colab** (Cloud GPUs)
- 🖧 **Server-Cluster** (Multi-GPU)

**Ein Befehl - Alle Plattformen!**

```bash
uarf run --model Qwen/Qwen2.5-0.5B --time 600
```

## ✨ Features

### Automatische Hardware-Erkennung
UARF analysiert dein System und passt automatisch an:
- Batch Size
- Präzision (FP32/FP16/BF16/INT8)
- Sequenzlänge
- Gradient Checkpointing
- Flash Attention

### Intelligente Modellauswahl
```bash
uarf suggest
```
Empfiehlt passende Modelle basierend auf deiner Hardware.

### Cross-Platform Training
- **Auto-Detect**: CUDA, MPS (Apple Silicon), CPU
- **Optimiert**: Für jede Plattform vorkonfiguriert
- **Robust**: Fallback-Mechanismen bei Fehlern

### Zeitgesteuertes Training
Setze ein Zeitbudget - UARF macht das Beste daraus:
```bash
uarf run --time 300  # 5 Minuten Training
```

## 📦 Installation

### Standard-Installation
```bash
pip install uarf
```

### Von Source
```bash
git clone https://github.com/yourusername/uarf.git
cd uarf
pip install -e .
```

### Mit optionalen Dependencies
```bash
# Für Export-Funktionen
pip install uarf[export]

# Für Google Colab
pip install uarf[colab]

# Für Entwicklung
pip install uarf[dev]
```

## 🎯 Schnellstart

### 1. Hardware erkennen
```bash
uarf auto-setup
```

Ausgabe:
```
============================================================
UARF HARDWARE DETECTION
============================================================
Plattform: Linux (x86_64)
CPU: 8 Kerne, 3.50 GHz max
RAM: 16.0 GB gesamt, 12.3 GB frei
GPU: NVIDIA GeForce RTX 3080 (10.0 GB VRAM)
     Compute Capability: (8, 6)
Speicher: 500.0 GB frei
Mobile: Nein
Colab: Nein
Cluster: Nein
============================================================

EMPFOHLENE KONFIGURATION:
  Maximale Modellgröße: 3B
  Batch Size: 64
  Präzision: fp16
  GPU Modus: medium
  Max Sequenzlänge: 2048
============================================================
```

### 2. Modellempfehlungen anzeigen
```bash
uarf suggest
```

### 3. Training starten
```bash
# Mit automatischer Konfiguration
uarf run

# Spezifisches Modell
uarf run --model Qwen/Qwen2.5-1.5B

# Angepasstes Zeitbudget
uarf run --time 600 --dataset karpathy/tinyshakespeare

# Custom Parameter
uarf run --batch-size 32 --lr 1e-4 --max-seq-len 512
```

### 4. Ergebnisse exportieren (in Entwicklung)
```bash
uarf export --checkpoint ./outputs/final --format gguf
```

## 💻 Verwendung als Python-Bibliothek

```python
from uarf import HardwareDetector, ModelSelector, UARFConfig, UniversalTrainer

# Hardware erkennen
detector = HardwareDetector()
detector.print_summary()

# Optimale Konfiguration
hardware_config = detector.get_optimal_config()

# Passendes Modell finden
selector = ModelSelector(detector.specs)
best_model = selector.get_best_model()
print(f"Empfohlenes Modell: {best_model.model_id}")

# Config erstellen
config = UARFConfig(
    model_id=best_model.model_id,
    time_budget_seconds=600,
    output_dir="./my-experiment"
)
config.update_from_hardware(hardware_config)

# Training starten
trainer = UniversalTrainer(config)
trainer.train()
```

## 📋 Verfügbare Befehle

| Befehl | Beschreibung |
|--------|-------------|
| `uarf auto-setup` | Hardware erkennen und optimale Config anzeigen |
| `uarf run` | Training starten |
| `uarf suggest` | Modellempfehlungen für deine Hardware |
| `uarf detect` | Hardware-Informationen anzeigen |
| `uarf export` | Modell exportieren (GGUF, ONNX, TFLite) |

## 🔧 Konfigurationsoptionen

### `uarf run` Parameter

```
--model MODEL          Hugging Face Model ID (default: Qwen/Qwen2.5-0.5B)
--dataset DATASET      Dataset Name (default: karpathy/tinyshakespeare)
--time SEKUNDEN        Zeitbudget in Sekunden (default: 300)
--batch-size SIZE      Batch Size (auto wenn nicht angegeben)
--max-seq-len LEN      Maximale Sequenzlänge
--lr RATE              Learning Rate (default: 2e-4)
--output-dir DIR       Output Verzeichnis (default: ./outputs)
--device DEVICE        auto, cuda, cpu, mps (default: auto)
--precision PRECISION  auto, fp32, fp16, bf16, int8 (default: auto)
--config FILE          JSON Config Datei laden
```

## 🎓 Beispiele

### Auf schwacher Hardware (4GB RAM)
```bash
uarf run --model TinyLlama/TinyLlama-1.1B --batch-size 8 --max-seq-len 256
```

### Auf GPU mit 8GB VRAM
```bash
uarf run --model Qwen/Qwen2.5-1.5B --time 600 --precision fp16
```

### Auf Google Colab (Free Tier)
```python
# In Colab Notebook
!pip install uarf
!uarf run --model microsoft/phi-2 --time 900
```

### Auf M1/M2 Mac
```bash
uarf run --device mps --precision fp16
```

## 📊 Unterstützte Modelle

UARF unterstützt alle Hugging Face Modelle. Empfohlene Starter-Modelle:

| Modell | Größe | RAM | VRAM | Plattform |
|--------|-------|-----|------|-----------|
| Qwen2.5-0.5B | 494M | 2GB | 1GB | Mobile/Edge |
| TinyLlama-1.1B | 1.1B | 3GB | 1.5GB | Desktop |
| Qwen2.5-1.5B | 1.5B | 4GB | 2GB | Desktop/Colab |
| Phi-2 | 2.7B | 6GB | 3GB | Desktop GPU |
| Qwen2.5-3B | 3.2B | 8GB | 4GB | GPU/Colab Pro |

## 🛠️ Entwicklung

### Tests ausführen
```bash
pytest tests/
```

### Code formatieren
```bash
black uarf/
ruff check uarf/
```

## 🤝 Contributing

Contributions sind willkommen! Bitte:

1. Fork das Repository
2. Erstelle einen Feature Branch (`git checkout -b feature/amazing-feature`)
3. Committe deine Änderungen (`git commit -m 'Add amazing feature'`)
4. Pushe den Branch (`git push origin feature/amazing-feature`)
5. Öffne einen Pull Request

## 📝 Roadmap

### MVP (v0.1.0) ✅
- [x] Hardware Detection
- [x] Model Selector
- [x] Universal Trainer
- [x] CLI Interface
- [ ] GGUF Export
- [ ] ONNX Export
- [ ] TFLite Export

### v0.2.0 (Geplant)
- [ ] Resume Training
- [ ] Multi-GPU Support
- [ ] Distributed Training
- [ ] Experiment Tracking
- [ ] More Datasets

### v1.0.0 (Vision)
- [ ] Android APK Export
- [ ] AutoML Features
- [ ] Cloud Integration (AWS, GCP, Azure)
- [ ] Web UI
- [ ] Plugin System

## 🐛 Bekannte Probleme

Siehe [Issues](https://github.com/yourusername/uarf/issues) für bekannte Probleme.

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE) Datei.

## 🙏 Danksagungen

- Inspiriert von [autoresearch](https://github.com/karpathy/autoresearch) von Andrej Karpathy
- Built with [PyTorch](https://pytorch.org/)
- Models from [Hugging Face](https://huggingface.co/)

## 📞 Kontakt

- GitHub Issues: [Issues](https://github.com/yourusername/uarf/issues)
- Email: your.email@example.com

---

**Made with ❤️ for the AI Community**

*"Demokratisierung von ML-Training - für jeden, überall."*
