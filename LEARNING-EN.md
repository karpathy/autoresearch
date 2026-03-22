# Auto-Research Learning Notes

> **Learning Date**: 2026-03-22 15:00
> **Project Author**: @karpathy (Andrej Karpathy)
> **Project Type**: AI Agent Autonomous Research Framework

---

## 🎯 Core Concepts

### **What is autoresearch?**

autoresearch is a framework that allows AI Agents to conduct LLM training experiments autonomously.

**Core Philosophy**:
> Give an AI Agent a real LLM training environment and let it experiment autonomously overnight. It modifies code, trains for 5 minutes, checks if results improved, keeps or discards, and repeats.

---

## 📁 Project Structure

```
autoresearch/
├── prepare.py      — Data preparation (do not modify)
├── train.py        — Model + training loop (Agent modifies this)
├── program.md      — Agent instructions (Human modifies this)
├── pyproject.toml  — Dependencies
└── README.md       — Documentation
```

**Key Design**:
- ✅ **Single file modification**: Agent only modifies train.py
- ✅ **Fixed time**: Training always runs for 5 minutes
- ✅ **Self-contained**: No external dependencies

---

## 🤖 Agent Workflow

### **1. Experiment Setup**

```bash
# 1. Create experiment branch
git checkout -b autoresearch/mar22

# 2. Verify data exists
ls ~/.cache/autoresearch/

# 3. Create results file
echo "commit	val_bpb	memory_gb	status	description" > results.tsv
```

### **2. Experiment Loop (Never Stops)**

```python
while True:
    # 1. Modify train.py
    # 2. git commit
    # 3. Run experiment (5 minutes)
    uv run train.py > run.log 2>&1

    # 4. Read results
    grep "^val_bpb:\|^peak_vram_mb:" run.log

    # 5. Record to results.tsv
    # 6. If improved, keep; otherwise, rollback
```

---

## 🏗️ Model Architecture

### **Core Components**

**1. GPT Model**
```python
class GPT(nn.Module):
    def __init__(self, config):
        # Transformer blocks
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "h": nn.ModuleList([Block(config, i)]),
        })

        # Value embeddings (ResFormer)
        self.value_embeds = nn.ModuleDict({...})

        # Rotary embeddings
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
```

**2. Attention Mechanism**
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.c_q = nn.Linear(n_embd, n_head * head_dim)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim)

        # Value embedding gate
        self.ve_gate = nn.Linear(32, n_kv_head)
```

**3. Flash Attention 3**
```python
# Use Flash Attention 3 for acceleration
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

---

## ⚙️ Training Configuration

### **Hyperparameters**

```python
# Model configuration
DEPTH = 8                # Transformer layers
HEAD_DIM = 128           # Attention head dimension
ASPECT_RATIO = 96        # Model dimension ratio

# Training configuration
TOTAL_BATCH_SIZE = 2**20  # Total batch size (~1M tokens)
DEVICE_BATCH_SIZE = 128   # Device batch size

# Learning rates
UNEMBEDDING_LR = 0.004
EMBEDDING_LR = 0.2
MATRIX_LR = 0.02
SCALAR_LR = 0.5

# Optimizer
ADAM_BETAS = (0.8, 0.95)
WEIGHT_DECAY = 0.0

# Scheduler
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.1
FINAL_LR_FRAC = 0.1

# Time budget
TIME_BUDGET = 300  # 5 minutes
```

---

## 📊 Evaluation Metrics

### **Primary Metric: val_bpb**

**val_bpb** = validation bits per byte (lower is better)

**Calculation**:
```python
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
```

**Goal**: Reduce val_bpb

---

## 🔄 Optimizer Setup

### **Muon + AdamW**

```python
optimizer = model.setup_optimizer(
    unembedding_lr=0.004,
    embedding_lr=0.2,
    matrix_lr=0.02,
    scalar_lr=0.5,
    adam_betas=(0.8, 0.95),
    weight_decay=0.0,
)
```

**Features**:
- ✅ Grouped learning rates (different LRs for different parameter groups)
- ✅ Muon optimizer (momentum adjustment)
- ✅ Dynamic learning rate scheduling

---

## 💡 Core Innovations

### **1. Value Embedding (ResFormer)**

```python
# Add Value Embedding to alternating layers
if has_ve(layer_idx, n_layer):
    ve = ve.view(B, T, n_kv_head, head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))
    v = v + gate.unsqueeze(-1) * ve
```

**Effect**: Improves model expressiveness

### **2. Windowed Attention**

```python
# Window pattern: "SSSL" = Short-Short-Short-Long
window_pattern = "SSSL"
```

**Effect**: Reduces computation, improves efficiency

### **3. RMS Norm**

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

**Effect**: More stable training

---

## 🎯 What Agents Can Modify

### **✅ Can Modify**

1. **Model Architecture**
   - DEPTH (number of layers)
   - HEAD_DIM (head dimension)
   - window_pattern (window mode)
   - Activation functions

2. **Hyperparameters**
   - Learning rates
   - Batch sizes
   - Weight decay
   - Optimizer parameters

3. **Training Strategies**
   - Learning rate scheduling
   - Gradient accumulation
   - Regularization

### **❌ Cannot Modify**

1. prepare.py (data loading)
2. Evaluation function (evaluate_bpb)
3. Time budget (5 minutes)
4. Dependencies

---

## 🚀 Quick Start

### **⚠️ Current Environment Check**

**Check Results**:
- ❌ CUDA not available
- ❌ GPU count: 0
- ❌ Cannot run directly

**Reason**: Current is a virtual machine environment without GPU

**Solutions**:
1. ✅ Can run on CPU (requires code modification)
2. ✅ Can run on cloud GPU (AWS/GCP)
3. ✅ Can learn architecture and concepts

---

### **Step 1: Install Dependencies** (if GPU available)

```bash
cd ~/.openclaw/workspace/autoresearch
uv sync
```

### **Step 2: Prepare Data**

```bash
uv run prepare.py  # ~2 minutes
```

### **Step 3: Run Baseline Experiment**

```bash
uv run train.py  # ~5 minutes
```

### **Step 4: Let Agent Experiment Autonomously**

```bash
# Start Agent (Claude/Codex)
# Point to program.md
# Agent will automatically start experiment loop
```

---

## 📈 Expected Results

### **Baseline Performance**

**H100 GPU**:
- val_bpb: ~1.0
- Training time: 300 seconds
- MFU: ~40%
- Parameters: ~50M

### **Improvement Directions**

1. ✅ **Architecture optimization**: Adjust layers, heads
2. ✅ **Hyperparameter tuning**: Learning rates, batch sizes
3. ✅ **Regularization**: Weight decay, dropout
4. ✅ **Optimizer**: Adjust Adam parameters

---

## 💡 Learning Value

### **1. Agent Application Scenarios**

**✅ Real AI Agent Application**:
- Agent autonomously modifies code
- Agent autonomously evaluates results
- Agent autonomously decides keep/discard

### **2. Research Paradigm**

**✅ Autonomous Research Paradigm**:
- Humans design program.md
- Agents execute experiment loops
- Humans wake up to check results

### **3. Ultimate Vibe Coding**

**✅ Natural Language Driven**:
- program.md describes in natural language
- Agent understands and executes
- Code auto-generated

---

## 🔧 Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **NVIDIA GPU** | ✅ Official | H100 best |
| **MacOS** | ✅ Community | [autoresearch-macos](https://github.com/miolini/autoresearch-macos) |
| **Windows RTX** | ✅ Community | [autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) |
| **AMD** | ✅ Community | [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) |

---

## 📝 Recommendations for Small-Scale Devices

**If running on Macbook or other small-scale devices**:

1. ✅ Use TinyStories dataset
2. ✅ Reduce vocab_size (8192→1024)
3. ✅ Reduce MAX_SEQ_LEN (→256)
4. ✅ Reduce DEPTH (8→4)
5. ✅ Use "L" pattern (not "SSSL")
6. ✅ Reduce TOTAL_BATCH_SIZE (→2^14)

---

## 🎯 Next Steps

### **Immediate Actions**

1. [ ] Check if GPU is available
2. [ ] Run first baseline experiment
3. [ ] Record baseline val_bpb
4. [ ] Try first improvement

### **Deep Understanding**

1. [ ] Read Flash Attention 3 paper
2. [ ] Understand Muon optimizer
3. [ ] Study Value Embedding
4. [ ] Analyze windowed attention

---

## 📚 Related Resources

- **Project**: https://github.com/karpathy/autoresearch
- **Tweet**: https://x.com/karpathy/status/2029701092347630069
- **Papers**: Flash Attention 3, Muon Optimizer
- **Forks**: MacOS, Windows, AMD versions

---

## 💭 Core Insights

### **1. Autonomous Research Paradigm**

**Traditional**: Humans design experiments → Humans execute → Humans analyze
**autoresearch**: Humans design instructions → Agents execute → Agents analyze

### **2. Vibe Coding Application**

**Core**: Drive complex experiments with natural language (program.md)
**Advantage**: Humans can sleep, Agents work all night

### **3. Quality Assurance**

**Simplification Principle**: Better simple than complex
**Evaluation Standard**: val_bpb is the only truth

---

**Boss, autoresearch learning complete! This is a very cool AI Agent autonomous research framework!** 🚀

---

## 🎮 Consumer GPU Support

### **Recommended Consumer GPUs**

| GPU | VRAM | Performance | Recommendation | Price |
|-----|------|-------------|----------------|-------|
| **RTX 4090** | 24GB | ⭐⭐⭐⭐⭐ | 🟢 Highly Recommended | ~$1500 |
| **RTX 3090** | 24GB | ⭐⭐⭐⭐⭐ | 🟢 Highly Recommended | ~$1000 |
| **RTX 4080** | 16GB | ⭐⭐⭐⭐ | ✅ Recommended | ~$800 |
| **RTX 3080** | 10GB | ⭐⭐⭐ | ✅ Usable | ~$350 |
| **GTX 1080 Ti** | 11GB | ⭐⭐⭐ | ✅ Usable | ~$700 |

---

### **Configuration Examples**

#### **RTX 4090 / RTX 3090** (Best)

```python
# train.py
DEPTH = 6              # From 8 to 6
TOTAL_BATCH_SIZE = 2**16  # ~65K tokens
DEVICE_BATCH_SIZE = 64   # From 128 to 64
WINDOW_PATTERN = "L"     # Simple pattern
```

#### **RTX 3080 / GTX 1080 Ti** (Minimum)

```python
# train.py
DEPTH = 4              # From 8 to 4
TOTAL_BATCH_SIZE = 2**13  # ~8K tokens
DEVICE_BATCH_SIZE = 16   # From 128 to 16
WINDOW_PATTERN = "L"     # Simple pattern

# prepare.py
MAX_SEQ_LEN = 256   # From 2048 to 256
vocab_size = 1024   # From 8192 to 1024
```

---

### **Performance Estimates**

| GPU | Training Time | val_bpb | MFU | Relative to H100 |
|-----|---------------|---------|-----|------------------|
| **H100** | 5 minutes | ~1.0 | ~40% | 1x |
| **RTX 4090** | 10-15 minutes | ~1.0-1.1 | ~30-35% | 2-3x |
| **RTX 3080** | 20-30 minutes | ~1.1-1.2 | ~20-25% | 4-6x |

---

### **Memory Requirements**

| GPU | VRAM | Model Parameters | Estimated Usage | Safe |
|-----|------|------------------|-----------------|------|
| **RTX 4090** | 24GB | ~25M | ~18GB | ✅ |
| **RTX 3080** | 10GB | ~10M | ~8GB | ✅ |

---

**Conclusion**: Consumer GPUs are fully usable! RTX 4090 or RTX 3090 are the best choices!
