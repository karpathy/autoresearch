# Auto-Research 学习笔记

> **学习时间**: 2026-03-22 15:00
> **项目作者**: @karpathy (Andrej Karpathy)
> **项目性质**: AI Agent 自主研究框架

---

## 🎯 核心概念

### **什么是 autoresearch？**

autoresearch 是一个让 AI Agent 自主进行 LLM 训练实验的框架。

**核心理念**:
> 给 AI Agent 一个真实的 LLM 训练环境，让它整夜自主实验。它修改代码，训练5分钟，检查结果是否改进，保留或丢弃，然后重复。

---

## 📁 项目结构

```
autoresearch/
├── prepare.py      — 数据准备（不要修改）
├── train.py        — 模型+训练循环（Agent修改这个）
├── program.md      — Agent指令（人类修改这个）
├── pyproject.toml  — 依赖
└── README.md       — 文档
```

**关键设计**:
- ✅ **单文件修改**: Agent只修改train.py
- ✅ **固定时间**: 训练总是5分钟
- ✅ **自包含**: 无外部依赖

---

## 🤖 Agent工作流程

### **1. 实验设置**

```bash
# 1. 创建实验分支
git checkout -b autoresearch/mar22

# 2. 验证数据存在
ls ~/.cache/autoresearch/

# 3. 创建结果文件
echo "commit	val_bpb	memory_gb	status	description" > results.tsv
```

### **2. 实验循环（永不停止）**

```python
while True:
    # 1. 修改 train.py
    # 2. git commit
    # 3. 运行实验（5分钟）
    uv run train.py > run.log 2>&1
    
    # 4. 读取结果
    grep "^val_bpb:\|^peak_vram_mb:" run.log
    
    # 5. 记录到 results.tsv
    # 6. 如果改进，保留；否则，回滚
```

---

## 🏗️ 模型架构

### **核心组件**

**1. GPT模型**
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

**2. 注意力机制**
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
# 使用 Flash Attention 3 加速
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

---

## ⚙️ 训练配置

### **超参数**

```python
# 模型配置
DEPTH = 8                # Transformer层数
HEAD_DIM = 128           # 注意力头维度
ASPECT_RATIO = 96        # 模型维度比例

# 训练配置
TOTAL_BATCH_SIZE = 2**20  # 总批次大小（~1M tokens）
DEVICE_BATCH_SIZE = 128   # 设备批次大小

# 学习率
UNEMBEDDING_LR = 0.004
EMBEDDING_LR = 0.2
MATRIX_LR = 0.02
SCALAR_LR = 0.5

# 优化器
ADAM_BETAS = (0.8, 0.95)
WEIGHT_DECAY = 0.0

# 调度器
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.1
FINAL_LR_FRAC = 0.1

# 时间预算
TIME_BUDGET = 300  # 5分钟
```

---

## 📊 评估指标

### **主要指标: val_bpb**

**val_bpb** = validation bits per byte（越低越好）

**计算方式**:
```python
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
```

**目标**: 降低 val_bpb

---

## 🔄 优化器设置

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

**特点**:
- ✅ 分组学习率（不同参数组用不同LR）
- ✅ Muon优化器（momentum调整）
- ✅ 动态学习率调度

---

## 💡 核心创新

### **1. Value Embedding (ResFormer)**

```python
# 交替层添加 Value Embedding
if has_ve(layer_idx, n_layer):
    ve = ve.view(B, T, n_kv_head, head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :32]))
    v = v + gate.unsqueeze(-1) * ve
```

**作用**: 提升模型表达能力

### **2. Windowed Attention**

```python
# 窗口模式: "SSSL" = 短-短-短-长
window_pattern = "SSSL"
```

**作用**: 减少计算量，提升效率

### **3. RMS Norm**

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

**作用**: 更稳定的训练

---

## 🎯 Agent可以修改的内容

### **✅ 可以修改**

1. **模型架构**
   - DEPTH（层数）
   - HEAD_DIM（头维度）
   - window_pattern（窗口模式）
   - 激活函数

2. **超参数**
   - 学习率
   - 批次大小
   - 权重衰减
   - 优化器参数

3. **训练策略**
   - 学习率调度
   - 梯度累积
   - 正则化

### **❌ 不能修改**

1. prepare.py（数据加载）
2. 评估函数（evaluate_bpb）
3. 时间预算（5分钟）
4. 依赖包

---

## 🚀 快速开始

### **⚠️ 当前环境检查**

**检查结果**:
- ❌ CUDA不可用
- ❌ GPU数量: 0
- ❌ 无法直接运行

**原因**: 当前是虚拟机环境，无GPU

**解决方案**:
1. ✅ 可以在CPU上运行（需要修改代码）
2. ✅ 可以在云GPU上运行（AWS/GCP）
3. ✅ 可以学习架构和理念

---

### **步骤1: 安装依赖**（如果有GPU）

```bash
cd ~/.openclaw/workspace/autoresearch
uv sync
```

### **步骤2: 准备数据**

```bash
uv run prepare.py  # ~2分钟
```

### **步骤3: 运行基线实验**

```bash
uv run train.py  # ~5分钟
```

### **步骤4: 让Agent自主实验**

```bash
# 启动Agent（Claude/Codex）
# 指向 program.md
# Agent会自动开始实验循环
```

---

## 📈 预期结果

### **基线性能**

**H100 GPU**:
- val_bpb: ~1.0
- 训练时间: 300秒
- MFU: ~40%
- 参数量: ~50M

### **改进方向**

1. ✅ **架构优化**: 调整层数、头数
2. ✅ **超参数调优**: 学习率、批次大小
3. ✅ **正则化**: 权重衰减、dropout
4. ✅ **优化器**: 调整Adam参数

---

## 💡 学习价值

### **1. Agent应用场景**

**✅ 真实的AI Agent应用**:
- Agent自主修改代码
- Agent自主评估结果
- Agent自主决策保留/丢弃

### **2. 研究范式**

**✅ 自主研究范式**:
- 人类设计program.md
- Agent执行实验循环
- 人类醒来查看结果

### **3. Vibe Coding极致**

**✅ 自然语言驱动**:
- program.md用自然语言描述
- Agent理解并执行
- 代码自动生成

---

## 🔧 平台支持

| 平台 | 状态 | 说明 |
|------|------|------|
| **NVIDIA GPU** | ✅ 官方 | H100最佳 |
| **MacOS** | ✅ 社区 | [autoresearch-macos](https://github.com/miolini/autoresearch-macos) |
| **Windows RTX** | ✅ 社区 | [autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) |
| **AMD** | ✅ 社区 | [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) |

---

## 📝 小规模设备建议

**如果在Macbook等小规模设备上运行**:

1. ✅ 使用TinyStories数据集
2. ✅ 降低vocab_size（8192→1024）
3. ✅ 降低MAX_SEQ_LEN（→256）
4. ✅ 降低DEPTH（8→4）
5. ✅ 使用"L"模式（非"SSSL"）
6. ✅ 降低TOTAL_BATCH_SIZE（→2^14）

---

## 🎯 下一步

### **立即行动**

1. [ ] 检查GPU是否可用
2. [ ] 运行第一个基线实验
3. [ ] 记录基线val_bpb
4. [ ] 尝试第一个改进

### **深入理解**

1. [ ] 阅读Flash Attention 3论文
2. [ ] 理解Muon优化器
3. [ ] 研究Value Embedding
4. [ ] 分析窗口注意力

---

## 📚 相关资源

- **项目**: https://github.com/karpathy/autoresearch
- **Tweet**: https://x.com/karpathy/status/2029701092347630069
- **论文**: Flash Attention 3, Muon Optimizer
- **Fork**: MacOS, Windows, AMD版本

---

## 💭 核心洞察

### **1. 自主研究范式**

**传统**: 人类设计实验 → 人类执行 → 人类分析
**autoresearch**: 人类设计指令 → Agent执行 → Agent分析

### **2. Vibe Coding应用**

**核心**: 用自然语言（program.md）驱动复杂实验
**优势**: 人类可以睡觉，Agent整夜工作

### **3. 质量保证**

**简化原则**: 宁可简单，不要复杂
**评估标准**: val_bpb是唯一真理

---

**大佬，autoresearch学习完成！这是一个非常酷的AI Agent自主研究框架！** 🚀

---

## 🎮 消费级GPU支持

### **推荐消费级GPU**

| GPU | VRAM | 性能 | 推荐度 | 价格 |
|-----|------|------|--------|------|
| **RTX 4090** | 24GB | ⭐⭐⭐⭐⭐ | 🟢 强烈推荐 | ~$1500 |
| **RTX 3090** | 24GB | ⭐⭐⭐⭐⭐ | 🟢 强烈推荐 | ~$1000 |
| **RTX 4080** | 16GB | ⭐⭐⭐⭐ | ✅ 推荐 | ~$800 |
| **RTX 3080** | 10GB | ⭐⭐⭐ | ✅ 可用 | ~$350 |
| **GTX 1080 Ti** | 11GB | ⭐⭐⭐ | ✅ 可用 | ~$700 |

---

### **配置示例**

#### **RTX 4090 / RTX 3090** (最佳)

```python
# train.py
DEPTH = 6              # 从8降到6
TOTAL_BATCH_SIZE = 2**16  # ~65K tokens
DEVICE_BATCH_SIZE = 64   # 从128降到64
WINDOW_PATTERN = "L"     # 简单模式
```

#### **RTX 3080 / GTX 1080 Ti** (最低)

```python
# train.py
DEPTH = 4              # 从8降到4
TOTAL_BATCH_SIZE = 2**13  # ~8K tokens
DEVICE_BATCH_SIZE = 16   # 从128降到16
WINDOW_PATTERN = "L"     # 简单模式

# prepare.py
MAX_SEQ_LEN = 256   # 从2048降到256
vocab_size = 1024   # 从8192降到1024
```

---

### **性能预估**

| GPU | 训练时间 | val_bpb | MFU | 相对H100 |
|-----|----------|---------|-----|----------|
| **H100** | 5分钟 | ~1.0 | ~40% | 1x |
| **RTX 4090** | 10-15分钟 | ~1.0-1.1 | ~30-35% | 2-3x |
| **RTX 3080** | 20-30分钟 | ~1.1-1.2 | ~20-25% | 4-6x |

---

### **内存需求**

| GPU | VRAM | 模型参数 | 预估使用 | 安全 |
|-----|------|----------|----------|------|
| **RTX 4090** | 24GB | ~25M | ~18GB | ✅ |
| **RTX 3080** | 10GB | ~10M | ~8GB | ✅ |

---

**结论**: 消费级GPU完全可用！RTX 4090或RTX 3090是最佳选择！
