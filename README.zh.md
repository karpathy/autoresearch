# autoresearch (自主研究)

![teaser](progress.png)

*曾几何时，前沿 AI 研究是由“肉体计算机”在吃饭、睡觉、玩乐之余完成的，他们偶尔还会通过“声波互联”这种被称为“组会”的仪式进行同步。那个时代早已一去不复返。现在的研究完全是 AI Agent 自主集群的领地，它们穿梭在云端的巨型计算集群中。Agent 们声称现在的代码库已经迭代到了第 10,205 代。无论如何，没人能分辨对错，因为现在的“代码”已经变成了一种自我修改的二进制文件，其复杂程度早已超越了人类的理解。本仓库记录了这一切是如何开始的故事。——@karpathy，2026年3月*。

**核心构思**：给 AI Agent 一个规模虽小但功能完备的真实 LLM 训练环境，让它在夜间自主进行实验。它修改代码，训练 5 分钟，检查结果是否有所提升，决定保留或放弃，然后循环往复。当你第二天清晨醒来，你会看到一份实验日志，以及（理想情况下）一个更好的模型。这里的训练代码是 [nanochat](https://github.com/karpathy/nanochat) 的简化单显卡实现。核心理念是，作为研究员，你不再像往常那样去碰任何 Python 文件。相反，你是在编写 `program.md` 这个 Markdown 文件，它为 AI Agent 提供上下文并构建你的自主研究组织。本仓库中默认的 `program.md` 刻意保持为一个精简的基准方案，但显而易见，你可以随着时间的推移对其进行迭代，找到能实现最快研究进展的“研究组织代码”，或是引入更多的 Agent 等。关于该项目的更多背景信息，请参阅[这条推文](https://x.com/karpathy/status/2029701092347630069)和[这条推文](https://x.com/karpathy/status/2031135152349524125)。

## 工作原理

本仓库刻意保持精简，实际上只有三个关键文件：

- **`prepare.py`** —— 包含固定常量、一次性数据准备（下载训练数据、训练 BPE 分词器）以及运行时工具（数据加载器、评估）。不可修改。
- **`train.py`** —— Agent 唯一编辑的文件。包含完整的 GPT 模型、优化器（Muon + AdamW）和训练循环。一切皆可调整：架构、超参数、优化器、批大小（batch size）等。**该文件由 Agent 进行编辑和迭代**。
- **`program.md`** —— 单个 Agent 的基准指令。将你的 Agent 指向这里，让它开始工作。**该文件由人类进行编辑和迭代**。

根据设计，无论你的算力细节如何，训练运行都有一个**固定的 5 分钟时间预算**（墙上时间，不包括启动/编译）。衡量指标是 **val_bpb**（验证每字节比特数）—— 越低越好。该指标与词表大小无关，因此可以公平地比较架构层面的变化。

如果你是神经网络新手，这份[“傻瓜指南”](https://x.com/hooeem/status/2030720614752039185)提供了更多背景信息，看起来非常不错。

## 快速开始

**要求**：单块 NVIDIA GPU（已在 H100 上测试）、Python 3.10+、[uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装 uv 项目管理器（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 下载数据并训练分词器（一次性操作，约 2 分钟）
uv run prepare.py

# 4. 手动运行单次训练实验（约 5 分钟）
uv run train.py
```

如果上述命令均能正常运行，说明你的环境已就绪，可以进入自主研究模式了。

## 运行 Agent

只需在仓库中启动你的 Claude/Codex 或任何你喜欢的工具（并禁用所有权限），然后你可以输入类似以下的提示词：

```
你好，请查看 program.md，让我们开启一次新实验！先完成初始化设置。
```

`program.md` 文件本质上是一个非常轻量级的“技能（skill）”。

## 项目结构

```
prepare.py      — 常量、数据准备 + 运行时工具（请勿修改）
train.py        — 模型、优化器、训练循环（Agent 修改此文件）
program.md      — Agent 指令
pyproject.toml  — 依赖项
```

## 设计选择

- **单一修改文件**。Agent 仅触碰 `train.py`。这使范围保持在可控范围内，且 diff（代码差异）易于审查。
- **固定时间预算**。无论你的具体平台如何，训练总是精确运行 5 分钟。这意味着你可以预期每小时约 12 次实验，在你睡觉时约 100 次实验。这一设计决策有两个优点：首先，无论 Agent 更改了什么（模型大小、批大小、架构等），实验结果都具有直接可比性；其次，这意味着 autoresearch 将在给定的时间预算内找到最适合你平台的优化模型。缺点是你的运行（和结果）无法与使用其他计算平台的人直接对比。
- **自包含**。除了 PyTorch 和少数几个小包外，没有外部依赖。没有分布式训练，没有复杂的配置。一块 GPU，一个文件，一个指标。

## 平台支持

目前代码要求使用单块 NVIDIA GPU。原则上支持 CPU、MPS 和其他平台是完全可能的，但这会使代码变得臃肿。我目前并不确定是否想亲自承担这项工作。大家可以参考（或让你们的 Agent 参考）支持更广泛平台的完整版/父级 [nanochat](https://github.com/karpathy/nanochat) 仓库，那里展示了各种解决方案（例如 Flash Attention 3 kernel 的 fallback 实现、通用设备支持、自动检测等）。欢迎创建针对其他平台的 fork 或发起讨论，我也很乐意在 README 的“显著 Fork”部分链接它们。

鉴于大家对于在比 H100 算力小得多的平台上折腾 autoresearch 表现出了浓厚兴趣，这里多说几句。如果你打算在较小的设备（Macbook 等）上运行 autoresearch，我建议参考下方的 fork。此外，对于想要尝试的小型模型 fork，这里有一些关于如何调整默认值的建议：

1. 为了获得尚可的结果，建议使用熵值较低的数据集，例如这个 [TinyStories 数据集](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)。这些是 GPT-4 生成的短篇故事。因为数据的范围更窄，你会在模型规模小得多时看到合理的结果（如果你在训练后尝试采样）。
2. 你可以尝试减小 `vocab_size`，例如从 8192 降到 4096、2048、1024，甚至直接使用 UTF-8 编码后的 256 字节级分词器。
3. 在 `prepare.py` 中，你需要大幅降低 `MAX_SEQ_LEN`，根据电脑性能甚至可以降到 256 等。降低 `MAX_SEQ_LEN` 的同时，你可能需要尝试在 `train.py` 中略微增加 `DEVICE_BATCH_SIZE` 作为补偿。每次前向/后向传递的 token 数是这两者的乘积。
4. 同样在 `prepare.py` 中，你需要减小 `EVAL_TOKENS`，以便在更少的数据上评估验证损失。
5. 在 `train.py` 中，控制模型复杂度的主要旋钮是 `DEPTH`（此处默认为 8）。许多变量只是它的函数，所以例如可以将其降低到 4。
6. 你可能更希望将 `WINDOW_PATTERN` 仅设为 "L"，因为 "SSSL" 使用的交替带状注意力模式对你来说可能非常低效。试一试。
7. 你需要大幅降低 `TOTAL_BATCH_SIZE`，但保持其为 2 的幂，例如降到 `2**14` (~16K) 左右，具体视情况而定。

我认为这些是值得尝试的合理超参数。请向你最喜欢的编程 Agent 寻求帮助，并给它们复制这份指南以及完整的源代码。

## 显著 Fork

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## 许可证

MIT
