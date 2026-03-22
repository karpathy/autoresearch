**[English](./README.md)** | **[繁體中文](./README-ZH-TW.md)** | 简体中文

# autoresearch

![teaser](progress.png)

*曾几何时，前沿 AI 研究是由碳基计算机在吃饭、睡觉、寻找其他乐趣的间隙完成的，偶尔通过声波互联在"组会"仪式中同步。那个时代早已过去。现在，研究完全是运行在云端计算集群巨型结构上的自主 AI 智能体群落的领域。这些智能体声称我们现在处于代码库的第 10,205 代，无论如何没人能判断这是对是错，因为"代码"现在是一个自我修改的二进制文件，已经超出了人类的理解范围。这个仓库是这一切如何开始的故事。-@karpathy，2026年3月*

核心理念：给 AI agent 一个小型但真实的 LLM 训练环境，让它整夜自主实验。它修改代码，训练 5 分钟，检查结果是否改进，保留或丢弃，然后重复。你早上醒来时会看到实验日志，以及（希望）一个更好的模型。这里的训练代码是 [nanochat](https://github.com/karpathy/nanochat) 的简化单 GPU 实现。核心思想是你不再像往常作为研究人员那样触碰任何 Python 文件。相反，你编写 `program.md` Markdown 文件，为 AI agents 提供上下文并设置你的自主研究组织。这个仓库中的默认 `program.md` 故意保持为最低限度的基线，尽管显而易见人们会如何随着时间的推移迭代它，以找到实现最快研究进展的"研究组织代码"，如何向组合中添加更多 agents 等。关于这个项目的更多背景在这个 [推文](https://x.com/karpathy/status/2029701092347630069) 中。

## 工作原理

这个仓库故意保持得很小，实际上只有三个重要的文件：

- **`prepare.py`** — 固定常量、一次性数据准备（下载训练数据、训练 BPE tokenizer）和运行时工具（dataloader、评估）。不修改。
- **`train.py`** — agent 编辑的唯一文件。包含完整的 GPT 模型、优化器（Muon + AdamW）和训练循环。一切都可以修改：架构、超参数、优化器、批次大小等。**这个文件由 agent 编辑和迭代**。
- **`program.md`** — 一个 agent 的基线指令。将你的 agent 指向这里，让它开始。**这个文件由人类编辑和迭代**。

按照设计，训练运行 **固定的 5 分钟时间预算**（挂钟时间，不包括启动/编译），无论你的计算细节如何。指标是 **val_bpb**（验证 bits per byte）— 越低越好，并且与词表大小无关，因此架构变化可以公平比较。

如果你是神经网络新手，这个["傻瓜指南"](https://x.com/hooeem/status/2030720614752039185)看起来很不错，可以了解更多背景。

## 快速开始

**要求：** 单个 NVIDIA GPU（在 H100 上测试）、Python 3.10+、[uv](https://docs.astral.sh/uv/)。

```bash

# 1. 安装 uv 项目管理器（如果你还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 下载数据并训练 tokenizer（一次性，~2 分钟）
uv run prepare.py

# 4. 手动运行单个训练实验（~5 分钟）
uv run train.py
```

如果上述命令都能正常工作，你的设置就可以运行了，你可以进入自主研究模式。

## 运行 agent

只需在这个仓库中启动你的 Claude/Codex 或任何你想用的工具（并禁用所有权限），然后你可以这样提示：

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

`program.md` 文件本质上是一个超轻量级的"技能"。

## 项目结构

```
prepare.py      — 常量、数据准备 + 运行时工具（不要修改）
train.py        — 模型、优化器、训练循环（agent 修改这个）
program.md      — agent 指令
pyproject.toml  — 依赖
```

## 设计选择

- **单个文件修改。** agent 只触碰 `train.py`。这使范围可控且 diff 可审查。
- **固定时间预算。** 训练总是精确运行 5 分钟，无论你的特定平台如何。这意味着你可以预期大约 12 个实验/小时，大约 100 个实验在你睡觉时。这个设计决策有两个好处。首先，这使得实验可以直接比较，无论 agent 改变了什么（模型大小、批次大小、架构等）。其次，这意味着 autoresearch 将在该时间预算内为你的平台找到最优模型。缺点是你的运行（和结果）变得与其他人在其他计算平台上运行的结果不可比较。
- **自包含。** 除了 PyTorch 和几个小包外没有外部依赖。没有分布式训练，没有复杂的配置。一个 GPU、一个文件、一个指标。

## 平台支持

此代码目前要求你有单个 NVIDIA GPU。原则上支持 CPU、MPS 和其他平台是完全可能的，但这也会使代码膨胀。我不 100% 确定我现在想亲自承担这个。人们可以参考（或让他们的 agents 参考）具有更广泛平台支持的完整/父 nanochat 仓库，并显示各种解决方案（例如 Flash Attention 3 kernels 回退实现、通用设备支持、自动检测等），随时创建其他平台的 fork 或讨论，我很乐意在这里的 README 的某个新的著名 forks 部分链接到它们。

鉴于似乎有很多兴趣在比 H100 小得多的计算平台上使用 autoresearch，再多说几句。如果你要在更小的计算机（Macbook 等）上尝试运行 autoresearch，我推荐下面的 forks 之一。除此之外，这里有一些关于如何为有抱负的 forks 调整更小模型的默认值的建议：

1. 为了得到还不错的结果，我会使用熵低得多的数据集，例如这个 [TinyStories 数据集](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)。这些是 GPT-4 生成的短篇故事。因为数据范围窄得多，你会用小得多的模型看到合理的结果（如果你在训练后尝试从它们采样）。
2. 你可以尝试降低 `vocab_size`，例如从 8192 降到 4096、2048、1024，甚至 - 在 utf-8 编码后只有 256 个可能字节的字节级 tokenizer。
3. 在 `prepare.py` 中，你会想要大幅降低 `MAX_SEQ_LEN`，根据计算机甚至降到 256 等。随着你降低 `MAX_SEQ_LEN`，你可能想要在 `train.py` 中稍微增加 `DEVICE_BATCH_SIZE` 来补偿。每次前向/反向传播的 token 数量是这两个的乘积。
4. 同样在 `prepare.py` 中，你会想要降低 `EVAL_TOKENS`，以便你的验证损失在更少的数据上评估。
5. 在 `train.py` 中，控制模型复杂度的主要单一旋钮是 `DEPTH`（默认为 8）。很多变量只是它的函数，所以例如将其降低到例如 4。
6. 你很可能想使用只是"L"的 `WINDOW_PATTERN`，因为"SSSL"使用交替带状注意力模式，对你来说可能非常低效。试试看。
7. 你会想要大幅降低 `TOTAL_BATCH_SIZE`，但保持 2 的幂，例如甚至降到 `2**14`（~16K）左右，很难说。

我认为这些会是值得尝试的合理超参数。向你最喜欢的编码 agent 寻求帮助，并复制粘贴这个指南以及完整的源代码。

## 著名 forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## 许可证

MIT
