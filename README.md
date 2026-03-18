# ctr-autoresearch

本仓库聚焦于 CTR 模型的自动化迭代，思路参考 FuxiCTR 的配置化训练流程。

## 工作原理

仓库结构保持精简，核心文件如下：

- **`prepare.py`** — CTR 数据准备、特征映射与缓存。
- **`train.py`** — CTR 模型与时间预算训练流程。
- **`program.md`** — 代理实验指引。

训练默认固定 5 分钟预算。关键指标为 **val_auc**（越高越好）与 **val_logloss**（越低越好）。

## 快速开始

**依赖要求：** Python 3.10+、[uv](https://docs.astral.sh/uv/)。CTR 需要本地数据集（CSV 或 Parquet），并在数据文件同目录提供 `features.json`。

```bash
# 1. 安装 uv 项目管理器（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 准备 CTR 数据
CTR_DATA_PATH=/path/to/data uv run prepare.py --data-path /path/to/data

# 4. 运行一次训练实验（约 5 分钟）
CTR_DATA_PATH=/path/to/data uv run train.py
```

## 运行自动化代理

将代理指向 `program.md`，按步骤完成设置后即可开始 CTR 实验循环。

## 项目结构

```
prepare.py      — CTR 数据准备与特征映射
train.py        — CTR 训练流程
program.md      — 代理指引
pyproject.toml  — 依赖声明
```

## 设计选择

- **单文件修改。** CTR 迭代集中在 `train.py`。
- **固定时间预算。** 每次实验限制为 5 分钟，保证可比性。
- **自包含。** 除 PyTorch 与少量依赖外，无额外外部依赖。

## 许可

MIT
