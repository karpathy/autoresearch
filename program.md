# ctr-autoresearch

本仓库聚焦于 CTR 模型的自动化迭代，思路参考 FuxiCTR 的配置化训练流程。

## 设置

要开始新的实验，请按以下步骤与用户协作：

1. **确定实验标签**：用当天日期生成标签（例如 `mar14`）。分支 `ctr-autoresearch/<tag>` 必须不存在。
2. **创建分支**：从当前 master 创建 `git checkout -b ctr-autoresearch/<tag>`。
3. **阅读范围文件**：仓库很小，完整阅读：
   - `README.md` — 项目说明。
   - `prepare.py` 与 `train.py`
4. **确认数据路径**：CTR 使用本地 CSV 数据，且在数据文件同目录放置 `features.json`。
5. **执行准备（CTR）**：
   - `CTR_DATA_PATH=/mlx_devbox/users/yangao.v/repo/28288/ModelByAI/data/Criteo_x1/sample_200k.csv uv run prepare.py --data-path /mlx_devbox/users/yangao.v/repo/28288/ModelByAI/data/Criteo_x1/sample_200k.csv --test-ratio 0.4 --force`
6. **初始化 results.tsv**：创建只包含表头的 `results.tsv`。

确认无误后，开始实验循环。

## 实验

每次实验在 CPU 上运行，默认固定 5 分钟预算。启动命令：`CTR_DATA_PATH=/mlx_devbox/users/yangao.v/repo/28288/ModelByAI/data/Criteo_x1/sample_200k.csv uv run train.py`。

**可以做的事：**
- 修改 `train.py` — 模型结构、优化器、超参、模型类型等。

**不可以做的事：**
- 安装新依赖或添加包，只能使用 `pyproject.toml` 中已有依赖。

目标：提升 `val_auc`（越高越好）并降低 `val_logloss`（越低越好）。

**首次运行**：第一轮必须先跑基线，直接运行默认训练脚本。

## 输出格式

脚本结束后会输出类似结果：

```
---
val_auc:          0.762100
val_logloss:      0.421300
training_seconds: 300.1
total_seconds:    300.1
peak_vram_mb:     1024.0
num_steps:        1200
num_params_M:     2.31
model:            DEEPFM
embed_dim:        16
```

## 结果记录

每次实验结束后记录到 `results.tsv`（制表符分隔）。

```
commit	val_auc	val_logloss	memory_gb	status	description
```

## 实验循环

实验在独立分支上进行（例如 `ctr-autoresearch/mar14`）。

LOOP FOREVER:

1. 查看当前分支/提交状态
2. 在对应训练文件中做实验性修改
3. git commit
4. 运行实验：`CTR_DATA_PATH=/mlx_devbox/users/yangao.v/repo/28288/ModelByAI/data/Criteo_x1/sample_200k.csv uv run train.py > run.log 2>&1`
5. 读取结果：`grep "^val_auc:\|^val_logloss:\|^peak_vram_mb:" run.log`
6. 记录结果到 tsv
7. 如果指标提升则保留提交，否则重置回退
