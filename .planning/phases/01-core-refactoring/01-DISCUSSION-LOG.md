# Phase 1: Core Refactoring - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-25
**Phase:** 01-core-refactoring
**Areas discussed:** Split boundary, Teacher model, Execution model, Backbone strategy

---

## Split Boundary (Augmentation)

| Option | Description | Selected |
|--------|-------------|----------|
| train.py 完全控制 | train augmentations 全部在 train.py，prepare.py 只提供基礎 dataset class | ✓ |
| prepare.py 定義 + train.py 注入 | prepare.py 定義 dataset class，train.py 透過 transform 參數注入 augmentation | |
| Claude 決定 | 讓 Claude 根據研究結果和程式碼分析決定最佳拆分方式 | |

**User's choice:** train.py 完全控制
**Notes:** Agent gets maximum flexibility over augmentation pipeline.

---

## Teacher Model

| Option | Description | Selected |
|--------|-------------|----------|
| ONNX (Trendyol) | 較快 inference，已有 disk cache 在 workspace/output/trendyol_teacher_cache2/ | ✓ |
| DINOv2 (Trendyol) | HuggingFace model，可能品質更高但較慢 | |
| 兩者都支援 | prepare.py 支援兩種 teacher，透過參數選擇 | |

**User's choice:** ONNX (Trendyol)
**Notes:** Existing cache available, faster inference.

---

## Execution Model

| Option | Description | Selected |
|--------|-------------|----------|
| 獨立執行 | `python train.py` 內部 import prepare.py，跟原版 autoresearch 一樣 | ✓ |
| prepare.py 呼叫 | `python prepare.py --train` 由 prepare.py 啟動 train.py 並評估結果 | |
| Claude 決定 | 讓 Claude 根據研究結果決定最佳執行模式 | |

**User's choice:** 獨立執行
**Notes:** Follows original autoresearch pattern.

---

## Backbone Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| train.py 控制 | unfreezing schedule 是 agent 可調整的實驗變數 | ✓ |
| prepare.py 固定 | 凍結策略不可變，確保實驗之間的公平比較 | |
| Claude 決定 | 讓 Claude 根據 domain 理解決定 | |

**User's choice:** train.py 控制
**Notes:** Backbone unfreezing is a high-leverage experiment axis for distillation.

---

## Claude's Discretion

- Exact split of helper functions (PadToSquare, collate functions)
- Whether ArcFace head setup goes in prepare.py or train.py

## Deferred Ideas

None
