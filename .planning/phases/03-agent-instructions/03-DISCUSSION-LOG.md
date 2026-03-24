# Phase 3: Agent Instructions - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-25
**Phase:** 03-agent-instructions
**Areas discussed:** Agent platform, Experiment hints, Search scope

---

## Agent Platform

| Option | Description | Selected |
|--------|-------------|----------|
| Claude Code | 用 Claude Code CLI 作為 agent | ✓ |
| OpenAI Codex | 跟原版 autoresearch 一樣用 Codex CLI | |
| Claude 決定 | 讓 Claude 決定 | |

**User's choice:** Claude Code
**Notes:** Direct terminal execution.

---

## Experiment Hints Detail

| Option | Description | Selected |
|--------|-------------|----------|
| 詳細實驗清單 | 列出具體實驗 idea 清單 | |
| 方向性指引 | 描述實驗方向，agent 自己生成具體實驗 | |
| Claude 決定 | 讓 Claude 決定最佳平衡 | ✓ |

**User's choice:** Claude 決定
**Notes:** Balance between guidance and agent autonomy.

---

## Search Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Loss 權重/組合 | distill, arcface, vat, separation loss | ✓ |
| Model 架構 | projection head, embedding dim, activation | ✓ |
| Optimizer/LR | optimizer, learning rate, scheduler, warmup | ✓ |
| Augmentation | training augmentation pipeline | ✓ |

**User's choice:** All four, with critical constraint
**Notes:** "此模型必須要跑在 edge device 上，所以不能把模型無限加大。更改 model 架構必須考量到模型的參數、GFLOPs，embedding dim 盡量不要跟 256 差太多，在 edge inference 也會有問題。"

---

## Claude's Discretion

- Exact experiment prioritization order
- Whether to include specific loss alternatives
- "What to try when stuck" section structure

## Deferred Ideas

None
