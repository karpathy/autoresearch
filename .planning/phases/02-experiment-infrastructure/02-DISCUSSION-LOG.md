# Phase 2: Experiment Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-25
**Phase:** 02-experiment-infrastructure
**Areas discussed:** Loop ownership, Metric output, Results logging

---

## Loop Ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Agent 直接跑 | Agent 用 shell 呼叫 python train.py，自己處理 git/log | ✓ |
| run.sh 包裝 | shell script 包裝執行 + log + crash 處理 | |
| Claude 決定 | 讓 Claude 決定 | |

**User's choice:** Agent 直接跑
**Notes:** Follows original autoresearch pattern.

---

## Metric Output

| Option | Description | Selected |
|--------|-------------|----------|
| train.py print | train.py 最後 print 指標，agent 從 stdout grep | |
| JSON 檔案 | train.py 寫入 metrics.json，agent 讀取 | ✓ |
| Claude 決定 | 讓 Claude 決定 | |

**User's choice:** JSON 檔案
**Notes:** More reliable than stdout parsing.

---

## Results Logging

| Option | Description | Selected |
|--------|-------------|----------|
| Agent 寫入 | Agent 讀取 stdout 後自己 append 到 results.tsv | |
| train.py 寫入 | train.py 結束時自動 append 一行到 results.tsv | |
| Claude 決定 | 讓 Claude 決定 | ✓ |

**User's choice:** Claude 決定
**Notes:** Key requirement is every experiment gets a row.

---

## Claude's Discretion

- results.tsv write ownership
- Git workflow details
- 3-consecutive-crash skip logic implementation

## Deferred Ideas

None
