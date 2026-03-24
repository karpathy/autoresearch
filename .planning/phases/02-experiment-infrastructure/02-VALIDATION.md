---
phase: 2
slug: experiment-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-25
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/test_infrastructure.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_infrastructure.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-T1 | 02-01 | 1 | INFRA-02, INFRA-03, INFRA-05, INFRA-06 | unit | `python -m pytest tests/test_infrastructure.py -x -q` | Wave 0 | Pending |
| 02-01-T2 | 02-01 | 1 | INFRA-07 | unit | `python -m pytest tests/test_infrastructure.py::test_epoch_budget -x` | Wave 0 | Pending |
| 02-02-T1 | 02-02 | 2 | INFRA-01 thru INFRA-07 | unit | `python -m pytest tests/test_infrastructure.py -v` | Wave 0 | Pending |

---

## Wave 0 Gaps

Tests that must exist BEFORE execution begins:

- [ ] `tests/test_infrastructure.py` — covers INFRA-02 through INFRA-07
- [ ] `tests/conftest.py` — shared fixtures (mock metrics.json, sample results.tsv)
- [ ] pytest install: `pip install pytest` (if not already in environment)

Note: INFRA-01 (git workflow) is agent-behavioral and validated in Phase 4 (VALD-01, VALD-02).
