# Agent compatibility

Each row is one real run of [`tests/harness_smoke/run.py`](../tests/harness_smoke/run.py). The driver prepares the toy lab as the host, tells the agent to run a baseline and one trial, then checks the result itself instead of trusting the agent's report.

| Agent | Version | Model | Date | Result |
| --- | --- | --- | --- | --- |
| Claude Code | 2.1.316 | Haiku | 2026-11-01 | Passed on Windows: baseline and one accepted trial, `mean_error` 2.000 to 0.000, measured project unchanged, 66 s |
| Codex CLI | 0.139.0 | | 2026-11-01 | Not verified: the provider rejected the request before the agent started, because this CLI version does not support the models available to the account. The skill never ran |
| Cursor | | | | Not verified |
| Gemini CLI | | | | Not verified |

## What a pass does and does not show

- The host prepared the lab and wrote the harness profile. A pass does not cover the agent preparing a lab from a goal.
- A pass covers one short offline evaluator on one machine.

## Run it yourself

Each run spends your agent's quota, so the driver is not part of the test suite or CI.

```text
python tests/harness_smoke/run.py claude
python tests/harness_smoke/run.py codex --model <model>
```
