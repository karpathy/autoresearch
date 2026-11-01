# Agent compatibility

Each row is one real run of [`tests/harness_smoke/run.py`](../tests/harness_smoke/run.py), in two phases. In phase 1 the agent gets only the toy project and evaluator, writes the lab spec from the skill's documentation and prepares the lab. The host then finds the lab config itself and writes the offline harness profile. In phase 2 the same agent session resumes and runs a baseline and one trial. The driver checks the result itself instead of trusting the agent's report.

| Agent | Version | Model | Date | Result |
| --- | --- | --- | --- | --- |
| Claude Code | 2.1.316 | Haiku | 2026-11-01 | Passed on Windows: the agent wrote the spec and prepared the lab, then ran a baseline and one accepted trial; `mean_error` 2.000 to 0.000, measured project unchanged, 114 s |
| Codex CLI | 0.166.0-alpha.5.1 | gpt-6-luna | 2026-11-01 | Passed on Windows: same scenario and result, 122 s |
| Cursor | | | | Not verified |
| Gemini CLI | | | | Not verified |

## What a pass does and does not show

- A pass covers one short offline evaluator on one machine.

## Run it yourself

Each run spends your agent's quota, so the driver is not part of the test suite or CI. Each phase has its own timeout (`--timeout`, default 900 seconds). Codex needs `--model`, since it has no default model; Claude defaults to Haiku. Output goes to `.tests/harness_smoke/<agent>-<timestamp>/`: per-phase stdout and stderr, and `result.json` with the exit code and seconds of each phase.

```text
python tests/harness_smoke/run.py claude
python tests/harness_smoke/run.py codex --model <model>
```
