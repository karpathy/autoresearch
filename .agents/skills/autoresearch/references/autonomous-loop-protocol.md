# Autonomous Loop Protocol for AI Agents

This protocol defines the recommended zero-file, bounded-context interaction loop for AI agents operating an AutoResearch lab.

## Core Philosophy: Zero-File Overhead & Direct Cockpit Diagnostics

AI agents should minimize tool round-trips and context overhead:
1. **Never write temporary hypothesis files to disk**: Use inline hypothesis flags (`--hypothesis-json` or `--title`/`--observation`/`--expected-effect`) directly on CLI commands.
2. **Never read evaluator log files for crashes**: Crash stack traces are automatically captured in `impact.error_tail` and `run_detail.error_tail` in the command response envelope.
3. **Never parse `cases.jsonl` for case regressions**: Regressed case details are automatically extracted into `case_samples` within the `case_regression` signal.
4. **Protect against idea repetition across context compaction**: The cockpit surfaces `dead_ends` (up to 8 rejected/crashed hypotheses) backed by a 100-run memory window.

---

## 1. Inspect State & Cockpit

Start every cycle by reading the compact agent cockpit:

```bash
python scripts/lab.py --agent status
# or for detailed guidance:
python scripts/lab.py --agent guide --json
```

The response envelope returns:
- `stage`: Current lab stage (`BASELINE_NEEDED`, `ITERATING`, `CONVERGED`, etc.).
- `recommended_action`: The recommended next action (`RUN_BASELINE`, `RUN_TRIAL`, etc.).
- `cockpit.dead_ends`: List of recent rejected or crashed hypothesis attempts from the current epoch (up to 8 unique entries), preventing repeating failed ideas even after conversation context compaction.
- `cockpit.signals`: Active research signals (e.g. `repeated_rejection`, `case_regression`, `near_miss`, `similar_hypothesis_rejected`).

---

## 2. Formulate Hypothesis & Edit Candidate

Review `cockpit.dead_ends` and active signals. Ensure your proposed change is:
- **Novel**: Distinct from any entry in `cockpit.dead_ends`.
- **Falsifiable**: Identifies an observation, proposed mechanism, and expected metric effect.
- **Isolated**: Exactly one atomic change in `candidate/`.

Edit the candidate files directly in `candidate/`.

---

## 3. Execute Trial with Zero-File Inline Hypotheses

Execute the trial using **zero-file inline arguments**. Do NOT call `write_file` to create a temporary `hypothesis.json`.

### Option A: Inline JSON (Recommended)

```bash
python scripts/lab.py --agent --confirmed step --run \
  --hypothesis-json '{"title":"Vectorize loop","observation":"Inner loop takes 80% CPU","expected_effect":"Reduce latency by 25%"}'
```

Or for asynchronous / background trials:

```bash
python scripts/lab.py --agent --confirmed trial start \
  --hypothesis-json '{"title":"Vectorize loop","observation":"Inner loop takes 80% CPU","expected_effect":"Reduce latency by 25%"}'
python scripts/lab.py --agent trial wait --timeout 60
```

### Option B: Inline Individual Flags

```bash
python scripts/lab.py --agent --confirmed step --run \
  --title "Vectorize loop" \
  --observation "Inner loop takes 80% CPU" \
  --expected-effect "Reduce latency by 25%"
```

All three fields (`title`, `observation`, `expected_effect`) are required and validated before execution.

---

## 4. Analyze Results Directly from Cockpit Diagnostics

Read the output JSON envelope `data.cockpit`:

### On Trial Acceptance / Rejection
- `cockpit.impact.status`: `"accepted"` or `"rejected"`.
- `cockpit.impact.score`: Candidate score.
- `cockpit.impact.delta`: Parent-relative delta.
- `cockpit.impact.is_new_best`: `true` if new best in epoch.

### On Crash Diagnostics (`error_tail`)
If the evaluator crashes (non-zero exit code or uncaught exception):
- `cockpit.impact.error_tail` and `cockpit.run_detail.error_tail` automatically contain the last 10 lines (up to 600 characters) of the evaluator's `stderr` / `stdout`.
- `cockpit.impact.reason` includes the exit code and the last line of the error trace.
- **Do NOT execute tool calls to read `run.log`** — all debugging context is present directly in `error_tail`.

### On Case Regression Diagnostics (`case_samples`)
If a trial regresses on test cases that previously passed in the parent run:
- A `case_regression` signal is emitted in `cockpit.signals`.
- The signal includes `case_samples`: up to 3 compact diagnostics containing:
  - `id`: The test case ID.
  - `status`: Failure status or outcome.
  - `error`: Error message preview (up to 120 chars).
  - `metrics`: Scalar metrics observed on that specific case (up to 3 metrics).
- **Do NOT execute tool calls to read `cases.jsonl`** — the failing case samples are directly visible in `signals`.

### On Long-Horizon Memory & Plateau (`dead_ends`)
- `cockpit.dead_ends` tracks up to 8 unique failed candidate hashes/titles from the current epoch.
- Signal history tracks up to 100 runs (expanded from 15), correctly identifying `same_candidate_rejected` and `similar_hypothesis_rejected` across long-running experiments.
- If `repeated_rejection` occurs (3+ consecutive rejected trials), consult the `openresearch` skill (`orx discover`) for external literature retrieval rather than continuing local permutations.
