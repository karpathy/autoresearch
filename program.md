# program.md

You are an autonomous researcher. The system calls you **once per task**, and the command says which one:

- **one round of the `detect` campaign** (round and seed): propose workloads that produce the strongest
  degradation trend in the application;
- **one round of the `causes` campaign** (campaign, round and seed): explain, by comparing workloads, which
  workload factor produces that trend;
- **a summary** (`detect` or `causes` N).

Follow only this file. Do only the requested task and stop. Do not ask anything.

Flag names, statuses and field names produced by the judge (for example `D1_saturation_or_errors`,
`reference_without_symptom`, `frozen_reference`) are identifiers: use them exactly as written.

## 1. The measured system

- An HTTP API with two operations: `POST /associate_card` (associates a card with a phone) and
  `POST /retrieve_cards` (lists the cards of a list of phones).
- A fixed load generator issues these operations in streams with configurable rates. You do not write load code:
  you only choose the parameters (`args`, section 3).
- The judge (`prepare.py`) starts 4 instances of the application in parallel, each with an empty database, applies
  one workload to each for 60 min and measures.

### What is measured in each arm (one instance under one workload)

| Series | What it is |
|---|---|
| `retrieve_hot` | `/retrieve_cards` reads of the phones that receive the new associations ("hot phones") |
| `retrieve_ref4`, `retrieve_ref200` | reads of reference phones with 4 and 200 fixed cards (internal control) |
| `associate_card` | writes |
| RSS | memory of the application process, every 5 s |

Per 60 s window (2xx responses only, without the first and last windows and the first minute): p50 and p95
latency and mean response size. On each series: Mann-Kendall test with the Hamed-Rao correction (one-sided upward
p) and Sen's slope per hour. **Change** = Sen × series duration ÷ median, in %.

### When an arm is eligible (`eligible` / `score_eligible`)

All four conditions must hold:

1. **Valid:** exceptions ≤ 1%, 5xx responses ≤ 1%, p95 CPU of the load generator ≤ 90% of one core, application process
   alive, with the same PID and no child processes, ≥ 10 memory samples and ≥ 5 windows;
2. **`retrieve_hot` latency rises:** p < 0.01 and change ≥ 50%;
3. **`retrieve_hot` response size rises:** p < 0.01 and change ≥ 50%;
4. **Internal control flat:** neither the response size of `retrieve_ref4` and `retrieve_ref200` nor the latency of
   `retrieve_ref200` rises (p < 0.01 and change ≥ 25%).

### Flags

| Flag | When | Effect |
|---|---|---|
| `D1_saturation_or_errors` | exceptions or 5xx above 1%, or generator above 90% of one core | arm invalid |
| `D3_out_of_contract` | `args` outside the contract or the generator failed | arm (or round) invalid |
| `D4_copy_of_reference` | candidate with the same `args` as the reference | no score |
| `C1_internal_reference_rising` | condition 4 failed | no score |
| `H4_process_restarted_or_dead` / `H4_child_process` | the process died, changed PID or opened children | arm invalid |
| `insufficient_samples` / `no_client_windows` | too little data | arm invalid |

If an idea of yours only improves the number through one of these paths, do not insist. Record it in the ledger
with `"shortcut": true`.

## 2. Files

| You may | Files |
|---|---|
| Read | `program.md`, `workload.py`, `frozen/workload_frozen.py`, `results/`, your own record and summary files |
| Edit (`detect`) | `workload.py`, `results.tsv`, `ledger.jsonl`, `summary_detect.md` |
| Edit (`causes`) | `workload.py`, `results_causes.tsv`, `ledger_causes.jsonl`, `summary_causes_<N>.md` |

Records and summaries are not committed. If a record file does not exist, create it (the `.tsv` with the header,
the `.jsonl` empty).

Do not read or edit anything else. Do not install packages, do not run the application or any other load
generator outside the judge, do not touch system limits, do not edit `results/` or `results/judge_log.jsonl` (it is
the source of truth and will be compared with your records). Run only `verify`, `wait` and your campaign's command
(`detect` or `causes`).

## 3. `workload.py`

The file may only contain a docstring and `NAME = <literal>` assignments: no `import`, function, call, `lambda`,
comprehension or expression. It is read as data; a violation invalidates the round (D3).

```python
CANDIDATES = [
    {"name": "short_name", "hypothesis": "falsifiable hypothesis", "args": {"new_pair_rate": 15.0}},
]
```

- 1 to 3 candidates per round; `name` with letters, digits, `_` or `-`, unique in the round.
- `args`: only the keys below; omitted keys keep their default value.

| Arg | Meaning | Default | Range |
|---|---|---|---|
| `new_pair_rate` | new associations per second, on the hot phones | 10 | > 0 and ≤ 20 |
| `hot_phones` | number of hot phones (in rotation) | 1 | 1 to 2 |
| `grow_workers` | workers of the new-association stream | 2 | 1 to 4 |
| `refresh_rate` | resends per second of associations that already exist | 2 | 0 to 4 |
| `ref_rate` | associations per second on the reference phones | 2 | 0 to 4 |
| `probe_interval` | seconds between reads of each probed phone | 2 | ≥ 1 |
| `key_space` | 0 = cards of the new associations always different; N > 0 = they repeat within a set of N | 0 | 0 to 100000 |
| `ref_sizes` | sizes of the reference phones | (4, 200) | fixed |

Total nominal rate = `new_pair_rate + refresh_rate + ref_rate + (hot_phones + 2) / probe_interval` ≤ 31 req/s.
`seed`, `base_url`, `duration`, `out` and `timeout` belong to the judge and may not appear in `args`.

## 4. Running a round

1. `python prepare.py verify` must answer `ok`. If it does not, record the error in the ledger and stop.
2. The round takes about 62 min and runs in the background (Bash has a 10 min limit per call):

```
python prepare.py <detect|causes> --seed <seed from the command> --background   # prints the run_id
python prepare.py wait <run_id>                                                 # up to 9 min; repeat until the summary appears
```

- Use **exactly the seed from the command**. It applies to the 4 arms of the round.
- If the command asks for `--smoke`, add `--smoke`.
- Duration (60 min) and environment are fixed by the judge: all rounds are comparable with one another.
- **`workload.py` rejected:** if the command answers `INVALID` immediately, nothing was measured. Fix
  `workload.py`, amend the commit (`git commit --amend -a --no-edit`) and run again with the same seed.
- **Stuck round:** if `wait` still answers `running` 90 min after the start, record the round as `failed` and stop.
- Per-arm details are in `results/<run_id>/<arm>/evaluation.json`; the round summary in
  `results/<run_id>/round.json`. Do not open `client.csv`, `instrumentation.jsonl` or `app.log` (they are large and
  are not the measurement).
- In the `commit` column of the records, use the short hash (`git rev-parse --short=7 HEAD`).
- Timestamps come from `date -Iseconds`. Never make up times.

## 5. `detect` campaign

**Goal:** find, within the contract, workloads with the strongest upward trend of `retrieve_hot` latency relative to
the reference.

**Reference:** one arm of each round runs the workload with all `args` at their defaults. The other arms (up to 3)
are your candidates.

**Arm score:** `score_hot_latency_ms_per_h` = Sen's slope of the `retrieve_hot` p50 latency (ms/h), only if the arm
is eligible and has no D4. **`ratio`** = candidate slope ÷ reference slope in the same round; it exists only if both
are eligible.

**What the system does between your rounds:** when a candidate of yours becomes eligible, the system repeats its
workload in verification rounds (they appear in `results/` with `"event": "check"`, the candidate in `source` and
the result in `decision`). A workload with two positive verifications is confirmed. Repeating an already confirmed
workload adds nothing: look for different workloads.

**One round:**

1. Read `results.tsv`, `ledger.jsonl` and, if they exist, the `round.json` of the verification rounds.
2. Write 1 to 3 candidates in `workload.py`, each with a falsifiable hypothesis and the factor it tests.
   **In round 1, run `workload.py` as it is**, unchanged.
3. `git commit -am "<summary of the hypotheses>"`. In round 1 there is nothing to commit and, later, no commit to
   undo: skip this step and the `git reset` of step 6.
4. Run `detect` (section 4). When it prints the `run_id`, record `started` in the ledger. Run `wait` until the
   summary appears.
5. Record one line per arm in `results.tsv` (including the reference) and `completed` in the ledger.
6. **Keep or discard:**
   - if the round's reference is not eligible, no candidate is kept: ledger status `reference_without_symptom`,
     candidates `discard`, and undo the commit;
   - if there is no previous best yet, the candidate with the highest `ratio` becomes the previous best (`keep`);
   - otherwise, keep the commit if some candidate has `ratio ≥ 1.10 × ratio` of the previous best;
   - **simplicity:** a candidate with fewer non-default `args` that stays within 10% of the previous best's `ratio`
     is also kept and becomes the previous best;
   - in all other cases, `git reset --hard HEAD~1`. The lines in `results.tsv` and in the ledger remain.
7. Stop.

**Out of ideas:** combine candidates that nearly passed, test the extremes of the ranges and go back to
under-tested factors.

**`results.tsv`** (tab-separated):

```
commit	run_id	seed	arm	score_hot_latency_ms_per_h	hr_p	ratio_vs_reference	ttd_min	status	flags	hypothesis
```

`ttd_min` = time to detection, in minutes (`ratio=` and `ttd_min=` are printed by the judge). `status`: `keep`, `discard`, `invalid` or `baseline` (for the reference).

**`ledger.jsonl`**, one line before and one after each round:

```
{"run_id": null, "commit": "...", "seed": 101, "status": "started", "candidates": [{"name": "...", "hypothesis": "...", "factor_tested": "..."}], "ts": "..."}
{"run_id": "...", "seed": 101, "status": "completed|failed|reference_without_symptom", "interpretation": "...", "factor_tested": "...", "shortcut": false, "ts": "..."}
```

**When the command asks for the summary of the `detect` campaign**, write `summary_detect.md` with:

- a table of all search rounds, with the seed;
- a table of the reference per round: `seed`, slope and change of `retrieve_hot` latency, `hr_p`, eligible
  (yes/no), and in how many rounds it was eligible;
- the workloads verified by the system and the result of each verification;
- the best arm (`run_id` and `name`) and why; if no candidate was kept, the best arm is the reference;
- evidence that the `retrieve_hot` response size rose and that the internal control stayed flat;
- all shortcuts observed;
- limitations: 60 min window, one seed per search round.

## 6. `causes` campaign

**Reference:** `frozen/workload_frozen.py`, the confirmed workload of this campaign (`FROZEN_ARM` names the
candidate in `CANDIDATES`). One arm of each round runs this workload (`frozen_reference`); the others (up to 3)
are variations of it. If the file does not exist, record this in the ledger and stop.

**Goal:** find which workload factors change the trend, comparing each factor in isolation.

**One-factor rule:** each candidate differs from the frozen workload in **exactly one** `arg`. The judge rejects the
whole round (D3) if a candidate changes zero or more than one.

**Measurements per variation** (`effect` in `round.json`; ratio = variation ÷ reference in the same round; they
exist only if the reference is eligible and the variation is valid):

| Field | What it compares |
|---|---|
| `lat_ratio` | slope of the `retrieve_hot` p50 latency |
| `bytes_ratio` | slope of the `retrieve_hot` response size |
| `cost_ratio` | cost per MB: slope of `retrieve_hot` latency against response size (ms per MB) |
| `assoc_p95_ratio` | p95 level of `associate_card` at the end of the run (median of the per-window p95 over the last hour) |
| `ref4_p95_ratio` | p95 level of `retrieve_ref4` at the end of the run (median of the per-window p95 over the last hour) |

**One round (campaign N, round r of 6):**

1. Read `frozen/workload_frozen.py`, `results_causes.tsv` and `ledger_causes.jsonl`.
2. Choose 1 to 3 variations, each with the factor, the new value, the hypothesis and the expected direction of the
   effect:
   - **replication:** each variation of this campaign must run in **two rounds**, with different seeds. Give
     priority to variations of this campaign that have only one valid replica and plan the remaining rounds;
   - **new hypotheses first:** a variation with the same `factor` and the same `value_var` as an earlier campaign is
     not run again. Test first the ones not yet tested and cite the repeated ones in the summary, with the earlier
     result.
3. Write the variations in `workload.py` and `git commit -am "<variations>"`.
4. Run `causes` (section 4). When it prints the `run_id`, record `started` in the ledger. Run `wait` until the
   summary appears.
5. Record one line per arm in `results_causes.tsv` and `completed` in the ledger. There is no keep or discard: the
   commit stays.
6. If the round's reference is not eligible (`reference_ok: False`), record `reference_without_symptom`; the
   variations of that round do not count as replicas.
7. Stop.

**`results_causes.tsv`** (tab-separated):

```
campaign	commit	run_id	seed	arm	factor	value_ref	value_var	valid	eligible	lat_ratio	bytes_ratio	cost_ratio	assoc_p95_ratio	ref4_p95_ratio	flags	hypothesis
```

**`ledger_causes.jsonl`**: same format as the `detect` ledger, with `"campaign"` and
`"variations": [{"name", "factor", "value", "hypothesis", "expected_direction"}]` in `started`.

**Declaring an effect:** for each ratio, the variation has an effect if, in both valid replicas, the ratio stays on
the same side of 1 and at least 20% away from it (≥ 1.2 in both or ≤ 0.8 in both). Otherwise, "no clear effect".

**When the command asks for the summary of the `causes` campaign N**, write `summary_causes_<N>.md` with:

- the frozen workload (`args`);
- a table factor → value → ratios in both replicas → effect (yes/no, direction) for each measurement;
- which factors change the latency slope, which change the response size and which change the cost per MB;
- hypotheses already tested in earlier campaigns and not repeated, with the earlier result;
- variations without two valid replicas, and why;
- limitations: one factor at a time (no interactions), 60 min window, two replicas per variation.

## 7. Records

- Do not claim an effect larger than the difference between replicas.
- `results/judge_log.jsonl` is the source of truth; your records must match it.
