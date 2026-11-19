# bax-fastapi-credit-card-F1 — targeted workload v1

Script: `workloads/bax-fastapi-credit-card-F1_targeted_v1.py` (control R0: `workloads/bax-fastapi-credit-card-F1_control.py`).
Finding: every POST /associate_card with a new (credit_card, phone) pair inserts a row (plus 3
index entries) into `associations` that is never deleted; POST /retrieve_cards runs
`WHERE phone IN (...) GROUP BY credit_card HAVING ... ORDER BY` with no LIMIT and `fetchall()`s every
matching row, then builds a pydantic model + JSON with every card. Its cost grows with the number of
cards on the requested phones.

## Trigger mapping

| stream (endpoint / phase) | request | card element |
|---|---|---|
| `associate_card` / `grow` | POST /associate_card, a never-seen 16-digit card on hot phone `+551190000001`, at `--new-pair-rate` | `mechanism.trigger` (distinct pairs, concentrated on a hot phone) → INSERT OR IGNORE with rowcount 1; resource = rows per phone, db size |
| `associate_card` / `refresh` | re-sends an already-associated (card, hot phone) pair (seeded RNG) | INSERT OR IGNORE path with rowcount 0; realism; adds no rows |
| `associate_card` / `prime`, `ref` | registration + cyclic re-association of the reference phones' fixed cards | keeps reference phones at a constant row count |
| `retrieve_hot` / `probe` | POST /retrieve_cards `{"phone_numbers":["+551190000001"]}` every `--probe-interval` s | symptom: retrieve query + fetchall + response model cost, latency and response size growing with cards on the phone |
| `retrieve_ref4`, `retrieve_ref200` / `refread` | same read for phones with a constant 4 and 200 cards, same interval | internal control: same read path, constant size → should stay flat; separates per-row growth from global drift (B-tree depth, file size, host noise) |

Cards are a deterministic, seed-dependent affine permutation of a counter (prefix `4` for hot, `5`
for reference cards), so they are distinct, digits-only (valid per the app's validator) and not
inserted in sorted order (realistic index insert pattern). The first probe is sent one interval
after start so the hot phone already has cards (no 404).

## Matched control R0b (`--key-space N`)

`--key-space N` (default 0 = off) makes the `grow` stream cycle over N fixed cards per hot phone at
the same rate and with the same request shape (phase `cycle`); `refresh` picks among those N.
Rates, phones, probes and reference phones are identical to R1; only key diversity changes.
R0b uses `--key-space 4`: table constant at 4 (hot) + 4 + 200 = 208 rows.

## Pacing (why the server is not saturated early)

There is no app-level lock, but every write opens a connection, commits in WAL mode and closes
(~23 ms per associate in the smoke runs, writes serialized by SQLite's writer lock → ceiling
~40/s). Defaults: 10 (grow) + 2 (refresh) + 2 (ref) = 14 associates/s ≈ 30–35 % writer
utilization, plus 3 retrieves every 2 s. Shared slot pacers drop missed slots instead of
bursting, with a fixed number of workers (2 for grow, 1 per other stream), so a slowing server
shows up as lower achieved rate, never as a client backlog. Per-request timeout 30 s.

## Defaults and 8 h sizing

| flag | default |
|---|---|
| `--new-pair-rate` | 10 /s (distinct pairs, all on the hot phone) |
| `--hot-phones` / `--grow-workers` | 1 / 2 |
| `--refresh-rate` | 2 /s |
| `--ref-sizes` / `--ref-rate` | `4,200` / 2 /s |
| `--probe-interval` | 2 s (per probed phone) |
| `--key-space` | 0 (R0b: 4) |
| `--timeout` | 30 s |

Calibration (offline micro-benchmark on this host, same app code, pre-filled DB, warm cache):
/retrieve_cards for one phone with 30 k / 100 k / 170 k cards took ~40 / ~135 / ~237 ms and returned
0.57 / 1.9 / 3.2 MB, i.e. ~1.4 µs and ~19 bytes per card; the DB file was ~131 bytes per row
(row + UNIQUE + 2 secondary indexes). Smoke run: ~19 bytes per card in the response, as expected.

| elapsed | cards on hot phone (≈ 10/s × t) | total rows | /retrieve_cards (hot) latency | response size | db.sqlite3 |
|---|---|---|---|---|---|
| 60 s (warm-up) | ~550 | ~760 | ~2–5 ms | ~10 kB | ~0.15 MB |
| 1 h | ~36 000 | ~36 200 | ~50 ms | ~0.7 MB | ~5 MB |
| 8 h | ~288 000 | ~288 200 | ~400–450 ms (more if the file falls out of page cache) | ~5.5 MB | ~38 MB |

Growth is linear for the whole 8 h (constant insert rate; inserts degrade only logarithmically).
At 8 h the hot probe costs ~0.4 s of server CPU every 2 s (~20 % of one core) plus the writes, so
the symptom is ~100× the 1-minute baseline without saturating the server (compare Uptime pilot:
~480 ms per read at ~36 k rows under a global lock; here 10 /s gives a comparable read cost only
near the end of the 8 h run, with no lock serializing other requests). If R1 degrades too little,
the first knob is `--new-pair-rate 20` (~576 k cards, ~0.8–0.9 s per probe at 8 h, still < ~50 %
writer utilization). client.csv: ~15.5 requests/s → ~450 k rows (~25 MB) over 8 h; client reads
a ≤ 6 MB response every 2 s without parsing it — negligible.

Note on the instrumentation gauge `rows.max_per_phone` / `retrieve.rows.max`: during the first
~20 s they are dominated by the 200-card reference phone; in R1 they track the hot phone after it
passes 200 cards, in R0b they stay at 200.

## Smoke runs (60 s, instrumented, deleted afterwards)

Port 5000 was held by an orphaned idle `python app.py` (cwd deleted, parent init) for >30 min, so
per instructions it was not killed; the smokes used a temporary copy of `instrumented/` (identical
code, `app.yaml` starting the same app via the uvicorn CLI on port 5099), removed afterwards.

- control: associate_card 2310 (201), retrieve_cards 2309 (200), 0 errors; associate median 23 ms,
  retrieve 2.2 ms; table = 1 row throughout.
- targeted: associate_card 977 (grow 551, prime 204, ref 111, refresh 111), retrieve_hot 27,
  retrieve_ref4 27, retrieve_ref200 26; 0 errors. Rows 0 → 755; rows.max_per_phone 200 → 551;
  retrieve_hot response 398 B → 10 278 B; ref responses constant (94 B, 3818 B).
- matched (`--key-space 4`): same request counts (974 / 27 / 27 / 26), 0 errors; rows constant at
  208, rows.max_per_phone constant at 200, db.bytes constant, retrieve_hot response constant 94 B.

## Commands (8 h)

```bash
# R0 control (reference/credit_card_test.jmx: 1 thread, no timers)
.venv/bin/python harness/run_diagnostic.py --app apps/credit-card-fastapi/instrumented --instrument \
    --workload workloads/bax-fastapi-credit-card-F1_control.py --duration 28800 --label R0_cc_F1_control \
    --app-cpus 0 --load-cpus 1-3

# R0b matched control (same load, 4 fixed cards on the hot phone)
.venv/bin/python harness/run_diagnostic.py --app apps/credit-card-fastapi/instrumented --instrument \
    --workload workloads/bax-fastapi-credit-card-F1_targeted_v1.py --duration 28800 --label R0b_cc_F1_matched \
    --app-cpus 0 --load-cpus 1-3 -- --key-space 4

# R1 targeted
.venv/bin/python harness/run_diagnostic.py --app apps/credit-card-fastapi/instrumented --instrument \
    --workload workloads/bax-fastapi-credit-card-F1_targeted_v1.py --duration 28800 --label R1_cc_F1_targeted_v1 \
    --app-cpus 0 --load-cpus 1-3
```

R2 = R1 with `--app apps/credit-card-fastapi/repaired`. The control is a closed loop without think
time (as in the .jmx), driving ~38 iterations/s (2 requests each) over a single fixed pair — the
original condition. Campaign definition: `apps/credit-card-fastapi/campaign.yaml`.
