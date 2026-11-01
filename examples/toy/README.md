# Toy example

A one-file offline project for the SQLite lab flow. The measured project is `project/`,
whose `rule.txt` holds a factor that `evaluator.py` applies to `cases.json`.
The primary metric is `mean_error` (min); `coverage` is a guard.

- `evaluator.py` is offline and deterministic. It stays outside the measured project.
- `spec.json` and `brief.md` are the inputs to `prepare`. `run_demo.py` writes
  `known_case.json` and fills in its hashes in the temp copy, because the hashes depend
  on the checked-out bytes.
- The lab state, the host profile and the temp copy are created at run time and removed afterwards.

Run the whole flow (prepare, guide, offline profile, baseline, one-line edit, trial) from any directory:

    python examples/toy/run_demo.py
