# Crash Recovery Evidence (VALD-03)

## Test Performed
- Inserted `_oom_trigger = torch.zeros(24_000, 1024, 1024, device="cuda", dtype=torch.float32)` into train.py main()
- Committed as `f2de967` ("test: intentional OOM trigger for crash recovery validation")
- Ran `python train.py > run.log 2>&1` -- exited with code 1 (clean exit, not hard crash)

## Verification
- `metrics.json` written with `{"status": "oom", "peak_vram_mb": 0.0, "error": "CUDA out of memory"}`
- `run.log` contains `status: OOM` (greppable output block)
- Crash logged to `results.tsv` with: `f2de967  0.000000  0.000000  0.000000  0.0  crash  intentional OOM trigger...`
- `git reset --hard HEAD~1` reverted train.py to pre-OOM state
- train.py does NOT contain `_oom_trigger` after reset
- Git working tree is clean
- Current HEAD is `7c880a9` (pre-OOM commit)

## Results
- OOM was caught by `except torch.cuda.OutOfMemoryError` handler in train.py
- Handler wrote metrics.json and exited cleanly (not a segfault/hard crash)
- Agent crash protocol (log + git reset) restored system to clean state
- System is ready to continue experiments
