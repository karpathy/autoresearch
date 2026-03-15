# Saved Benchmark Findings

这份文件只保留后续最值得复用的结论，适合在重新开始性能优化前快速回顾。

## 1. 吞吐基线

- 当前 throughput-first 基线是 `baseline/default` profile。
- 代表记录：`generated/2026-03-15/baseline_default_20_serial.txt`
- 配置：
  - `n_embd=512`
  - `depth=8`
  - `MAX_SEQ_LEN=2048`
  - `WINDOW_PATTERN=SSSL`
  - `DEVICE_BATCH_SIZE=12`
  - `compile_backend=inductor`
  - `compile_mode=default`
  - `compile_scope=model`
  - `optimizer_compile_backend=inductor`
- 参考结果：
  - `73,537 tok/s`
  - `17.58 train TFLOPS`
  - `40.16% MFU`
  - `4610.3 MB peak VRAM`

## 2. MFU 基线

- 当前 utilization-first 基线是 `mfu50` profile。
- 代表记录：`generated/2026-03-15/640d_b5_llll_4096_bench20.txt`
- 配置：
  - `n_embd=640`
  - `depth=8`
  - `MAX_SEQ_LEN=4096`
  - `WINDOW_PATTERN=LLLL`
  - `DEVICE_BATCH_SIZE=5`
  - `compile_backend=inductor`
  - `compile_mode=default`
  - `compile_scope=model`
  - `optimizer_compile_backend=inductor`
- 参考结果：
  - `44,001 tok/s`
  - `22.84 train TFLOPS`
  - `52.17% MFU`
  - `4976.6 MB peak VRAM`

## 3. 已经确定的方向判断

- `max-autotune` 在当前 Windows + `triton-windows` 路径下不适合作默认值。
- `max-autotune-no-cudagraphs` 比 `max-autotune` 更实用，但整体仍不如 `default`。
- 提高 MFU 的有效杠杆不是单纯增大 batch，而是：
  - 保持 `640d` 量级
  - 使用 `LLLL`
  - 拉长 `MAX_SEQ_LEN`
  - 同时把 batch 控制在不会掉进显存退化区的范围内
- 旧版 `896d`, `10L`, `batch=8` 的 `mfu50` 路线已经证伪，不建议继续投入。

## 4. 读旧日志时要注意

- `3` 步和 `5` 步日志主要用于筛方向，不是最终性能排名。
- `baseline_default_20.txt` 是并发占卡导致的无效样本，不要引用。
- `modular_baseline_3_after_warmup.txt` 和 `modular_mfu50_3.txt` 的价值是“重构后性能未退化”，不是替代 20 步或 100 步主 benchmark。

## 5. 详细记录入口

- 详细表格版总结见 `SUMMARY_2026-03-15.md`
- 历史归档日志见 `archive/`
- 本地生成日志见 `generated/`
