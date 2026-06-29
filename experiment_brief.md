# Experiment Brief: Autoresearch Optimization Benchmarking

This brief sets the objectives, constraints, target hypotheses, and validation metrics for the automated machine learning research agent running on Karpathy's autoresearch benchmark environment.

---

## 1. Objective

The primary objective is to **minimize validation bits per byte (`val_bpb`)** on the pretraining task within a strict training time budget.
* **Default Time Budget**: 300 seconds (5 minutes) for standard validation.
* **Fast Testing Time Budget**: 60 seconds (or 10 seconds for mechanical sanity checks).
* The model must be trained from scratch within this budget, and evaluated on the pinned validation shard.

---

## 2. Hard Constraints

All experiments conducted by the agent must strictly adhere to the following constraints:
1. **Single-Device Execution**: Canonical benchmark runs target a single NVIDIA GPU. Local CPU/MPS fast-mode runs are useful for smoke tests only and are not comparable to published BPB numbers.
2. **No New Dependencies**: Do not import or add any external libraries to the project. All dependencies must be satisfied by standard PyTorch or existing libraries in `pyproject.toml`.
3. **Do Not Change Evaluation Semantics**: The validation shard, BPB calculation, byte counting, and target/label alignment in `prepare.py` are fixed. Configuration knobs such as `VOCAB_SIZE`, `MAX_SEQ_LEN`, and `EVAL_TOKENS` may be environment-overridden for local smoke tests, but canonical comparisons should use the documented 5-minute setup.
4. **Time Budget Enforcement**: The training loop must stop immediately when the elapsed training time exceeds the `TIME_BUDGET` (excluding compilation/startup time, i.e., after step 10).
5. **No Loss of Invariants**: Do not silently compare fast-mode runs against published baselines. Record any non-default budget, sequence length, vocabulary size, device, or step cap before interpreting BPB.
---

## 2.1 Published Reference Points

Use these as context, not as proof of parity unless the run uses the same canonical evaluation protocol and multi-seed validation:

| Solution | Mean BPB | Notes |
|---|---:|---|
| Recursive best (`optimized_from_karpathy`) | 0.9109 | 10 seeds, single B200, 5-minute budget |
| Recursive from vanilla Transformer | 0.9344 | 10 seeds, single B200, 5-minute budget |
| autoresearch@home community best | 0.9372 | Reward-hack-removed, 10 seeds, single B200 |
| Karpathy seed | 0.9751 | Re-run by Recursive for comparability |
| Vanilla Transformer | 1.0587 | Recursive plain-transformer baseline |

Local single-seed or fast-mode runs around 2.x BPB are useful for agent workflow debugging, but they are not competitive benchmark results.

---

## 3. Target Hypotheses

The agent should propose and evaluate experiments testing the following hypotheses:

### Hypothesis A: N-gram Value Embeddings Improve Local Context
* **Statement**: Recursive's best public solution adds hashed bigram and trigram value-embedding tables mixed into the attention value path through learned gates. Porting a minimal variant may improve short-context modeling at low compute cost.
* **Variables**: Bigram/trigram table size, active layers, hash functions, gate initialization, optimizer group.

### Hypothesis B: Current GQA Needs Tuning, Not Re-Discovery
* **Statement**: The current `train.py` already selects grouped-query attention when the head count permits it. Experiments should tune `n_kv_head` or compare against full MHA deliberately, not propose "add GQA" as a new feature.
* **Variables**: `n_kv_head` in `GPTConfig`, fallback compatibility, step throughput.

### Hypothesis C: Muon Matrix Learning Rate Tuning Accelerates Convergence
* **Statement**: The default Muon learning rate (`MATRIX_LR = 0.04`) is tuned for a specific dimension. Tuning `MATRIX_LR` (e.g. between `0.02` and `0.08`) will yield faster initial convergence and lower final validation loss.
* **Variables**: `MATRIX_LR`.

### Hypothesis D: Sliding Window Attention Patterns Trade Context for Speed
* **Statement**: Modifying the sliding window pattern `WINDOW_PATTERN` to have more short context layers (e.g. `"SSSS"`) will speed up steps and allow more training updates within the time budget, but might hurt long-range dependency modeling. An optimal trade-off (e.g. `"SLSL"` or `"SSSL"`) minimizes `val_bpb`.
* **Variables**: `WINDOW_PATTERN` string.

### Hypothesis E: Current Normalization and Regularization Need Targeted Ablations
* **Statement**: The current `train.py` already uses learnable RMSNorm modules. Future normalization work should ablate specific placement or scaling behavior rather than reintroducing a generic "learnable RMSNorm" idea. Dropout remains risky in a 5-minute budget and should be tested only with a clear speed/generalization rationale.
* **Variables**: RMSNorm placement, q/k normalization order, residual scaling, Dropout rate in blocks.

---

## 4. Validation Metrics and Reporting

Every run must log the following metrics into `results.tsv`:

1. **`val_bpb`**: Validation bits per byte (Primary Metric). Must be strictly lower than the best run so far to be kept.
2. **`training_seconds`**: Real elapsed training time (excluding step 0-10 compilation).
3. **`memory_gb`**: Peak memory allocation in GB (`peak_vram_mb / 1024`).
4. **`status`**: Outcome of the run:
   * `keep`: Successful run with lower validation loss (changes merged into baseline).
   * `discard`: Successful run but with higher/equal validation loss (changes reverted).
   * `crash`: Code failed to compile or run, or loss exploded/NaN (reverted).
5. **`description`**: A concise explanation of the hypothesis and change tested (e.g., "SwiGLU MLP activation").
