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
1. **Single-Device Execution**: The system must run on a single device (typically MPS on macOS or CUDA on Linux).
2. **No New Dependencies**: Do not import or add any external libraries to the project. All dependencies must be satisfied by standard PyTorch or existing libraries in `pyproject.toml`.
3. **Do Not Edit `prepare.py`**: The data prep, tokenization, and validation evaluation logic in `prepare.py` is fixed and must not be altered.
4. **Time Budget Enforcement**: The training loop must stop immediately when the elapsed training time exceeds the `TIME_BUDGET` (excluding compilation/startup time, i.e., after step 10).
5. **No Loss of Invariants**: Do not break critical mathematical structures like sequence length (fixed at 2048) or model vocabulary size (8192).

---

## 3. Target Hypotheses

The agent should propose and evaluate experiments testing the following hypotheses:

### Hypothesis A: SwiGLU Gating is Superior to Squared ReLU
* **Statement**: Replacing the baseline `ReLU^2` activation in the MLP module with `SwiGLU` (while parameter-matching hidden dimension to $d_{ff} \approx \frac{8}{3} d_{model}$) will improve representation capacity and yield a lower `val_bpb` within the same training time budget.
* **Variables**: Activation function (`F.silu` vs `F.relu().square()`), MLP hidden dimension scaling.

### Hypothesis B: Grouped-Query Attention (GQA) Speeds up Training and Improves BPB
* **Statement**: Tuning the number of KV heads (`n_kv_head`) to be less than query heads (`n_head`) reduces parameter counts and sequence processing overhead, resulting in more gradient steps per second. This higher step count under the time budget will lead to a better `val_bpb`.
* **Variables**: `n_kv_head` in `GPTConfig`.

### Hypothesis C: Muon Matrix Learning Rate Tuning Accelerates Convergence
* **Statement**: The default Muon learning rate (`MATRIX_LR = 0.04`) is tuned for a specific dimension. Tuning `MATRIX_LR` (e.g. between `0.02` and `0.08`) will yield faster initial convergence and lower final validation loss.
* **Variables**: `MATRIX_LR`.

### Hypothesis D: Sliding Window Attention Patterns Trade Context for Speed
* **Statement**: Modifying the sliding window pattern `WINDOW_PATTERN` to have more short context layers (e.g. `"SSSS"`) will speed up steps and allow more training updates within the time budget, but might hurt long-range dependency modeling. An optimal trade-off (e.g. `"SLSL"` or `"SSSL"`) minimizes `val_bpb`.
* **Variables**: `WINDOW_PATTERN` string.

### Hypothesis E: Normalization Variants and Regularization Prevent Overfitting
* **Statement**: Swapping the default functional `RMSNorm` with functional `LayerNorm` or adding dropout (`nn.Dropout`) to the attention and MLP residuals will improve generalizability and lower `val_bpb`.
* **Variables**: `norm(x)` function, Dropout rate in blocks.

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
