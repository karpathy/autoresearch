# LoopGraph & Baseline Agent Design Spec

We are implementing a comparative evaluation framework for automated ML research agents on Karpathy's `autoresearch` benchmark. The system must run on a macOS arm64 host (using MPS) or a Linux GPU host (using CUDA).

## Files to Implement

1. `method_pack.md`: A structured reference document detailing valid optimization techniques, architectures, hyperparameters, code templates, and constraints.
2. `experiment_brief.md`: A document setting the goals, baseline values, compute constraints, and target hypotheses for the agent's run.
3. `run_agent.py`: A Python script implementing the autonomous propose-train-evaluate loop for both the Baseline and LoopGraph agents.
4. `test_runner.py`: A testing suite to verify the runner's parsing, LLM call, code extraction, and git state management logic.

---

## 1. `method_pack.md` Structure
It must cover:
- **MLP Gating / Activation**: RELU^2, SwiGLU, GeGLU, showing how to change the `MLP` class in `train.py`.
- **Attention Variants**: Multi-Head, Grouped-Query (GQA), Multi-Query (MQA).
- **Optimizer & Hyperparameters**: Learning rates for `matrix_lr` (Muon), `embedding_lr` (Adam), `unembedding_lr` (Adam), and schedules.
- **Normalization & Regularization**: RMSNorm, LayerNorm, Weight Decay, Dropout.

Provide clear code templates showing exactly how to swap structures in `train.py`.

---

## 2. `experiment_brief.md` Structure
- **Objective**: Minimize `val_bpb` within a 60-second budget (for local fast testing) or 300-second budget.
- **Constraints**: Single device, no new dependencies, do not edit `prepare.py`.
- **Target Hypotheses**: e.g., "SwiGLU yields lower loss than ReLU^2", "Tuning Muon LR improves convergence speed".

---

## 3. `run_agent.py` Implementation Details
The script should accept the following command-line flags:
- `--mode`: `baseline` or `loopgraph` (required).
- `--model`: LLM model name (default: `gpt-4o-mini`).
- `--max-experiments`: Number of loop iterations (default: 5).
- `--time-budget`: Override training time budget in seconds (default: 300).
- `--api-provider`: `openai` or `openrouter` (default: `openai`).

### Main Loop Logic:
1. **Setup**:
   - Determine target branch name (e.g. `autoresearch/exp_<mode>_<timestamp>`).
   - Create and checkout the branch: `git checkout -b <branch_name>`.
   - Initialize `results.tsv` (headers: `commit\tval_bpb\tmemory_gb\tstatus\tdescription`).
2. **First Run (Baseline)**:
   - Run `uv run train.py` with `--time-budget` injected (we can modify `train.py` temporarily or set an environment variable, but wait—`train.py` reads `TIME_BUDGET` from `prepare.py`. Wait, since `prepare.py` has `TIME_BUDGET = 300`, how can the runner override it?
     Ah! In `train.py`, `TIME_BUDGET` is imported from `prepare.py`. If we want to override it, we can check for an environment variable `TIME_BUDGET` in `train.py`, or modify `train.py` to support `os.environ.get("TIME_BUDGET")`!
     Let's check `train.py`:
     Line 26: `from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb, DEVICE`
     We should modify `train.py` to load `TIME_BUDGET` from `os.environ` if present, falling back to the imported one:
     ```python
     TIME_BUDGET = int(os.environ.get("TIME_BUDGET", TIME_BUDGET))
     ```
     This is extremely simple and clean!)
   - Log the baseline run as commit `baseline`, status `keep`, along with the `val_bpb` and `memory_gb`.
3. **Iterative Propose-Train-Evaluate Loop**:
   - Read the current `train.py` content.
   - Read `results.tsv` to construct the history context.
   - Construct Prompt:
     - **Common**: System instructions for code modification, current `train.py` code, experiment history from `results.tsv`.
     - **LoopGraph Only**: Inject `method_pack.md` and `experiment_brief.md` into the system prompt.
   - Call LLM via HTTP API:
     - OpenAI endpoint: `https://api.openai.com/v1/chat/completions` using `OPENAI_API_KEY`.
     - OpenRouter endpoint: `https://openrouter.ai/api/v1/chat/completions` using `OPENROUTER_API_KEY`.
   - Extract code: Look for ` ```python ` block in the LLM response.
   - Apply change: Overwrite `train.py` with the extracted code.
   - Commit change: `git commit -am "experiment <N>"` (we commit before running so we have a clean commit hash to log and can easily revert).
   - Execute: `uv run python train.py` (redirect output to a unique log file, e.g. `run_<N>.log`, set env var `TIME_BUDGET` to the configured budget).
   - Parse:
     - If the process exits successfully, grep for `val_bpb:` and `peak_vram_mb:` in the log.
     - If it crashes, extract the traceback from the log and log status `crash`.
   - Decide Keep/Discard:
     - If the run succeeded and `val_bpb` is strictly lower than the best `val_bpb` so far: keep the commit, update the best loss.
     - Else (crashed or worse/equal loss): `git reset --hard HEAD~1` (revert the commit and code changes), delete or ignore the commit, status `discard` or `crash`.
   - Log to `results.tsv`.

---

## 4. `test_runner.py` Implementation Details
Tests should verify:
- LLM response parsing: extracting python code blocks from markdown text.
- Log file parsing: extracting `val_bpb` and `peak_vram_mb` from successful runs, or identifying crashes.
- Git operation wrapper calls (optional/mocked to avoid polluting git state during unit tests).
- Prompt assembly: ensuring `method_pack` and `experiment_brief` are correctly included in `loopgraph` mode and excluded in `baseline` mode.
