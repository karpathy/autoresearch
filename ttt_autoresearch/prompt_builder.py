from __future__ import annotations

from typing import Any


CONSTRUCTION_SECTION = (
    "You must start your search from the current training script shown above.\n"
    "This is the current starting point selected by the search procedure.\n"
    "Preserve a working script, but do not limit yourself to tiny hyperparameter tweaks.\n"
    "Pursue bold, high-upside changes when they are technically coherent and likely to materially improve val_bpb.\n"
    "You are encouraged to explore meaningfully different directions if the current approach appears saturated."
)

CODE_SECTION = (
    "Reason about how you could further improve this training script under the fixed 5-minute training budget.\n"
    "Hyperparameter tuning is allowed, but do not stop there: pursue stronger algorithmic, architectural, data-flow, attention, optimization, or systems ideas when they could deliver a step-change improvement.\n"
    "Prefer edits that are technically coherent and high-upside, even if they are more ambitious than simple hill-climbing.\n"
    "Try different algorithmic ideas, architecture changes, optimizer and schedule changes, batching changes, or other training heuristics.\n"
    "Minor increases in VRAM are acceptable if they lead to meaningful gains.\n"
    "Do not refactor unrelated code, but do make all integration edits required for the new idea to work cleanly.\n"
    "Unless you make a meaningful improvement in `val_bpb`, you will not be rewarded."
)


def build_rollout_prompt(
    *,
    state_ctx: str,
    construction_section: str,
    code_section: str,
) -> str:
    return f"""You are an expert machine learning researcher and systems engineer optimizing a single-file language-model training script.

Your task is to edit `train.py` so that, when run under the fixed AutoResearch harness, it achieves a lower validation bits-per-byte (`val_bpb`).

You are producing exactly one candidate patch to the current `train.py`.

## Objective

Modify `train.py` to improve final `val_bpb` under the fixed 5-minute training budget.

Lower `val_bpb` is better.

Your goal is not to make the code cleaner, more general, or more reusable. Your goal is to improve the measured validation result under the fixed harness while preserving a working script.

## Fixed Project Facts

This is a simplified single-GPU AutoResearch / nanochat-style setup in which:
- `train.py` is the only editable file
- runs are compared under a fixed wall-clock budget
- the optimization target is validation bits-per-byte (`val_bpb`)
- lower `val_bpb` is the only thing that matters

## Non-Negotiable Constraints

Treat all of the following as fixed:
- only `train.py` may be edited
- the evaluation harness is fixed
- the dataset is fixed
- the tokenizer setup is fixed
- do not add dependencies
- do not rely on new packages or changes to environment setup
- use only Python standard library modules and packages already clearly part of this codebase

## Fixed Training / Evaluation Facts

- the training budget is exactly 300 seconds of wall-clock training time, excluding startup / compilation
- maximum sequence length is fixed at `2048`
- evaluation token budget is fixed at `40 * 524288`
- validation is pinned to shard `06542`
- tokenizer vocabulary size is fixed at `8192`
- the BOS token and BOS-aligned packing behavior are fixed
- `val_bpb` is vocabulary-size-independent
- results must remain comparable under the fixed harness

## Fixed Data Pipeline Contract

Your edited `train.py` must remain compatible with the existing pipeline.

In particular:
- the dataloader uses BOS-aligned packing
- every row starts with BOS
- documents are packed with best-fit packing to minimize cropping
- when no document fits, the shortest buffered document is cropped to fill the row exactly
- the dataloader is designed for full utilization without padding
- `make_dataloader(tokenizer, B, T, split)` must continue to work with your script

Do not introduce assumptions that would break BOS alignment, packed rows, or the existing input / target layout.

## Fixed Model / Evaluation Contract

Your edited script must preserve all of the following:
- the model must continue to support `forward(x, y, reduction='none')`
- when `targets` are provided, the model must still return a loss compatible with the current evaluator
- the final summary prints must remain present
- especially preserve the line beginning exactly with `val_bpb:`
- keep `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0`

## What You May Change

Everything inside `train.py` is fair game, including:
- architecture
- optimizer
- schedules
- hyperparameters
- training loop
- batch geometry
- model size
- attention pattern
- initialization
- regularization
- precision and memory-management choices inside `train.py`

## Technical Guidance

Treat `train.py` as the full research surface: it contains the model, optimizer, batching choices, and training loop.

The fixed 5-minute budget exists so experiments remain comparable even when model size, batch size, architecture, or optimization strategy change. The target metric is `val_bpb`; lower is better.

The most important practical knobs in this setup are:
- `DEVICE_BATCH_SIZE`
- `TOTAL_BATCH_SIZE`
- `DEPTH`
- `WINDOW_PATTERN`
- hidden width / head geometry
- optimizer and schedule choices
- activation and memory behavior inside `train.py`

Important practical facts:
- the number of tokens per forward/backward pass is `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`
- `DEPTH` is a primary model-complexity knob, and many other dimensions scale with it
- `WINDOW_PATTERN = "SSSL"` may be less efficient than `"L"` on some systems
- sequence length, dataset, vocabulary size, tokenizer, and evaluation protocol are fixed outside `train.py`, so only optimize through knobs that still live in `train.py`

## Budget-Aware Reasoning Requirement

Reason about the script as a 5-minute optimization problem, not as an unlimited-training problem.

Before choosing a change, consider:
- what is most likely limiting performance right now: optimization, throughput, memory, model size, or architecture
- whether the current script is undertrained, overbuilt, memory-inefficient, unstable, or throughput-limited
- whether your change increases or decreases:
  - activation memory
  - optimizer-state memory
  - parameter memory
  - step time
  - tokens/sec
  - total useful optimization completed within 5 minutes

Important practical principles:
- sequence length is fixed at `2048`
- larger `DEVICE_BATCH_SIZE`, deeper models, wider models, extra residual branches, extra embedding streams, and more expensive attention patterns can all hurt memory or throughput
- if you increase one major cost driver, compensate elsewhere
- a somewhat smaller but faster or stabler model can beat a larger model that gets fewer useful steps
- avoid changes that are likely to OOM or materially reduce useful training under the fixed budget
- bold changes are allowed, but they must be coherent, dependency-safe, memory-aware, and budget-aware

## What Good Edits Usually Look Like

Good candidates usually do one coherent thing well:
- improve quality-per-token under a short training horizon
- improve throughput under the fixed wall-clock budget
- improve early optimization stability
- improve memory-efficiency enough to unlock a better tradeoff
- choose a more coherent scaling point across depth, width, heads, batch, and optimizer
- simplify components if the current design is overbuilt for the budget
- make a targeted architectural change with a clear expected payoff

Prefer one coherent direction over many unrelated tweaks.

## What To Avoid

Avoid:
- elegant changes with weak expected effect on `val_bpb`
- changes that mainly help only at much longer training horizons
- fragile edits likely to break compilation or runtime
- edits that silently violate the data pipeline or output contract
- extra complexity without a clear budget-aware reason
- memory increases without a clear speed/quality payoff
- changes to fixed knobs controlled outside `train.py`

## Output Requirements

Return only exact search-and-replace patch blocks for `train.py`.

Do not return the full file.
Do not return standalone code fragments.
Do not return JSON.
Do not use markdown code fences.
Do not include commentary, rationale, summary, or any prose before or after the patch.
Do not abbreviate with `...` or placeholders.
Each replacement must be fully expanded source code.

Each patch block must use exactly this format:

<search>
[exact existing text from the current train.py]
</search>
<replace>
[new replacement text]
</replace>

Patch rules:
- each `<search>` block must copy exact contiguous text from the current `train.py`
- each `<search>` block must match the current file exactly once
- include enough surrounding context to make each patch unique and apply cleanly
- use as few patch blocks as possible, but as many as necessary for correctness
- ensure the final result is a working script

Optimize for the lowest `val_bpb` under the fixed evaluation budget, subject to actually running successfully.

## Current Starting Point

Below is the current `train.py` state, current score context, target score, and previous run context:

{state_ctx}
"""


def build_prompt_for_state(state: Any, target: float) -> str:
    state_ctx = state.to_prompt(target, metric_name="val_bpb", maximize=False, language="python")
    return build_rollout_prompt(
        state_ctx=state_ctx,
        construction_section=CONSTRUCTION_SECTION,
        code_section=CODE_SECTION,
    )
