from __future__ import annotations

from typing import Any


CONSTRUCTION_SECTION = (
    "You may want to start your search from the current training script shown above.\n"
    "This is the current starting point selected by the search procedure.\n"
    "Make one focused experimental change at a time and preserve a working script.\n"
    "You are encouraged to explore meaningfully different directions if the current approach appears saturated."
)

CODE_SECTION = (
    "Reason about how you could further improve this training script under the fixed 5-minute training budget.\n"
    "Prefer small, local hill-climbing edits over broad rewrites.\n"
    "Try different algorithmic ideas, architecture changes, optimizer and schedule changes, batching changes, or other training heuristics.\n"
    "Moderate increases in VRAM are acceptable if they lead to meaningful gains.\n"
    "Do not refactor unrelated code.\n"
    "Unless you make a meaningful improvement in `val_bpb`, you will not be rewarded."
)


def build_rollout_prompt(
    *,
    state_ctx: str,
    construction_section: str,
    code_section: str,
) -> str:
    return f"""You are an expert machine learning researcher and systems engineer optimizing a language-model training script.

Your task is to improve `train.py` so that it achieves a lower `val_bpb` under the fixed AutoResearch evaluation budget.

## Problem

Improve the `train.py` program so that the resulting training run achieves a lower validation bits-per-byte (`val_bpb`).

Everything in `train.py` is fair game:
- architecture
- optimizer
- hyperparameters
- training loop
- batch size
- model size

**Lower `val_bpb` values are better** - they indicate a stronger model under the fixed evaluation budget.

## Budget & Resources
- **Time budget**: 5 minutes of wall-clock training time
- **Evaluation harness**: fixed AutoResearch runner
- **VRAM**: moderate increases are acceptable for meaningful gains, but avoid wasteful blowups

## AutoResearch Invariants
- `prepare.py` and the evaluation protocol are fixed and cannot be changed
- Maximum sequence length is `2048`
- Validation uses the pinned shard `06542`
- The tokenizer / vocabulary setup is fixed at vocab size `8192`
- The training script must remain compatible with the existing BOS-aligned bin-packing data pipeline
- The model implementation must continue to support `forward(x, y, reduction='none')`
- Keep `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0`
- Preserve the final summary prints, especially the line that starts with `val_bpb:`

## Rules
- You may only edit `train.py`
- Do not modify `prepare.py`, dependencies, or the evaluation harness
- Return only one or more exact SEARCH/REPLACE edit blocks for `train.py`
- Prefer 1-3 small patch blocks
- Each SEARCH block must copy exact contiguous text from the current `train.py`
- If you change constants or a small code region, include enough surrounding context in SEARCH to make the patch unique
- Treat each SEARCH block like an exact `old_string` tool argument: it must match exactly once
- Do not return the full file
- Do not return standalone code fragments
- Do not wrap the answer in JSON
- Do not wrap the answer in markdown code fences
- Do not include any commentary, rationale, summary, or prose before or after the patch
- Do not abbreviate with `...` or placeholders; each replacement must be fully expanded source code
- Each patch block must use exactly this format:
<<<<<<< SEARCH
[exact existing text from the current train.py]
=======
[new replacement text]
>>>>>>> REPLACE
- The SEARCH text must match the current starting `train.py` exactly
- Propose exactly one candidate for this rollout
- Optimize for the lowest `val_bpb` under the fixed time budget
- Prefer simpler changes when improvement is similar

## Example Response
<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 524288
=======
TOTAL_BATCH_SIZE = 393216
>>>>>>> REPLACE

{state_ctx}
{construction_section}
{code_section}
"""


def build_prompt_for_state(state: Any, target: float) -> str:
    state_ctx = state.to_prompt(target, metric_name="val_bpb", maximize=False, language="python")
    return build_rollout_prompt(
        state_ctx=state_ctx,
        construction_section=CONSTRUCTION_SECTION,
        code_section=CODE_SECTION,
    )
