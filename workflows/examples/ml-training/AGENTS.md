# ml-training (RTX 4060)
GPT language model training. Optimize train.py for lowest val_bpb in 5 minutes.

## Files
- `train.py` | model + training loop | agent-edited
- `prepare.py` | data prep + tokenizer | read-only
- `workflow.yaml` | manifest | read-only
- `program.md` | strategy | read-only

## Constraints
- 8GB VRAM -- OOM risk; estimate memory before scaling
- No Flash Attention 3 -- use PyTorch SDPA fallback
- No torch.compile -- no Triton on Windows
- MAX_SEQ_LEN = 512 -- fixed in prepare.py
- 5-minute time budget (TIME_BUDGET = 300) -- fixed in prepare.py

## Run
`uv sync && uv run prepare.py --num-shards 2 && uv run train.py`
