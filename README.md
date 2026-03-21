# autoresearch (MPS / Apple Silicon fork)

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for **Apple Silicon** (M3 Ultra, MPS backend). Trains a small GPT on Bach 2-part & 3-part inventions in a custom music notation format, then generates new compositions.

## What's different from upstream

- **MPS backend** instead of CUDA — no NVIDIA GPU required
- **PyTorch native SDPA** instead of Flash Attention 3
- **Bach inventions dataset** (`experiments/bach/input_inventions_v2.txt`) — 75 pieces in a custom symbolic music notation
- **Monkey-patched `prepare.py`** — the upstream file is kept read-only; MPS compatibility is injected at import time from `train.py`
- **Post-training sample generation** — set `SAMPLE_OUTPUT_PATH` env var to generate Bach-style output after training

## Quick start

**Requirements:** Apple Silicon Mac (tested on M3 Ultra, 96 GB), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Prepare data (converts Bach text to parquet, trains tokenizer)
python -c "
import pyarrow as pa, pyarrow.parquet as pq, os
with open('experiments/bach/input_inventions_v2.txt') as f: text = f.read()
pieces, current = [], []
for line in text.split('\n'):
    if line.startswith('bwv') and current: pieces.append('\n'.join(current)); current = [line]
    elif line.strip(): current.append(line)
    else: current.append(line) if current else None
if current: pieces.append('\n'.join(current))
n_val = max(1, len(pieces) // 10)
d = os.path.expanduser('~/.cache/autoresearch/data'); os.makedirs(d, exist_ok=True)
pq.write_table(pa.table({'text': pieces[:-n_val]}), f'{d}/shard_00000.parquet')
pq.write_table(pa.table({'text': pieces[-n_val:]}), f'{d}/shard_06542.parquet')
print(f'Wrote {len(pieces)-n_val} train / {n_val} val pieces')
from prepare import train_tokenizer; train_tokenizer()
"

# 4. Train (~5 min on M3 Ultra)
uv run train.py

# 5. Train + generate samples
SAMPLE_OUTPUT_PATH=samples.txt uv run train.py
```

## Results

| commit | val_bpb | memory_gb | description |
|--------|---------|-----------|-------------|
| ef1e262 | 2.961 | 14.1 | baseline: DEPTH=4 dim=256 |
| dab8187 | 1.500 | 52.2 | DEPTH=8 dim=512 + dropout=0.1 |
| e3bea63 | 0.687 | 103.3 | DEPTH=12 dim=768 |

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
experiments/
  bach/
    input_inventions_v2.txt    — Bach inventions dataset (custom notation)
    txt_to_mid_music21_v2.py   — convert text notation to MIDI
    min/                       — best generated samples
```

## How the agent loop works

See `program.md` for full details. In short: the agent modifies `train.py`, trains for 5 minutes, checks val_bpb, keeps improvements and discards regressions, then repeats indefinitely.

## License

MIT
