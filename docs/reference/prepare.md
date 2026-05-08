# Reference: `prepare.py`

`prepare.py` has two jobs:

1. **One-time setup** (run once via the CLI): download data shards, train a BPE tokenizer, build the `token_bytes` lookup.
2. **Runtime utilities imported by `train.py`**: the `Tokenizer` class, the `make_dataloader` generator, the `evaluate_bpb` metric, and the fixed constants.

By contract, **`prepare.py` is read-only during experiments** — agents must not modify it. Changing it would invalidate cross-experiment comparisons.

## Constants

These are the public knobs the rest of the system depends on. Defined at module level near the top of `prepare.py`.

| Name | Value | Meaning |
|---|---|---|
| `MAX_SEQ_LEN` | `2048` | Context length used everywhere. `train.py` imports this as the model's `sequence_len`. |
| `TIME_BUDGET` | `300` | Seconds of training (excluding compilation/warmup) before the loop exits. |
| `EVAL_TOKENS` | `40 * 524288` (~21 M) | Number of tokens evaluated by `evaluate_bpb`. |
| `CACHE_DIR` | `~/.cache/autoresearch` | Root for all persisted state. |
| `DATA_DIR` | `<CACHE_DIR>/data` | Parquet shards live here. |
| `TOKENIZER_DIR` | `<CACHE_DIR>/tokenizer` | `tokenizer.pkl` and `token_bytes.pt` live here. |
| `BASE_URL` | `https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main` | Source of shards. |
| `MAX_SHARD` | `6542` | The dataset has shards `shard_00000.parquet` … `shard_06542.parquet`. |
| `VAL_SHARD` | `MAX_SHARD` (= `6542`) | Pinned validation shard. Always held out. |
| `VAL_FILENAME` | `f"shard_{VAL_SHARD:05d}.parquet"` | Filename derived from `VAL_SHARD`. |
| `VOCAB_SIZE` | `8192` | Target vocab size including special tokens. |
| `SPLIT_PATTERN` | GPT-4-style regex (with `\p{N}{1,2}` instead of `{1,3}`) | Pre-tokenization regex used by `rustbpe`. |
| `SPECIAL_TOKENS` | `["<|reserved_0|>", …, "<|reserved_3|>"]` | Reserved tokens appended after the BPE vocab. |
| `BOS_TOKEN` | `"<|reserved_0|>"` | Inserted at the start of every packed row. |

`MAX_SEQ_LEN`, `TIME_BUDGET`, `EVAL_TOKENS`, and `VOCAB_SIZE` are the contract surface — agents and forks must not touch them.

## CLI

```bash
uv run prepare.py [--num-shards N] [--download-workers M]
```

| Flag | Default | Effect |
|---|---|---|
| `--num-shards` | `10` | How many training shards to download. `-1` means "all 6,542". The validation shard is downloaded in addition. |
| `--download-workers` | `8` | Parallel HTTP workers. Auto-clamped to the number of files actually needed. |

Steps the CLI executes (in order):

1. `download_data(num_shards, download_workers)` — fetch missing shards.
2. `train_tokenizer()` — train BPE if `tokenizer.pkl` and `token_bytes.pt` aren't both present.

Both steps are idempotent.

## Module-level functions

### `download_single_shard(index) -> bool`

Downloads `shard_{index:05d}.parquet` to `DATA_DIR`. Skips if already present. Retries with exponential backoff up to 5 attempts, cleaning up the `.tmp` partial file between retries. Returns `True` on success.

### `download_data(num_shards, download_workers=8)`

Downloads shards `[0, num_shards)` plus `VAL_SHARD`, in parallel via `multiprocessing.Pool`. Prints progress and a final summary. Creates `DATA_DIR` if missing.

### `list_parquet_files() -> list[str]`

Returns sorted absolute paths of all `*.parquet` files in `DATA_DIR`, excluding `.tmp` partials. Used by the dataloader and tokenizer training.

### `text_iterator(max_chars=1_000_000_000, doc_cap=10_000)`

Generator that yields raw document strings from the **training split only** (val shard excluded). Each document is truncated to `doc_cap` characters and the generator stops once `max_chars` total characters have been yielded. Used by `train_tokenizer`.

### `train_tokenizer()`

Trains a BPE tokenizer with `rustbpe` over `text_iterator()`, then wraps it as a `tiktoken.Encoding` with the four `<|reserved_i|>` special tokens appended after the merge vocabulary. Saves:

- `tokenizer.pkl` — pickled `tiktoken.Encoding`.
- `token_bytes.pt` — `torch.int32` tensor of length `enc.n_vocab` where entry `i` is `len(enc.decode([i]).encode("utf-8"))`. Special tokens get length `0` so `evaluate_bpb` excludes them.

Skips entirely if both files already exist. Asserts that at least 2 parquet files are present (1 train + 1 val). Includes a roundtrip sanity check on a fixed test string.

### `_document_batches(split, tokenizer_batch_size=128)`

Internal infinite generator over `(documents, epoch)` tuples. `split` is `"train"` or `"val"`. The val branch yields *only* `VAL_FILENAME`; the train branch yields every other shard.

### `make_dataloader(tokenizer, B, T, split, buffer_size=1000) -> generator`

Returns an infinite generator yielding `(inputs, targets, epoch)` tuples on CUDA.

- `inputs` and `targets` are `torch.long` tensors of shape `(B, T)`.
- Inputs and targets are offset by 1: `inputs = row[:-1]`, `targets = row[1:]`.
- Every row starts with the BOS token id.
- Documents are packed using **best-fit-decreasing**: the largest doc that fits the remaining slot is taken; if none fits, the shortest doc is cropped to fill exactly. This achieves 100% utilization with no padding.
- The buffer is refilled to `buffer_size` documents whenever it drops below.
- Internally allocates a pinned CPU buffer and a CUDA buffer once, then re-uses them.

The dataloader is a public boundary: `train.py` uses it for both training (`split="train"`) and inside `evaluate_bpb` (`split="val"`).

### `get_token_bytes(device="cpu") -> torch.Tensor`

Loads `token_bytes.pt` to the requested device. `evaluate_bpb` calls it with `device="cuda"`.

### `evaluate_bpb(model, tokenizer, batch_size) -> float`

The fixed metric. **Do not modify.**

```python
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0          # exclude special tokens
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
```

Properties:

- Always `MAX_SEQ_LEN`-long contexts so different model configs evaluate identically.
- Special tokens (`token_bytes == 0`) are masked out from both numerator and denominator.
- Returns nats/byte → bits/byte via the `1/log(2)` factor.
- Vocab-size-independent: the result is interpreted in bits per UTF-8 byte of the original text.

`train.py` calls this once at the end of each run, inside the `bf16` autocast context.

## `Tokenizer` class

A thin wrapper around the pickled `tiktoken.Encoding`. `train.py` instantiates one via `Tokenizer.from_directory()` and passes it to `make_dataloader`.

| Method | Purpose |
|---|---|
| `Tokenizer.from_directory(tokenizer_dir=TOKENIZER_DIR)` | Load `tokenizer.pkl` and return an instance. |
| `get_vocab_size() -> int` | Returns `enc.n_vocab` (= `VOCAB_SIZE`). |
| `get_bos_token_id() -> int` | Returns the encoded id of `BOS_TOKEN`. |
| `encode(text, prepend=None, num_threads=8)` | Encodes a `str` or a `list[str]`. If `prepend` is given (id or string token), inserts it at the start of every output sequence. Uses `tiktoken`'s batched, threaded path for lists. |
| `decode(ids) -> str` | Standard `tiktoken` decode. |

The `Tokenizer` exists so `train.py` can stay independent of `tiktoken`'s exact API.

## Where things break if you violate the contract

- Editing `evaluate_bpb` or `EVAL_TOKENS`: cross-experiment metrics become incomparable.
- Editing `MAX_SEQ_LEN`: changes the batch token budget and the rotary embedding range; old runs are no longer comparable.
- Editing `VAL_SHARD`: contaminates val with prior train data — invalidates the metric.
- Editing `VOCAB_SIZE` after the tokenizer is built: the model embedding shape will mismatch the saved tokenizer.

The canonical fix for any of those is to regenerate the cache: delete `~/.cache/autoresearch/` and re-run `uv run prepare.py`.
