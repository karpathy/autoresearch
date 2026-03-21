"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import json
import torch

# ---------------------------------------------------------------------------
# Constants — loaded from ground.json (required); fail with clear message
# ---------------------------------------------------------------------------

_GROUND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground.json")
try:
    with open(_GROUND_PATH, "r", encoding="utf-8") as _f:
        _ground = json.load(_f)
except FileNotFoundError:
    sys.exit(f"ERROR: ground.json not found at {_GROUND_PATH}. "
             "This file is required — see README.md for setup instructions.")
except json.JSONDecodeError as _e:
    sys.exit(f"ERROR: ground.json is malformed: {_e}")

_required_keys = ["training", "data", "tokenizer", "mode", "processor"]
_missing = [k for k in _required_keys if k not in _ground]
if _missing:
    sys.exit(f"ERROR: ground.json missing required keys: {_missing}")

_training = _ground["training"]
_data = _ground["data"]
_tok = _ground["tokenizer"]
_mode = _ground["mode"]

MAX_SEQ_LEN = _training["max_seq_len"]
TIME_BUDGET = _training["time_budget_test"] if _mode == "test" else _training["time_budget_train"]
EVAL_TOKENS = _training["eval_tokens_multiplier"] * _training["eval_tokens_unit"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.expanduser(_data["cache_dir"])
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = _data["base_url"]
MAX_SHARD = _data["max_shard"]
VAL_SHARD = _data["val_shard"]
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = _tok["vocab_size"]

SPLIT_PATTERN = _tok["split_pattern"]

_n_special = _tok["special_tokens_count"]
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(_n_special)]
BOS_TOKEN = _tok["bos_token"]

# ---------------------------------------------------------------------------
# Platform detection — auto-detect GPU, with ground.json processor overrides
# Exports PLATFORM dict consumed by train.py alongside MAX_SEQ_LEN, TIME_BUDGET
# ---------------------------------------------------------------------------

_proc = _ground["processor"]

# FP16/BF16 tensor-core ops per cycle per SM, by (major, minor) compute capability.
# Used with SM count + boost clock to compute peak FLOPS at runtime.
_GPU_OPS_PER_CYCLE_PER_SM = {
    (7, 0): 128,   # Volta (V100)
    (7, 5): 128,   # Turing (RTX 20xx, Quadro RTX, T4)
    (8, 0): 512,   # Ampere GA100 (A100)
    (8, 6): 256,   # Ampere GA10x (RTX 30xx, A40)
    (8, 7): 256,   # Ampere GA10B (Jetson Orin)
    (8, 9): 512,   # Ada Lovelace (RTX 40xx, L40)
    (9, 0): 1024,  # Hopper (H100 SXM)
    (10, 0): 1024, # Blackwell (B100/B200) — provisional
}

def _estimate_peak_flops(device_idx=0):
    """Compute peak FP16/BF16 tensor TFLOPS from SM count and clock."""
    import torch as _torch
    props = _torch.cuda.get_device_properties(device_idx)
    cc = (props.major, props.minor)
    sm_count = props.multi_processor_count
    clock_ghz = props.clock_rate / 1e6  # clock_rate is in kHz
    ops_per_cycle = _GPU_OPS_PER_CYCLE_PER_SM.get(cc)
    if ops_per_cycle is None:
        major_fallbacks = {7: 128, 8: 256, 9: 1024, 10: 1024}
        ops_per_cycle = major_fallbacks.get(cc[0], 128)
        print(f"Warning: unknown GPU CC {cc}, using fallback ops/cycle={ops_per_cycle}")
    peak = sm_count * ops_per_cycle * clock_ghz * 1e9 * 2  # *2 for FMA
    print(f"GPU: {props.name} | CC {cc[0]}.{cc[1]} | {sm_count} SMs | "
          f"{clock_ghz:.2f} GHz | peak {peak/1e12:.1f} TFLOPS")
    return peak


def _detect_platform():
    import torch as _torch
    p = {"device": "cpu", "dtype": "fp32", "use_grad_scaler": False,
         "attention": "naive", "compile": False, "embedding_dtype": "fp32",
         "fa3_repo": None, "peak_flops": 0.0}
    if _torch.cuda.is_available():
        p["device"] = "cuda"
        cc = _torch.cuda.get_device_capability(0)
        p["peak_flops"] = _estimate_peak_flops(0)
        if cc[0] >= 9:
            # Hopper+ — bf16, FA3, compile
            p["dtype"] = "bf16"
            p["embedding_dtype"] = "bf16"
            p["fa3_repo"] = "varunneal/flash-attention-3"
            p["attention"] = "flash"
            try:
                import triton  # noqa: F401
                p["compile"] = sys.platform != "win32"
            except ImportError:
                print("Warning: triton not found — torch.compile disabled (Hopper)")
        elif cc[0] >= 8:
            # Ampere/Ada — bf16, FA3, compile
            p["dtype"] = "bf16"
            p["embedding_dtype"] = "bf16"
            p["fa3_repo"] = "kernels-community/flash-attn3"
            p["attention"] = "flash"
            try:
                import triton  # noqa: F401
                p["compile"] = sys.platform != "win32"
            except ImportError:
                print("Warning: triton not found — torch.compile disabled (Ampere/Ada)")
        else:
            # Turing / older — fp16, SDPA, no compile
            p["dtype"] = "fp16"
            p["use_grad_scaler"] = True
            p["attention"] = "sdpa"
            p["embedding_dtype"] = "fp32"

    # Apply ground.json processor overrides (non-"auto" values)
    if _proc["dtype"] != "auto":
        p["dtype"] = _proc["dtype"]
        p["use_grad_scaler"] = p["dtype"] == "fp16"
        p["embedding_dtype"] = "bf16" if p["dtype"] == "bf16" else "fp32"
    if _proc["compile"] != "auto":
        p["compile"] = bool(_proc["compile"])
    if _proc["flash_attention"] != "auto":
        fa = _proc["flash_attention"]
        if fa is False or fa == "sdpa":
            p["attention"] = "sdpa"
            p["fa3_repo"] = None
        elif isinstance(fa, str) and fa not in ("auto", "sdpa"):
            p["attention"] = "flash"
            p["fa3_repo"] = fa
    pf = _proc.get("peak_flops", "auto")
    if pf != "auto":
        p["peak_flops"] = float(pf)
    return p

PLATFORM = _detect_platform()

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError as cleanup_err:
                        print(f"  Warning: failed to clean up {path}: {cleanup_err}")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    # Count what's already downloaded
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
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
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=_data["num_shards"], help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=_data["download_workers"], help="Number of parallel download workers")
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
