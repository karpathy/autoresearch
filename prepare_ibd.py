#!/usr/bin/env python3
"""
prepare_ibd.py

Download and prepare IBD-relevant pathology/clinical text for autoresearch.

Data sources (both CC BY 4.0 — commercial and open-source use permitted):
  1. TCGA-Reports (Kefeli et al., 2024)
     9,523 surgical pathology reports; GI tract reports (COAD/READ) included.
     https://data.mendeley.com/datasets/hyg5xkznpx/1
  2. MultiCaRe (Bitterman et al., 2023)
     96,000+ PMC open-access clinical case reports, filtered for IBD.
     https://zenodo.org/records/10079370

Output: ~/.cache/autoresearch/data/ — train + val parquet shards
  shard_00000.parquet  (train)
  shard_06542.parquet  (pinned val — must match VAL_SHARD in prepare.py)

After running this script, run:
  uv run prepare.py   # trains BPE tokenizer on the new data

Usage:
  uv run prepare_ibd.py
"""

import io
import json
import math
import os
import pickle
import random
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
RAW_DIR = CACHE_DIR / "ibd_raw"
VAL_SHARD_IDX = 6542   # must match VAL_SHARD in prepare.py
VAL_FRACTION = 0.10    # 10% held out for validation
DOCS_PER_SHARD = 5000  # max documents per train shard

IBD_KEYWORDS = [
    "inflammatory bowel disease",
    "crohn's disease", "crohn disease", "crohns disease",
    "ulcerative colitis",
    "indeterminate colitis",
    " ibd ",
    "ileocolitis",
    "proctocolitis",
    "pouchitis",
    "ileitis",
    "colitis",
]

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url, dest, desc=""):
    """Download url → dest, with MB progress. Skip if dest already exists."""
    if dest.exists():
        print(f"  Cached: {dest.name}")
        return
    print(f"  Downloading {desc or dest.name} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)",
                          end="", flush=True)
    print()
    tmp.rename(dest)


# ---------------------------------------------------------------------------
# Source 1: TCGA-Reports (Mendeley Data, CC BY 4.0)
# ---------------------------------------------------------------------------

MENDELEY_DATASET_ID = "hyg5xkznpx"
# Direct download URL for the TCGA-Reports CSV (Mendeley Data, CC BY 4.0).
# If this URL breaks, download manually from:
#   https://data.mendeley.com/datasets/hyg5xkznpx/1
TCGA_DIRECT_URL = (
    "https://data.mendeley.com/public-files/datasets/hyg5xkznpx/files/"
    "b0c9a16b-2b24-4b0e-a849-bb80d0b3d8f9/file_downloaded"
)

def fetch_tcga_reports():
    print("\n=== Source 1: TCGA-Reports (Mendeley Data, CC BY 4.0) ===")
    dest_dir = RAW_DIR / "tcga"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / "tcga_reports.csv"
    try:
        download_file(TCGA_DIRECT_URL, dest, desc="tcga_reports.csv")
    except Exception as e:
        print(f"  WARNING: direct download failed: {e}")
        print(f"  Manual fallback: download from https://data.mendeley.com/datasets/{MENDELEY_DATASET_ID}/1")
        print(f"  and place CSV/parquet files in {dest_dir}")

    return _load_tcga_from_dir(dest_dir)


def _load_tcga_from_dir(directory):
    """Load all CSV/TSV/ZIP/parquet files in directory and extract pathology report text."""
    docs = []
    paths = (list(directory.glob("*.csv")) + list(directory.glob("*.tsv"))
             + list(directory.glob("*.zip")) + list(directory.glob("*.parquet")))
    if not paths:
        print(f"  No data files found in {directory}. Skipping TCGA-Reports.")
        return docs
    for path in paths:
        docs.extend(_read_tabular_file(path, source="TCGA"))
    print(f"  TCGA-Reports: {len(docs)} documents loaded")
    return docs


def _read_tabular_file(path, source=""):
    """Read a CSV/TSV/ZIP/parquet and extract free-text strings from the best text column."""
    docs = []
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            df = pq.read_table(path).to_pandas()
            docs.extend(_df_to_texts(df, source))
            return docs
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith((".csv", ".tsv")):
                        with zf.open(name) as fh:
                            df = _read_df(fh, name)
                            docs.extend(_df_to_texts(df, source))
        else:
            df = _read_df(path, str(path))
            docs.extend(_df_to_texts(df, source))
    except Exception as e:
        print(f"  WARNING: could not read {path.name}: {e}")
    return docs


def _read_df(path_or_fh, name):
    sep = "\t" if str(name).endswith(".tsv") else ","
    return pd.read_csv(path_or_fh, sep=sep, low_memory=False)


def _df_to_texts(df, source=""):
    """Find the richest text column and return non-empty strings."""
    # Priority list of column names likely to hold free-text reports
    candidates = [
        "report_text", "text", "path_report", "pathology_report",
        "report", "narrative", "diagnosis", "findings", "abstract",
        "case_text", "clinical_text",
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        # Fall back: pick the object column with the longest average text
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if not str_cols:
            return []
        col = max(str_cols, key=lambda c: df[c].dropna().str.len().mean())
        print(f"  Auto-selected column '{col}' in {source}")
    texts = df[col].dropna().astype(str).str.strip().tolist()
    return [t for t in texts if len(t) > 80]


# ---------------------------------------------------------------------------
# Source 2: MultiCaRe (Zenodo, CC BY 4.0) — IBD-filtered
# ---------------------------------------------------------------------------

ZENODO_RECORD_ID = "10079370"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Text columns in MultiCaRe that contain clinical narrative
MULTICARE_TEXT_COLS = [
    "abstract", "background", "case_presentation", "case presentation",
    "clinical presentation", "discussion", "conclusion", "text",
    "body", "history", "findings", "report",
]


def fetch_multicare_ibd():
    print("\n=== Source 2: MultiCaRe (Zenodo, CC BY 4.0) — IBD filter ===")
    dest_dir = RAW_DIR / "multicare"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("  Querying Zenodo API ...")
    try:
        r = requests.get(ZENODO_API, timeout=30)
        r.raise_for_status()
        record = r.json()
    except Exception as e:
        print(f"  ERROR: could not reach Zenodo API: {e}")
        print(f"  Manual fallback: download from https://zenodo.org/records/{ZENODO_RECORD_ID}")
        print(f"  and place files in {dest_dir}")
        return _load_multicare_ibd_from_dir(dest_dir)

    # Zenodo API v1 format: record["files"] list
    # Zenodo API v2 format: record["entries"] list (newer deposits)
    files = record.get("files") or record.get("entries", [])
    print(f"  Found {len(files)} file(s) in Zenodo record {ZENODO_RECORD_ID}")

    for f in files:
        # Support both Zenodo v1 and v2 key naming
        fname = f.get("filename") or f.get("key", "")
        size_bytes = f.get("filesize") or f.get("size", 0)
        size_mb = size_bytes / 1e6

        # Skip non-tabular files (images, PDFs etc)
        if not any(fname.lower().endswith(ext) for ext in (".csv", ".tsv", ".zip", ".json", ".parquet")):
            print(f"  Skipping non-text file: {fname} ({size_mb:.0f} MB)")
            continue

        # Build download URL (v1: links.download, v2: links.content)
        links = f.get("links", {})
        url = (links.get("download")
               or links.get("content")
               or links.get("self")
               or f.get("download_url", ""))
        if not url:
            print(f"  WARNING: no download URL for {fname}")
            continue

        print(f"  {fname}  ({size_mb:.0f} MB)")
        download_file(url, dest_dir / fname, desc=fname)

    return _load_multicare_ibd_from_dir(dest_dir)


def _load_multicare_ibd_from_dir(directory):
    """Load MultiCaRe files, combine text columns per case, filter for IBD."""
    all_docs = []
    paths = (list(directory.glob("*.csv")) + list(directory.glob("*.tsv"))
             + list(directory.glob("*.zip")) + list(directory.glob("*.json"))
             + list(directory.glob("*.parquet")))
    if not paths:
        print(f"  No data files found in {directory}. Skipping MultiCaRe.")
        return []

    for path in paths:
        all_docs.extend(_parse_multicare_file(path))

    ibd_docs = [d for d in all_docs if _is_ibd(d)]
    print(f"  MultiCaRe total cases loaded: {len(all_docs)}")
    print(f"  MultiCaRe IBD-relevant after filter: {len(ibd_docs)}")
    return ibd_docs


def _parse_multicare_file(path):
    """Parse one MultiCaRe file → list of combined case-text strings."""
    docs = []
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            df = pq.read_table(path).to_pandas()
            # Handle MultiCaRe cases.parquet: "cases" column is array of dicts
            if "cases" in df.columns:
                for cases_arr in df["cases"]:
                    if cases_arr is None:
                        continue
                    for case in cases_arr:
                        if not isinstance(case, dict):
                            continue
                        text = case.get("case_text", "")
                        if isinstance(text, str) and len(text) > 100:
                            docs.append(text)
            else:
                # Generic parquet: find text columns
                text_cols = [c for c in df.columns
                             if any(kw in c.lower() for kw in MULTICARE_TEXT_COLS)]
                if not text_cols:
                    text_cols = df.select_dtypes(include="object").columns.tolist()
                for _, row in df.iterrows():
                    parts = []
                    for c in text_cols:
                        val = row[c]
                        try:
                            if val is not None and str(val).strip() not in ("nan", ""):
                                parts.append(str(val).strip())
                        except Exception:
                            pass
                    combined = "\n\n".join(parts)
                    if len(combined) > 100:
                        docs.append(combined)
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.endswith((".csv", ".tsv", ".json")):
                        with zf.open(name) as fh:
                            docs.extend(_parse_multicare_buffer(fh, name))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    text = _combine_case_fields(item)
                    if len(text) > 100:
                        docs.append(text)
        else:
            with open(path, "rb") as fh:
                docs.extend(_parse_multicare_buffer(fh, str(path)))
    except Exception as e:
        print(f"  WARNING: failed to parse {path.name}: {e}")
    return docs


def _parse_multicare_buffer(fh, name):
    """Read a CSV/TSV buffer, merge relevant text columns per row."""
    try:
        sep = "\t" if str(name).endswith(".tsv") else ","
        df = pd.read_csv(fh, sep=sep, low_memory=False)
        # Find text-rich columns
        text_cols = [c for c in df.columns
                     if any(kw in c.lower() for kw in MULTICARE_TEXT_COLS)]
        if not text_cols:
            text_cols = df.select_dtypes(include="object").columns.tolist()
        docs = []
        for _, row in df.iterrows():
            parts = [str(row[c]).strip() for c in text_cols
                     if pd.notna(row[c]) and str(row[c]).strip() not in ("nan", "")]
            combined = "\n\n".join(parts)
            if len(combined) > 100:
                docs.append(combined)
        return docs
    except Exception as e:
        print(f"  WARNING: parse error in {name}: {e}")
        return []


def _combine_case_fields(item):
    """Merge relevant fields from a JSON case dict into one string."""
    parts = []
    for key in ["abstract", "background", "case_presentation", "discussion",
                "conclusion", "text", "body", "clinical_presentation"]:
        val = item.get(key, "")
        if val and isinstance(val, str) and len(val.strip()) > 20:
            parts.append(val.strip())
    return "\n\n".join(parts)


def _is_ibd(text):
    t = text.lower()
    return any(kw in t for kw in IBD_KEYWORDS)


# ---------------------------------------------------------------------------
# Build shards
# ---------------------------------------------------------------------------

def build_shards(docs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(docs)

    n_val = max(50, int(len(docs) * VAL_FRACTION))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]

    print(f"\n=== Building shards ===")
    print(f"  Total: {len(docs)}  Train: {len(train_docs)}  Val: {len(val_docs)}")

    def write_shard(shard_docs, idx):
        path = DATA_DIR / f"shard_{idx:05d}.parquet"
        df = pd.DataFrame({"text": shard_docs})
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        kb = path.stat().st_size / 1024
        print(f"  Wrote {path.name}  ({len(shard_docs)} docs, {kb:.0f} KB)")

    # Pinned val shard
    write_shard(val_docs, VAL_SHARD_IDX)

    # Train shards (chunked)
    shard_idx = 0
    for i in range(0, len(train_docs), DOCS_PER_SHARD):
        write_shard(train_docs[i:i + DOCS_PER_SHARD], shard_idx)
        shard_idx += 1


# ---------------------------------------------------------------------------
# Runtime constants and utilities (imported by train.py)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288

TOKENIZER_DIR = str(Path.home() / ".cache" / "autoresearch" / "tokenizer")
VAL_SHARD = VAL_SHARD_IDX  # 6542
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
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
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Run prepare_ibd.py first.")
        sys.exit(1)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

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

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

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

    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


class Tokenizer:
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
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare_ibd.py first."
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
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


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
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("IBD Pathology Text — Data Preparation")
    print("=" * 40)
    print(f"Output: {DATA_DIR}")
    print()
    print("Licenses: CC BY 4.0 (both sources) — commercial and open-source use permitted.")
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    docs = []
    docs.extend(fetch_tcga_reports())
    docs.extend(fetch_multicare_ibd())

    if not docs:
        print("\nERROR: No documents collected. Check the download errors above.")
        print("You may need to manually download the files:")
        print(f"  TCGA-Reports: https://data.mendeley.com/datasets/{MENDELEY_DATASET_ID}/1")
        print(f"  MultiCaRe:    https://zenodo.org/records/{ZENODO_RECORD_ID}")
        sys.exit(1)

    build_shards(docs)

    print()
    print("Done! Next step:")
    print("  uv run prepare.py   # trains BPE tokenizer on your IBD text corpus")
