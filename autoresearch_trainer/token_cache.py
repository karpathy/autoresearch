from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from prepare import CACHE_DIR, TOKENIZER_DIR, VAL_FILENAME, Tokenizer, list_parquet_files


CACHE_VERSION = 1
TOKENIZER_BATCH_SIZE = 128


@dataclass(frozen=True)
class TokenCacheInfo:
    cache_path: Path
    meta_path: Path
    dtype_name: str
    num_tokens: int
    fingerprint: str
    built: bool


def _cache_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _train_shard_paths() -> list[Path]:
    return [Path(path) for path in list_parquet_files() if not path.endswith(VAL_FILENAME)]


def _fingerprint_inputs(train_paths: list[Path], tokenizer_path: Path, tokenizer: Tokenizer) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"token-cache-v{CACHE_VERSION}".encode("utf-8"))
    hasher.update(str(tokenizer.get_vocab_size()).encode("utf-8"))
    hasher.update(str(tokenizer.get_bos_token_id()).encode("utf-8"))
    tokenizer_stat = tokenizer_path.stat()
    hasher.update(str(tokenizer_path).encode("utf-8"))
    hasher.update(str(tokenizer_stat.st_size).encode("utf-8"))
    hasher.update(str(tokenizer_stat.st_mtime_ns).encode("utf-8"))
    for path in train_paths:
        stat = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def _meta_matches(meta: dict, cache_path: Path, *, fingerprint: str, dtype_name: str) -> bool:
    if meta.get("version") != CACHE_VERSION:
        return False
    if meta.get("fingerprint") != fingerprint:
        return False
    if meta.get("dtype_name") != dtype_name:
        return False
    if not cache_path.exists():
        return False
    num_tokens = int(meta.get("num_tokens", 0))
    if num_tokens <= 1:
        return False
    expected_size = num_tokens * np.dtype(dtype_name).itemsize
    return cache_path.stat().st_size == expected_size


def ensure_train_token_cache(tokenizer: Tokenizer, *, verbose: bool = True) -> TokenCacheInfo:
    train_paths = _train_shard_paths()
    if not train_paths:
        raise RuntimeError("No training shards found. Run prepare.py first.")

    tokenizer_path = Path(TOKENIZER_DIR) / "tokenizer.pkl"
    dtype = _cache_dtype(tokenizer.get_vocab_size())
    fingerprint = _fingerprint_inputs(train_paths, tokenizer_path, tokenizer)

    cache_dir = Path(CACHE_DIR) / "token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"train_tokens_v{CACHE_VERSION}_{dtype.name}.bin"
    meta_path = cache_dir / f"train_tokens_v{CACHE_VERSION}_{dtype.name}.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if _meta_matches(meta, cache_path, fingerprint=fingerprint, dtype_name=dtype.name):
            return TokenCacheInfo(
                cache_path=cache_path,
                meta_path=meta_path,
                dtype_name=dtype.name,
                num_tokens=int(meta["num_tokens"]),
                fingerprint=fingerprint,
                built=False,
            )

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    bos_token = tokenizer.get_bos_token_id()
    num_tokens = 0
    num_docs = 0

    try:
        with open(tmp_path, "wb") as handle:
            for shard_idx, shard_path in enumerate(train_paths, start=1):
                shard_tokens = 0
                parquet_file = pq.ParquetFile(shard_path)
                for row_group_idx in range(parquet_file.num_row_groups):
                    row_group = parquet_file.read_row_group(row_group_idx, columns=["text"])
                    texts = row_group.column("text").to_pylist()
                    for batch_start in range(0, len(texts), TOKENIZER_BATCH_SIZE):
                        doc_batch = texts[batch_start:batch_start + TOKENIZER_BATCH_SIZE]
                        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
                        flat_count = sum(len(doc) for doc in token_lists)
                        flat_tokens = np.fromiter(
                            chain.from_iterable(token_lists),
                            dtype=dtype,
                            count=flat_count,
                        )
                        flat_tokens.tofile(handle)
                        num_tokens += int(flat_tokens.size)
                        shard_tokens += int(flat_tokens.size)
                        num_docs += len(token_lists)
                if verbose:
                    print(
                        f"Train token cache: encoded {shard_path.name} -> {shard_tokens:,} tokens "
                        f"({shard_idx}/{len(train_paths)})",
                        flush=True,
                    )
        if num_tokens <= 1:
            raise RuntimeError("Train token cache is empty after encoding.")
        os.replace(tmp_path, cache_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    meta = {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "dtype_name": dtype.name,
        "num_tokens": num_tokens,
        "num_docs": num_docs,
        "bos_token_id": bos_token,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return TokenCacheInfo(
        cache_path=cache_path,
        meta_path=meta_path,
        dtype_name=dtype.name,
        num_tokens=num_tokens,
        fingerprint=fingerprint,
        built=True,
    )


class TokenWindowLoader:
    def __init__(
        self,
        cache_info: TokenCacheInfo,
        batch_size: int,
        sequence_len: int,
        *,
        device: str,
        seed: int,
    ) -> None:
        self.cache_info = cache_info
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.row_capacity = sequence_len + 1
        self.device = device
        self.dtype = np.dtype(cache_info.dtype_name)
        self.tokens = np.memmap(cache_info.cache_path, dtype=self.dtype, mode="r", shape=(cache_info.num_tokens,))
        if cache_info.num_tokens <= self.row_capacity:
            raise RuntimeError(
                f"Train token cache has only {cache_info.num_tokens} tokens, which is not enough for "
                f"a training window of length {self.row_capacity}."
            )
        self.max_start = cache_info.num_tokens - self.row_capacity
        self.tokens_sampled = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.host_windows = np.empty((batch_size, self.row_capacity), dtype=self.dtype)
        self.host_windows_tensor = torch.from_numpy(self.host_windows)
        self.row_buffer = torch.empty((batch_size, self.row_capacity), dtype=torch.long)
        self.cpu_buffer = torch.empty(2 * batch_size * sequence_len, dtype=torch.long, pin_memory=True)
        self.gpu_buffer = torch.empty(2 * batch_size * sequence_len, dtype=torch.long, device=device)
        self.cpu_inputs = self.cpu_buffer[:batch_size * sequence_len].view(batch_size, sequence_len)
        self.cpu_targets = self.cpu_buffer[batch_size * sequence_len:].view(batch_size, sequence_len)
        self.inputs = self.gpu_buffer[:batch_size * sequence_len].view(batch_size, sequence_len)
        self.targets = self.gpu_buffer[batch_size * sequence_len:].view(batch_size, sequence_len)

    def __iter__(self) -> TokenWindowLoader:
        return self

    def __next__(self):
        starts = torch.randint(0, self.max_start + 1, (self.batch_size,), generator=self.generator)
        for row_idx, start in enumerate(starts.tolist()):
            self.host_windows[row_idx] = self.tokens[start:start + self.row_capacity]
        self.row_buffer.copy_(self.host_windows_tensor)
        self.cpu_inputs.copy_(self.row_buffer[:, :-1])
        self.cpu_targets.copy_(self.row_buffer[:, 1:])
        self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=True)
        self.tokens_sampled += self.batch_size * self.sequence_len
        epoch = 1 + (self.tokens_sampled // max(self.cache_info.num_tokens, 1))
        return self.inputs, self.targets, epoch


def make_token_window_loader(
    cache_info: TokenCacheInfo,
    batch_size: int,
    sequence_len: int,
    *,
    device: str,
    seed: int,
) -> TokenWindowLoader:
    return TokenWindowLoader(
        cache_info,
        batch_size,
        sequence_len,
        device=device,
        seed=seed,
    )
