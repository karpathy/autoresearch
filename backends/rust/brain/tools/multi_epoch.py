"""
Multi-epoch data iteration for the feeder pipeline.

Instead of streaming through all 6542 parquets (seeing only ~9% with 700 steps),
use a smaller curated subset and iterate over it multiple times. Two passes over
high-quality data beats one pass over mixed data, especially during WSD warmdown.

Usage with feeder.py:
    from multi_epoch import multi_epoch_doc_iter
    doc_iter = streaming_document_batches(indices, cache_dir, cleanup=False)
    wrapped = multi_epoch_doc_iter(doc_iter, num_parquets_to_use=100, num_epochs=2)
    for row in pack_rows(wrapped, enc, bos_token_id):
        ...
"""

import os
import sys
import json
import random
import tempfile
import time


# 4GB threshold -- above this, spill docs to a temp file instead of RAM
_MEM_THRESHOLD = 4 * 1024 * 1024 * 1024


def estimate_parquets_needed(
    target_rows: int,
    avg_docs_per_parquet: int = 1000,
    avg_row_capacity: float = 0.97,
    tokens_per_doc: float = 500,
    row_capacity: int = 2049,
    num_epochs: int = 2,
) -> int:
    """Estimate how many parquets to use for a target number of packed rows.

    With multi-epoch, each parquet's documents are seen `num_epochs` times,
    so we need fewer parquets than single-epoch.

    Args:
        target_rows: Total packed rows needed (e.g. 179200 for 700 steps).
        avg_docs_per_parquet: Average documents per parquet shard.
        avg_row_capacity: Average fraction of row capacity used by packing.
        tokens_per_doc: Average tokens per document (including BOS).
        row_capacity: Tokens per packed row (2049).
        num_epochs: Number of passes over the data.

    Returns:
        Number of parquets to use.
    """
    tokens_per_row = row_capacity * avg_row_capacity
    total_tokens_needed = target_rows * tokens_per_row
    tokens_per_parquet = avg_docs_per_parquet * tokens_per_doc
    # Each parquet contributes its tokens num_epochs times
    parquets = total_tokens_needed / (tokens_per_parquet * num_epochs)
    return max(1, int(parquets + 0.5))  # round to nearest


def multi_epoch_doc_iter(doc_iter, num_parquets_to_use: int, num_epochs: int = 2):
    """Collect documents from doc_iter, then yield them num_epochs times with reshuffling.

    Args:
        doc_iter: Iterator yielding batches (lists) of text strings, as produced
                  by streaming_document_batches or jsonl_document_batches.
        num_parquets_to_use: How many parquets worth of data to consume from
                             doc_iter before stopping collection. Since each parquet
                             yields ~1000 docs in batches of 128, we count docs.
        num_epochs: Number of times to iterate over the collected documents.

    Yields:
        Batches (lists) of text strings, same interface as doc_iter.
    """
    DOCS_PER_PARQUET = 1000
    max_docs = num_parquets_to_use * DOCS_PER_PARQUET
    BATCH_SIZE = 128

    # Phase 1: collect all documents
    print(f"[multi_epoch] collecting docs from ~{num_parquets_to_use} parquets "
          f"(max {max_docs} docs)...", file=sys.stderr)
    t0 = time.time()

    all_docs = []
    total_bytes = 0
    collected = 0

    for batch in doc_iter:
        for doc in batch:
            if collected >= max_docs:
                break
            all_docs.append(doc)
            total_bytes += len(doc.encode("utf-8")) if isinstance(doc, str) else len(doc)
            collected += 1
        if collected >= max_docs:
            break

    elapsed = time.time() - t0
    n_docs = len(all_docs)
    print(f"[multi_epoch] collected {n_docs} docs ({total_bytes / 1e6:.1f} MB) "
          f"in {elapsed:.1f}s", file=sys.stderr)

    if n_docs == 0:
        print("[multi_epoch] WARNING: no documents collected", file=sys.stderr)
        return

    # Phase 1.5: if too large for memory, spill to temp file
    use_disk = total_bytes > _MEM_THRESHOLD
    tmp_path = None

    if use_disk:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="multi_epoch_", suffix=".jsonl")
        os.close(tmp_fd)
        print(f"[multi_epoch] data exceeds 2GB, spilling to {tmp_path}", file=sys.stderr)
        t0 = time.time()
        with open(tmp_path, "w", encoding="utf-8") as f:
            for doc in all_docs:
                f.write(json.dumps(doc) + "\n")
        del all_docs
        all_docs = None
        print(f"[multi_epoch] spill complete in {time.time() - t0:.1f}s", file=sys.stderr)

    # Phase 2: yield documents num_epochs times with per-epoch shuffling
    try:
        for epoch in range(num_epochs):
            rng = random.Random(42 + epoch)
            print(f"[multi_epoch] === epoch {epoch + 1}/{num_epochs} "
                  f"({n_docs} docs) ===", file=sys.stderr)
            t_epoch = time.time()

            if use_disk:
                # Re-read from disk, build shuffled index, yield in shuffled order
                # Read all lines into memory temporarily for shuffling
                with open(tmp_path, "r", encoding="utf-8") as f:
                    docs = [json.loads(line) for line in f]
                rng.shuffle(docs)
                batch = []
                for doc in docs:
                    batch.append(doc)
                    if len(batch) >= BATCH_SIZE:
                        yield batch
                        batch = []
                if batch:
                    yield batch
                del docs
            else:
                # In-memory path: shuffle indices, yield in batches
                indices = list(range(n_docs))
                rng.shuffle(indices)
                batch = []
                for idx in indices:
                    batch.append(all_docs[idx])
                    if len(batch) >= BATCH_SIZE:
                        yield batch
                        batch = []
                if batch:
                    yield batch

            elapsed_epoch = time.time() - t_epoch
            print(f"[multi_epoch] epoch {epoch + 1} done in {elapsed_epoch:.1f}s",
                  file=sys.stderr)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"[multi_epoch] cleaned up {tmp_path}", file=sys.stderr)


if __name__ == "__main__":
    # Quick estimate for our training setup
    target = 700 * 128 * 2  # 179,200 rows
    for epochs in [1, 2, 3]:
        n = estimate_parquets_needed(target, num_epochs=epochs)
        print(f"epochs={epochs}: need ~{n} parquets for {target} rows")
