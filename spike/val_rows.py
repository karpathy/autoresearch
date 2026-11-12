"""
Spike (pi-autoresearch-serverless, plan 17a, Trial 3): the rows upstream's evaluate_bpb scores, for
spike/onnx_check.py's full eval. Runs prepare.py's own code only, never train.py: the val rows come
from the fixed side, as the harness's will.

    python spike/val_rows.py OUT_DIR

Writes x.npy and y.npy, int32 [EVAL_TOKENS // MAX_SEQ_LEN, MAX_SEQ_LEN], and token_bytes.npy, int64
[V]. The rows are evaluate_bpb's: make_dataloader's "val" split at train.py's DEVICE_BATCH_SIZE, for
EVAL_TOKENS // (batch * MAX_SEQ_LEN) batches. The dataloader fills its rows in order, whatever the
batch, and puts them on the GPU, which this needs.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare import EVAL_TOKENS, MAX_SEQ_LEN, Tokenizer, get_token_bytes, make_dataloader  # noqa: E402

BATCH = 128  # train.py's DEVICE_BATCH_SIZE, which it passes to evaluate_bpb


def main(out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = Tokenizer.from_directory()
    loader = make_dataloader(tokenizer, BATCH, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (BATCH * MAX_SEQ_LEN)
    xs, ys = [], []
    for _ in range(steps):
        x, y, _ = next(loader)  # views of buffers the next batch overwrites: copied by .cpu()
        xs.append(x.cpu().to(torch.int32).numpy())
        ys.append(y.cpu().to(torch.int32).numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    tb = get_token_bytes("cpu").numpy().astype(np.int64)
    np.save(os.path.join(out_dir, "x.npy"), x)
    np.save(os.path.join(out_dir, "y.npy"), y)
    np.save(os.path.join(out_dir, "token_bytes.npy"), tb)
    print("SPIKE_VAL " + json.dumps({
        "rows": int(x.shape[0]), "T": int(x.shape[1]), "vocab": int(tb.shape[0]),
        "tokens": int(x.size), "bytes": int(tb[y].sum()), "seconds": time.time() - t0,
    }), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
