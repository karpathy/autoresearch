"""
Spike (pi-autoresearch-serverless, plan 17a): the trained model as ONNX, and what spike/onnx_check.py
compares onnxruntime with: val rows, the token bytes, and PyTorch's own float32 logits and bpb on
them. Nothing is pickled. Prints one line, SPIKE_EXPORT, with the export's own numbers.

The model is exported uncompiled, on the CPU, in float32, with SDPA attention and explicit band masks
for the short windows, by torch.onnx.export(dynamo=True) at opset 18 with no graph optimizer, as 16a's
export: one int64 input [B, T] with B dynamic (or 1, if a dynamic batch fails to export), one float
output [B, T, V].
"""

import json
import logging
import math
import os
import time

import numpy as np
import torch

for _name in ("onnxscript", "onnx_ir", "torch.onnx"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def to_onnx(model, T, dynamic):
    # A dynamic dimension can't be traced at size 1, which torch.export specializes.
    idx = torch.zeros((2 if dynamic else 1, T), dtype=torch.long)
    extra = {"dynamic_shapes": {"idx": {0: torch.export.Dim("batch", min=1, max=1024)}}} if dynamic else {}
    with torch.no_grad():
        program = torch.onnx.export(
            model, (idx,), dynamo=True, opset_version=18,
            input_names=["idx"], output_names=["logits"], verbose=False, optimize=False, **extra,
        )
    return program.model_proto.SerializeToString()


def bpb_terms(logits, targets, token_bytes):
    """Nats over the targets whose token has bytes, and those bytes: upstream's evaluate_bpb, in
    float64, with the log-softmax over the first V columns."""
    V = token_bytes.shape[0]
    z = logits[..., :V].astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll = -np.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    nbytes = token_bytes[targets]
    return float((nll * (nbytes > 0)).sum()), int(nbytes.sum())


def export(model, x, y, token_bytes, out_dir):
    """Writes model.onnx and the reference into out_dir. x, y: [rows, T] int64 tensors on the CPU;
    token_bytes: [V] int tensor on the CPU."""
    os.makedirs(out_dir, exist_ok=True)
    info = {}
    model = model.to("cpu").float().eval()
    T = x.size(1)
    model.set_attention("sdpa", T, torch.device("cpu"))

    t0 = time.time()
    try:
        data = to_onnx(model, T, dynamic=True)
        info["batch"] = "dynamic"
    except Exception as e:  # noqa: BLE001 - the spike records why, and falls back
        info["dynamic_error"] = f"{type(e).__name__}: {e}"[:4000]
        data = to_onnx(model, T, dynamic=False)
        info["batch"] = "static"
    info["export_seconds"] = time.time() - t0
    info["model_bytes"] = len(data)
    with open(os.path.join(out_dir, "model.onnx"), "wb") as f:
        f.write(data)

    t0 = time.time()
    with torch.no_grad():
        logits = torch.cat([model(x[i : i + 1]) for i in range(x.size(0))]).numpy()
    info["reference_seconds"] = time.time() - t0
    xs, ys, tb = x.numpy().astype(np.int64), y.numpy().astype(np.int64), token_bytes.numpy().astype(np.int64)
    nats, nbytes = bpb_terms(logits, ys, tb)
    np.save(os.path.join(out_dir, "x.npy"), xs)
    np.save(os.path.join(out_dir, "y.npy"), ys)
    np.save(os.path.join(out_dir, "token_bytes.npy"), tb)
    np.save(os.path.join(out_dir, "logits0.npy"), logits[0])
    reference = {"rows": int(x.size(0)), "T": int(T), "vocab": int(tb.shape[0]), "bpb": nats / (math.log(2) * nbytes)}
    with open(os.path.join(out_dir, "reference.json"), "w") as f:
        json.dump(reference, f)
    info.update(reference)
    print("SPIKE_EXPORT " + json.dumps(info), flush=True)
    return info
