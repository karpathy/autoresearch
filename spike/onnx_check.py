"""
Spike (pi-autoresearch-serverless, plan 17a): what scoring the exported model in onnxruntime would
take, on the CPU and on the GPU. Run in the scorer's own environment, as 16a's scorer is: numpy, onnx
and onnxruntime only, never torch or the training code.

    python -I spike/onnx_check.py EXPORT_DIR

For each provider (CUDA with TF32 off and on, then the CPU): the session's creation, its logits on
the first row against PyTorch's float32 ones, its bpb on the rows against PyTorch's, one causality
probe, and the time of a forward at batch 1 and 8, from which the time to score upstream's 40 x 524288
val tokens follows. Prints one line, SPIKE_ONNX.
"""

import json
import math
import os
import statistics
import sys
import time
from collections import Counter

# onnxruntime's telemetry opens a network socket from a thread of its own on Linux (plan 16a).
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

EVAL_TOKENS = 40 * 524288  # upstream's prepare.py


def threads():
    """The CPUs the container's cgroup allows (cgroup2, or v1 as on Cloud Run), or the machine's."""
    for quota_file, period_file in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            with open(quota_file) as f:
                words = f.read().split()
            if period_file is not None:
                with open(period_file) as f:
                    words.append(f.read().strip())
            quota, period = words[0], words[-1]
            if quota not in ("max", "-1"):
                return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError, IndexError):
            pass
    return os.cpu_count() or 1


def graph_facts(proto):
    ops = Counter()
    domains = set()

    def walk(graph):
        for node in graph.node:
            ops[node.op_type] += 1
            domains.add(node.domain or "ai.onnx")
            for attr in node.attribute:
                if attr.g is not None and attr.g.node:
                    walk(attr.g)
                for g in attr.graphs:
                    walk(g)

    walk(proto.graph)
    external = sum(1 for t in proto.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL)
    return {
        "opsets": {o.domain or "ai.onnx": o.version for o in proto.opset_import},
        "domains": sorted(domains),
        "functions": len(proto.functions),
        "external_initializers": external,
        "initializers": len(proto.graph.initializer),
        "nodes": sum(ops.values()),
        "ops": dict(ops.most_common()),
        "input": [d.dim_param or d.dim_value for d in proto.graph.input[0].type.tensor_type.shape.dim],
    }


def bpb_terms(logits, targets, token_bytes):
    V = token_bytes.shape[0]
    z = logits[..., :V].astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll = -np.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    nbytes = token_bytes[targets]
    return float((nll * (nbytes > 0)).sum()), int(nbytes.sum())


def timed(fn, repeats):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def check(name, data, providers, options, x, y, tb, logits0, ref_bpb, dynamic):
    out = {"name": name}
    t0 = time.perf_counter()
    session = ort.InferenceSession(data, options, providers=providers)
    out["create_seconds"] = time.perf_counter() - t0
    out["providers"] = session.get_providers()
    inp = session.get_inputs()[0].name
    run = lambda batch: session.run(None, {inp: batch})[0]  # noqa: E731

    got0 = run(x[:1])[0]
    out["finite"] = bool(np.isfinite(got0).all())
    out["logits_max_abs_diff"] = float(np.abs(got0[:, : logits0.shape[1]] - logits0).max())

    nats = nbytes = 0
    for i in range(x.shape[0]):
        n, b = bpb_terms(run(x[i : i + 1]), y[i : i + 1], tb)
        nats, nbytes = nats + n, nbytes + b
    out["bpb"] = nats / (math.log(2) * nbytes)
    out["bpb_diff"] = out["bpb"] - ref_bpb

    # Causality: row 0 with every token after p replaced by row 1's; the logits up to p unchanged.
    T = x.shape[1]
    p = T // 2
    spliced = x[:1].copy()
    spliced[0, p + 1 :] = x[1, p + 1 :]
    out["splice_prefix_max_abs_diff"] = float(np.abs(run(spliced)[0, : p + 1] - got0[: p + 1]).max())

    rows = EVAL_TOKENS // T
    per_row = out["batch1_seconds"] = timed(lambda: run(x[:1]), 3)
    if dynamic and x.shape[0] >= 8:
        out["batch8_seconds"] = timed(lambda: run(x[:8]), 3)
        per_row = min(per_row, out["batch8_seconds"] / 8)
    out["eval_rows"] = rows
    out["projected_eval_seconds"] = per_row * rows
    return out


def main(directory):
    with open(os.path.join(directory, "model.onnx"), "rb") as f:
        data = f.read()
    x = np.load(os.path.join(directory, "x.npy"))
    y = np.load(os.path.join(directory, "y.npy"))
    tb = np.load(os.path.join(directory, "token_bytes.npy"))
    logits0 = np.load(os.path.join(directory, "logits0.npy"))
    with open(os.path.join(directory, "reference.json")) as f:
        ref = json.load(f)

    result = {"ort": ort.__version__, "available_providers": ort.get_available_providers(), "model_bytes": len(data)}
    proto = onnx.load_model_from_string(data)
    result["graph"] = graph_facts(proto)
    dynamic = isinstance(result["graph"]["input"][0], str)
    result["reference_bpb"] = ref["bpb"]

    runs = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:  # noqa: BLE001
                result["preload_dlls_error"] = f"{type(e).__name__}: {e}"
        for tf32 in ("0", "1"):
            runs.append((f"cuda_tf32_{tf32}", [("CUDAExecutionProvider", {"use_tf32": tf32}), "CPUExecutionProvider"]))
    runs.append(("cpu", ["CPUExecutionProvider"]))

    result["checks"] = []
    for name, providers in runs:
        options = ort.SessionOptions()
        if name == "cpu":
            options.intra_op_num_threads = threads()
            result["cpu_threads"] = options.intra_op_num_threads
        try:
            result["checks"].append(check(name, data, providers, options, x, y, tb, logits0, ref["bpb"], dynamic))
        except Exception as e:  # noqa: BLE001 - the spike records why
            result["checks"].append({"name": name, "error": f"{type(e).__name__}: {e}"[:4000]})
        print(f"{name}: {json.dumps(result['checks'][-1])}", flush=True)

    print("SPIKE_ONNX " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
