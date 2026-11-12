"""
Spike (pi-autoresearch-serverless, plan 17a, Trial 4): are FlexAttention's outputs and gradients on
this GPU as close to the truth as SDPA's? Against float64, with train.py's own attention, masks and
model (run from its source above the hyperparameters, as spike/local_check.py does).

1. The attention op alone, at training's shapes (B 8, T 2048, H 4, D 128, laid out (B, T, H, D) as
   train.py's `attention` takes them), for each window of the model (left 1024 and 2048) and two
   regimes of RMS-normed q and k: diffuse (independent) and peaked (each key near its own query, so
   each query fixes on its own position). Flex and SDPA through torch.compile, as in training; each
   SDPA backend forced, eagerly; against an explicit band-masked softmax in float64 written here, not
   train.py's masks. The relative error of out, dq, dk and dv, and the largest absolute one.
2. The whole model, one backward on 8 rows of the train split: Flex and SDPA compiled under bf16
   autocast, as train.py trains, against a float64 copy, eager, on SDPA's math backend. The weights:
   init with noise (upstream zero-initialises every c_proj, so at init no gradient reaches attention),
   or a state dict that train.py saved (AR_WEIGHTS). Each parameter's relative gradient error.

    python spike/grad_check.py [WEIGHTS]

Prints one line, SPIKE_GRAD. spike/local_check.py runs the same functions on the CPU in float32.
"""

import contextlib
import copy
import enum
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from spike.local_check import train_model_code  # noqa: E402

TENSORS = ("out", "dq", "dk", "dv")
REGIMES = ("diffuse", "peaked")
GPU_BACKENDS = (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH)
NOISE = 0.02  # local_check's: enough for every c_proj to pass gradients to attention


class Precision(enum.Enum):
    BF16 = "bf16"        # train.py's: its parameters' dtypes, bf16 autocast; on the GPU
    FLOAT32 = "float32"  # everything in float32, no autocast; for local_check on the CPU


def error(e):
    return f"{type(e).__name__}: {e}"[:1000]


def rel_err(a, ref):
    return float((a.double() - ref).norm() / ref.norm())


def max_abs(a, ref):
    return float((a.double() - ref).abs().max())


def reference_attention(q, k, v, left):
    """FA3's window (left, 0), from its definition: query i attends to keys j with i - left <= j <= i.
    (B, T, H, D) in and out, in the inputs' dtype."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    i = torch.arange(q.size(-2), device=q.device)
    d = i[:, None] - i[None, :]
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    s = s.masked_fill(~((d >= 0) & (d <= left)), float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2)


def attention_inputs(shape, regime, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    randn = lambda: torch.randn(shape, dtype=torch.float64, device=device, generator=g)  # noqa: E731
    q = randn()
    k = q + 0.5 * randn() if regime == "peaked" else randn()
    D = shape[-1]
    return F.rms_norm(q, (D,)), F.rms_norm(k, (D,)), randn(), randn()


def fwd_bwd(fn, q, k, v, dout, dtype):
    """fn's output and its gradients in q, k and v, from float64 inputs cast to dtype."""
    q, k, v = (t.detach().clone().to(dtype).requires_grad_() for t in (q, k, v))
    out = fn(q, k, v)
    out.backward(dout.to(dtype))
    return {"out": out.detach(), "dq": q.grad, "dk": k.grad, "dv": v.grad}


def compare(got, want):
    return {
        "rel": {t: rel_err(got[t], want[t]) for t in TENSORS},
        "max_abs": {t: max_abs(got[t], want[t]) for t in TENSORS},
    }


def sdpa_choice(q, k, v, mask):
    """The backend train.py's SDPA call gets by default, for these inputs: (B, T, H, D)."""
    try:
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return SDPBackend(torch._fused_sdp_choice(q, k, v, mask, 0.0, mask is None)).name
    except Exception as e:  # noqa: BLE001 - a private API: its failure is a result, not a crash
        return error(e)


def attention_errors(code, B, T, H, D, lefts, device, precision, compile, backends):
    """For each window and regime, each implementation's errors against float64; and SDPA's default
    backend for each window."""
    dtype = torch.bfloat16 if precision is Precision.BF16 else torch.float32
    attention = torch.compile(code["attention"], dynamic=False) if compile else code["attention"]
    results, defaults = {}, {}
    for left in lefts:
        masks = {impl: code["window_mask"](impl, left, T, device) for impl in ("flex", "sdpa")}
        for seed, regime in enumerate(REGIMES):
            q, k, v, dout = attention_inputs((B, T, H, D), regime, device, seed)
            want = fwd_bwd(lambda q, k, v: reference_attention(q, k, v, left), q, k, v, dout, torch.float64)
            impls = {}
            for impl in ("flex", "sdpa"):
                def trained(q, k, v, impl=impl):  # as train.py's compiled model calls it
                    return attention(q, k, v, impl, masks[impl])
                impls[impl] = trained
            for backend in backends:
                def forced(q, k, v, backend=backend):
                    with sdpa_kernel([backend]):
                        return code["attention"](q, k, v, "sdpa", masks["sdpa"])
                impls[f"sdpa_{backend.name.lower()}"] = forced
            for name, fn in impls.items():
                try:
                    results[f"{left}/{regime}/{name}"] = compare(fwd_bwd(fn, q, k, v, dout, dtype), want)
                except Exception as e:  # noqa: BLE001 - a backend that refuses the inputs is a result
                    results[f"{left}/{regime}/{name}"] = {"error": error(e)}
            del want
        defaults[str(left)] = sdpa_choice(*(t.to(dtype) for t in (q, k, v)), masks["sdpa"])
    return {"results": results, "sdpa_default": defaults}


def build_model(code, config, device, state):
    """train.py's model: its init (seed 42, as train.py), then the given weights, or noise."""
    torch.manual_seed(42)
    with torch.device("meta"):
        model = code["GPT"](config)
    model.to_empty(device=device)
    model.init_weights()
    if state is not None:
        model.load_state_dict(state)
        return model
    g = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_((torch.randn(p.shape, device=device, generator=g) * NOISE).to(p.dtype))
    return model


def grads(model, x, y, autocast, compile):
    fn = torch.compile(model, dynamic=False) if compile else model
    ctx = torch.amp.autocast(device_type=x.device.type, dtype=autocast) if autocast else contextlib.nullcontext()
    with ctx:
        loss = fn(x, y)
    loss.backward()
    return float(loss.detach()), {n: p.grad.detach().double() for n, p in model.named_parameters()}


def model_errors(code, config, x, y, state, precision, compile):
    """One backward of the model with Flex and with SDPA, against a float64 copy on SDPA's math
    backend; each parameter's relative gradient error."""
    device, T = x.device, x.size(1)
    base = build_model(code, config, device, state)
    reference = copy.deepcopy(base).double()
    reference.set_attention("sdpa", T, device)
    with sdpa_kernel([SDPBackend.MATH]):
        ref_loss, ref_grads = grads(reference, x, y, None, compile=False)
    del reference
    out = {"loss": {"reference": ref_loss}}
    per_param = {n: {} for n in ref_grads}
    for impl in ("flex", "sdpa"):
        try:
            model = copy.deepcopy(base)
            if precision is Precision.FLOAT32:
                model.float()
            model.set_attention(impl, T, device)
            loss, g = grads(model, x, y, torch.bfloat16 if precision is Precision.BF16 else None, compile)
            del model
            out["loss"][impl] = loss
            errs = {n: rel_err(g[n], ref_grads[n]) for n in ref_grads if ref_grads[n].norm() > 0}
            for n, e in errs.items():
                per_param[n][impl] = e
            worst = max(errs, key=errs.get)
            out[impl] = {"median": statistics.median(errs.values()), "max": errs[worst], "worst": worst}
        except Exception as e:  # noqa: BLE001 - one implementation failing still reports the other
            out[impl] = {"error": error(e)}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out["per_param"] = per_param
    return out


def summary(result):
    """A few lines for a human reading the log, before the JSON."""
    lines = []
    for key, r in result.get("attention", {}).get("results", {}).items():
        if "error" in r:
            lines.append(f"attention {key}: {r['error'][:200]}")
        else:
            lines.append(f"attention {key}: " + " ".join(f"{t} {r['rel'][t]:.2e}" for t in TENSORS))
    m = result.get("model", {})
    for impl in ("flex", "sdpa"):
        if impl in m:
            r = m[impl]
            lines.append(f"model {impl}: " + (r["error"][:200] if "error" in r else
                         f"loss {m['loss'][impl]:.6f} (reference {m['loss']['reference']:.6f}), "
                         f"grad rel err median {r['median']:.2e}, max {r['max']:.2e} ({r['worst']})"))
    return "\n".join(lines)


def main(weights):
    t0 = time.time()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")  # train.py's
    code = train_model_code()
    tokenizer = code["Tokenizer"].from_directory()
    depth = 8  # upstream's DEPTH, sized by train.py's build_model_config
    dim = (depth * 64 + 127) // 128 * 128
    config = code["GPTConfig"](
        sequence_len=code["MAX_SEQ_LEN"], vocab_size=tokenizer.get_vocab_size(), n_layer=depth,
        n_head=dim // 128, n_kv_head=dim // 128, n_embd=dim, window_pattern="SSSL",
    )
    result ={"weights": weights or f"init + noise {NOISE}", "device": torch.cuda.get_device_name(),
              "torch": torch.__version__}
    try:
        with torch.device("meta"):
            lefts = sorted({left for left, _ in code["GPT"](config).window_sizes})
        result["attention"] = attention_errors(
            code, 8, config.sequence_len, config.n_head, dim // config.n_head, lefts, device,
            Precision.BF16, compile=True, backends=GPU_BACKENDS,
        )
    except Exception as e:  # noqa: BLE001
        result["attention"] = {"error": error(e)}
    try:
        x, y, _ = next(code["make_dataloader"](tokenizer, 8, config.sequence_len, "train"))
        state = torch.load(weights, map_location=device, weights_only=True) if weights else None
        result["model"] = model_errors(code, config, x.clone(), y.clone(), state, Precision.BF16, compile=True)
    except Exception as e:  # noqa: BLE001
        result["model"] = {"error": error(e)}
    result["seconds"] = time.time() - t0
    print(summary(result), flush=True)
    print("SPIKE_GRAD " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
