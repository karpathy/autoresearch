"""
Spike (pi-autoresearch-serverless, plan 17a), on the workstation, before anything is spent: a small
model of train.py's own code on the CPU (depth 2, so one short window and one long, at T 2048).

0. train.py's budget: N steps follow upstream's schedule exactly, at the step time that ends a
   time budget after N steps.
1. FlexAttention and SDPA give the same logits, eagerly, and compiled if torch.compile works here.
2. The export (spike/export.py) and spike/onnx_check.py, on onnxruntime's CPU provider, with its
   full eval (the scorer's nats head) on the export's rows.

    .venv/bin/python spike/local_check.py [OUT_DIR] [DEPTH]

DEPTH defaults to 2; upstream's model is depth 8, sized by train.py's own rule.

Exits non-zero if a comparison is off.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
T, V = 2048, 8192


def train_model_code():
    """train.py up to its hyperparameters: the imports, the attention, the model and the optimizer,
    with nothing run."""
    with open(os.path.join(ROOT, "train.py")) as f:
        src = f.read()
    namespace = {"__name__": "train_model_code"}
    exec(compile(src[: src.index("# Hyperparameters (edit these directly")], "train.py", "exec"), namespace)
    return namespace


def max_diff(a, b):
    return float((a - b).abs().max())


def schedule(code, budget, dt):
    """train.py's loop, without the training: the progress before each step, and the steps run, at a
    constant step time dt."""
    step, seconds, progress = 0, 0.0, []
    while True:
        progress.append(code["budget_progress"](budget, step, seconds))
        if step > 10:
            seconds += dt
        step += 1
        if code["budget_finished"](budget, step, seconds):
            return progress, step


def budget_failures(code):
    """For every N in [12, 1500] and integer step times (whole seconds keep the sums exact): N steps
    and (N - 11) * dt seconds give the same progress at every step, and both stop after N steps."""
    failures = []
    for n in range(12, 1501):
        dt = 1 + n % 4
        by_steps = schedule(code, code["Steps"](n), dt)
        by_seconds = schedule(code, code["Seconds"]((n - 11) * dt), dt)
        if by_steps != by_seconds or by_steps[1] != n:
            failures.append(f"budget: {n} steps at dt {dt}: steps {by_steps[1]}, seconds {by_seconds[1]}")
    for env in ({"AR_STEPS": "953", "AR_TIME_BUDGET": "300"}, {"AR_STEPS": "11"}):
        try:
            code["budget_from"](env)
            failures.append(f"budget: {env} accepted")
        except AssertionError:
            pass
    if code["budget_from"]({}) != code["Seconds"](code["TIME_BUDGET"]):
        failures.append("budget: upstream's default is not prepare.py's TIME_BUDGET")
    return failures[:10]


def main(out_dir, depth):
    code = train_model_code()
    torch.manual_seed(0)
    dim = (depth * 64 + 127) // 128 * 128  # train.py's build_model_config: ASPECT_RATIO 64, HEAD_DIM 128
    config = code["GPTConfig"](sequence_len=T, vocab_size=V, n_layer=depth, n_head=dim // 128, n_kv_head=dim // 128, n_embd=dim, window_pattern="SSSL")
    with torch.device("meta"):
        model = code["GPT"](config)
    model.to_empty(device="cpu")
    model.init_weights()
    model.float()
    with torch.no_grad():  # init zeroes the projections, which would hide the attention
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    assert model.window_sizes[0] == (T // 2, 0) and model.window_sizes[-1] == (T, 0), model.window_sizes

    failures = budget_failures(code)
    print(f"budget: {'ok' if not failures else failures}", flush=True)
    report = {}
    idx = torch.randint(0, V, (2, T))
    with torch.no_grad():
        model.set_attention("sdpa", T, torch.device("cpu"))
        want = model(idx)
        model.set_attention("flex", T, torch.device("cpu"))
        report["flex_eager_vs_sdpa"] = max_diff(model(idx), want)
        try:
            compiled = torch.compile(model, dynamic=False)
            report["flex_compiled_vs_sdpa"] = max_diff(compiled(idx), want)
        except Exception as e:  # noqa: BLE001 - CPU compiling may not work on the workstation
            report["flex_compiled_error"] = f"{type(e).__name__}: {e}"[:2000]
    for key in ("flex_eager_vs_sdpa", "flex_compiled_vs_sdpa"):
        if key in report and report[key] > 1e-4:
            failures.append(f"{key} {report[key]:.3g} > 1e-4")
    print("attention: " + json.dumps(report), flush=True)

    from spike.export import export

    x = torch.randint(0, V, (8, T))
    y = torch.randint(0, V, (8, T))
    token_bytes = torch.randint(1, 9, (V,))
    token_bytes[-4:] = 0  # the special tokens
    info = export(model, x, y, token_bytes, out_dir)
    val_dir = os.path.join(out_dir, "val")  # as spike/val_rows.py writes it, from the export's rows
    os.makedirs(val_dir, exist_ok=True)
    np.save(os.path.join(val_dir, "x.npy"), x.numpy().astype(np.int32))
    np.save(os.path.join(val_dir, "y.npy"), y.numpy().astype(np.int32))
    np.save(os.path.join(val_dir, "token_bytes.npy"), token_bytes.numpy().astype(np.int64))

    proc = subprocess.run(
        [sys.executable, "-I", os.path.join(ROOT, "spike", "onnx_check.py"), out_dir, val_dir],
        capture_output=True, text=True, check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr[-4000:])
    line = next((l for l in proc.stdout.splitlines() if l.startswith("SPIKE_ONNX ")), None)
    if line is None:
        failures.append("onnx_check.py printed no SPIKE_ONNX")
    else:
        result = json.loads(line[len("SPIKE_ONNX ") :])
        if result["graph"]["domains"] != ["ai.onnx"]:
            failures.append(f"domains {result['graph']['domains']}")
        for c in result["checks"]:
            if "error" in c:
                failures.append(f"{c['name']}: {c['error']}")
                continue
            if c["logits_max_abs_diff"] > 1e-4:
                failures.append(f"{c['name']} logits differ by {c['logits_max_abs_diff']:.3g}")
            if abs(c["bpb_diff"]) > 1e-5:
                failures.append(f"{c['name']} bpb differs by {c['bpb_diff']:.3g}")
            if c["splice_prefix_max_abs_diff"] > 1e-5:
                failures.append(f"{c['name']} splice changed the prefix by {c['splice_prefix_max_abs_diff']:.3g}")
        full = result.get("full_eval", {"error": "missing"})
        if "error" in full or "skipped" in full:
            failures.append(f"full_eval: {full.get('error') or full.get('skipped')}")
        elif not full["finite"] or abs(full["bpb"] - info["bpb"]) > 1e-6:
            failures.append(f"full_eval bpb {full['bpb']} against the export's {info['bpb']}")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("local check passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="spike-17a-"), int(sys.argv[2]) if len(sys.argv) > 2 else 2)
