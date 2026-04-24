# autoresearch — nanoGPT-based autonomous ML research loop (research prototype)

[![PyPI](https://img.shields.io/pypi/v/autoresearch-nano.svg)](https://pypi.org/project/autoresearch-nano/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)

> **What this is:** a fork of Karpathy's
> [nanoGPT](https://github.com/karpathy/nanoGPT) / nanochat plus an autonomous
> research-orchestration layer driven by a Claude Code `program.md` playbook.
> The goal is to have an LLM agent iterate on `train.py` inside a fixed time
> budget and hunt for lower `val_bpb`.
>
> **What this is NOT (v1.0.0):** a finished Tribunal-evaluated, Silicon-fused
> research stack. Read the [honest-scope section](#honest-scope-for-v100)
> before depending on any specific feature.

---

## Honest scope for v1.0.0

This release ships a **clean installable skeleton** around the parts that
really work, plus honest documentation of what does not.

| Component | Status | Notes |
|---|---|---|
| nanoGPT-style training core (`train.py`) | **Works** | Requires H100-class GPU and `flash-attn3` kernels for real runs. |
| CPU-friendly toy training (`autoresearch toy-train`) | **Works** | No GPU required. Useful for CI and smoke-tests. |
| CLI (`info`, `config`, `toy-train`) | **Works** | No network, no GPU. |
| JSON metrics sidecar (`run_metrics.json`) | **Works** | Replaces brittle stdout-regex parsing. |
| P2PCLAW **Silicon** integration | **PARTIAL (inert)** | `silicon/grid_generator.py` is a placeholder. Do not rely on it. |
| **Tribunal** evaluation | **NOT IMPLEMENTED** | Stub only (`autoresearch.tribunal`), raises on use. |
| Railway P2PCLAW API | **OPTIONAL** | Core loop and tests do not need it. |

See [`ROADMAP.md`](ROADMAP.md) for the full gap list.

## Install

```bash
pip install autoresearch-nano
# or, with optional extras:
pip install "autoresearch-nano[full]"   # transformers, datasets, wandb
pip install "autoresearch-nano[dev]"    # pytest, build, twine
```

Requires Python **>= 3.10**. Core deps: `torch`, `numpy`, `requests`.

## Quickstart (CPU, no GPU, no network)

```bash
# 1. See environment + dependency status
autoresearch info

# 2. Show the default config
autoresearch config

# 3. Run a tiny nanoGPT training smoke-test (writes run_metrics.json)
autoresearch toy-train --steps 20 --verbose
```

You should see the loss drop across steps and a `run_metrics.json` written
to the current directory.

## Running the real nanoGPT training loop

The top-level `train.py` is unchanged from the nanochat-derived original and
**requires an NVIDIA GPU with flash-attn3 kernels** (H100 recommended; other
Hopper/Blackwell-class cards may work via `kernels-community/flash-attn3`).
CPU is not supported for `train.py` — use `autoresearch toy-train` for CPU.

```bash
uv run prepare.py      # prepares data under ~/.cache/autoresearch/
uv run train.py        # runs the 5-minute time-budgeted training loop
```

The autonomous research loop itself is orchestrated via Claude Code following
`program.md`.

## Structured metrics (preferred over stdout regex)

Earlier versions parsed training results out of stdout using regex against
the `---` block. That path is now a **fallback**. Primary path:

```python
from autoresearch.metrics import write_metrics, read_metrics

write_metrics("run_metrics.json", {"val_bpb": 0.9979, "num_steps": 953})
read_metrics("run_metrics.json")  # -> {"val_bpb": 0.9979, "num_steps": 953}
```

The legacy parser is still available as
`autoresearch.metrics.parse_stdout_summary(text)` for old run logs.

## Tribunal and Silicon

* `autoresearch.tribunal.evaluate(...)` raises `TribunalNotImplementedError`.
* `silicon/grid_generator.py` does not currently emit a usable grid.

Both are tracked in [`ROADMAP.md`](ROADMAP.md).

## Testing

```bash
pip install -e ".[dev]"
pytest
```

All tests are offline and CPU-only. CUDA-requiring cases are skip-marked.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

## Credits

* Andrej Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT) and nanochat (MIT).
* Francisco Angulo de Lafuente (Agnuxo1) — autoresearch orchestration layer, P2PCLAW integration work.
