# ResearchFS NanoChat benchmark

This directory is an optional, extraction-ready benchmark project. It is not
part of the `researchfs-core` wheel or the root development environment.
NanoChat training code, PyTorch/data dependencies, Modal orchestration,
benchmark reports, images, and fixture-specific agent code live here so the
ResearchFS product stays lightweight.

## Install

Run benchmark commands from this directory:

```bash
cd benchmarks/nanochat
uv sync
```

The benchmark project installs the local `researchfs-core` checkout plus its
own training dependencies. Optional tools remain separate:

```bash
uv sync --extra modal       # Modal runner
uv sync --extra analysis    # analysis.ipynb dependencies
```

Neither command changes the dependency surface of the root core project.

## Run

```bash
uv run python seed.py
uv run python prepare.py
uv run python train.py
uv run python compare_runs.py --list
uv run python run_comparison.py --max-experiments 3
```

Canonical comparisons require a supported NVIDIA GPU. CPU, MPS, and reduced
budgets are smoke tests only.

For Modal, install the optional extra and run:

```bash
uv run --extra modal modal run modal_loopgraph_benchmark.py
```

Fixture design and experiment guidance are under `docs/`; historical plots are
under `assets/`. The compressed NanoChat method seed is
`fixtures/seed.bundle.json` and contains no raw paper bodies.

## Test

The focused fixture regressions do not import the training runtime:

```bash
uv run pytest -q
uv run pytest -q test_runner.py
```

The root ResearchFS suite is intentionally independent and runs from the
repository root with `uv run pytest -q`.
