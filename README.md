# Test Time RL Discover + Auto Research

![teaser](progress.png)

This repo is a focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that replaces the outer experiment loop with [TTT-Discover](https://github.com/test-time-training/discover).

The setup is:

- The **inner loop** is still AutoResearch: edit [`train.py`](train.py), run a fixed-budget training job, and measure `val_bpb`.
- The **outer loop** is TTT-Discover: the model proposes full replacements for `train.py`, sees the resulting metric, and is reinforced online from that reward.
- The reward is strictly the inner-loop outcome: `current_best_val_bpb - candidate_val_bpb`.

This fork keeps the original AutoResearch target and uses TTT-Discover as the policy improvement layer.

## Credits

This project builds on:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- [test-time-training/discover](https://github.com/test-time-training/discover)

The RL recipe stays with upstream `discover`. This repo provides the AutoResearch-specific environment, reward, runner, configs, and practical launch workflow.

## What This Repo Optimizes

The repo has two layers:

1. **Inner optimization target**
   - [`prepare.py`](prepare.py) downloads data and trains the tokenizer.
   - [`train.py`](train.py) is the only file the outer model edits.
   - `val_bpb` is the optimization metric. Lower is better.

2. **Outer TTT-Discover loop**
   - [`run_ttt_discover.py`](run_ttt_discover.py) launches the run.
   - [`ttt_autoresearch/`](ttt_autoresearch/) adapts AutoResearch to the `discover` environment interface.
   - Each candidate `train.py` is executed in an isolated workspace.
   - Reward is computed from the measured improvement over the current best state.

## Repository Layout

```text
prepare.py                  Fixed data prep and runtime utilities
train.py                    Inner training program edited by the outer model
program.md                  Human-authored research instructions/context
run_ttt_discover.py         Main TTT-Discover entrypoint
ttt_autoresearch/           Adapter layer for environment, reward, runner, config
configs/                    Practical preset YAML configs
tests/                      Smoke and unit coverage for the adapter
```

## How The RL Loop Works

At each outer-loop step:

1. TTT-Discover samples grouped candidate replacements for `train.py`.
2. Each candidate is evaluated by running a real AutoResearch training job.
3. The run logs are parsed for `val_bpb`.
4. Reward is computed from improvement over the current best state.
5. Upstream `discover` performs the online RL update.
6. If a candidate improves `val_bpb`, it becomes the new best `train.py`.

Important details:

- The **action** is the full replacement contents of `train.py`.
- The **reward** is the inner-loop metric outcome, not a heuristic about the patch text.
- `groups_per_step` controls how many rollout groups are sampled at each RL step.
- `samples_per_step` controls how many rollouts are sampled inside each group.
- `max_concurrent_evaluations` controls how many expensive inner `train.py` jobs may run at once.

## Quick Start

**Requirements**

- Linux
- NVIDIA GPUs
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

Install and prepare the base AutoResearch environment:

```bash
uv sync
uv run prepare.py
uv run train.py
```

Then launch the default practical TTT-Discover mode:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch.yaml
```

## Training Presets

This repo now ships with three practical presets instead of a paper-scale default.

### Small

File: [`configs/ttt_discover_autoresearch_small.yaml`](configs/ttt_discover_autoresearch_small.yaml)

- `groups_per_step: 2`
- `samples_per_step: 4`
- `max_steps: 12`
- total evaluations: `96`

Use this when:

- you want a quick sanity run
- you are testing a new model backend
- you are on a single GPU and want something that finishes in a reasonable time

### Medium

File: [`configs/ttt_discover_autoresearch_medium.yaml`](configs/ttt_discover_autoresearch_medium.yaml)

- `groups_per_step: 2`
- `samples_per_step: 8`
- `max_steps: 12`
- total evaluations: `192`

This is the **recommended main mode** for the repo.

It is also the checked-in default at [`configs/ttt_discover_autoresearch.yaml`](configs/ttt_discover_autoresearch.yaml).

### Large

File: [`configs/ttt_discover_autoresearch_large.yaml`](configs/ttt_discover_autoresearch_large.yaml)

- `groups_per_step: 2`
- `samples_per_step: 8`
- `max_steps: 20`
- total evaluations: `320`

Use this when the medium run is already stable and you want more policy updates without moving into paper-scale compute.

## Recommended Modes

For this fork, the most realistic settings are:

- **Small:** `2 x 4 x 12`
- **Medium:** `2 x 8 x 12`
- **Large:** `2 x 8 x 20`

These are intentionally sized around the practical AutoResearch regime, where each rollout is a real GPU training job. They keep grouped rollouts and online RL updates from TTT-Discover, but avoid the extreme compute profile of the original paper.

## Hardware Recommendation

If your goal is to push `val_bpb` seriously, the inner loop should run on **H100 80GB** class GPUs.

Why:

- [`train.py`](train.py) uses Hopper-specific FA3 kernels when available.
- [`program.md`](program.md) shows representative peak VRAM around `45 GB`.
- `A100 40GB` is therefore not viable for the intended setup.

Recommended inner-loop target:

- **Best cost/performance:** H100 PCIe 80GB
- **Best absolute performance:** H100 SXM 80GB

For these practical presets, I recommend:

- **Small (`2x4x12`)**: rent `8x H100 80GB`
- **Medium (`2x8x12`)**: rent `16x H100 80GB`
- **Large (`2x8x20`)**: rent `16x H100 80GB`

That gives one GPU per rollout in a step wave. If you rent fewer GPUs, the run still works, but each step is split into multiple waves and takes longer.

To run with rented GPUs, set:

```yaml
max_concurrent_evaluations: 16
gpu_devices: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
```

The runner pins each candidate subprocess to one configured `CUDA_VISIBLE_DEVICES` slot.

## Cost Model

There are two cost buckets:

1. **Inner-loop GPU rental**
   - pays for the real `train.py` runs
   - dominates total cost in this repo

2. **Outer-loop Tinker cost**
   - pays for prompt prefill, sampling, and RL training tokens
   - is much smaller than the inner-loop GPU cost here

### Cost Assumptions

The estimates below use:

- `Qwen/Qwen3.5-35B-A3B` on Tinker
- H100 PCIe 80GB at about `$2.86 / GPU / hour`
- about `325.9s` per inner rollout
- about `$0.020` Tinker cost per rollout as a practical midpoint for this repo

### Preset Cost Estimates

| Mode | Shape | Total evals | GPU rental | Tinker | Total |
|---|---:|---:|---:|---:|---:|
| Small | `2x4x12` | 96 | about `$25` | about `$1.9` | about `$27` |
| Medium | `2x8x12` | 192 | about `$50` | about `$3.8` | about `$54` |
| Large | `2x8x20` | 320 | about `$83` | about `$6.4` | about `$89` |

### Cost Distribution

For these realistic runs, the cost split is still roughly:

- **~92% GPU rental**
- **~8% Tinker**

That is the core difference between this repo and cheaper code-generation tasks: each rollout is a real training job.

## How I Recommend Running It

### Single GPU

Use the small preset, and keep evaluation serialized:

```yaml
groups_per_step: 2
samples_per_step: 4
max_steps: 12
max_concurrent_evaluations: 1
gpu_devices: null
```

This is the safest way to stay close to the original one-GPU AutoResearch style while still using the TTT-Discover framework.

### Practical Rented Run

Use the medium preset:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch_medium.yaml
```

Recommended provisioning:

- `16x H100 PCIe 80GB`
- `max_concurrent_evaluations: 16`
- `gpu_devices` set to the visible devices on the host

This is the main mode I recommend if your goal is to beat the baseline without exploding compute.

### Larger Budget Run

Use the large preset:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch_large.yaml
```

This keeps the same grouped structure as medium, but increases the number of RL updates from `12` to `20`.

## Model and Renderer Configuration

The model is configurable, but the prompt and response format must match a supported renderer.

Known-good renderer values:

- `qwen3`
- `qwen3_instruct`
- `gpt_oss_no_sysprompt`
- `gpt_oss_low_reasoning`
- `gpt_oss_medium_reasoning`
- `gpt_oss_high_reasoning`

Examples:

```yaml
model_name: Qwen/Qwen3.5-35B-A3B
renderer_name: qwen3
```

```yaml
model_name: openai/gpt-oss-120b
renderer_name: gpt_oss_high_reasoning
```

If you use an unknown model family, set `renderer_name` explicitly. The config fails fast if it cannot infer a compatible renderer.

## Output Artifacts

Each run writes artifacts under `runs/<timestamp>/`:

- `baseline.json`
- `resolved_config.json`
- `history.jsonl`
- `best/train.py`
- `best/metrics.json`
- `candidates/`
- `discover_log/`

## Plain AutoResearch Mode Still Works

This fork does not remove the original AutoResearch workflow. You can still use it directly:

```bash
uv run prepare.py
uv run train.py
```

The TTT-Discover path is an additional outer loop, not a replacement for the inner codebase.

## Current Readiness

What is tested locally:

- config loading and override behavior
- reward mapping
- candidate parsing
- environment prompt and state flow
- CLI wiring into upstream `discover`
- concurrency gating for inner evaluations

What is still environment-dependent:

- a true end-to-end production run on the target Linux/CUDA machine
- provider-specific model serving details
- long-run throughput and stability on rented multi-GPU hardware

So the repo is structurally ready for the intended setup, but final operational confidence still comes from a real GPU run on target hardware.

## License

MIT
