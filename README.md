# Test Time RL Discover + Auto Research

![teaser](progress.png)

This repo is a focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) that replaces the outer experiment loop with [TTT-Discover](https://github.com/test-time-training/discover).

The checked-in defaults now support two primary outer-loop modes:

- **Kimi mode:** Tinker + `moonshotai/Kimi-K2.5` with `qwen3`
- **GPT-OSS mode:** Tinker + `openai/gpt-oss-120b` with `gpt_oss_high_reasoning`
- **Inner loop:** one Hyperbolic on-demand node with `8x H100`
- **Main preset:** `2 groups x 8 rollouts x 12 steps`
- **Launch mode:** detached remote controller so the run survives your local machine disconnecting

The core objective stays the same as the original AutoResearch repo: improve [`train.py`](train.py) to lower `val_bpb`.

## Credits

This project builds on:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- [test-time-training/discover](https://github.com/test-time-training/discover)

The RL recipe stays with upstream `discover`. This repo provides the AutoResearch-specific environment, reward, runner, Hyperbolic execution/launch backend, and practical launch workflow.

## How The System Works

There are two loops:

1. **Inner loop**
   - [`train.py`](train.py) is the only file the outer model edits.
   - Every rollout runs a real fixed-budget AutoResearch training job.
   - The score is `val_bpb`, and lower is better.

2. **Outer loop**
   - TTT-Discover samples strict SEARCH/REPLACE patches against the current working `train.py`.
   - Each candidate is evaluated by the inner loop.
   - Reward is a direct transformed task score: `1 / (1e-8 + val_bpb)`.
   - Failed or invalid candidates receive `0.0` reward.
   - Upstream `discover` updates the outer model online.

Before any H100 evaluation, every candidate now goes through a CPU-side preflight:

- patch-only parsing
- AST / `py_compile` validation
- batch-divisibility checks
- final `val_bpb` summary preservation
- `forward(... reduction=...)` compatibility checks

Only preflight-passing candidates reach the GPU evaluator.

The checked-in workflow launches the entire controller onto the Hyperbolic node itself. After launch, the outer loop and all inner evaluations keep running on the remote machine even if your laptop sleeps, disconnects, or closes.

## What “Unattended” Means Here

This repo is designed so that:

- you start the run once from your local machine
- the repo bootstraps a Hyperbolic `8x H100` node over SSH
- it uploads a remote config and launches `run_ttt_discover.py` under `nohup` on that machine
- the remote controller runs the outer loop and the inner `train.py` evaluations locally on the node’s 8 GPUs
- the run continues until the configured `groups_per_step x samples_per_step x max_steps` budget is completed

The important implication is that the remote node is now the source of truth for checkpoints and artifacts. Your local machine is only used to kick the run off.

The launcher also starts a local background mirror process by default:

- while your laptop is online, it continuously pulls the full remote run directory back to `runs/<name>/mirror/`
- when the remote controller exits, it performs one final sync
- if your laptop is offline, the mirror pauses implicitly and the remote node remains the source of truth

## Primary Model Modes

### Kimi K2.5

The default config at [`configs/ttt_discover_autoresearch.yaml`](configs/ttt_discover_autoresearch.yaml) uses:

- `model_name: moonshotai/Kimi-K2.5`
- `renderer_name: qwen3`
- `groups_per_step: 2`
- `samples_per_step: 8`
- `max_steps: 12`

### GPT-OSS 120B

The shipped medium/large presets use:

- `model_name: openai/gpt-oss-120b`
- `renderer_name: gpt_oss_high_reasoning`

### Shared Practical Defaults

Both primary model modes use the same practical unattended search shape:

- `target_val_bpb: 0.85`
- `execution_backend: hyperbolic`
- `groups_per_step: 2`
- `samples_per_step: 8`
- `max_steps: 12`
- `max_concurrent_evaluations: 8`
- `gpu_devices: ["0", "1", "2", "3", "4", "5", "6", "7"]`
- `hyperbolic_detached_controller: true`

That means:

- `16` rollouts per outer step
- `12` outer RL updates
- `192` rollout evaluations total
- `1` extra baseline run before RL starts
- `193` total inner jobs
- evaluations run in two waves per outer step on a single `8x H100` node

## Presets

The repo ships with three practical presets:

### Small

File: [`configs/ttt_discover_autoresearch_small.yaml`](configs/ttt_discover_autoresearch_small.yaml)

- `2 x 4 x 12`
- `96` RL rollouts
- `8` concurrent GPU slots on the Hyperbolic node

### Medium

File: [`configs/ttt_discover_autoresearch_medium.yaml`](configs/ttt_discover_autoresearch_medium.yaml)

- `2 x 8 x 12`
- `192` RL rollouts
- `8` concurrent GPU slots on the Hyperbolic node

This is the recommended main mode and matches the default config.

### Large

File: [`configs/ttt_discover_autoresearch_large.yaml`](configs/ttt_discover_autoresearch_large.yaml)

- `2 x 8 x 20`
- `320` RL rollouts
- `8` concurrent GPU slots on the Hyperbolic node

Use this only after the medium run is stable.

## Hyperbolic Backend

The inner-loop executor now supports two backends:

- `local`
- `hyperbolic`

The `hyperbolic` backend does two different things depending on where the controller is running:

1. **Detached controller launch**
   - connects to your Hyperbolic node over SSH
   - uploads the repo snapshot
   - runs `uv sync`
   - runs `uv run prepare.py --num-shards 10`
   - writes a remote config with `execution_backend: local`
   - starts the full TTT controller under `nohup`

2. **Inner evaluations on the remote node**
   - the remote controller pins rollouts to `CUDA_VISIBLE_DEVICES=0..7`
   - each rollout runs in an isolated workspace
   - logs, metrics, and manifests are saved under the remote `run_dir`

## Prerequisites

You need:

- Linux or macOS for the launch machine
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- a Tinker-enabled account for the outer loop
- a Hyperbolic account with:
  - one running on-demand `8x H100` node
  - an SSH public key registered on the account
  - the node’s SSH host / IP
  - enough credits to keep the machine alive for the full run

Environment:

```bash
export OPENAI_API_KEY=...
```

Or equivalently:

```bash
export TINKER_API_KEY=...
export OPENAI_API_KEY="$TINKER_API_KEY"
```

Optional for Kimi runs on fresh nodes:

```bash
export HF_TOKEN=...
```

`HF_TOKEN` is not required for correctness. It only reduces Hugging Face rate-limit and cold-start friction when the Kimi tokenizer/custom code is downloaded on a fresh machine.

## Quick Start

Launch the default unattended medium run:

```bash
uv sync
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch.yaml
```

Or explicitly choose the medium preset:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch_medium.yaml
```

When the launch succeeds, the script prints the remote run directory and remote controller log path, and it writes the same metadata to `hyperbolic_launch.json` in the local launch directory.

The launcher also refuses to start if another detached AutoResearch controller or `train.py` process is already active on the same Hyperbolic node. This prevents overlapping runs from silently OOMing each other.

It also starts a local mirror process and records:

- `local_mirror_dir`
- `local_mirror_log_path`
- `local_mirror_pid`

## Cost And Runtime Shape

For this repo, the expensive part is the inner loop. Each rollout is a real five-minute AutoResearch training job.

The repo’s own reference timing in [`program.md`](program.md) shows:

- `total_seconds: 325.9` per rollout

That means the default medium run has:

- `192` RL rollouts
- `1` baseline run
- `193` total inner runs
- about `17.47` total GPU-hours

### Example Medium Budget

With a single `8x H100` node, the total GPU-hours are the same `17.47`, but they are spread across the 8 GPUs on that machine.

If your node price is `P` dollars per GPU-hour, the GPU line item is approximately:

- `17.47 x P`

If Hyperbolic bills a full `8x H100` node at a flat hourly rate `N`, the same run costs approximately:

- `(2.5 to 3.5 hours) x N`

Tinker is the smaller cost bucket here. The exact amount depends on current Tinker pricing and token usage, but for this repo it is materially smaller than the H100 rental line item.

### Wall Clock

Approximate medium-run wall clock on one `8x H100` node:

- about `2.5-3.5h`

## Model And Renderer

One first-class outer-loop mode is:

```yaml
model_name: moonshotai/Kimi-K2.5
renderer_name: qwen3
```

The other first-class outer-loop mode is:

```yaml
model_name: openai/gpt-oss-120b
renderer_name: gpt_oss_high_reasoning
```

This dual support is intentional:

- Kimi K2.5 and GPT-OSS-120B are both treated as primary outer-loop modes
- both are explicitly supported by the renderer mapping and tokenizer compatibility patches
- both use the same strict patch-only rollout/evaluation pipeline

Other models still work, but if the model family is not recognized automatically you must set `renderer_name` explicitly.

## Important Config Knobs

The main knobs for unattended Hyperbolic execution are:

- `execution_backend`
  - use `hyperbolic` to launch a detached remote controller
  - use `local` for direct local GPU execution
- `max_concurrent_evaluations`
  - number of simultaneous inner evaluations on the remote node
  - number of local simultaneous inner runs for `local`
- `hyperbolic_ssh_host`
  - SSH host or IP for your Hyperbolic node
- `hyperbolic_ssh_user`
  - defaults to `ubuntu`
- `hyperbolic_ssh_private_key_path`
  - optional explicit SSH private key path
- `hyperbolic_detached_controller`
  - default `true`; launches the whole controller remotely under `nohup`
- `gpu_devices`
  - defaults to all eight GPUs on the remote node

## Fixed Prompt Target

The checked-in presets use:

```yaml
target_val_bpb: 0.85
```

This is a prompt-side benchmark target, not a reward cap.

- the model is shown the current starting state and the gap to `0.85`
- the RL reward still comes from the actual achieved `val_bpb`
- if a rollout beats `0.85`, it is still rewarded more for going even lower

This mirrors how upstream `discover` environments use fixed benchmark targets in the prompt while computing reward from the evaluated task score.

## Repository Layout

```text
prepare.py                  Fixed data prep and runtime utilities
train.py                    Inner training program edited by the outer model
program.md                  Human-authored research instructions/context
run_ttt_discover.py         Main TTT-Discover entrypoint
ttt_autoresearch/           Adapter layer for environment, reward, runner, Hyperbolic, config
configs/                    Practical preset YAML configs
tests/                      Smoke and unit coverage for the adapter
```

## Output Artifacts

Each run writes artifacts under `runs/<timestamp>/`:

- `baseline.json`
- `resolved_config.json`
- `history.jsonl`
- `best/train.py`
- `best/metrics.json`
- `candidates/`
- `discover_log/`
- `hyperbolic_launch.json`

`hyperbolic_launch.json` records the remote launch metadata for the detached Hyperbolic controller, including the remote run directory and remote log path.

The important resume/checkpoint files are:

- `baseline.json`
  - cached baseline result; if it already exists, the CLI reuses it instead of rerunning baseline
- `baseline/train.py`
  - stored baseline script snapshot for reproducible resume
- `best/train.py`
  - best discovered script so far
- `best/metrics.json`
  - best discovered `val_bpb` plus artifact paths
- `history.jsonl`
  - append-only candidate evaluation log
- `candidates/<step>_<id>/train.py`
  - exact candidate script evaluated for that rollout
- `candidates/<step>_<id>/stdout.log`
  - raw stdout from the inner AutoResearch run
- `candidates/<step>_<id>/stderr.log`
  - raw stderr from the inner AutoResearch run
- `candidates/<step>_<id>/metrics.json`
  - parsed metrics sidecar for that rollout
- `candidates/<step>_<id>/rollout_manifest.json`
  - self-contained rollout record with the starting state, candidate payload, evaluation result, reward, and promotion outcome
- `candidates/<step>_<id>/prompt.txt`
  - exact prompt sent to the outer model for that rollout
- `candidates/<step>_<id>/response.txt`
  - raw model response for that rollout
- invalid or malformed model outputs are also persisted under `candidates/` with a `rollout_manifest.json`, `metrics.json`, and raw `response.txt`
- `discover_log/checkpoints.jsonl`
  - upstream TTT-Discover checkpoint index
- `discover_log/`
  - LoRA/training state and sampler checkpoints used for resume
- `hyperbolic_launch.json`
  - local launch metadata, including the remote run dir and local mirror info
- `mirror/`
  - best-effort local mirror of the remote run directory while your laptop is reachable

## Resuming A Stopped Run

To continue a stopped run, reuse the same `run_dir`.

Example:

```yaml
run_dir: runs/my-main-run
```

Then rerun the same command:

```bash
uv run python run_ttt_discover.py --config configs/ttt_discover_autoresearch.yaml
```

Resume behavior:

- the CLI reuses `baseline.json` and `baseline/train.py` if they already exist
- upstream `discover` reloads the latest training checkpoint from `discover_log/checkpoints.jsonl`
- upstream sampler state is reloaded from the matching sampler checkpoint step
- every evaluated rollout remains on disk under `candidates/`, so prompt/response/result provenance is preserved even if the run is interrupted

If you stopped at `12` steps and want to continue farther, increase `max_steps` above the completed count before rerunning.

For example, to continue a finished medium run out to `20` steps:

- keep the same `run_dir`
- change `max_steps: 20`
- rerun the command

Important resume rule:

- resume with the same code revision, model, renderer, rollout structure, and run directory whenever possible
- changing those mid-run is not guaranteed to be meaningful or stable

## Local Mode Still Exists

If you want to run without Hyperbolic, set:

```yaml
execution_backend: local
```

and configure `gpu_devices` if you want more than one local concurrent evaluation.

## Current Readiness

What is covered in tests:

- config loading and normalization
- reward mapping
- candidate parsing
- CLI wiring into upstream `discover`
- local concurrency gating
- Hyperbolic detached launch wiring
- Hyperbolic runner/backend cleanup behavior
- malformed candidate persistence and rollout manifests

What is still operationally environment-dependent:

- SSH access from your launch machine to the Hyperbolic node
- a working Hyperbolic `8x H100` node with enough disk and credits
- real Tinker credentials and provider setup
- long-run stability on your specific Hyperbolic account and node

So the repo is structurally ready for unattended Tinker + Hyperbolic operation, but the final production proof is still a real run on your account.

## License

MIT
