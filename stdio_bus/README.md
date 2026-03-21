# stdio_bus Workers

Workers for [stdio Bus](https://stdiobus.com) kernel — a deterministic C runtime for AI agent message routing.

## What is stdio Bus?

stdio Bus is a transport layer that routes JSON-RPC messages between clients and worker processes via NDJSON. See [stdiobus.com](https://stdiobus.com) for the full specification.

This directory contains **workers** — Python processes that stdio Bus spawns and routes messages to.

## Workers

| Worker | Config | Description |
|--------|--------|-------------|
| `dispatcher.py` | `configs/dispatcher.json` | GPU experiment dispatcher for `train.py` |
| `workers/benchmark.py` | `configs/benchmark.json` | MPS vs MLX benchmark runner |

## Quick Start

### 1. Get stdio Bus kernel

```bash
# Download binary for your platform
curl -L -o stdio_bus https://github.com/stdiobus/stdiobus/raw/main/releases/v2.0.3/darwin-arm64/stdio_bus
chmod +x stdio_bus
```

Available platforms: `darwin-amd64`, `darwin-arm64`, `linux-amd64`, `linux-arm64`, `linux-armv7`, `linux-musl-amd64`, `linux-musl-arm64`

### 2. Install Python dependencies

```bash
uv sync
```

### 3. Run

```bash
# stdio mode (single client)
echo '{"jsonrpc":"2.0","id":1,"method":"status","params":{}}' | ./stdio_bus -c configs/dispatcher.json

# TCP mode (multiple clients)
./stdio_bus -c configs/dispatcher.json --tcp 127.0.0.1:9000
```

## Docker

```bash
docker run -d --gpus all -p 9000:9000 \
  -v $(pwd)/stdio_bus/dispatcher.py:/stdio_bus/dispatcher.py:ro \
  -v $(pwd)/stdio_bus/configs/dispatcher.json:/stdio_bus/configs/dispatcher.json.json:ro \
  -v $(pwd)/pyproject.toml:/stdio_bus/pyproject.toml:ro \
  -v $(pwd)/train.py:/stdio_bus/train.py:ro \
  -v $(pwd)/prepare.py:/stdio_bus/prepare.py:ro \
  -v ~/.cache/autoresearch:/root/.cache/autoresearch:ro \
  stdiobus/stdiobus:python312 \
  --config /stdio_bus/configs/dispatcher.json --stdio
```

With TCP:

```bash
docker run -d --gpus all -p 9000:9000 \
  -v $(pwd)/stdio_bus/dispatcher.py:/stdio_bus/dispatcher.py:ro \
  -v $(pwd)/stdio_bus/configs/dispatcher.json:/stdio_bus/configs/dispatcher.json.json:ro \
  -v $(pwd)/pyproject.toml:/stdio_bus/pyproject.toml:ro \
  -v $(pwd)/train.py:/stdio_bus/train.py:ro \
  -v $(pwd)/prepare.py:/stdio_bus/prepare.py:ro \
  -v ~/.cache/autoresearch:/root/.cache/autoresearch:ro \
  stdiobus/stdiobus:python312 \
  --config /stdio_bus/configs/dispatcher.json --tcp 0.0.0.0:9000
```

Tags: `latest`, `v2.0.3`

Platforms: `linux/amd64`, `linux/arm64`, `linux/arm/v7`

## Config Format

```json
{
  "pools": [
    {
      "id": "dispatcher",
      "command": "uv",
      "args": ["run", "stdio_bus/dispatcher.py"],
      "instances": 1
    }
  ]
}
```

## GPU Dispatcher API

### status

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"status","params":{}}' | ./stdio_bus -c configs/dispatcher.json
```

### run

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"run","params":{"agentId":"agent-1"}}' | ./stdio_bus -c configs/dispatcher.json
```

### history / sync

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"history","params":{"limit":10}}' | ./stdio_bus -c configs/dispatcher.json
echo '{"jsonrpc":"2.0","id":1,"method":"sync","params":{}}' | ./stdio_bus -c configs/dispatcher.json
```

## Environment

| Variable | Description |
|----------|-------------|
| `SWARM_GPU_IDS` | Override GPU detection (e.g., `0,1,2,3`) |


## Test

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"status","params":{},"sessionId":"s1"}' | nc -i 1 localhost 9000
```

Response:
```json
{"jsonrpc":"2.0","id":1,"sessionId":"s1","result":{"total":0,"active":0,"best":null,"gpus":{"ids":[0],"busy":[]}}}
```

- `total` — completed experiments
- `active` — running experiments
- `best` — best val_bpb achieved
- `gpus.ids` — available GPU IDs
- `gpus.busy` — currently busy GPU IDs

## JSON-RPC Methods

| Method | Description |
|--------|-------------|
| `status` | GPU pool status |
| `sync` | Full state sync with last 50 results |
| `history` | Experiment history (params: `limit`) |
| `run` | Run experiment (params: `agentId`, `branch`, `blocking`) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SWARM_GPU_IDS` | auto-detect | Comma-separated GPU IDs (e.g., "0,1,2") |

## Standalone Testing

Test workers directly without kernel:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"status","params":{}}' | uv run stdio_bus/dispatcher.py
```

## Resources

- [stdio Bus Docs](https://stdiobus.com)
- [Kernel Source](https://github.com/stdiobus/stdiobus)
- [Workers Registry](https://github.com/stdiobus/workers-registry)
