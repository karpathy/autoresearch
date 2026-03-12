# GPU Dispatcher for stdio_bus

Distributes train.py experiments across multiple GPUs via stdio_bus.

## Quick Start

```bash
# With NVIDIA GPUs
docker run -d --gpus all -p 9000:9000 \
  -v $(pwd)/stdio_bus/dispatcher.py:/stdio_bus/dispatcher.py:ro \
  -v $(pwd)/stdio_bus/stdio-bus-config.json:/stdio_bus/stdio-bus-config.json:ro \
  -v $(pwd)/pyproject.toml:/stdio_bus/pyproject.toml:ro \
  -v $(pwd)/train.py:/stdio_bus/train.py:ro \
  -v $(pwd)/prepare.py:/stdio_bus/prepare.py:ro \
  stdiobus/stdiobus:python312 \
  --config /stdio_bus/stdio-bus-config.json --tcp 0.0.0.0:9000

# Without GPU (testing only)
docker run -d -p 9000:9000 \
  -v $(pwd)/stdio_bus/dispatcher.py:/stdio_bus/dispatcher.py:ro \
  -v $(pwd)/stdio_bus/stdio-bus-config.json:/stdio_bus/stdio-bus-config.json:ro \
  -v $(pwd)/pyproject.toml:/stdio_bus/pyproject.toml:ro \
  stdiobus/stdiobus:python312 \
  --config /stdio_bus/stdio-bus-config.json --tcp 0.0.0.0:9000
```

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
| `DEVICE_BACKEND` | auto-detect | `cuda` or `mps` |
