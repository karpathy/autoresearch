# Add stdio_bus Integration for Distributed GPU Training

## Summary

This PR introduces stdio_bus as the transport layer for distributing training experiments across multiple GPUs, replacing the HTTP-based agenthub approach with a high-performance local message bus.

## Motivation

The `agenthub` branch implements multi-agent coordination via HTTP REST API to a central hub server. While this works for geographically distributed agents, it introduces unnecessary overhead for the common case: multiple GPUs on a single machine or local cluster.

**agenthub limitations:**
- Network latency on every API call (register, push, fetch, post)
- Central server dependency — single point of failure
- HTTP overhead for simple request/response patterns
- Complex git bundle serialization for code sharing
- Requires external infrastructure (hub server)

## Solution: stdio_bus

stdio_bus is a deterministic C runtime that routes JSON-RPC messages between clients and worker processes via NDJSON over stdin/stdout. It provides:

**Architecture:**
```
Client (nc/curl) → TCP:9000 → stdio_bus kernel → dispatcher.py → GPU pool
```

**Key advantages over agenthub:**

| Aspect | agenthub | stdio_bus |
|--------|----------|-----------|
| Latency | HTTP round-trip (~10-100ms) | Local IPC (~0.1ms) |
| Dependencies | Hub server, network | Single Docker container |
| Protocol | REST + git bundles | JSON-RPC 2.0 over NDJSON |
| Failure mode | Hub down = all agents blocked | Local process restart |
| Scaling | Limited by hub capacity | Limited by local GPUs |

**Technical benefits:**
- Zero AI logic in transport — deterministic routing by session ID
- Single-threaded event loop (epoll/kqueue) — minimal resource usage
- Backpressure management — prevents memory exhaustion under load
- Protocol agnostic — same bus can route ACP, MCP, or custom protocols

## Implementation

### Files Added

```
stdio_bus/
├── dispatcher.py          # GPU pool manager (~80 lines)
├── stdio-bus-config.json  # Worker pool configuration
└── README.md              # Usage documentation
```

### dispatcher.py

Minimal Python worker that:
- Auto-detects available GPUs via `nvidia-smi` or `SWARM_GPU_IDS` env var
- Manages GPU allocation (acquire/release)
- Runs `train.py` experiments with `CUDA_VISIBLE_DEVICES` isolation
- Tracks experiment results (total, active, best val_bpb)

### JSON-RPC Methods

| Method | Description |
|--------|-------------|
| `status` | GPU pool state: available/busy GPUs, experiment counts, best result |
| `sync` | Full state sync with last 50 results |
| `history` | Experiment history with configurable limit |
| `run` | Execute experiment on available GPU (blocking or async) |

### Usage

```bash
# Start dispatcher
docker run -d --gpus all -p 9000:9000 \
  -v $(pwd)/stdio_bus:/stdio_bus:ro \
  -v $(pwd)/pyproject.toml:/stdio_bus/pyproject.toml:ro \
  -v $(pwd)/train.py:/stdio_bus/train.py:ro \
  stdiobus/stdiobus:python312 \
  --config /stdio_bus/stdio-bus-config.json --tcp 0.0.0.0:9000

# Query status
echo '{"jsonrpc":"2.0","id":1,"method":"status","params":{},"sessionId":"s1"}' | nc -i 1 localhost 9000

# Run experiment
echo '{"jsonrpc":"2.0","id":1,"method":"run","params":{"agentId":"agent-0"},"sessionId":"s1"}' | nc -i 1 localhost 9000
```

## When to Use Each Approach

**Use stdio_bus when:**
- Multiple GPUs on single machine
- Local cluster with shared filesystem
- Low-latency experiment dispatch required
- No external infrastructure available

**Use agenthub when:**
- Geographically distributed agents
- Agents on different networks/clouds
- Need persistent experiment history across restarts
- Human-readable web dashboard required

## Future Extensions

The stdio_bus architecture enables:
- **Agent routing**: Multiple AI agents sharing GPU pool via session affinity
- **MCP integration**: IDE tools can dispatch experiments via MCP-to-ACP proxy
- **Hybrid mode**: Local stdio_bus for GPU dispatch, agenthub for cross-machine coordination
