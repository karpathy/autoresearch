#!/usr/bin/env python3
"""
GPU dispatcher. stdio_bus worker, JSON-RPC 2.0 over NDJSON.
Usage: echo '{"jsonrpc":"2.0","id":1,"method":"status","params":{}}' | python3 dispatcher.py
"""
import json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# Project root: /stdio_bus in Docker, parent of stdio_bus/ locally
_dispatcher_dir = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/stdio_bus") if _dispatcher_dir == Path("/stdio_bus") else _dispatcher_dir.parent

def _detect_gpus():
    if ids := os.environ.get("SWARM_GPU_IDS"):
        return [int(x) for x in ids.split(",")]
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and (lines := r.stdout.strip()):
            return [int(x.strip()) for x in lines.split("\n") if x.strip()]
    except: pass
    return [0]

GPU_IDS = _detect_gpus()

state = {"best": float('inf'), "total": 0, "active": 0, "results": []}
state_lock = threading.Lock()
gpu_busy = dict.fromkeys(GPU_IDS, False)
gpu_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=len(GPU_IDS))

def log(msg): print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)

def _parse_output(stdout, key):
    return next((float(l.split(":")[1]) for l in stdout.split("\n") if l.startswith(f"{key}:")), 0.0)

def run_experiment(gpu_id, agent_id, branch):
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    log(f"[GPU {gpu_id}] Start {agent_id}")
    base = {"agent_id": agent_id, "gpu_id": gpu_id, "timestamp": datetime.now().isoformat(), "branch": branch}
    try:
        r = subprocess.run(["uv", "run", "train.py"], capture_output=True, text=True, timeout=600, env=env, cwd=PROJECT_ROOT)
        val_bpb = _parse_output(r.stdout, "val_bpb")
        mem = _parse_output(r.stdout, "peak_vram_mb")
        commit = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                               capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip() or "?"
        log(f"[GPU {gpu_id}] Done val_bpb={val_bpb:.6f}")
        return {**base, "commit": commit, "val_bpb": val_bpb, "memory_gb": mem/1024,
                "status": "keep" if val_bpb > 0 else "crash"}
    except subprocess.TimeoutExpired:
        return {**base, "commit": "timeout", "val_bpb": 0, "memory_gb": 0, "status": "crash"}
    except Exception as e:
        return {**base, "commit": "error", "val_bpb": 0, "memory_gb": 0, "status": "crash", "error": str(e)}

def experiment_wrapper(gpu_id, agent_id, branch):
    with state_lock: state["active"] += 1
    try:
        result = run_experiment(gpu_id, agent_id, branch)
        with state_lock:
            state["total"] += 1
            state["active"] -= 1
            state["results"].append(result)
            if result["status"] == "keep" and 0 < result["val_bpb"] < state["best"]:
                state["best"] = result["val_bpb"]
        return result
    finally:
        with gpu_lock: gpu_busy[gpu_id] = False

def handle(method, params):
    match method:
        case "status":
            with state_lock:
                return {"total": state["total"], "active": state["active"],
                        "best": state["best"] if state["best"] != float('inf') else None,
                        "gpus": {"ids": GPU_IDS, "busy": [g for g, b in gpu_busy.items() if b]}}
        case "sync":
            with state_lock:
                return {"best": state["best"] if state["best"] != float('inf') else None,
                        "total": state["total"], "results": state["results"][-50:]}
        case "history":
            with state_lock: return {"results": state["results"][-params.get("limit", 100):]}
        case "run":
            with gpu_lock:
                gpu_id = next((g for g in GPU_IDS if not gpu_busy[g]), None)
                if gpu_id is None: return {"error": "All GPUs busy"}
                gpu_busy[gpu_id] = True
            agent_id, branch = params.get("agentId", "agent-0"), params.get("branch", f"agent/{params.get('agentId', 'agent-0')}")
            if params.get("blocking", True):
                return experiment_wrapper(gpu_id, agent_id, branch)
            executor.submit(experiment_wrapper, gpu_id, agent_id, branch)
            return {"queued": True, "gpu_id": gpu_id}
    return None

def main():
    log(f"Dispatcher | GPUs: {GPU_IDS}")
    for line in sys.stdin:
        if not (line := line.strip()): continue
        try: msg = json.loads(line)
        except: print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}), flush=True); continue
        if msg.get("jsonrpc") != "2.0" or "method" not in msg:
            if msg.get("id"): print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32600, "message": "Invalid"}}), flush=True)
            continue
        result = handle(msg["method"], msg.get("params", {}))
        if msg.get("id") is not None:
            out = {"jsonrpc": "2.0", "id": msg["id"]}
            if msg.get("sessionId"): out["sessionId"] = msg["sessionId"]
            out |= {"result": result} if result is not None else {"error": {"code": -32601, "message": "Not found"}}
            print(json.dumps(out), flush=True)

if __name__ == "__main__": main()
