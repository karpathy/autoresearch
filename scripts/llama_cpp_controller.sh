#!/usr/bin/env bash
set -euo pipefail

# Logger helpers
info() { echo -e "\e[34m[INFO]\e[0m $*"; }
warn() { echo -e "\e[33m[WARN]\e[0m $*"; }
success() { echo -e "\e[32m[SUCCESS]\e[0m $*"; }

# Environment Defaults
MODEL_NAME="${MODEL_NAME:-unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ2_M}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8081}"
LLAMA_SERVER_HOST="${LLAMA_SERVER_HOST:-0.0.0.0}"
AUTO_CTX="${AUTO_CTX:-204800}"
AUTO_NP="${AUTO_NP:-1}"
AUTO_THREADS="${AUTO_THREADS:-16}"
IS_MODEL_MTP="${IS_MODEL_MTP:-true}"
IS_HIDDEN="${IS_HIDDEN:-true}"

kill_port() {
    local port="${1:?kill_port requires a port number}"
    local pids=()

    if command -v lsof &>/dev/null; then
        mapfile -t pids < <(lsof -t -i "TCP:${port}" 2>/dev/null || true)
    elif command -v fuser &>/dev/null; then
        mapfile -t pids < <(fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true)
    fi

    if [[ ${#pids[@]} -eq 0 ]]; then
        return 0
    fi

    info "Freeing port $port — sending SIGTERM to PID(s): ${pids[*]}"
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done

    local waited=0
    while lsof -t -i "TCP:${port}" &>/dev/null 2>&1 || fuser "${port}/tcp" &>/dev/null 2>&1; do
        sleep 1; (( waited++ )) || true
        if (( waited >= 5 )); then
            warn "Port $port still busy after ${waited}s — escalating to SIGKILL..."
            for pid in "${pids[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
            break
        fi
    done
    success "Port $port is now free."
}

run_model() {
    local is_hidden="${1:-$IS_HIDDEN}"
    local model="${2:-$MODEL_NAME}"
    local port="${3:-$LLAMA_SERVER_PORT}"
    local is_mtp="${4:-$IS_MODEL_MTP}"

    if [[ -z "$model" ]]; then
        warn "No model specified and MODEL_NAME is empty."
        exit 1
    fi

    pkill llama-server 2>/dev/null || true
    kill_port "$port" || true
    mkdir -p ./cache_dir

    local model_flag="-hf"
    [[ -f "$model" ]] && model_flag="-m"

    info "Starting llama-server for '$model' on port $port (${model_flag})..."

    local llama_args=(
        "$model_flag" "$model"
        --host "$LLAMA_SERVER_HOST"
        --port "$port"
        --alias "$model"
        -c "$AUTO_CTX"
        -np "$AUTO_NP"
        --threads "$AUTO_THREADS"
        --threads-batch "$AUTO_THREADS"
        -ngl -1
        --chat-template-kwargs '{"enable_thinking": true}'
        --reasoning on
        --mlock
        --mmap
        --cache-prompt
        --slot-save-path ./cache_dir
        --cache-type-k q4_0
        --cache-type-v q4_0
        --cache-reuse 0.7
        # --reasoning-budget 1000
        --flash-attn on
        --no-warmup
        --temp 1.0
        --top_p 0.95
        --top_k 64
        -cb
    )

    if [ "${is_mtp,,}" == "true" ]; then
        llama_args+=( --spec-type draft-mtp --spec-draft-n-max 4)
    fi

    if [[ "${is_hidden,,}" == "true" ]]; then
        mkdir -p ./logs
        llama-server "${llama_args[@]}" >> ./logs/api-server.log 2>&1 &
        local server_pid=$!
        info "llama-server started in background (PID: $server_pid). Logs: ./logs/api-server.log"
    else
        llama-server "${llama_args[@]}"
    fi

    # Optional quick health check
    if [[ "${is_hidden,,}" == "true" ]]; then
        info "Waiting for server health check..."
        local waited=0
        while ! curl -s "http://${LLAMA_SERVER_HOST}:${port}/health" &>/dev/null; do
            sleep 1
            (( waited++ )) || true
            if ! kill -0 "$server_pid" 2>/dev/null; then
                warn "llama-server process exited unexpectedly."
                break
            fi
            if (( waited >= 15 )); then
                warn "Server did not respond to /health within 15 seconds."
                break
            fi
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            success "llama-server is running and healthy on port $port."
        fi
    fi
}

Stage="${1:-start}"
shift || true

case "$Stage" in
    "start")
        run_model "$@"
        ;;
    "stop")
        pkill llama-server 2>/dev/null || true
        # Clean up port (default or first argument remaining)
        kill_port "${1:-$LLAMA_SERVER_PORT}" || true
        ;;
    "restart")
        pkill llama-server 2>/dev/null || true
        # Clean up port (default or 3rd argument from original start if present)
        # In restart, arguments are passed as: restart [model] [is_hidden] [port] [is_mtp]
        target_port="${3:-$LLAMA_SERVER_PORT}"
        kill_port "$target_port" || true
        run_model "$@"
        ;;
    *)
        warn "Unknown action: $Stage. Usage: $0 {start|stop|restart} [is_hidden] [model] [port] [is_mtp]"
        exit 1
        ;;
esac