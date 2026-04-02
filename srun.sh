#!/bin/bash
# Usage: srun.sh [host_path] <command>
#   host_path  Directory containing autoresearch.sif (default: current directory)
#   command    Command to run inside /opt/autoresearch in the container
#
# Examples:
#   ./srun.sh uv run train.py
#   ./srun.sh /scratch/mydir uv run train.py

HOST_PATH="$(pwd)"

# If the first argument is an existing directory, treat it as the host path
if [ -d "$1" ]; then
    HOST_PATH="$1"
    shift
fi

SIF="${HOST_PATH}/autoresearch.sif"

if [ ! -f "$SIF" ]; then
    echo "Error: SIF file not found at ${SIF}" >&2
    exit 1
fi

REPO="${HOST_PATH}/autoresearch"

if [ ! -d "$REPO" ]; then
    echo "Error: repo not found at ${REPO}" >&2
    exit 1
fi

singularity exec --nv \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --bind "${REPO}:/opt/autoresearch" \
    "$SIF" \
    bash -c "cd /opt/autoresearch && $*"
