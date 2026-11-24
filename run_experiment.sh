#!/usr/bin/env bash
# Dispara o experimento completo, sem humano no meio (ver docs/DESENHO-EXPERIMENTAL.md):
#   Fase 0 (controles) -> busca (agente) -> verificação -> causas (agente) -> retomada -> relatório.
# Uso: ./run_experiment.sh <tag> [--smoke]
# O relatório sai em results/experiment-<tag>/report.md, e os resumos do agente em summary_*.md.
set -euo pipefail
TAG="${1:?tag do experimento}"; shift || true
cd "$(dirname "$0")"
python prepare.py verify
# Barreira real (além das permissões do Claude Code): juiz, oráculos, app e cargas sem escrita.
# .judge/ e frozen/ ficam com escrita: o orquestrador congela as cargas confirmadas e refaz o manifesto.
chmod -R a-w prepare.py harness oracle apps workloads findings 2>/dev/null || true
mkdir -p logs
python prepare.py experiment --tag "$TAG" "$@" 2>&1 | tee "logs/experiment-${TAG}-$(date +%Y%m%d-%H%M).log"
