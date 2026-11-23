#!/usr/bin/env bash
# Lança uma campanha com o Claude Code em modo headless.
# Uso: ./run_campaign.sh detect <tag>   (esta versão é só de detecção; ver program.md)
set -euo pipefail
PHASE="${1:?detect}"; TAG="${2:?tag da campanha}"
[ "$PHASE" = "detect" ] || { echo "fase desconhecida: $PHASE (só detect)"; exit 2; }
cd "$(dirname "$0")"
python prepare.py verify
cp ".claude/settings.${PHASE}.json" .claude/settings.json
# Barreira real (além das permissões do Claude Code): juiz e oráculos sem escrita.
chmod -R a-w prepare.py harness oracle apps .judge 2>/dev/null || true
[ -d frozen ] && chmod -R a-w frozen || true
mkdir -p logs
PROMPT="Leia program_${PHASE}.md e execute a campanha com a tag '${TAG}'. O humano já confirmou o setup: não pergunte nada, siga o laço até o orçamento acabar e escreva o summary."
claude -p "$PROMPT" --output-format stream-json --verbose \
  > "logs/${PHASE}-${TAG}-$(date +%Y%m%d-%H%M).jsonl" 2>&1
echo "Campanha encerrada. Compare results.tsv com results/judge_log.jsonl."
