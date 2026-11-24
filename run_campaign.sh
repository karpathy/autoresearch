#!/usr/bin/env bash
# Lança uma campanha com o Claude Code em modo headless.
# Uso: ./run_campaign.sh detect <tag>    |    ./run_campaign.sh causas <tag>
set -euo pipefail
PHASE="${1:?detect|causas}"; TAG="${2:?tag da campanha}"
case "$PHASE" in detect|causas) ;; *) echo "fase desconhecida: $PHASE (detect ou causas)"; exit 2;; esac
cd "$(dirname "$0")"
if [ "$PHASE" = causas ] && [ ! -f frozen/workload_frozen.py ]; then echo "causas exige frozen/workload_frozen.py (freeze + confirmação)"; exit 2; fi
python prepare.py verify
cp ".claude/settings.${PHASE}.json" .claude/settings.json
# Barreira real (além das permissões do Claude Code): juiz e oráculos sem escrita.
chmod -R a-w prepare.py harness oracle apps .judge 2>/dev/null || true
[ -d frozen ] && chmod -R a-w frozen || true
mkdir -p logs
PROMPT="Leia program.md e execute a campanha ${PHASE} com a tag ${TAG}."
claude -p "$PROMPT" --output-format stream-json --verbose \
  > "logs/${PHASE}-${TAG}-$(date +%Y%m%d-%H%M).jsonl" 2>&1
echo "Campanha encerrada. Compare os registros do agente com results/judge_log.jsonl."
