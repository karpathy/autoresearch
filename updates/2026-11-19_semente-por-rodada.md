# 2026-11-19 — Semente diferente por rodada de detecção e keep pela razão contra a referência

**Objetivo:** confirmar o comportamento do F1 com sementes independentes antes de liberar o reparo.

**Antes:** todas as rodadas da campanha 1 usavam a mesma semente (o padrão do juiz, `--seed 1`), e o keep
comparava o score bruto (`score_hot_latency_ms_per_h`, em ms/h) com o do melhor anterior.

**Agora:** cada rodada usa uma semente nova, e o keep compara a razão contra a referência da própria rodada.

## O que mudou

| Arquivo | Mudança |
|---|---|
| `program_detect.md` | Rodada *n* usa `--seed 100+n` (101 … 108): `python prepare.py detect --seed <semente> --background`. Rodadas que falham também consomem a semente; nenhuma é reutilizada. Coluna `seed` no `results.tsv` (depois de `run_id`) e campo `seed` nas linhas `started`/`completed` do `ledger.jsonl`. Keep por `ratio_vs_reference` (≥ 1,10 × o do melhor anterior; na primeira rodada, o maior `ratio` elegível define o melhor anterior). Referência não elegível (`ratio = NA`) → nenhum keep na rodada, status `referencia_sem_sintoma` no ledger e commit desfeito. O `summary_detect.md` traz a tabela da R1 por rodada (semente, `hot_lat_slope`, `hot_lat_change`, `hr_p`, elegível). Limitação reescrita: "uma semente por rodada (sementes diferentes entre rodadas); cada candidato roda com uma única semente" |
| `README.md` | Seção "Semente por rodada e keep pela razão"; linha na tabela "De onde vem cada decisão"; `confirm` com `--seed` explícito, uma semente por execução, sementes distintas se for repetida (repetir não é obrigatório) |
| `updates/criterio-de-deteccao.md` | Etapas B e C da liberação com a semente; subseção "Semente por rodada e keep"; linha na tabela "O que o código impõe e o que não impõe" |

Nada em `prepare.py`, `harness/`, `oracle/`, `workloads/`, `findings/` ou `apps/` mudou: `detect` e `confirm`
já aceitam `--seed`, aplicam a mesma semente aos 4 braços, gravam-na em `round.json` e `judge_log.jsonl` e já
calculam o `ratio_vs_reference`. Não houve rehash; `prepare.py verify` continua `ok`.

## Por quê

- **Independência da semente:** a R1 roda no 4º slot de toda rodada. Com semente nova a cada rodada, ela vira
  uma réplica independente, e 8 rodadas dão até 8 observações de que o sintoma do F1 aparece qualquer que
  seja o conjunto de cartões gerado pela semente.
- **Keep pela razão:** com sementes diferentes, a inclinação absoluta varia de rodada para rodada também por
  causa da semente. A razão contra a referência da mesma rodada (mesma semente, mesma máquina, mesmo momento)
  tira esse efeito comum. Se a referência não mostrou o sintoma, não há base de comparação, e a rodada não
  seleciona nada.

## O que ainda falta

- **O juiz não impõe o protocolo:** a sequência de sementes e a decisão de keep ficam a cargo do agente. O
  humano confere no fim: as sementes em `results/judge_log.jsonl` (campo `seed` de cada `detect`) devem ser
  101, 102, … sem repetição, e os keeps devem bater com os `ratio` do `round.json` de cada rodada.
- **Cada candidato roda com uma semente só:** um keep pode ainda refletir a interação entre os `args` do
  candidato e a semente daquela rodada. A confirmação de 8 h (`confirm --seed`) é a segunda checagem.
- **Ruído da razão não medido:** a margem de 10% foi mantida do Piloto 2. A dispersão do `ratio` entre sementes
  ainda não foi medida (a tabela da R1 por rodada dá a dispersão da referência, não a da razão).
