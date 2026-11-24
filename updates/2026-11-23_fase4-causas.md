# 2026-11-23 — Fase 4: investigação de causas só pela carga

**Objetivo:** depois de o aging ser confirmado, descobrir **o que na carga** causa o envelhecimento, comparando
variações que mudam um fator de cada vez. Sem ler nem alterar o código da aplicação, para não sobrepor a pesquisa do
Cesar (que acha a causa por análise de código).

## Como funciona

| Peça | Regra |
|---|---|
| Comando | `prepare.py causes --seed <s> [--background]`; lançado por `./run_campaign.sh causas <tag>` |
| Referência | slot 0 = carga congelada (`referencia_congelada`), a mesma confirmada nas duas execuções de 8 h |
| Variações | slots 1–3; cada uma muda **exatamente um** `arg` da carga congelada. Zero ou mais de um → a rodada inteira é recusada (D3), por `harness/load.py:one_factor_diff` |
| Sementes | 301 a 306 (6 rodadas); cada variação roda em **duas réplicas** com sementes diferentes |
| Medidas (razão variação ÷ referência da mesma rodada) | `lat_ratio` (inclinação da latência de `retrieve_hot`), `bytes_ratio` (inclinação do tamanho da resposta), `cost_ratio` (custo por MB), `assoc_p95_ratio` e `ref4_p95_ratio` (interferência) |
| Quando as razões existem | só se a referência da rodada for elegível (`reference_ok`) e a variação for válida |
| Efeito declarado | a razão fica do mesmo lado de 1, a pelo menos 20% dele (≥ 1,2 ou ≤ 0,8), nas duas réplicas |
| Saídas | `round.json` (`event: causes`, `factor`, `effect`), `judge_log.jsonl`, e do agente: `results_causas.tsv`, `ledger_causas.jsonl`, `summary_causas.md` |

## Medida nova: custo por MB

`hot_cost_ms_per_mb` em `harness/evaluate.py`: Sen's slope da latência p50 de `retrieve_hot` **contra** o tamanho
médio da resposta (MB), janela a janela. Separa "a resposta ficou maior" (mais dados acumulados) de "cada MB ficou
mais caro". Por exemplo, uma variação que só muda a taxa de acúmulo muda `bytes_ratio` e `lat_ratio`, mas deveria
manter `cost_ratio` perto de 1.

## Ajuste no contrato: `ref_sizes` fixo

Os telefones de referência (4 e 200 cartões) são o **controle interno**: a regra C1 e a medida de interferência
leem `retrieve_ref4` e `retrieve_ref200`. Se o agente mudasse `ref_sizes`, essas séries sumiriam e o C1 passaria
sem avisar. Agora `ref_sizes` só aceita (4, 200) e não é um fator da carga.

## Verificação

- `tests/test_causes.py`: regra de um fator (aceita 1, recusa 0 e 2), custo por MB numa série sintética
  (~30 ms/MB), razões contra a referência;
- uma rodada `causes` simulada (sem subir a app): a variação de um fator foi medida e a de dois foi recusada;
- `tests/test_c1.py` continua passando; `rehash` + `verify` ok.

## O que falta

- Calibrar o limiar de 20% para declarar efeito com as primeiras réplicas.
- Um fator de cada vez não captura interações entre fatores.
