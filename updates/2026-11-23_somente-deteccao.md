# 2026-11-23 — Versão só de detecção (branch `deteccao-f1`)

**Objetivo:** seguir a decisão com o Cesar em 22/11: o autoresearch roda sozinho, com um número mínimo de rodadas,
e foca em **causar** o aging do F1. O reparo e a análise de código saem do escopo, porque são o tema da pesquisa do
Cesar. O desenho completo está em [`docs/DESENHO-EXPERIMENTAL.md`](../docs/DESENHO-EXPERIMENTAL.md).

## O que saiu

| Item | Onde estava |
|---|---|
| Campanha de reparo | `program_repair.md`, `.claude/settings.repair.json`, comandos `repair`, `repair-init`, `promote` e `confirm --phase repair` em `prepare.py` |
| Nota e critério `fixed` do reparo | `score_repair`, `interference_check`, `FIXED_MIN_REDUCTION`, `FIXED_P95_TOL`, `tests/test_repair_fixed.py` |
| Portões de reparo e replay | `harness/gates.py` (trocado por `harness/functional_check.py`, só o teste funcional do Cesar) |
| App reparada do Cesar (R2) | `apps/credit-card-fastapi/repaired/` |
| Testes escondidos | manifesto e `check-oracles` (só serviam aos portões de reparo) |
| Teste A/A e desenho antigo | `workload_aa.py`, `docs/METODOLOGIA.md` |

## O que ficou igual

Juiz de detecção (`harness/instance.py`, `load.py`, `evaluate.py`, `stats.py`, `workload_check.py`), critérios de
detecção (4 condições, C1, D1–D4, H4 do processo), sementes 101–108, keep pela razão, confirmação 2 × 8 h com
sementes 201 e 202. A medida de interferência continua, como diagnóstico e para a Fase 4.

## Verificação

`py_compile` de todo o código, `tests/test_c1.py` passando e `prepare.py rehash` + `verify` (21 arquivos fixos).
Nenhuma medição com a app foi feita nesta pasta.

## O que ainda falta

- Orquestrador que encadeia as fases sozinho (a Fase 4 foi implementada depois, no mesmo dia: `prepare.py causes`).
- Decidir com o Cesar: `N_MIN`, `N_CAUSAS` e se o agente pode ler o código da app e o card do F1.
- As notas de 19/11 abaixo descrevem o desenho anterior, com reparo; ficam como histórico.

## Atualização: Fase 4 e `program.md` único (23/11)

- **Fase 4 implementada:** `prepare.py causes` roda a carga congelada como referência e até 3 variações de um
  fator (o juiz recusa as outras), e grava as razões contra a referência (`lat_ratio`, `bytes_ratio`,
  `cost_ratio`, `assoc_p95_ratio`, `ref4_p95_ratio`). Nova medida `hot_cost_ms_per_mb` em `evaluate.py`.
  `ref_sizes` ficou fixo em (4, 200), porque é o controle interno.
- **`program.md` só com instruções do agente** (campanhas `detect` e `causas`); o desenho foi para
  `docs/DESENHO-EXPERIMENTAL.md` e o `program_detect.md` foi absorvido.
- **Menos viés:** o agente só lê `program.md`, `workload.py`, `frozen/`, `results/` e os próprios registros; textos
  que ele vê nos resultados ficaram neutros. `run_campaign.sh causas <tag>` lança a Fase 4.
- **Verificado:** `tests/test_c1.py`, `tests/test_causes.py` e uma rodada `causes` simulada (sem a app).
