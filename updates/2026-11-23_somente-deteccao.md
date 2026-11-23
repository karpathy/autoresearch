# 2026-11-23 — Versão só de detecção (branch `deteccao-f1`)

**Objetivo:** seguir a decisão com o Cesar em 22/11: o autoresearch roda sozinho, com um número mínimo de rodadas,
e foca em **causar** o aging do F1. O reparo e a análise de código saem do escopo, porque são o tema da pesquisa do
Cesar. O desenho completo está em [`program.md`](../program.md).

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

- Orquestrador que encadeia as fases sozinho e a Fase 4 (investigação de causas): previstos no `program.md`.
- Decidir com o Cesar: `N_MIN`, `N_CAUSAS` e se o agente pode ler o código da app e o card do F1.
- As notas de 19/11 abaixo descrevem o desenho anterior, com reparo; ficam como histórico.
