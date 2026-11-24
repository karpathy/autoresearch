# 2026-11-23 — Resumo: da `juiz-v2-bd0b0ad` para a `deteccao-f1`

Visão geral de tudo que mudou na branch `deteccao-f1` (criada a partir de `juiz-v2-bd0b0ad` @ `75a657e`).
Detalhes em cada nota.

| # | Mudança | Por quê | Nota |
|---|---|---|---|
| 1 | Desenho do experimento sem reparo: detecção autônoma, confirmação 2 × 8 h, investigação de causas pela carga | decisão com o Cesar em 22/11 (o reparo e a análise de código são a pesquisa dele) | [`docs/DESENHO-EXPERIMENTAL.md`](../docs/DESENHO-EXPERIMENTAL.md) |
| 2 | Removidos campanha de reparo, portões H1–H3/H5–H7, nota `fixed`, app reparada (R2), testes escondidos, A/A e o desenho antigo | fora do escopo | [2026-11-23_somente-deteccao.md](2026-11-23_somente-deteccao.md) |
| 3 | `harness/gates.py` → `harness/functional_check.py` (só o teste funcional, no preparo) | os portões só serviam ao reparo | idem |
| 4 | Fase 4: `prepare.py causes`, regra de um fator, custo por MB, razões contra a carga congelada; `ref_sizes` fixo | explicar a causa sem ler código | [2026-11-23_fase4-causas.md](2026-11-23_fase4-causas.md) |
| 5 | `program.md` só com instruções do agente; desenho em `docs/`; `program_detect.md` absorvido | alinhar com o autoresearch e evitar viés | [2026-11-23_program-md-e-autoresearch.md](2026-11-23_program-md-e-autoresearch.md) |
| 6 | Leitura do agente restrita e textos neutros | o agente foca na temática, sem pistas do mecanismo | idem |
| 7 | Conceitos do autoresearch reforçados no `program.md` (simplicidade, crash, timeout, primeira rodada, contexto) | conferência com o original | idem |
| 8 | `README.md` reescrito | estado atual | — |
| 9 | Confirmação dentro da campanha (2 rodadas com R1, R0, R0b), sem humano no meio; opção de 3 campanhas em stand-by | decisão de 23/11 | [2026-11-23_confirmacao-em-rodadas.md](2026-11-23_confirmacao-em-rodadas.md) |
| 10 | Orquestrador: Fase 0 → busca → verificação → causas → retomada → relatório; `run_experiment.sh`; sai o `confirm` de 8 h e o `run_campaign.sh` | implementa o item 9 | [2026-11-23_orquestrador.md](2026-11-23_orquestrador.md) |

## O que não mudou

Juiz de detecção (instância, carga, Hamed-Rao + Sen, as 4 condições de elegibilidade, C1, D1–D4, H4) e keep pela
razão contra a referência da rodada. A confirmação de 2 × 8 h foi trocada pela verificação em rodadas (item 9).

## Estado

Código compila; `tests/test_c1.py`, `tests/test_causes.py` e `tests/test_orchestrator.py` passam; simulação de
ponta a ponta do experimento ok; manifesto refeito (`verify: ok`). **Nenhuma medição com a app foi feita** nesta
branch.

## Pendências

1. Controles de 60 min (Fase 0) e calibração dos limiares.
2. Primeiro `./run_experiment.sh <tag> --smoke` com a app.
3. Confirmar com o Cesar: 8 rodadas de busca, 6 de causas e o agente sem acesso ao código da app e ao card do F1.
4. Atualizar a figura do fluxograma (o Cesar monta quando as mudanças fecharem).
