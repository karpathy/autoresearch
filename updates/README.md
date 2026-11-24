# updates/ — notas de atualização e estado do repositório

Uma nota por mudança relevante, mais recente no topo. Cada nota diz o que mudou, por quê e o que ainda falta.

| Data | Nota | Resumo |
|---|---|---|
| 2026-11-23 | [2026-11-23_traducao-ingles.md](2026-11-23_traducao-ingles.md) | Instruções, prompts e mensagens que o agente vê em inglês; regras e identificadores iguais |
| 2026-11-23 | [2026-11-23_orquestrador.md](2026-11-23_orquestrador.md) | **Orquestrador implementado:** experimento completo sem humano no meio (`./run_experiment.sh <tag>`) |
| 2026-11-23 | [2026-11-23_confirmacao-em-rodadas.md](2026-11-23_confirmacao-em-rodadas.md) | Decisão (implementada): confirmação dentro da campanha (3 rodadas), sem humano no meio; opção de 3 campanhas em stand-by |
| 2026-11-23 | [2026-11-23_resumo-das-mudancas.md](2026-11-23_resumo-das-mudancas.md) | **Comece por aqui:** tudo que mudou da `juiz-v2-bd0b0ad` para a `deteccao-f1` |
| 2026-11-23 | [2026-11-23_program-md-e-autoresearch.md](2026-11-23_program-md-e-autoresearch.md) | `program.md` só com instruções do agente, leitura restrita e conferência com o `karpathy/autoresearch` |
| 2026-11-23 | [2026-11-23_fase4-causas.md](2026-11-23_fase4-causas.md) | Fase 4: investigação de causas pela carga (`prepare.py causes`, um fator por vez, custo por MB) |
| 2026-11-23 | [2026-11-23_somente-deteccao.md](2026-11-23_somente-deteccao.md) | Versão só de detecção: reparo, portões, R2 e testes escondidos removidos |
| 2026-11-23 | [contexto.md](contexto.md) | Resumo do contexto para retomar o trabalho |
| 2026-11-19 | [2026-11-19_confirmacao-duas-execucoes.md](2026-11-19_confirmacao-duas-execucoes.md) | Confirmação de 8 h em duas execuções (sementes 201 e 202) |
| 2026-11-19 | [2026-11-19_semente-por-rodada.md](2026-11-19_semente-por-rodada.md) | Semente diferente por rodada de detecção (101–108) e keep pela razão |
| 2026-11-19 | [2026-11-19_alvo-cesar.md](2026-11-19_alvo-cesar.md) | Troca do alvo Node pelo cenário do Cesar (`credit-card-fastapi`, finding F1) |
| 2026-11-19 | [criterio-de-deteccao.md](criterio-de-deteccao.md) | Critério de detecção (as seções sobre reparo são do desenho anterior) |

As notas de 19/11 descrevem o desenho anterior, com reparo, e ficam como histórico.

## Estado atual (2026-11-23)

**Branch:** `deteccao-f1` (local, sem push), criada a partir de `juiz-v2-bd0b0ad` @ `75a657e`.

| Peça | Estado |
|---|---|
| Alvo | `apps/credit-card-fastapi/instrumented`, copiado do repo do Cesar (commit `97addef`) |
| Juiz de detecção | pronto; mesmo critério da `juiz-v2-bd0b0ad` |
| Orquestrador | **implementado** (`harness/orchestrator.py`, `prepare.py experiment`, `./run_experiment.sh`); testado com juiz e agente simulados |
| Fase 0, busca, verificação, causas | prontas e encadeadas; **nenhuma rodada real ainda** |
| Testes | `tests/test_c1.py`, `tests/test_causes.py` e `tests/test_orchestrator.py` passam |
