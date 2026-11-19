# updates/ — notas de atualização e estado do repositório

Uma nota por mudança relevante, mais recente no topo. Cada nota diz o que mudou, por quê e o que ainda falta.

| Data | Nota | Resumo |
|---|---|---|
| 2026-11-19 | [2026-11-19_confirmacao-duas-execucoes.md](2026-11-19_confirmacao-duas-execucoes.md) | **Confirmação de 8 h em duas execuções** (sementes 201 e 202); reparo só se as duas passarem |
| 2026-11-19 | [2026-11-19_semente-por-rodada.md](2026-11-19_semente-por-rodada.md) | **Semente diferente por rodada de detecção (101–108)** e keep pela razão contra a referência da rodada |
| 2026-11-19 | [contexto.md](contexto.md) | **Resumo do contexto para retomar o trabalho** (pedido, decisões, próximos passos) |
| 2026-11-19 | [2026-11-19_alvo-cesar.md](2026-11-19_alvo-cesar.md) | Troca do alvo Node pelo cenário do Cesar (`credit-card-fastapi`, finding F1); juiz, carga, métrica e portões adaptados |
| 2026-11-19 | [criterio-de-deteccao.md](criterio-de-deteccao.md) | **Quando a detecção é considerada positiva e libera a campanha de reparo** |

## Estado atual (2026-11-19)

**Branch:** `juiz-v2-bd0b0ad`, com push. Commits desta mudança: `19cc4bd` em diante (ver `git log bd0b0ad..HEAD`).

| Peça | Estado |
|---|---|
| Alvo | `apps/credit-card-fastapi/{instrumented,repaired}`, copiados do repo do Cesar (commit `97addef`, em `apps/credit-card-fastapi/SOURCE.json`) |
| venv | `.cache/venv`, com o `requirements.txt` do Cesar + dependências do juiz; `prepare.py` se re-executa nele |
| Oráculos | `check-oracles` passa em `instrumented` e `repaired` (funcional do Cesar, escondidos, persistência) |
| Detecção (campanha 1) | Código pronto; smoke de 2 min ok. Protocolo: semente 101–108 por rodada e keep pelo `ratio_vs_reference` ([nota](2026-11-19_semente-por-rodada.md)). **A campanha em si não foi rodada** |
| Controles R0/R0b | Smoke de 2 min ok (R0 e R0b planos, R1 sobe). A rodada de 60 min (`results/controls-1119-0743-9b5e`) foi **interrompida a pedido em ~18 min**; nos dados parciais, R0 e R0b planos e R1 elegível. **A rodada completa de 60 min não foi feita** |
| Reparo (campanha 2) | Código pronto; smoke de 2 min dos portões ok. Não rodado |
| Confirmação 8 h | Código pronto (`confirm`, R0/R0b/R1/R2). Protocolo: **duas execuções** (sementes 201 e 202), as duas precisam passar ([nota](2026-11-19_confirmacao-duas-execucoes.md)). Não rodada |
| `run_campaign.sh` | Não rodado |

### O que está pendente

1. **Rodada de 60 min dos controles (refazer, foi interrompida):** confirmar que R1 sobe em `retrieve_hot` e que R0/R0b ficam planos, como no
   resultado do Cesar. Os limiares da detecção (abaixo) são de partida e podem precisar de ajuste por causa dela.
2. **Validar com o Cesar:** (a) D2 desligada no F1; (b) nota baseada na latência e no tamanho de resposta de
   `retrieve_hot`, além do RSS; (c) stall do `ps`: **resolvido, não se aplica** ao Credit Card (só existe no Monitor). O cenário dele para a credit-card não tem
   injeção; nada foi implementado.
3. **O critério de liberação do reparo ainda não é imposto pelo código:** `freeze` aceita qualquer braço, e nada
   impede o `repair-init` antes das duas confirmações aprovadas. Ver
   [criterio-de-deteccao.md](criterio-de-deteccao.md#o-que-o-código-impõe-e-o-que-não-impõe).
4. **Sem controle positivo neste cenário:** um resultado negativo não diz nada sobre a sensibilidade do juiz.
