# Resumo do contexto (para retomar o trabalho)

Última atualização: 2026-11-23. Comece por [2026-11-23_resumo-das-mudancas.md](2026-11-23_resumo-das-mudancas.md).

## O que é o experimento

O autoresearch (agente propõe, juiz fixo mede, laço mantém ou descarta) aplicado ao cenário do Cesar: app
`credit-card-fastapi`, finding F1, carga `F1_targeted_v1`, controles negativos R0 e R0b. Escopo: **detectar,
confirmar e explicar pela carga** o aging do F1. Reparo e análise de código ficam de fora (pesquisa do Cesar).
Desenho em [`docs/DESENHO-EXPERIMENTAL.md`](../docs/DESENHO-EXPERIMENTAL.md); instruções do agente em
[`program.md`](../program.md).

## Onde estamos

- Branch `deteccao-f1`, local em `C:\Users\rafae\Documents\GitHub\autoresearch-deteccao-f1`, **sem push**.
- A branch `juiz-v2-bd0b0ad` (versão com reparo) continua no GitHub, sem estas mudanças.
- Nenhuma medição com a app foi feita nesta branch; para rodar, o preparo (`setup`) tem de ser feito no Linux/WSL
  com o repo do Cesar clonado ao lado.

## Decisões vigentes

| Quem | Decisão |
|---|---|
| Rafael e Cesar (22/11) | O autoresearch roda sozinho, com um número mínimo de rodadas, e foca em causar o aging; reparo fora |
| Rafael (23/11) | Investigação de causas só por comparação entre cargas, um fator por vez, duas réplicas |
| Rafael (23/11) | `program.md` só com instruções; o agente não lê código da app, card do F1, cargas do Cesar, juiz nem docs |
| Rafael e Ermeson (19/11) | Sementes independentes; confirmação com parâmetros fixos, banco limpo, R1 e controles com a mesma semente, unanimidade |
| Rafael (23/11) | Sem humano no meio; confirmação dentro da campanha: a carga que der aging é repetida nas 2 rodadas seguintes (com R0 e R0b), e só conta se as 3 derem aging; depois vem a campanha de causas. Opção de 3 campanhas completas em stand-by ([nota](2026-11-23_confirmacao-em-rodadas.md)). **Implementado** ([orquestrador](2026-11-23_orquestrador.md)) |
| Rafael | `workload.py` só com `args` literais; taxas ≤ 2× os padrões; `ref_sizes` fixo (controle interno) |
| Rafael e Cesar | D2 desligada no F1 (crescimento de linhas é ativação); stall do `ps` não se aplica |

## Próximos passos

1. Rodar os controles de 60 min (Fase 0) e calibrar os limiares.
2. Confirmar com o Cesar: `N_MIN = 8`, `N_CAUSAS = 6` e a restrição de leitura do agente.
3. Rodar `./run_experiment.sh <tag> --smoke` no WSL e, se o fluxo passar, o experimento completo.
4. Campanhas só quando o Rafael pedir.
