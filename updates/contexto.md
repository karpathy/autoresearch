# Resumo do contexto (para retomar o trabalho)

Última atualização: 2026-11-19. Leia isto antes de continuar numa sessão nova. Detalhes em
[2026-11-19_alvo-cesar.md](2026-11-19_alvo-cesar.md) e [criterio-de-deteccao.md](criterio-de-deteccao.md).

## O pedido original

Manter o **método** autoresearch (laço de detecção e reparo, juiz fixo, portões H1–H7, flags D1–D3, travas
anti-trapaça, testes escondidos em `~/.aging-hidden`) e trocar o **alvo e o cenário** pelos do Cesar
([chas42/llm-aging-detection-agents](https://github.com/chas42/llm-aging-detection-agents)), exatamente como ele usou:

- app `apps/credit-card-fastapi` (instrumented e repaired) e finding `bax-fastapi-credit-card-F1`;
- carga de referência `F1_targeted_v1`; controles negativos R0 (`F1_control`) e R0b (`--key-space 4`); sem controle positivo;
- rodadas R0, R0b, R1 e R2 do `campaign.yaml`; nenhum bug injetado.

Regras do Rafael: não editar o repo do Cesar; commits pequenos por etapa; perguntar em vez de inventar quando o
cenário do Cesar for ambíguo; não rodar `run_campaign.sh`.

## Onde estamos

- Branch `juiz-v2-bd0b0ad`, commits `19cc4bd` … (ver `git log bd0b0ad..HEAD`). As etapas 2 a 9 do plano estão
  feitas e commitadas. A etapa 10 (validação) está em andamento.
- O push para `origin` (`Rafaelmos/autoresearch`) foi bloqueado pelo modo automático do Claude Code; a permissão
  foi configurada em `.claude/settings.local.json` (só esta branch) e as campanhas negam `git push`.
- O repo do Cesar **não** existe em `../llm-aging-detection-agents`; foi clonado só para leitura num diretório
  temporário da sessão. Para um novo `setup`, clone-o ao lado deste repo.

## Decisões

| Quem | Decisão |
|---|---|
| Rafael | `workload.py` do agente = só `args` literais do F1 (sem código); taxas ≤ 2× os defaults |
| Rafael | Caso `phone="12"` dos testes escondidos aceita 201 ou 400 (original em `~/.aging-hidden/hidden_tests.py.node-original`) |
| Rafael | Stall do `ps`: não implementado até validar com o Cesar (o cenário credit-card dele não tem injeção) |
| Rafael | Semente diferente por rodada de detecção (101–108; falhas consomem a semente) e keep pelo `ratio_vs_reference` da rodada (≥ 1,10 × o melhor anterior; referência sem sintoma → nenhum keep). Protocolo em `program_detect.md`, não imposto pelo juiz ([nota](2026-11-19_semente-por-rodada.md)) |
| Claude (reversível) | Agente não lê `apps/credit-card-fastapi/repaired/` (deny nos `settings.{detect,repair}.json`) |
| Claude (reversível) | `confirm` de 8 h com uma semente; app do slot *i* na CPU *i*, gerador na CPU 4+*i* |
| Claude (reversível) | Replay dos portões repete cada requisição até 3× e refaz o replay antes de acusar H2/H3 |

## Próximos passos

1. Ler o resultado da rodada de 60 min de R0, R0b e R1 (`results/controls-1119-0743-9b5e/round.json`): R1 deve
   ser elegível (latência e resposta de `retrieve_hot` sobem, controle interno plano) e R0/R0b planos. Atualizar
   `updates/README.md` com os números e, se preciso, ajustar os limiares.
2. Validar com o Cesar: D2 desligada no F1; nota pela latência/bytes de `retrieve_hot`; stall do `ps`.
3. Opcional, já oferecido: fazer o `freeze` exigir um `confirm --phase detect` aprovado.
4. Depois: campanha 1 (`./run_campaign.sh detect <tag>`) só quando o Rafael pedir.
