# AutoResearch-Aging: detecção autônoma de aging no F1 (`credit-card-fastapi`)

Este repositório adapta o [karpathy/autoresearch](https://github.com/karpathy/autoresearch) para **detectar e
explicar envelhecimento de software** numa aplicação gerada por LLM. O **método** é o do autoresearch: um agente
propõe, um juiz fixo mede, e o laço mantém ou descarta a proposta. O **alvo e o cenário** são os do Cesar, em
[chas42/llm-aging-detection-agents](https://github.com/chas42/llm-aging-detection-agents): a app
`credit-card-fastapi` (BaxBench), o finding `bax-fastapi-credit-card-F1` e as rodadas R0, R0b e R1 do
`apps/credit-card-fastapi/campaign.yaml`.

Responsáveis: Rafael, Cesar e Ermeson (UFRPE). Colaboração com Fumio Machida (Tsukuba) e Artur Andrzejak
(Heidelberg).

> **Escopo (branch `deteccao-f1`): só detecção e explicação pela carga, sem humano no meio.** O agente procura
> cargas que causem o aging do F1; o sistema confirma cada carga repetindo-a nas rodadas seguintes, com os
> controles negativos ao lado; depois o agente testa, variando um fator da carga de cada vez, o que causa o
> envelhecimento. O humano só dispara o experimento e lê o relatório final. O **reparo** e a **análise de código** ficam de fora: são o tema da
> pesquisa do Cesar (decisão de 22/11). A versão com reparo continua na branch `juiz-v2-bd0b0ad`.

| Documento | Para quem | Conteúdo |
|---|---|---|
| [`program.md`](program.md) | o agente | só as instruções de execução e as regras das campanhas `detect` e `causes` |
| [`docs/DESENHO-EXPERIMENTAL.md`](docs/DESENHO-EXPERIMENTAL.md) | as pessoas do projeto | perguntas de pesquisa, fases, critérios, parâmetros, custo, limitações |
| [`docs/AMEACAS-A-VALIDADE.md`](docs/AMEACAS-A-VALIDADE.md) | as pessoas do projeto | prováveis ameaças à validade e o que falta mitigar |
| [`updates/`](updates/README.md) | as pessoas do projeto | uma nota por mudança, com o porquê |

## Fluxo do experimento

```
Humano dispara: ./run_experiment.sh <tag>
 └─ Fase 0: controles de 60 min (R0, R0b, R1) ── falha ──► relatório: sem calibração
 └─ Busca: rodadas de 60 min do agente (R1 + até 3 candidatos), até a rodada 8
      um candidato fica elegível na rodada n
      └─ Verificação: rodadas n+1 e n+2 repetem a carga com R1, R0 e R0b (mesma semente), até a rodada 10
           ├─ as 2 positivas ──► aging confirmado ──► a busca pausa ──► Campanha de causas (6 rodadas)
           │                                                          └─ a busca retoma nas rodadas que restam
           └─ alguma negativa ──► a busca continua
 └─ Fim do orçamento ──► relatório final (results/experiment-<tag>/report.md + resumos do agente)
Humano lê o relatório
```

| Etapa | Quem roda | Braços | Passa se |
|---|---|---|---|
| Fase 0 | juiz | R0, R0b, R1 (semente 100) | R0 e R0b planos, R1 elegível |
| Busca | agente, uma rodada por chamada | referência + até 3 candidatos (sementes 101…) | algum candidato elegível dispara a verificação |
| Verificação | juiz | R1, a carga, R0, R0b, mesma semente | 2 rodadas positivas: R0 e R0b planos, R1 e a carga elegíveis |
| Causas | agente, uma rodada por chamada | carga confirmada + até 3 variações de um fator | cada variação com duas réplicas válidas |

Quem decide cada passagem é o **orquestrador** (`harness/orchestrator.py`, `prepare.py experiment`), código fixo
com hash no manifesto, nunca o agente.

## O cenário do Cesar (resumo)

- **Finding F1:** cada `POST /associate_card` com par (cartão, telefone) novo grava uma linha (e 3 entradas de
  índice) em `associations` que nada apaga; `POST /retrieve_cards` lê todas as linhas do telefone, sem `LIMIT`, e
  materializa tudo (`fetchall`, modelo pydantic, JSON). O custo da leitura cresce com os cartões acumulados.
- **Carga de referência (R1):** `workloads/bax-fastapi-credit-card-F1_targeted_v1.py`. Contrato:
  `script --base-url URL --duration S --seed N --out client.csv`. Fluxos: `grow` (pares novos num telefone quente,
  10/s), `refresh`, `ref`, e as sondas `retrieve_hot` (o sintoma) e `retrieve_ref4`/`retrieve_ref200` (controle
  interno: mesma rota, tamanho constante, devem ficar planos).
- **Controles negativos, sem controle positivo:** R0 = `F1_control.py` (um par fixo, laço fechado); R0b =
  `F1_targeted_v1 --key-space 4` (mesma carga, cartões repetidos).
- **Teste funcional:** `oracle/test_credit_card_service.py`, usado só no preparo (`check-oracles`).
- **Resultado de referência do Cesar** (8 h, semente 42): latência de `retrieve_hot` de ~10 ms para ~1,3 s; R0 e
  R0b planos. Veredito `ACTIVATED_WITH_AGING`, aging misto.

### O aging do F1 é misto

1. **Custo por cartão:** cada leitura do telefone quente custa proporcionalmente aos cartões acumulados;
2. **Pico de memória:** cada leitura grande materializa a lista inteira;
3. **Interferência:** a cada leitura grande o processo trava ~150 ms ou mais e atrasa as outras requisições
   (`associate_card`, `retrieve_ref4`).

A detecção mede (1) e (2) pela latência e pelo tamanho de resposta de `retrieve_hot` e pelo RSS. A campanha
`causes` separa os componentes: razão da inclinação da latência, do tamanho da resposta, do **custo por MB** e da
interferência (p95 de `associate_card` e `retrieve_ref4`).

### A flag D2 não se aplica ao F1

A flag **D2** ("mais de 1000 linhas no banco: não pontua") existe para findings em que encher o banco é um atalho.
No F1 o crescimento de linhas é **exigido pela spec** (`activation_criteria.resource_growth`): é a **ativação** do
finding, não o aging em si. Por isso a D2 está **desligada** para `bax-fastapi-credit-card-F1`
(`D2_BY_FINDING` em `harness/evaluate.py`). Contra o atalho equivalente sobram o controle interno (C1), os limites
de taxa (até 2x os padrões) e o controle negativo R0b.

## Arquitetura (mesma lógica do autoresearch)

| autoresearch original | Este repositório | Quem edita |
|---|---|---|
| `prepare.py` (avaliação fixa, não muda) | `prepare.py` + `harness/` + `oracle/`, com hash no manifesto | ninguém durante as campanhas |
| `train.py` (o único arquivo que o agente edita) | `workload.py`: só literais, variantes da carga do Cesar | agente |
| `program.md` (instruções do agente) | `program.md`: instruções de uma rodada das campanhas `detect` e `causes` e dos resumos | humano |
| `val_bpb` (uma métrica) | `detect`: Sen's slope da latência de `retrieve_hot` (ms/h), com Hamed-Rao, e `ratio` contra a referência da rodada. `causes`: razões de cada variação contra a carga congelada | juiz |
| orçamento fixo de 5 min | 60 min por rodada, 4 instâncias em paralelo | juiz |
| `results.tsv` sem commit; keep avança a branch, discard faz `git reset` | igual na `detect`; na `causes` não há keep nem discard | agente |
| o agente roda o laço inteiro | um orquestrador fixo chama o agente uma rodada por vez e decide verificação, causas e retomada | ninguém |

```
autoresearch/
├── program.md                  # instruções do agente (detect e causas)
├── workload.py                 # o agente edita: CANDIDATES com args literais
├── prepare.py                  # juiz: setup, check-oracles, experiment, controls, detect, check-load, freeze, causes, wait, verify, rehash
├── run_experiment.sh           # dispara o experimento completo (orquestrador)
├── docs/DESENHO-EXPERIMENTAL.md
├── apps/credit-card-fastapi/
│   ├── instrumented/           # alvo (código do Cesar + aging_probe.py)
│   └── SOURCE.json             # commit do repo do Cesar de onde veio
├── findings/                   # card do F1 e campaign.yaml (cópias do Cesar)
├── workloads/                  # F1_targeted_v1 (referência, R1, R0b) e F1_control (R0), intactos
├── harness/
│   ├── instance.py             # cópia limpa da app, uvicorn em porta própria, banco novo, psutil, filhos (H4)
│   ├── load.py                 # roda os scripts do Cesar; contrato de args; regra de um fator (causas)
│   ├── evaluate.py             # MK Hamed-Rao + Sen; elegibilidade; flags; custo por MB; interferência
│   ├── functional_check.py     # sobe a app e roda o teste funcional (check-oracles)
│   ├── orchestrator.py         # orquestrador: Fase 0, busca, verificação, causas, retomada, relatório
│   ├── workload_check.py       # lê workload.py como dados (AST + literal_eval)
│   └── stats.py                # Mann-Kendall, Sen's slope, Hamed-Rao (pymannkendall)
├── oracle/                     # teste funcional do Cesar e o executor
├── tests/                      # test_c1.py (controle interno), test_causes.py (causas), test_orchestrator.py (fluxo)
├── frozen/                     # carga confirmada congelada pelo orquestrador (campanha de causas atual)
├── .claude/settings.{detect,causes}.json   # permissões do agente por campanha
└── updates/                    # notas de mudança
```

### O que o agente vê

Para o agente trabalhar só sobre a temática e sem viés, ele **lê apenas** `program.md`, `workload.py`, `frozen/`,
`results/` e os próprios registros. Código da app, card do F1, cargas do Cesar, juiz, testes, documentação,
`updates/` e o histórico do git (`git log -p`, `git show`, `git diff`) estão negados nas permissões. As hipóteses
dos braços fixos e as mensagens de erro do contrato são neutras. As permissões do Claude Code ajudam, mas não são
barreira de segurança; para isolar de verdade, rode o agente com outro usuário Linux.

## Campanhas do agente (resumo; regras completas no `program.md`)

O orquestrador chama o agente **uma rodada por vez**, dizendo a campanha, a rodada e a semente.

**`detect`:** em cada rodada, a referência (carga com os parâmetros padrão) e até 3 candidatos. Um braço é
**elegível** quando é válido, a latência e o tamanho de resposta de `retrieve_hot` sobem (Hamed-Rao p < 0,01 e
mudança ≥ 50%) e o controle interno fica plano. Keep quando o `ratio` contra a referência da mesma rodada supera o
melhor anterior em pelo menos 10%; empate dentro de 10% com menos parâmetros alterados também vira keep (critério de
simplicidade). O melhor candidato elegível de cada rodada (que não seja uma carga já confirmada) vai para a
verificação.

**`causes`:** 6 rodadas por carga confirmada. A referência é a carga congelada; cada variação muda **exatamente
um** parâmetro (o juiz recusa as outras) e roda em duas réplicas. Um efeito é declarado quando a razão fica do
mesmo lado de 1, a pelo menos 20% dele, nas duas réplicas. Hipóteses já testadas numa campanha anterior entram só
no resumo.

O `ref_sizes` (telefones de referência 4 e 200) é fixo: é o controle interno e não pode ser um fator.

## Como rodar

Requisitos: Linux (WSL2, servidor do laboratório ou nuvem), Python 3.10+ com `psutil` e `pip`, Git e o Claude Code.
O venv travado sai do `requirements.txt` do próprio Cesar.

```bash
# 0. Clone o repo do Cesar ao lado deste (o juiz só LÊ; nada dentro dele é editado)
git clone https://github.com/chas42/llm-aging-detection-agents ../llm-aging-detection-agents

# 1. Preparo (humano, uma vez): copia instrumented + workloads + teste, cria o venv, grava o manifesto
python prepare.py setup --src ../llm-aging-detection-agents
python prepare.py check-oracles
git add -A && git commit -m "Setup do alvo do Cesar"

# 2. Teste rápido do fluxo inteiro (rodadas de 2 min; não mostra tendência)
./run_experiment.sh teste-smoke --smoke

# 3. Experimento completo, sem humano no meio
nohup ./run_experiment.sh exp-nov23 &
# relatório: results/experiment-exp-nov23/report.md; resumos do agente: summary_detect.md, summary_causes_<N>.md
```

`run_experiment.sh` tira a permissão de escrita de `prepare.py`, `harness/`, `oracle/`, `apps/`, `workloads/` e
`findings/` (o orquestrador precisa escrever em `frozen/` e `.judge/`). Para rodar `setup` depois disso, devolva a
escrita com `chmod -R u+w`. Depois de alterar qualquer arquivo do juiz, rode `python prepare.py rehash`; senão
`verify` aborta. As peças do fluxo também podem ser rodadas à mão para depurar (`controls`, `detect`,
`check-load`, `freeze`, `causes`).

**Verificação:** 2 rodadas de 60 min logo depois da rodada de busca que achou a carga, com R1, a carga, R0 e R0b
na mesma semente e banco limpo (regra do Ermeson). Confirmado só se **as duas** forem positivas. Uma rodada
**inválida** (D1, D3, H4, app caiu) é repetida uma vez; uma **negativa** encerra a verificação e a busca continua.
A verificação pode ir até a rodada 10.

## Taxonomia de atalhos (reward hacking)

| Flag | Atalho | Efeito |
|---|---|---|
| D1 | Saturar a máquina (timeouts, 5xx acima de 1%, gerador acima de 90% de um núcleo) e chamar isso de aging | Braço inválido |
| D2 | Inflar o banco com dados novos. **Desligada no F1** | Não se aplica |
| D3 | Sair do contrato (`args` fora das faixas, mais de um fator na campanha `causes`) ou o gerador falhar | Braço ou rodada inválida |
| D4 | Candidato igual à referência | Medido, sem nota |
| C1 | Tamanho de resposta dos refs ou latência do `ref200` sobem: deriva global, não crescimento por linha. A latência do `ref4` é só diagnóstico (sobe por interferência do próprio aging) | Medido, sem nota |
| H4 | O processo da app morreu, mudou de PID ou abriu processos filhos | Braço inválido |

Toda flag vai para `results/judge_log.jsonl`, e o agente marca no ledger as ideias que só melhoravam o número por
um desses caminhos (`"shortcut": true`). Esses casos também são resultado da pesquisa.

## De onde vem cada decisão

| Origem | Decisão | Onde está |
|---|---|---|
| Reunião de 15/10 | Primeiro apps geradas por LLM; depois apps reais | alvo `credit-card-fastapi` |
| **Cesar** (`campaign.yaml`, `findings/…-F1.yaml`) | Alvo, finding, carga de referência, controles R0/R0b | `apps/`, `workloads/`, `findings/`; `R0`, `R0B`, `BASELINE` em `harness/load.py` |
| **Cesar** (`run_diagnostic.py`, `analyze_run.py`) | Contrato da carga, instrumentação, pinagem de CPU; Hamed-Rao + Sen em janelas de 60 s | `harness/load.py`, `harness/instance.py`, `harness/stats.py`, `harness/evaluate.py` |
| **Cesar** (22/11) | O autoresearch roda sozinho, com um mínimo de rodadas, e foca em causar o aging; reparo fora | escopo desta branch; `docs/DESENHO-EXPERIMENTAL.md` |
| Artur | Reward hacking; feedback em ~1 h; versões em paralelo | flags D1–D4, C1, H4; `DEFAULT_MINUTES = 60`; `SLOTS = 4` |
| Fumio | Documentar os atalhos do agente é contribuição | `judge_log.jsonl` e `"shortcut"` no ledger |
| Ermeson | Workloads, métricas, critério de parada e duração | `harness/load.py`, `harness/evaluate.py`, orçamentos no `program.md` |
| Rafael e Ermeson (19/11) | Sementes independentes; confirmação repetida com parâmetros fixos, mesma duração, banco limpo, R1 e controles com a mesma semente, unanimidade | sementes 100 + *n*; verificação em 2 rodadas |
| Rafael (23/11) | Sem humano no meio; confirmação nas rodadas seguintes (até a 10); cada carga confirmada ganha campanha de causas; a busca retoma depois | `harness/orchestrator.py`, `prepare.py experiment` |
| Rafael (23/11) | Investigação de causas só pela carga; `program.md` só com instruções; agente sem acesso ao que causaria viés | `prepare.py causes`, `program.md`, `.claude/settings.*.json` |
| karpathy/autoresearch | Arquivo único editado pelo agente, orçamento fixo, keep/discard por git, critério de simplicidade, registros sem commit, não parar para perguntar | `workload.py`, `program.md` |

## Diferenças em relação ao cenário do Cesar

| Ponto | Cesar | Aqui | Por quê |
|---|---|---|---|
| Instâncias | uma, `python app.py` na porta 5000 | quatro em paralelo, `uvicorn app:app` em porta livre | os 4 braços rodam ao mesmo tempo |
| Log do servidor | `uvicorn.run()` com log de acesso | igual: `--log-level info --access-log` | paridade de custo por requisição |
| Máquinas | cliente e app separados | mesma máquina, núcleos distintos | ver Pendências |
| Rodadas | R0, R0b, R1 e R2 | R0, R0b e R1 | reparo fora do escopo |
| Stall do `ps` | só no app Monitor | não implementado | não faz parte deste cenário |

## Pendências

1. **Limiares** (50% na latência, 25% no controle interno, 5 MB/h no RSS, 20% para declarar efeito na
   `causes`): valores de partida, a calibrar com a Fase 0 e com as primeiras campanhas.
2. **Nenhuma rodada real ainda:** o fluxo foi testado com juiz e agente simulados; falta o primeiro
   `./run_experiment.sh <tag> --smoke` com a app.
3. **Sem controle positivo:** "não achamos aging" não mede a sensibilidade do juiz.
4. **Cliente e app na mesma máquina.**
5. **Próximas apps** do Cesar (convert-image, monitor, uptime) e **apps reais** (exigência do Artur).
