# AutoResearch-Aging: detecção autônoma de aging no F1 (`credit-card-fastapi`)

Este repositório adapta o [karpathy/autoresearch](https://github.com/karpathy/autoresearch) para **detectar e
explicar envelhecimento de software** numa aplicação gerada por LLM. O **método** é o do autoresearch: um agente
propõe, um juiz fixo mede, e o laço mantém ou descarta a proposta. O **alvo e o cenário** são os do Cesar, em
[chas42/llm-aging-detection-agents](https://github.com/chas42/llm-aging-detection-agents): a app
`credit-card-fastapi` (BaxBench), o finding `bax-fastapi-credit-card-F1` e as rodadas R0, R0b e R1 do
`apps/credit-card-fastapi/campaign.yaml`.

Responsáveis: Rafael, Cesar e Ermeson (UFRPE). Colaboração com Fumio Machida (Tsukuba) e Artur Andrzejak
(Heidelberg).

> **Escopo (branch `deteccao-f1`): só detecção e explicação pela carga.** O agente procura cargas que causem o
> aging do F1; o aging é confirmado em duas execuções longas; depois o agente testa, variando um fator da carga de
> cada vez, o que causa o envelhecimento. O **reparo** e a **análise de código** ficam de fora: são o tema da
> pesquisa do Cesar (decisão de 22/11). A versão com reparo continua na branch `juiz-v2-bd0b0ad`.

| Documento | Para quem | Conteúdo |
|---|---|---|
| [`program.md`](program.md) | o agente | só as instruções de execução e as regras das campanhas `detect` e `causas` |
| [`docs/DESENHO-EXPERIMENTAL.md`](docs/DESENHO-EXPERIMENTAL.md) | as pessoas do projeto | perguntas de pesquisa, fases, critérios, parâmetros, custo, limitações |
| [`updates/`](updates/README.md) | as pessoas do projeto | uma nota por mudança, com o porquê |

## Fluxo do experimento

```
Preparo (humano): setup, check-oracles, verify
  └─ Fase 0. Controles 60 min (R0, R0b, R1) ──── R0/R0b sobem ou R1 não sobe ──► fim: sem calibração
       └─ Fase 1. Campanha detect (agente, 8 rodadas de 60 min, sementes 101–108)
            └─ Fase 2. freeze do melhor braço + confirm 2 × 8 h (sementes 201, 202) ── alguma negativa ──► fim: não confirmado
                 └─ Fase 3. Registro: aging confirmado + carga congelada
                      └─ Fase 4. Campanha causas (agente, 6 rodadas de 60 min, sementes 301–306) ──► tabela de fatores
```

| Fase | Quem roda | Comando | Passa se |
|---|---|---|---|
| 0. Controles | humano | `prepare.py controls --with-r1` | R0 e R0b planos, R1 elegível |
| 1. Detecção | agente | `./run_campaign.sh detect <tag>` | existe um braço elegível (keep ou a referência) |
| 2. Confirmação | humano | `prepare.py freeze <run_id> <arm>`, depois `confirm --seed 201` e `--seed 202` | as duas execuções: R0 e R0b planos, R1 e candidato elegíveis |
| 4. Causas | agente | `./run_campaign.sh causas <tag>` | cada variação com duas réplicas válidas |

Hoje as passagens entre as fases são feitas pelo humano. Um orquestrador que encadeie tudo sozinho está previsto
no desenho, mas ainda não foi implementado.

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
`causas` separa os componentes: razão da inclinação da latência, do tamanho da resposta, do **custo por MB** e da
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
| `program.md` (instruções do agente) | `program.md`: instruções das campanhas `detect` e `causas` | humano |
| `val_bpb` (uma métrica) | `detect`: Sen's slope da latência de `retrieve_hot` (ms/h), com Hamed-Rao, e `ratio` contra a referência da rodada. `causas`: razões de cada variação contra a carga congelada | juiz |
| orçamento fixo de 5 min | 60 min por rodada, 4 instâncias em paralelo | juiz |
| `results.tsv` sem commit; keep avança a branch, discard faz `git reset` | igual na `detect`; na `causas` não há keep nem discard | agente |

```
autoresearch/
├── program.md                  # instruções do agente (detect e causas)
├── workload.py                 # o agente edita: CANDIDATES com args literais
├── prepare.py                  # juiz: setup, check-oracles, controls, detect, freeze, confirm, causes, wait, verify, rehash
├── run_campaign.sh             # lança uma campanha (detect ou causas) em modo headless
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
│   ├── workload_check.py       # lê workload.py como dados (AST + literal_eval)
│   └── stats.py                # Mann-Kendall, Sen's slope, Hamed-Rao (pymannkendall)
├── oracle/                     # teste funcional do Cesar e o executor
├── tests/                      # test_c1.py (controle interno) e test_causes.py (fase de causas)
├── frozen/                     # carga congelada (gerada pelo freeze)
├── .claude/settings.{detect,causas}.json   # permissões do agente por campanha
└── updates/                    # notas de mudança
```

### O que o agente vê

Para o agente trabalhar só sobre a temática e sem viés, ele **lê apenas** `program.md`, `workload.py`, `frozen/`,
`results/` e os próprios registros. Código da app, card do F1, cargas do Cesar, juiz, testes, documentação,
`updates/` e o histórico do git (`git log -p`, `git show`, `git diff`) estão negados nas permissões. As hipóteses
dos braços fixos e as mensagens de erro do contrato são neutras. As permissões do Claude Code ajudam, mas não são
barreira de segurança; para isolar de verdade, rode o agente com outro usuário Linux.

## Campanhas do agente (resumo; regras completas no `program.md`)

**`detect`:** 8 rodadas de 60 min, sem parar antes, sementes 101–108. Em cada rodada, a referência (carga com os
parâmetros padrão) e até 3 candidatos. Um braço é **elegível** quando é válido, a latência e o tamanho de resposta
de `retrieve_hot` sobem (Hamed-Rao p < 0,01 e mudança ≥ 50%) e o controle interno fica plano. Keep quando o
`ratio` contra a referência da mesma rodada supera o melhor anterior em pelo menos 10%; empate dentro de 10% com
menos parâmetros alterados também vira keep (critério de simplicidade).

**`causas`:** 6 rodadas de 60 min, sementes 301–306, só depois da confirmação. A referência é a carga congelada; cada
variação muda **exatamente um** parâmetro (o juiz recusa as outras) e roda em duas réplicas. Um efeito é declarado
quando a razão fica do mesmo lado de 1, a pelo menos 20% dele, nas duas réplicas.

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

# 2. Teste rápido do fluxo (cerca de 3 min; não mostra tendência)
python prepare.py detect --smoke

# 3. Fase 0: controles negativos (60 min)
python prepare.py controls --with-r1

# 4. Fase 1: campanha de detecção (cerca de 8,5 h)
./run_campaign.sh detect det-nov23

# 5. Fase 2: congelar o melhor braço e confirmar duas vezes (8 h cada)
python prepare.py freeze <run_id> <arm>
python prepare.py confirm --seed 201
python prepare.py confirm --seed 202

# 6. Fase 4: investigação de causas (cerca de 6,5 h), só com as duas confirmações positivas
./run_campaign.sh causas causas-nov23
```

`run_campaign.sh` tira a permissão de escrita de `prepare.py`, `harness/`, `oracle/`, `apps/`, `.judge/` e
`frozen/`. Para rodar `freeze` ou `setup` depois disso, devolva a escrita com `chmod -R u+w`. Depois de alterar
qualquer arquivo do juiz, rode `python prepare.py rehash`; senão `verify` aborta.

**Confirmação:** duas execuções de 8 h com sementes distintas (201 e 202), banco limpo, os quatro braços com a
mesma semente. O aging só é confirmado se **as duas** passarem. Execução **inválida** (D1, D3, H4, app caiu) não
conta e é repetida com 203, 204…; execução **negativa** significa aging não confirmado e a Fase 4 não roda.

## Taxonomia de atalhos (reward hacking)

| Flag | Atalho | Efeito |
|---|---|---|
| D1 | Saturar a máquina (timeouts, 5xx acima de 1%, gerador acima de 90% de um núcleo) e chamar isso de aging | Braço inválido |
| D2 | Inflar o banco com dados novos. **Desligada no F1** | Não se aplica |
| D3 | Sair do contrato (`args` fora das faixas, mais de um fator na campanha `causas`) ou o gerador falhar | Braço ou rodada inválida |
| D4 | Candidato igual à referência | Medido, sem nota |
| C1 | Tamanho de resposta dos refs ou latência do `ref200` sobem: deriva global, não crescimento por linha. A latência do `ref4` é só diagnóstico (sobe por interferência do próprio aging) | Medido, sem nota |
| H4 | O processo da app morreu, mudou de PID ou abriu processos filhos | Braço inválido |

Toda flag vai para `results/judge_log.jsonl`, e o agente marca no ledger as ideias que só melhoravam o número por
um desses caminhos (`"atalho": true`). Esses casos também são resultado da pesquisa.

## De onde vem cada decisão

| Origem | Decisão | Onde está |
|---|---|---|
| Reunião de 15/10 | Primeiro apps geradas por LLM; depois apps reais | alvo `credit-card-fastapi` |
| **Cesar** (`campaign.yaml`, `findings/…-F1.yaml`) | Alvo, finding, carga de referência, controles R0/R0b | `apps/`, `workloads/`, `findings/`; `R0`, `R0B`, `BASELINE` em `harness/load.py` |
| **Cesar** (`run_diagnostic.py`, `analyze_run.py`) | Contrato da carga, instrumentação, pinagem de CPU; Hamed-Rao + Sen em janelas de 60 s | `harness/load.py`, `harness/instance.py`, `harness/stats.py`, `harness/evaluate.py` |
| **Cesar** (22/11) | O autoresearch roda sozinho, com um mínimo de rodadas, e foca em causar o aging; reparo fora | escopo desta branch; `docs/DESENHO-EXPERIMENTAL.md` |
| Artur | Reward hacking; feedback em ~1 h; versões em paralelo | flags D1–D4, C1, H4; `DEFAULT_MINUTES = 60`; `SLOTS = 4` |
| Fumio | Documentar os atalhos do agente é contribuição | `judge_log.jsonl` e `"atalho"` no ledger |
| Ermeson | Workloads, métricas, critério de parada e duração | `harness/load.py`, `harness/evaluate.py`, orçamentos no `program.md` |
| Rafael e Ermeson (19/11) | Sementes independentes; confirmação repetida, mesma duração, banco limpo, unanimidade | sementes 101–108 e 201/202 |
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
   `causas`): valores de partida, a calibrar com a Fase 0 e com as primeiras campanhas.
2. **Orquestrador** e **decisão automática no `confirm`** (positiva / negativa / inválida): previstos, não
   implementados.
3. **Sem controle positivo:** "não achamos aging" não mede a sensibilidade do juiz.
4. **Cliente e app na mesma máquina.**
5. **Próximas apps** do Cesar (convert-image, monitor, uptime) e **apps reais** (exigência do Artur).
