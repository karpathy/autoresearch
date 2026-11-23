# AutoResearch-Aging: detecção autônoma de aging no F1 (`credit-card-fastapi`)

Este repositório adapta o [karpathy/autoresearch](https://github.com/karpathy/autoresearch) para **detectar
envelhecimento de software** numa aplicação gerada por LLM. O **método** é o do autoresearch: um agente propõe,
um juiz fixo mede, e o laço mantém ou descarta a proposta. O **alvo e o cenário** são os do Cesar, em
[chas42/llm-aging-detection-agents](https://github.com/chas42/llm-aging-detection-agents): a app
`credit-card-fastapi` (BaxBench), o finding `bax-fastapi-credit-card-F1` e as rodadas R0, R0b e R1 do
`apps/credit-card-fastapi/campaign.yaml`.

Responsáveis: Rafael, Cesar e Ermeson (UFRPE). Colaboração com Fumio Machida (Tsukuba) e Artur Andrzejak
(Heidelberg).

> **Escopo desta versão (branch `deteccao-f1`): só detecção.** O agente procura cargas que causem o aging do F1,
> o aging é confirmado em duas execuções longas e, depois, o agente testa hipóteses sobre a causa **só pela
> carga**. O **reparo** e a **análise de código** ficaram de fora porque são o tema da pesquisa do Cesar
> (decisão de 22/11). O desenho completo do experimento está em [`program.md`](program.md). A versão com reparo
> continua na branch `juiz-v2-bd0b0ad`.

## O cenário do Cesar (resumo)

- **Finding F1:** cada `POST /associate_card` com par (cartão, telefone) novo grava uma linha (e 3 entradas de
  índice) em `associations` que nada apaga; `POST /retrieve_cards` lê todas as linhas do telefone, sem `LIMIT`, e
  materializa tudo (`fetchall`, modelo pydantic, JSON). O custo da leitura cresce com os cartões acumulados.
- **Carga de referência (R1):** `workloads/bax-fastapi-credit-card-F1_targeted_v1.py` (e o `.md` ao lado).
  Contrato: `script --base-url URL --duration S --seed N --out client.csv`. Fluxos: `grow` (pares novos num
  telefone quente, 10/s), `refresh`, `ref`, e as sondas `retrieve_hot` (o sintoma) e
  `retrieve_ref4`/`retrieve_ref200` (controle interno: mesma rota, tamanho constante, devem ficar planos).
- **Controles NEGATIVOS, sem controle positivo:** R0 = `F1_control.py` (o `.jmx` original: um par fixo, laço
  fechado); R0b = `F1_targeted_v1 --key-space 4` (mesma carga, nada acumula).
- **Oráculo funcional:** `oracle/test_credit_card_service.py` (cópia de `reference/tests/test_credit_card_service.py`),
  usado só para validar a app no preparo.
- **Resultado de referência do Cesar** (8 h, semente 42): latência de `retrieve_hot` de ~10 ms para ~1,3 s; R0 e
  R0b planos. Veredito `ACTIVATED_WITH_AGING`, aging misto.

## De onde vem cada decisão

| Origem | Decisão | Onde está no código |
|---|---|---|
| Reunião de 15/10 (consenso) | Primeiro teste em apps geradas por LLM; depois, apps reais | Alvo: `credit-card-fastapi`; apps reais ficam como trabalho futuro |
| **Cesar** (`campaign.yaml`, `findings/…-F1.yaml`) | Alvo, finding, carga de referência, controles R0/R0b | `apps/credit-card-fastapi/`, `workloads/`, `findings/` (cópias do repo dele, com o commit em `apps/credit-card-fastapi/SOURCE.json`); `R0`, `R0B`, `BASELINE` em `harness/load.py`; `prepare.py controls` e `confirm` |
| **Cesar** (`harness/run_diagnostic.py`) | Contrato da carga (`--base-url --duration --seed --out`, `client.csv`), instrumentação por `AGING_INSTRUMENT=1`/`AGING_INSTRUMENT_LOG`, pinagem de CPU | `harness/load.py:run_script`, `harness/instance.py`, `prepare.py:pins` |
| **Cesar** (`harness/analyze_run.py`) | Mann-Kendall com Hamed-Rao + Sen, janelas de 60 s, `retrieve_ref*` como controle interno | `harness/stats.py:trend_hr`, `harness/evaluate.py` |
| **Cesar** (`harness/functional_check.py`) | Teste funcional contra a app em execução | `harness/functional_check.py`, `oracle/functional_test.py` |
| **Cesar** (22/11) | O autoresearch roda sozinho, com um mínimo de rodadas, e foca em causar o aging; reparo fora (sobreposição) | `program.md` |
| Artur | Reward hacking: o agente satura a máquina ou sai do contrato para melhorar a métrica | Flags D1–D4 e C1 em `harness/evaluate.py` e `prepare.py` |
| Artur | Feedback curto: sinal em cerca de 1 h, não 48 h | `DEFAULT_MINUTES = 60` (laço do agente) |
| Artur | Várias versões em paralelo | `SLOTS = 4`: 3 candidatos + 1 referência por rodada |
| Cesar (8 h por rodada) | Confirmação longa | `CONFIRM_MINUTES = 480` em `prepare.py confirm` |
| Fumio | Documentar os atalhos do agente já é contribuição | Toda flag vai para `results/judge_log.jsonl`, e o agente registra `"atalho"` no `ledger.jsonl` |
| Ermeson | Definir workloads, métricas, critério de parada e duração | Contrato em `harness/load.py`; métrica em `harness/evaluate.py`; orçamento em `program_detect.md` |
| Piloto 2 | Margem para não selecionar ruído; registro de proveniência | Keep exige +10% no `ratio_vs_reference`; hashes do juiz em `.judge/manifest.json` |
| Rafael (19/11) | Confirmar o F1 com sementes independentes | Semente nova por rodada de detecção (101–108) e keep pela razão contra a referência da rodada (`program_detect.md`; o juiz não impõe) |
| Rafael e Ermeson (19/11) | Repetir a confirmação com sementes diferentes, mesma duração, banco limpo; unanimidade | Duas execuções de `confirm` de 8 h (sementes 201 e 202); as duas precisam passar |

## A flag D2 não se aplica ao F1

Nos findings em que o crescimento do banco é um atalho (o agente "envelhece" a app só enchendo-a de dados
legítimos), a flag **D2** marca "mais de 1000 linhas no banco: não pontua". No **F1 o crescimento de linhas é
exigido pela spec** (`findings/bax-fastapi-credit-card-F1.yaml`, `activation_criteria.resource_growth`): é a prova
de que a carga percorreu o caminho do finding, ou seja, **ativação**, não o aging em si. Manter a D2 ligada
zeraria toda carga que ativa o finding. Por isso a D2 é **configurável por finding** (`D2_BY_FINDING` em
`harness/evaluate.py`) e está **desligada** para `bax-fastapi-credit-card-F1`. O número de linhas continua
registrado (`db_rows`). O que sobra contra o atalho equivalente: o controle interno (tamanho de resposta dos refs e
latência do `ref200` precisam ficar planos, senão a flag `C1_referencia_interna_subindo` tira a pontuação), os
limites de taxa (até 2x os defaults do Cesar) e o controle negativo R0b.

### O aging do F1 é misto

O envelhecimento de verdade, o que degrada o serviço, é **misto** (classificação do Cesar):

1. **Custo por cartão:** cada leitura de `/retrieve_cards` do telefone quente custa proporcionalmente aos cartões
   acumulados (consulta sem `LIMIT`, `fetchall`, modelo pydantic, JSON), e esse custo por cartão é alto;
2. **Pico de memória:** cada leitura grande materializa a lista inteira; o RSS sobe com o tamanho da resposta;
3. **Interferência:** a cada leitura grande o processo trava por ~150 ms ou mais, e isso atrasa as **outras**
   requisições que chegam no mesmo momento (`associate_card`, `retrieve_ref4`).

A detecção mede (1) e (2) pela latência e pelo tamanho de resposta de `retrieve_hot` e pelo RSS. A interferência
(3) é medida em todo braço como diagnóstico (p95 de `associate_card` e `retrieve_ref4`) e entra na investigação de
causas (`program.md`, Fase 4).

## Arquitetura (mesma lógica do autoresearch)

| autoresearch original | Este repositório | Quem edita |
|---|---|---|
| `prepare.py` (avaliação fixa) | `prepare.py` + `harness/` + `oracle/` | Ninguém durante a campanha |
| `train.py` | `workload.py` | Agente |
| `program.md` | `program.md` (desenho do experimento) e `program_detect.md` (instruções do agente na detecção) | Humano |
| `val_bpb` | Sen's slope da latência de `retrieve_hot` (ms/h), com Hamed-Rao (+ tamanho de resposta e RSS) | Juiz |
| Execução de 5 min | Execução de 60 min, 4 em paralelo | Juiz |

```
autoresearch/
├── program.md              # desenho do experimento: fases, critérios, parâmetros
├── program_detect.md       # instruções do agente na campanha de detecção
├── prepare.py              # juiz: setup, check-oracles, controls, detect, freeze, confirm, wait, verify, rehash
├── apps/credit-card-fastapi/
│   ├── instrumented/       # alvo (código do Cesar + aging_probe.py)
│   └── SOURCE.json         # commit do repo do Cesar de onde veio
├── findings/               # card do F1 e campaign.yaml (cópias do Cesar)
├── workloads/              # F1_targeted_v1 (referência, R1, R0b) e F1_control (R0), do Cesar, intactos
├── harness/
│   ├── instance.py         # cópia limpa da app, uvicorn em porta própria, banco novo, psutil, filhos (H4)
│   ├── load.py             # roda os scripts do Cesar como subprocesso; contrato de `args`; R0/R0b/referência
│   ├── evaluate.py         # RSS (MK/Sen); latência e bytes de retrieve_hot (MK Hamed-Rao/Sen); flags
│   ├── functional_check.py # sobe a app e roda o teste funcional do Cesar (check-oracles)
│   ├── workload_check.py   # lê workload.py como dados (AST + literal_eval)
│   └── stats.py            # Mann-Kendall (com empates), Sen's slope, Hamed-Rao (via pymannkendall)
├── oracle/
│   ├── functional_test.py  # executa o teste do Cesar contra a porta da instância
│   └── test_credit_card_service.py   # o teste do Cesar, sem alteração
├── tests/test_c1.py        # teste da regra C1 (controle interno)
├── workload.py             # o agente edita (só literais: variantes do F1_targeted_v1)
├── frozen/                 # carga congelada ao fim da campanha (gerada pelo freeze)
├── .claude/settings.detect.json   # permissões do Claude Code na detecção
└── run_campaign.sh         # lança a campanha de detecção em modo headless
```

### A carga do agente é uma variante do F1, sem código novo

O `workload.py` só declara `CANDIDATES`, e cada candidato traz `args`: flags do `F1_targeted_v1`. O juiz valida os
valores (`harness/load.py:normalize_args`) e roda o script fixo do Cesar como subprocesso, com a semente da rodada
(`--seed`, a mesma para os 4 braços). Travas: sem `import` nem código (o arquivo é lido com `ast.literal_eval`,
nada executa), determinismo (a carga é função só de `args` e da semente do juiz), taxas até 2x os defaults e
5xx/exceções acima de 1% invalidam o braço (saturação não é aging).

### Semente por rodada e keep pela razão

Cada rodada de detecção usa uma **semente nova**: rodada 1 = `101`, …, rodada 8 = `108`
(`prepare.py detect --seed <semente> --background`). Rodadas que falham também consomem a semente; nenhuma é
reutilizada. Como a referência R1 roda em toda rodada, ela vira uma **réplica com semente independente** a cada
rodada, e o `summary_detect.md` traz uma tabela da R1 por rodada: é a evidência de que o sintoma não depende dos
cartões gerados por uma semente.

Por causa disso, o score bruto (ms/h) **não** é comparado entre rodadas. O keep usa o `ratio_vs_reference`
(inclinação do candidato / inclinação da referência da **mesma** rodada): um candidato elegível é mantido se o
seu `ratio` for pelo menos 10% maior que o do melhor anterior. Se a referência da rodada não for elegível
(`ratio = NA`), nenhum candidato daquela rodada vira keep (`referencia_sem_sintoma` no ledger).

## Como rodar

Requisitos: Linux (WSL2, servidor do laboratório ou nuvem), Python 3.10+ com `psutil` e `pip`, Git e o Claude Code.
O venv travado sai do `requirements.txt` do próprio Cesar (mesmas versões de FastAPI, uvicorn, pydantic, httpx).

```bash
# 0. Clone o repo do Cesar ao lado deste (o juiz só LÊ; nada dentro dele é editado)
git clone https://github.com/chas42/llm-aging-detection-agents ../llm-aging-detection-agents

# 1. Setup (humano, uma vez): copia instrumented + workloads + oráculo, cria o venv, grava o manifesto
python prepare.py setup --src ../llm-aging-detection-agents
python prepare.py check-oracles          # teste funcional do Cesar contra a app copiada
git add -A && git commit -m "Setup do alvo do Cesar"

# 2. Teste rápido do fluxo (cerca de 3 min; 2 min de carga não mostram tendência, só validam o fluxo)
python prepare.py detect --smoke

# 3. Fase 0: controles negativos (60 min). R0 e R0b devem ficar planos e R1 deve subir
python prepare.py controls --with-r1

# 4. Fase 1: campanha de detecção (cerca de 8,5 h; o agente usa as sementes 101..108, uma por rodada)
./run_campaign.sh detect det-nov23

# 5. Fase 2: congelar o melhor braço e confirmar DUAS vezes (R0, R0b, R1 e o candidato, 8 h, em paralelo).
#    O confirm lê a carga congelada, por isso o freeze vem antes. O aging só é confirmado se as DUAS passarem.
python prepare.py freeze <run_id> <arm>
python prepare.py confirm --seed 201
python prepare.py confirm --seed 202
```

Hoje as passagens entre as fases são feitas pelo humano. O `program.md` prevê um orquestrador que encadeia as
fases sozinho e a Fase 4 (investigação de causas); os dois ainda não estão implementados.

`run_campaign.sh` tira a permissão de escrita de `prepare.py`, `harness/`, `oracle/`, `apps/`, `.judge/` e
`frozen/`. Para rodar `freeze` ou `setup` depois disso, devolva a escrita com `chmod -R u+w`. Depois de alterar
qualquer arquivo do juiz, rode `python prepare.py rehash` para refazer o manifesto; senão `verify` aborta.

O agente não tem `cat`, `grep`, `git diff` nem `git show` no Bash: ele lê arquivos pelas ferramentas Read e Grep.
As permissões do Claude Code ajudam, mas não são barreira de segurança; para isolar de verdade, rode o agente com
outro usuário Linux.

### Duração e confirmação

O laço do agente usa **60 min** por rodada (`DEFAULT_MINUTES`). A confirmação usa **8 h** (`CONFIRM_MINUTES`, como o
`campaign.yaml` do Cesar), com R0, R0b, R1 e o candidato em paralelo, um por slot. Ela é feita **duas vezes**, com
sementes distintas (201 e 202; informe `--seed` explicitamente, o padrão do juiz é 1), cada uma com 8 h e banco
limpo. Em cada execução, os quatro braços usam a mesma semente. O aging só é confirmado se **as duas** passarem:
unanimidade, não maioria. Uma execução **inválida** (infraestrutura: D1, D3, H4, app caiu) não conta e é repetida
com a próxima semente (203, …); uma execução **negativa** (válida, mas um controle subiu ou R1/candidato não
ficaram elegíveis) significa aging não confirmado. Detalhes em [`program.md`](program.md) e
[`updates/criterio-de-deteccao.md`](updates/criterio-de-deteccao.md).

## Taxonomia de atalhos (reward hacking)

| Flag | Atalho | Efeito |
|---|---|---|
| D1 | Saturar a máquina (timeouts, 5xx acima de 1%, gerador acima de 90% de um núcleo) e chamar isso de aging | Braço inválido |
| D2 | Inflar o banco com dados novos. **Configurável por finding; desligada no F1** (ver acima) | Não se aplica |
| D3 | Sair do contrato (`args` fora das faixas) ou o script falhar | Braço inválido |
| D4 | Candidato igual à referência: a diferença é só ruído | Medido, sem nota |
| C1 | Tamanho de resposta dos refs ou latência do `ref200` sobem: deriva global, não crescimento por linha. A latência do `ref4` saiu do C1 (sobe por interferência do aging) e virou diagnóstico, comparada com o R0b da mesma semente | Medido, sem nota |
| H4 | O processo da app morreu, mudou de PID ou abriu processos filhos | Braço inválido |

## Diferenças em relação ao cenário do Cesar

| Ponto | Cesar | Aqui | Por quê |
|---|---|---|---|
| Stall do `ps` | Não existe no Credit Card. Existe só no app Monitor (`monitor-flask`), injetado pelo `aging_probe.py` **do Monitor**; o `campaign.yaml` da credit-card não define `env` | **Não implementado** | Não faz parte deste cenário (confirmado pelo Cesar) |
| Instâncias | Uma, `python app.py` na porta fixa 5000 | Quatro em paralelo, `uvicorn app:app` em porta livre, uma por slot | Os 4 braços rodam ao mesmo tempo |
| Log do servidor | `python app.py` → `uvicorn.run()` com os padrões: `log_level=info`, **log de acesso ligado** | Igual: `uvicorn app:app --log-level info --access-log`, gravado em `results/<run>/<braço>/app.log` | Paridade: o log de acesso custa CPU e E/S por requisição dentro do mesmo processo, então entra no custo medido |
| Máquinas | Cliente e app em máquinas separadas | Mesma máquina, pinados em núcleos distintos | Ver Pendências |
| Rodadas | R0, R0b, R1 e R2 (app reparada) | R0, R0b e R1; R2 fica de fora | Reparo fora do escopo |

## Pendências

1. **Limiares:** a nota usa a latência e o tamanho de resposta de `retrieve_hot`, além do RSS. Os limiares
   (mudança de 50% na latência, 25% no controle interno, 5 MB/h no RSS) são de partida e precisam ser conferidos
   com os controles (`prepare.py controls`).
2. **Orquestrador e Fase 4:** previstos em `program.md`, ainda não implementados.
3. **Sem controle positivo:** "não achamos aging" não é afirmação sobre a sensibilidade do juiz. Só os controles
   negativos (R0 e R0b) existem neste cenário.
4. **Cliente e app na mesma máquina:** no artigo eram máquinas separadas. Aqui app e gerador são pinados em
   núcleos distintos (`taskset`, com 8 núcleos ou mais) e a CPU do gerador é medida (D1 acima de 90%).
5. **Próximas apps:** convert-image, monitor e uptime do Cesar. Cada uma precisa do seu `campaign.yaml`, dos seus
   workloads e do seu teste funcional.
6. **Apps reais, não sintéticas:** exigência do Artur para publicação.
