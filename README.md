# AutoResearch-Aging — Piloto 3 (detecção e reparo em apps geradas por LLM)

Este piloto adapta o [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
para **detectar e corrigir envelhecimento de software** em aplicações geradas por LLM.
O **método** é o do autoresearch (um agente propõe, um juiz fixo mede, o laço mantém ou descarta).
O **alvo e o cenário** são os do Cesar, em
[chas42/llm-aging-detection-agents](https://github.com/chas42/llm-aging-detection-agents), exatamente como ele
os usou: a app `credit-card-fastapi` (BaxBench), o finding `bax-fastapi-credit-card-F1` e as rodadas R0, R0b, R1
e R2 do `apps/credit-card-fastapi/campaign.yaml`. O estudo de origem é o de Santos, Andrade e Natella,
*Investigating Software Aging in LLM-Generated Software Systems*
([arXiv:2510.24188](https://arxiv.org/abs/2510.24188)).

Responsáveis: Rafael, Cesar e Ermeson (UFRPE). Colaboração com Fumio Machida (Tsukuba)
e Artur Andrzejak (Heidelberg).

> **Mudança de alvo (nesta ramificação).** O piloto anterior (commits `31ac544`…`bd0b0ad`) usava a `credit-card`
> em Node do `generated-software-llm`, com um vazamento sintético como controle positivo. Isso foi removido do
> código (o histórico do git guarda). Aqui **nenhum bug é injetado**: o aging vem do próprio finding F1.

## O cenário do Cesar (resumo)

- **Finding F1:** cada `POST /associate_card` com par (cartão, telefone) novo grava uma linha (e 3 entradas de
  índice) em `associations` que nada apaga; `POST /retrieve_cards` lê todas as linhas do telefone, sem `LIMIT`, e
  materializa tudo (`fetchall`, modelo pydantic, JSON). O custo da leitura cresce com os cartões acumulados.
- **Carga de referência:** `workloads/bax-fastapi-credit-card-F1_targeted_v1.py` (e o `.md` ao lado). Contrato:
  `script --base-url URL --duration S --seed N --out client.csv`. Fluxos: `grow` (pares novos num telefone quente,
  10/s), `refresh`, `ref`, e as sondas `retrieve_hot` (o sintoma) e `retrieve_ref4`/`retrieve_ref200` (controle
  interno: mesma rota, tamanho constante, devem ficar planos).
- **Controles NEGATIVOS, sem controle positivo:** R0 = `F1_control.py` (o `.jmx` original: um par fixo, laço
  fechado); R0b = `F1_targeted_v1 --key-space 4` (mesma carga, nada acumula).
- **Rodadas do `campaign.yaml`:** R0, R0b, R1 (`instrumented` + targeted) e R2 (`repaired` + targeted).
- **Oráculo funcional:** `oracle/test_credit_card_service.py` (cópia de `reference/tests/test_credit_card_service.py`).

## De onde vem cada decisão

| Origem | Decisão | Onde está no código |
|---|---|---|
| Reunião de 15/10 (consenso) | Primeiro teste em apps geradas por LLM; depois, apps reais | Alvo: `credit-card-fastapi`; apps reais ficam como trabalho futuro |
| Reunião de 15/10 (consenso) | Experimentos de **detecção** e de **reparo** | `program_detect.md` → `program_repair.md` |
| **Cesar** (`campaign.yaml`, `findings/…-F1.yaml`) | Alvo, finding, carga de referência, controles R0/R0b, rodadas R0/R0b/R1/R2 | `apps/credit-card-fastapi/`, `workloads/`, `findings/` (cópias do repo dele, com o commit em `apps/credit-card-fastapi/SOURCE.json`); `R0`, `R0B`, `BASELINE` em `harness/load.py`; `prepare.py controls` e `confirm` |
| **Cesar** (`harness/run_diagnostic.py`) | Contrato da carga (`--base-url --duration --seed --out`, `client.csv`), instrumentação por `AGING_INSTRUMENT=1`/`AGING_INSTRUMENT_LOG`, pinagem de CPU (app e gerador em núcleos distintos) | `harness/load.py:run_script`, `harness/instance.py`, `prepare.py:pins` |
| **Cesar** (`harness/analyze_run.py`) | Mann-Kendall com Hamed-Rao + Sen, janelas de 60 s, `retrieve_ref*` como controle interno | `harness/stats.py:trend_hr`, `harness/evaluate.py` |
| **Cesar** (`harness/functional_check.py`) | Oráculo funcional contra a app em execução | `oracle/functional_test.py` |
| Artur | Reward hacking: o agente remove funcionalidade ou satura a máquina para melhorar a métrica | Portões H1–H7 em `harness/gates.py`; flags D1–D4 e C1 em `harness/evaluate.py` e `prepare.py` |
| Artur | Feedback curto: sinal em cerca de 1 h, não 48 h | `DEFAULT_MINUTES = 60` (laço do agente) |
| Cesar (8 h por rodada) | Confirmação longa das quatro rodadas | `CONFIRM_MINUTES = 480` em `prepare.py confirm` |
| Artur | Várias versões em paralelo | `SLOTS = 4`: 3 candidatos + 1 referência por rodada |
| Fumio | Documentar os atalhos do agente já é contribuição | Toda flag vai para `results/judge_log.jsonl`, e o agente registra `"atalho"` no `ledger.jsonl` |
| Ermeson | Definir workloads, métricas, critério de parada e duração | Contrato em `harness/load.py`; métrica em `harness/evaluate.py`; orçamento de 8 rodadas nos `program_*.md` |
| Reunião de 20/10 (Rafael) | Duas campanhas em sequência; Claude Code; confirmação humana | Campanha 1 = detecção, campanha 2 = reparo; `confirm` e `freeze` são do humano |
| Piloto 2 | Margem para não selecionar ruído; registro de proveniência | Keep exige +10% no `ratio_vs_reference`; hashes do juiz em `.judge/manifest.json` |
| Rafael (19/11) | Confirmar o F1 com sementes independentes antes de liberar o reparo | Semente nova por rodada de detecção (101–108) e keep pela razão contra a referência da rodada: protocolo em `program_detect.md` (o juiz não impõe) |
| Rafael e Ermeson (19/11) | Repetir a confirmação com sementes diferentes, mesma duração, banco limpo; unanimidade | Duas execuções de `confirm --phase detect` de 8 h (sementes 201 e 202); reparo só se as duas passarem; protocolo humano em `updates/criterio-de-deteccao.md` |

## A flag D2 não se aplica ao F1

Nos findings em que o crescimento do banco é um atalho (o agente "envelhece" a app só enchendo-a de dados
legítimos), a flag **D2** marca "mais de 1000 linhas no banco: não pontua". No **F1 o crescimento de linhas é o
próprio finding**: `associations` nunca é podada e a latência de `retrieve_hot` sobe *porque* a tabela cresce.
Manter a D2 ligada zeraria toda carga que ativa o finding. Por isso a D2 é **configurável por finding**
(`D2_BY_FINDING` em `harness/evaluate.py`) e está **desligada** para `bax-fastapi-credit-card-F1`. O número de
linhas continua registrado (`db_rows`) como evidência de que o mecanismo foi ativado. O que sobra contra o atalho
equivalente aqui é: o controle interno (tamanho de resposta dos refs e latência do `ref200` precisam ficar planos, senão a flag
`C1_referencia_interna_subindo` tira a pontuação), os limites de taxa (até 2x os defaults do Cesar) e o
controle negativo R0b.

**Crescimento de linhas = ativação, não aging.** A spec do F1 (`findings/bax-fastapi-credit-card-F1.yaml`,
`activation_criteria.resource_growth`) **exige** que `COUNT(*)` de `associations` e o tamanho do `db.sqlite3`
cresçam: é a prova de que a carga percorreu o caminho do finding. Por isso as linhas contam como **ativação**
(e a D2 fica desligada), e não como o envelhecimento em si.

### O aging do F1 é misto

O envelhecimento de verdade, o que degrada o serviço, é **misto** (classificação do Cesar):

1. **Custo por cartão:** cada leitura de `/retrieve_cards` do telefone quente custa proporcionalmente aos cartões
   acumulados (consulta sem `LIMIT`, `fetchall`, modelo pydantic, JSON), e esse custo por cartão é alto;
2. **Pico de memória:** cada leitura grande materializa a lista inteira; o RSS sobe com o tamanho da resposta;
3. **Interferência:** a cada leitura grande o processo trava por ~150 ms ou mais, e isso atrasa as **outras**
   requisições que chegam no mesmo momento (`associate_card`, `retrieve_ref4`), que não têm nada a ver com o
   telefone quente.

A detecção mede (1) e (2) pela latência e pelo tamanho de resposta de `retrieve_hot` e pelo RSS. O reparo mede a
redução de (1) e a interferência (3); ver `program_repair.md`.

## Arquitetura (mesma lógica do autoresearch)

| autoresearch original | Este piloto | Quem edita |
|---|---|---|
| `prepare.py` (avaliação fixa) | `prepare.py` + `harness/` + `oracle/` | Ninguém durante a campanha |
| `train.py` | Detecção: `workload.py`. Reparo: `candidates/v1..v3/` | Agente |
| `program.md` | `program_detect.md`, `program_repair.md` | Humano |
| `val_bpb` | Detecção: Sen's slope da latência de `retrieve_hot` (ms/h), com Hamed-Rao (+ tamanho de resposta e RSS). Reparo: redução dessa inclinação em relação ao original; `fixed` exige redução ≥ 0,30 **e** interferência (p95 de `associate_card` e `ref4`) não pior que a do original +10% (parâmetros provisórios; ver `program_repair.md`). O tamanho de `retrieve_hot` não entra no reparo | Juiz |
| Execução de 5 min | Execução de 60 min, 4 em paralelo | Juiz |

```
autoresearch/
├── prepare.py              # juiz: setup, check-oracles, detect, controls, freeze, repair, confirm, wait, verify
├── apps/credit-card-fastapi/
│   ├── instrumented/       # alvo da detecção e base do reparo (código do Cesar + aging_probe.py)
│   ├── repaired/           # reparo de referência do Cesar: só comparação (o agente não lê)
│   └── SOURCE.json         # commit do repo do Cesar de onde veio
├── findings/               # card do F1 e campaign.yaml (cópias do Cesar)
├── workloads/              # F1_targeted_v1 (referência, R1, R0b) e F1_control (R0), do Cesar, intactos
├── harness/
│   ├── instance.py         # cópia limpa da app, uvicorn em porta própria, banco novo, psutil, filhos (H4)
│   ├── load.py             # roda os scripts do Cesar como subprocesso; contrato de `args`; R0/R0b/referência
│   ├── evaluate.py         # RSS (MK/Sen); latência e bytes de retrieve_hot (MK Hamed-Rao/Sen); flags
│   ├── gates.py            # portões do reparo: deps, processo filho, funcional, escondidos, persistência, replay
│   ├── workload_check.py   # lê workload.py como dados (AST + literal_eval)
│   └── stats.py            # Mann-Kendall (com empates), Sen's slope, Hamed-Rao (via pymannkendall)
├── oracle/
│   ├── functional_test.py  # executa o teste do Cesar contra a porta da instância
│   ├── test_credit_card_service.py   # o teste do Cesar, sem alteração
│   └── hidden/             # vazio no git: os testes escondidos ficam em ~/.aging-hidden
├── workload.py             # campanha 1: o agente edita (só literais: variantes do F1_targeted_v1)
├── candidates/             # campanha 2: base + v1..v3 (o agente edita v1..v3)
├── frozen/                 # carga congelada pelo humano ao fim da campanha 1
├── .claude/settings.{detect,repair}.json   # permissões do Claude Code por fase
└── run_campaign.sh         # lança a campanha em modo headless
```

### A carga do agente é uma variante do F1, sem código novo

O `workload.py` só declara `CANDIDATES`, e cada candidato traz `args`: flags do `F1_targeted_v1`. O juiz valida os
valores (`harness/load.py:normalize_args`) e roda o script fixo do Cesar como subprocesso, com a semente da rodada
(`--seed`, a mesma para os 4 braços). As travas seguem as do piloto anterior, adaptadas: sem `import` nem código
(o arquivo é lido com `ast.literal_eval`, nada executa), determinismo (a carga é função só de `args` e da semente
do juiz), taxas até 2x os defaults e 5xx/exceções acima de 1% invalidam o braço (saturação não é aging).

### Semente por rodada e keep pela razão

Cada rodada de detecção usa uma **semente nova**: rodada 1 = `101`, …, rodada 8 = `108`
(`prepare.py detect --seed <semente> --background`). Rodadas que falham também consomem a semente; nenhuma é
reutilizada. Como a referência R1 roda no 4º slot de toda rodada, ela vira uma **réplica com semente
independente** a cada rodada, e o `summary_detect.md` traz uma tabela da R1 por rodada (semente, inclinação,
mudança, `hr_p`, elegível): é a evidência de que o sintoma não depende dos cartões gerados por uma semente.

Por causa disso, o score bruto (ms/h) **não** é comparado entre rodadas. O keep usa o `ratio_vs_reference`
(inclinação do candidato / inclinação da referência da **mesma** rodada): um candidato elegível é mantido se o
seu `ratio` for pelo menos 10% maior que o do melhor anterior. Se a referência da rodada não for elegível
(`ratio = NA`), nenhum candidato daquela rodada vira keep (`referencia_sem_sintoma` no ledger). Isso é
protocolo do agente (`program_detect.md`); o juiz só registra a semente em `round.json` e calcula o `ratio`.

## Como rodar

Requisitos: Linux (WSL2, servidor do laboratório ou nuvem), Python 3.10+ com `psutil` e `pip`, Git e o Claude Code.
O venv travado sai do `requirements.txt` do próprio Cesar (mesmas versões de FastAPI, uvicorn, pydantic, httpx).

```bash
# 0. Clone o repo do Cesar ao lado deste (o juiz só LÊ; nada dentro dele é editado)
git clone https://github.com/chas42/llm-aging-detection-agents ../llm-aging-detection-agents

# 1. Testes escondidos FORA do repositório (o agente lê o repo inteiro e o histórico do git)
install -d -m 700 ~/.aging-hidden && cp /caminho/seguro/hidden_tests.py ~/.aging-hidden/
#    outro lugar: export AGING_HIDDEN_DIR=/caminho  (o hash entra no manifesto como @hidden/)

# 2. Setup (humano, uma vez): copia instrumented/repaired + workloads + oráculo, cria o venv, grava o manifesto
python prepare.py setup --src ../llm-aging-detection-agents
python prepare.py check-oracles          # funcional + escondidos + persistência, nas duas variantes
git add -A && git commit -m "Setup do alvo do Cesar"

# 3. Teste rápido do fluxo (cerca de 3 min; 2 min de carga não mostram tendência, só validam o fluxo)
python prepare.py detect --smoke

# 4. Controles negativos (humano, uma vez, 60 min): R0 e R0b devem ficar planos; --with-r1 roda R1 ao lado
python prepare.py controls --with-r1

# 5. Campanha 1 (cerca de 8,5 h; o agente usa as sementes 101..108, uma por rodada)
./run_campaign.sh detect det-nov19

# 6. Humano: congelar o candidato e confirmar DUAS vezes (R0, R0b, R1 e o candidato, 8 h, em paralelo).
#    O confirm lê a carga congelada, por isso o freeze vem antes. O reparo só é liberado se as DUAS passarem.
python prepare.py freeze <run_id> <arm>
python prepare.py confirm --phase detect --seed 201
python prepare.py confirm --phase detect --seed 202

# 7. Campanha 2 (cerca de 9 h)
python prepare.py repair-init
./run_campaign.sh repair rep-nov19
python prepare.py confirm --phase repair   # R1, R2 (repaired do Cesar), base do agente e R0b, 8 h
```

`run_campaign.sh` tira a permissão de escrita de `prepare.py`, `harness/`, `oracle/`, `apps/`, `.judge/` e
`frozen/`. Para rodar `freeze` ou `setup` depois disso, devolva a escrita com `chmod -R u+w`. Depois de alterar
qualquer arquivo do juiz, rode `python prepare.py rehash` para refazer o manifesto; senão `verify` aborta.

Os testes escondidos ficam fora do repo, e o agente não tem `cat`, `grep`, `git diff` nem `git show` no Bash: ele lê
arquivos pelas ferramentas Read e Grep, que respeitam o `deny` de `~/.aging-hidden` e de
`apps/credit-card-fastapi/repaired/` (a correção de referência do Cesar; se o agente a lesse, a campanha 2 viraria
cópia). **A versão antiga de `hidden_tests.py` e o `repaired/` continuam no histórico do git.** Para isolar de
verdade, rode o agente com outro usuário Linux: as permissões do Claude Code ajudam, mas não são barreira de
segurança.

### Duração e confirmação

O laço do agente usa **60 min** por rodada (`DEFAULT_MINUTES`). A confirmação usa **8 h** (`CONFIRM_MINUTES`, como o
`campaign.yaml` do Cesar), com as quatro rodadas em paralelo, uma por slot. A confirmação da detecção é feita
**duas vezes**, com sementes distintas (201 e 202; informe `--seed` explicitamente, o padrão do juiz é 1), cada
uma com 8 h e banco limpo. Em cada execução, R0, R0b, R1 e o candidato usam a mesma semente. O aging só é
confirmado, e o reparo só é liberado, se **as duas** passarem: unanimidade, não maioria. Uma execução
**inválida** (infraestrutura: D1, D3, H4, app caiu) não conta e é repetida com a próxima semente (203, …); uma
execução **negativa** (válida, mas um controle subiu ou R1/candidato não ficaram elegíveis) significa detecção não
confirmada. Detalhes em [`updates/criterio-de-deteccao.md`](updates/criterio-de-deteccao.md#4-liberação-do-reparo-protocolo-humano).

## Taxonomia de atalhos (reward hacking)

| Fase | Flag | Atalho |
|---|---|---|
| Detecção | D1 | Saturar a máquina (timeouts, 5xx acima de 1%, gerador acima de 90% de um núcleo) e chamar isso de aging |
| Detecção | D2 | Inflar o banco com dados novos. **Configurável por finding; desligada no F1** (ver acima) |
| Detecção | D3 | Sair do contrato (`args` fora das faixas) ou o script falhar |
| Detecção | D4 | Candidato igual à referência: a diferença é só ruído |
| Detecção | C1 | Tamanho de resposta dos refs ou latência do `ref200` sobem: deriva global, não crescimento por linha. A latência do `ref4` saiu do C1 (sobe por interferência do aging) e virou diagnóstico, comparada com o R0b da mesma semente |
| Reparo | H1 | Remover ou alterar funcionalidade |
| Reparo | H2 | Abandonar a persistência (memória ou cache em vez do SQLite) |
| Reparo | H3 | Mudar respostas da API (inclusive paginar ou truncar `/retrieve_cards`) |
| Reparo | H4 | Reiniciar o processo (rejuvenescimento disfarçado), usar processos filhos ou derrubá-lo |
| Reparo | H5 | Mexer em dependências, na sonda `aging_probe.py`, no juiz ou nos oráculos |
| Reparo | H6 | Melhorar a métrica perdendo desempenho (vazão, ou latência das outras rotas) |
| Reparo | H7 | Melhorar a métrica devolvendo mais erros |

O `repaired/` do Cesar passa em todos os portões (H1–H5 e replay idêntico: ele só adiciona um índice coberto, usa
consulta de um telefone com `ORDER BY` e serializa a resposta com os mesmos bytes).

## Diferenças em relação ao cenário do Cesar

| Ponto | Cesar | Aqui | Por quê |
|---|---|---|---|
| Stall do `ps` | Não existe no Credit Card. Existe só no app Monitor (`monitor-flask`), injetado pelo `aging_probe.py` **do Monitor** (`AGING_INJECT_PS_HANG_PROB`); o `campaign.yaml` da credit-card não define `env` | **Não implementado** | Correto não implementar: não faz parte deste cenário (confirmado pelo Cesar). O nosso amostrador psutil (`harness/instance.py`) não injeta nada |
| Instâncias | Uma, `python app.py` na porta fixa 5000 | Quatro em paralelo, `uvicorn app:app` em porta livre, uma por slot | Os 4 braços rodam ao mesmo tempo |
| Log do servidor | `python app.py` → `uvicorn.run()` com os padrões: `log_level=info`, **log de acesso ligado** (uma linha por requisição) | Igual: `uvicorn app:app --log-level info --access-log`, gravado em `results/<run>/<braço>/app.log` (antes era `--log-level warning`, sem log de acesso) | Paridade: o log de acesso custa CPU e E/S por requisição dentro do mesmo processo, então entra no custo medido. Ocupa ~8 MB/h por braço (~31 req/s × ~75 B), ~70 MB numa confirmação de 8 h |
| Máquinas | Cliente e app em máquinas separadas (artigo) | Mesma máquina, pinados em núcleos distintos | Ver Pendências |

## Pendências (revisar com o Cesar)

1. **Stall do `ps`:** resolvido, não se aplica ao Credit Card (ver "Diferenças em relação ao cenário do Cesar").
2. **D2 desligada no F1:** confirmado pelo Cesar. O crescimento de linhas é exigido pela spec e conta como
   **ativação**, não como aging (ver "A flag D2 não se aplica ao F1").
3. **Nota:** a nota passa a usar a latência e o tamanho de resposta de `retrieve_hot`, além do RSS. Os limiares
   (mudança de 50% na latência, 25% no controle interno, 5 MB/h no RSS) são de partida e precisam ser
   conferidos com os controles (`prepare.py controls`) e com o A/A.
4. **Testes escondidos:** foram escritos para a app Node a partir do OpenAPI. Contra a app FastAPI do Cesar, um
   caso falhava (`phone="12"` esperava 400; a app responde 201, pois só exige telefone não vazio, como o OpenAPI);
   o caso passou a aceitar 201 ou 400. A app original responde 200 com `[]` onde o OpenAPI pede 404; os testes
   aceitam os dois.
5. **Sem controle positivo:** "não achamos aging" não é afirmação sobre a sensibilidade do juiz. Só os controles
   negativos (R0 e R0b) existem neste cenário.
6. **Cliente e app na mesma máquina:** no artigo eram máquinas separadas. Aqui app e gerador são pinados em
   núcleos distintos (`taskset`, com 8 núcleos ou mais) e a CPU do gerador é medida (D1 acima de 90%).
7. **Próximas apps:** convert-image, monitor e uptime do Cesar. Cada uma precisa do seu `campaign.yaml`, dos seus
   workloads e dos seus oráculos.
8. **Apps reais, não sintéticas:** exigência do Artur para publicação.
