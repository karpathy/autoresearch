# 2026-11-19 — Alvo e cenário do Cesar (`credit-card-fastapi`, finding F1)

**Objetivo:** manter o método autoresearch (agente propõe, juiz fixo mede, laço mantém ou descarta) e trocar o
alvo e o cenário pelos de [chas42/llm-aging-detection-agents](https://github.com/chas42/llm-aging-detection-agents),
exatamente como o Cesar usou (`apps/credit-card-fastapi/campaign.yaml`).

**Antes:** `credit-card` em Node (`generated-software-llm`), sonda de GC `gc_probe.js`, carga gerada dentro do juiz
(`request(rng, i)`), controle positivo sintético (`leak_inject.js`, `prepare.py control`).

**Agora:** app FastAPI do Cesar, scripts de carga dele como subprocesso, controles negativos R0/R0b, sem bug
injetado.

## Mudanças por etapa

| Etapa | Commit | O que mudou |
|---|---|---|
| 2. Alvo | `19cc4bd` | `prepare.py setup --src <repo do Cesar>` copia `instrumented/` e `repaired/`, o card do F1, o `campaign.yaml`, os workloads F1 e o teste funcional; cria `.cache/venv` com o `requirements.txt` do Cesar; grava `SOURCE.json` com o commit de origem. Sai o caminho do zip Node |
| 3. Instância | `63b1e2e` | `harness/instance.py` sobe `uvicorn app:app` em porta livre, com banco novo por cópia e `AGING_INSTRUMENT=1` / `AGING_INSTRUMENT_LOG` (a `aging_probe.py` do Cesar). Mantém psutil e a contagem de filhos (H4). Sai `gc_probe.js`. Novo comando `check-oracles` |
| 4. Carga | `633e87b` | `harness/load.py` roda os scripts do Cesar com o contrato dele (`--base-url --duration --seed --out`). A referência do 4º slot é o `F1_targeted_v1` padrão (R1). O `workload.py` do agente só declara `args` literais do F1 (lido com `ast.literal_eval`); taxas até 2× os defaults e 31 req/s nominais no total |
| 6. Métrica | `4f80daf` | MK com Hamed-Rao + Sen na latência e no tamanho de resposta de `retrieve_hot`; `retrieve_ref4/ref200` como controle interno (flag C1); D2 configurável por finding e desligada no F1; 5xx/exceções > 1% invalidam o braço (D1). Sai `leak_inject.js` |
| 5 e 7. Controles e reparo | `9df581b` | `prepare.py controls` roda R0 e R0b (sai `control`). Reparo: `candidates/base` = `instrumented`; portões em Python (deps e `aging_probe.py` travados, processo filho, funcional, escondidos, persistência, replay diferencial no formato do F1, com repetição antes de acusar). Nota = redução da inclinação da latência de `retrieve_hot` |
| 8. Duração | `9df581b` | 60 min no laço do agente; `confirm` roda as quatro rodadas do `campaign.yaml` em 8 h, em paralelo |
| — | `f3a9630` | Remove artefatos de teste (`frozen/`, `candidates/`) commitados por engano |
| 9. Docs | `d0cb986` | `program_detect.md`, `program_repair.md`, README (tabela "de onde vem cada decisão" com o Cesar e o `campaign.yaml`), `.claude/settings.*.json` |
| 10. Validação | `0a9a6e2` | Correções achadas no smoke: corrida no sampler ao ler o `-wal`; R0 avaliado pela leitura `retrieve_cards` |

## Decisões tomadas com o Rafael

- **Workload do agente = parâmetros do F1**, não um script livre: o agente não escreve código que roda na máquina.
- **Testes escondidos:** o caso `phone="12"` aceita 201 ou 400. A app do Cesar responde 201 e o OpenAPI não define
  formato de telefone. O original fica em `~/.aging-hidden/hidden_tests.py.node-original`.
- **Stall do `ps`:** não implementado. O cenário do Cesar para a credit-card não tem injeção (o stall é do app
  `monitor-flask`). Fica pendente de validação com o Cesar.

## Decisões tomadas por mim (reversíveis)

- **Agente não lê `apps/credit-card-fastapi/repaired/`** (deny nos dois `settings`): é a correção de referência; se
  o agente a lesse, a campanha de reparo viraria cópia. Continua no histórico do git.
- **Confirmação com uma semente** nas quatro rodadas de 8 h, em vez das 3 sementes de 20/10 (`--seed` permite repetir).
- **Pinagem:** app do slot *i* na CPU *i*, gerador na CPU 4+*i* (com ≥ 8 núcleos), como `--app-cpus`/`--load-cpus`
  do Cesar.

## Validação feita

| Teste | Resultado |
|---|---|
| `check-oracles` | instrumented e repaired: funcional, escondidos e persistência ok |
| Contrato da carga, 20 s por braço | R0: 1 linha; R0b: 208 constantes; R1 acumula; 0% de erro (igual ao smoke do Cesar) |
| Travas de `args` e AST | recusam `seed`, taxas acima de 2×, `import`, chamadas, lambdas e compreensões |
| `detect --smoke` (2 min, 4 braços) | fluxo ok; achou e corrigiu o bug de `hot_phones=2` (`retrieve_hot1`, `retrieve_hot2`) |
| `repair --smoke` | `repaired/` do Cesar passa nos portões; candidato sem mudança → `sem_alteracao`; `LIMIT 100` → H3 |
| `controls --smoke --with-r1` | R0 e R0b planos, R1 sobe (latência +37%, resposta +165%). 2 min só validam o fluxo |
| `controls --with-r1`, 60 min | **em andamento** (`results/controls-1119-0743-9b5e`) |

## Limitações conhecidas

- O critério de liberação do reparo é um protocolo humano; o código não o impõe (ver
  [criterio-de-deteccao.md](criterio-de-deteccao.md)).
- Limiares de partida, ainda não calibrados com os controles de 60 min e 8 h.
- Sem controle positivo; app e cliente na mesma máquina (em núcleos distintos).
- Replay dos portões leva cerca de 1 min por candidato (as escritas são serializadas pelo SQLite).
