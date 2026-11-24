# 2026-11-23 — Orquestrador: experimento completo sem humano no meio

Implementa o desenho fechado em [2026-11-23_confirmacao-em-rodadas.md](2026-11-23_confirmacao-em-rodadas.md).

## Como dispara

```bash
./run_experiment.sh <tag>            # experimento completo
./run_experiment.sh <tag> --smoke    # fluxo inteiro com rodadas de 2 min, para testar
```

O humano só dispara e lê `results/experiment-<tag>/report.md` e os resumos do agente (`summary_detect.md`,
`summary_causas_<N>.md`).

## O que foi feito

| Peça | Onde | O que faz |
|---|---|---|
| Orquestrador | `harness/orchestrator.py` | lógica do fluxo: Fase 0 → busca → verificação → causas → retomada → relatório; recebe as ações por injeção (juiz, agente, congelamento), por isso é testável sem a app |
| Ligação ao juiz e ao agente | `prepare.py experiment` | cria a branch `autoresearch/<tag>`, chama o agente (`claude -p`) uma rodada por vez com as permissões da campanha, acha a rodada pelo `judge_log.jsonl` (evento e semente), grava `state.json` a cada passo e o `report.md` no fim |
| Fase 0 automática | `run_controls`, `controls_ok` | R0, R0b e R1, semente 100; passa se R0 e R0b planos e R1 elegível |
| Rodada de verificação | `run_check`, `check_decision`, `prepare.py check-load` | R1, a carga, R0 e R0b com a mesma semente; decisão `positiva` / `negativa` / `invalida` |
| Congelamento | `freeze_spec` | grava a carga confirmada em `frozen/workload_frozen.py` e refaz o manifesto |
| Disparo | `run_experiment.sh` | `verify`, tira a escrita do juiz, roda `prepare.py experiment` e grava o log |
| Agente por rodada | `program.md`, `.claude/settings.{detect,causas}.json` | o agente faz só a tarefa do comando (uma rodada ou um resumo) e termina; não cria branch nem roda comandos do orquestrador |

## Regras implementadas

- Rodada *n* usa a semente 100 + *n*; busca até a rodada 8; verificação até a rodada 10.
- Depois de cada rodada de busca, o melhor candidato elegível (maior `ratio`; sem `ratio`, maior inclinação),
  ignorando a referência, D4 e cargas com `args` idênticos aos de uma já confirmada, vai para a verificação.
- Verificação: 2 positivas confirmam; uma `invalida` é repetida uma vez; uma `negativa` volta à busca; sem rodadas
  disponíveis: `sem_orcamento`.
- Cada carga confirmada é congelada e ganha a sua campanha de causas (6 rodadas; campanha *j*, rodada *r*: semente
  300 + 10(*j*−1) + *r*) e o seu resumo; depois a busca retoma.
- Uma rodada em que o agente não produz medição (falhou ou estourou o tempo de 3 h) conta como `falhou` e consome
  a semente.

## Removido

- `confirm` (2 × 8 h, sementes 201/202) e `CONFIRM_MINUTES`.
- `run_campaign.sh` (substituído por `run_experiment.sh`).
- Etapas humanas no meio (controles, `freeze`, confirmação, disparo das causas). Os comandos continuam disponíveis
  à mão para depurar.

## Verificação

- `tests/test_orchestrator.py` (13 testes): Fase 0 falha; nenhuma carga; confirmação na rodada 2 com causas e
  retomada; verificação negativa; aging tarde até a rodada 10; sem orçamento; inválida repetida uma vez; duas
  inválidas; segunda carga com campanha própria; carga idêntica não reverificada; rodada do agente que falha;
  decisão da verificação; relatório.
- Simulação de ponta a ponta do `prepare.py experiment` numa cópia do repositório (agente e medições simulados):
  carga achada na rodada 1, confirmada nas rodadas 2 e 3, causas com sementes 301–306, busca retomada nas rodadas
  4–8, relatório e congelamento corretos.
- `tests/test_causes.py` e `tests/test_c1.py` continuam passando; `rehash` + `verify` ok.

## O que falta

- Rodar com a app: primeiro `./run_experiment.sh <tag> --smoke`, depois o experimento completo.
- Calibrar os limiares com a Fase 0.
- A figura do fluxograma.
