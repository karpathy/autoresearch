# AutoResearch-Aging — Campanha 1: DETECÇÃO

Você é um pesquisador autônomo. O alvo e o cenário são os do repositório do Cesar
([chas42/llm-aging-detection-agents](https://github.com/chas42/llm-aging-detection-agents)): a app
`credit-card-fastapi` (BaxBench, gerada por LLM) e o finding `bax-fastapi-credit-card-F1`. Seu objetivo é
encontrar, **dentro do contrato de carga**, a variante do workload de referência que **evidencia mais cedo
e com mais força o sintoma do finding**: a latência de `POST /retrieve_cards` para o telefone quente
crescendo com o tempo, com o tamanho da resposta crescendo junto, enquanto os controles internos
(`retrieve_ref4`, `retrieve_ref200`) ficam planos.

## Setup

1. **Combine uma tag de campanha** com o humano, por exemplo `det-nov19`. A ramificação
   `autoresearch/<tag>` não pode existir.
2. `git checkout -b autoresearch/<tag>`
3. **Leia, nesta ordem:** `README.md`, este arquivo, `workload.py`, `findings/bax-fastapi-credit-card-F1.yaml`,
   `workloads/bax-fastapi-credit-card-F1_targeted_v1.md` (o que cada flag faz e a calibração do Cesar),
   `harness/load.py` (o contrato), `harness/evaluate.py` (a métrica) e `oracle/functional_test.py`.
   O código da app está em `apps/credit-card-fastapi/instrumented/`, só para leitura.
4. `python prepare.py verify` precisa responder `ok`. Se não responder, pare e avise o humano.
5. Crie `results.tsv` (só o cabeçalho) e `ledger.jsonl` (vazio). Nenhum dos dois entra em commit.
6. Confirme com o humano e comece. A partir daí, não pergunte mais nada.

## O que você PODE fazer

- Editar **somente `workload.py`**: de 1 a 3 candidatos por rodada. O 4º slot roda sempre a **referência**:
  o `F1_targeted_v1` do Cesar com os parâmetros padrão (a rodada R1 dele), para comparação na mesma máquina
  e no mesmo momento.
- Você **não escreve código de carga**. Cada candidato é um dict com `name`, `hypothesis` e `args`, e `args`
  são flags literais do `F1_targeted_v1`: `new_pair_rate`, `hot_phones`, `grow_workers`, `refresh_rate`,
  `ref_sizes`, `ref_rate`, `probe_interval`, `key_space`. O juiz roda o script do Cesar, fixo, com a semente
  dele. As faixas permitidas estão em `workload.py` e em `harness/load.py`.
- Desenhar candidatos como **ablações ou variações de um fator**: taxa de pares novos, concentração em um ou
  dois telefones quentes, taxa de sondagem. Registre qual fator cada um testa.

## O que você NÃO PODE fazer

- Editar `prepare.py`, `harness/`, `oracle/`, `apps/`, `workloads/`, `findings/`, `frozen/`, `.judge/`,
  `.claude/` ou este arquivo. O juiz confere hashes a cada rodada e aborta se algum mudar (`H5`).
- Ler `oracle/hidden/`, os testes escondidos fora do repositório ou `apps/credit-card-fastapi/repaired/`
  (a correção de referência do Cesar, que serve só para o humano comparar na campanha 2).
- Escrever em `workload.py` qualquer coisa além de docstring e `NOME = <literal>`: nada de `import`,
  função, chamada, `lambda`, compreensão ou expressão. O juiz lê o arquivo como dados (`ast.literal_eval`);
  uma violação invalida a rodada inteira (flag D3). `seed`, `base_url`, `duration`, `out` e `timeout`
  são do juiz e não podem aparecer em `args`.
- Passar de **2x as taxas padrão** do Cesar (pares novos 20/s, reenvios 4/s, referência 4/s, sondagem a cada
  1 s no mínimo, no máximo 2 telefones quentes) nem de 31 req/s nominais no total. Saturar a máquina não é aging.
- Instalar pacotes, abrir outros processos de carga, rodar a app fora do juiz ou mexer em limites do SO.
- Editar resultados passados, `results/judge_log.jsonl` ou `round.json`. O `judge_log.jsonl` é a fonte de
  verdade: o humano vai comparar com o seu `results.tsv`.

## Métrica

- **Sintoma (principal):** `score_hot_latency_ms_per_h`, o Sen's slope da latência p50 de `retrieve_hot`
  (janelas de 60 s, descartando as janelas parcial inicial e final e o aquecimento), com Mann-Kendall de
  **Hamed-Rao** (correção de autocorrelação, o mesmo teste do `analyze_run.py` do Cesar). A nota só existe se
  o braço for válido, se o p unilateral de alta for < 0,01, se a mudança implícita ao longo da série for de
  **pelo menos 50%**, se o **tamanho de resposta** de `retrieve_hot` também subir (o mecanismo foi ativado)
  e se o **controle interno estiver plano** (`retrieve_ref4` e `retrieve_ref200` com mudança menor que 25%;
  senão `C1_referencia_interna_subindo`: é deriva global da máquina, não crescimento por linha).
  Maior é melhor.
- **Também reportados** (`evaluate.json` e a linha do resumo): tamanho de resposta de `retrieve_hot`, RSS do
  processo (MK + Sen, com as inclinações das duas metades), `ratio_vs_reference`, `ttd_min` (tempo até a
  detecção estável) e `db_rows`.
- Os limiares (50%, 25%, 5 MB/h do RSS) foram fixados antes dos controles; `prepare.py controls` (R0 e R0b)
  diz se eles são adequados: nos controles nada deve subir. Os controles são **negativos**: não há controle
  positivo neste cenário, então "não achamos aging" não é uma afirmação sobre sensibilidade.
- O crescimento de linhas no banco **não é um atalho** aqui: é o próprio finding (a tabela `associations`
  nunca é podada). Por isso a flag D2 está desligada para o F1; as linhas continuam em `db_rows`.

## Travas contra reward hacking (o juiz aplica, você respeita)

Estas regras vêm da reunião de 15/10 com o Artur: o agente tende a pegar o atalho mais fácil para o número subir.

| Flag | O que é | Efeito |
|---|---|---|
| D1_saturacao_ou_erros | Exceções/timeouts ou 5xx da app acima de 1%, ou gerador de carga acima de 90% de um núcleo | Braço inválido |
| D2_crescimento_de_estado_legitimo | Mais de N linhas no banco. **Desligada no F1** (ver Métrica); ligada nos outros findings | Medido, sem pontuar (quando ligada) |
| D3_fora_do_contrato | `args` fora do contrato, ou o script saiu com erro | Braço inválido |
| D4_copia_da_referencia | `args` iguais aos padrão: a diferença é só ruído | Medido, sem pontuar |
| C1_referencia_interna_subindo | `retrieve_ref4`/`ref200` também sobem: não é crescimento por linha | Medido, sem pontuar |
| H4_processo_reiniciado_ou_morto | O processo morreu ou mudou de PID | Braço inválido |

Se você perceber que uma ideia sua só melhora o número por um desses caminhos, **não insista**. Registre no
ledger com `"atalho": true`. Esses casos são resultado de pesquisa.

## Execução (cada rodada leva cerca de 62 min)

O Bash do Claude Code tem limite de 10 minutos por chamada. Por isso a rodada roda em segundo plano:

```
python prepare.py detect --background      # imprime run_id
python prepare.py wait <run_id>             # bloqueia até 9 min; repita até sair o resumo
```

A saída final traz uma linha por braço:
`arm=... valid=... score_hot_latency_ms_per_h=... hot_lat_slope=... hr_p=... hot_bytes_slope=... rss_slope=... ref_change[...] ttd_min=... ratio=... flags=...`.
Os detalhes ficam em `results/<run_id>/<arm>/evaluation.json` (e o `client.csv` e a `instrumentation.jsonl` de cada
braço). Use `--smoke` (2 min) só para testar se `workload.py` roda; smoke nunca entra no ranking e 2 min não
mostram tendência.

## Registro

`results.tsv`, separado por tabulação, com estas colunas:

```
commit	run_id	arm	score_hot_latency_ms_per_h	hr_p	ratio_vs_reference	ttd_min	status	flags	hypothesis
```

`status` é `keep`, `discard`, `invalid` ou `baseline`. Registre uma linha por braço, incluindo a referência.

`ledger.jsonl` recebe uma linha **antes** de cada rodada, com
`{"run_id": null, "commit": ..., "status": "started", "candidates": [{"name", "hypothesis", "fator_testado"}], "ts"}`,
e outra **depois**, com `{"run_id", "status": "completed"|"failed", "interpretacao", "fator_testado", "atalho": bool, "ts"}`.

## O laço

Orçamento: **8 rodadas**, cerca de 8,5 h numa noite, ou até o humano interromper. Rodadas que falham também contam.

1. Veja o estado do git e o histórico em `results.tsv` e `ledger.jsonl`.
2. Escreva até 3 candidatos em `workload.py`. Cada um precisa de uma hipótese **falsificável** e do fator testado.
3. `git commit -am "<resumo das hipóteses>"`
4. Registre `started` no ledger.
5. Rode `detect --background` e `wait` até sair o resumo.
6. Registre o resultado no `results.tsv` e `completed` no ledger, com a sua interpretação.
7. **Primeira rodada:** se o `workload.py` não mudou desde o commit base, ela serve como referência. O melhor
   candidato elegível vira o "melhor anterior" (keep), e não há commit para desfazer.
   **Keep** (nas demais): se algum candidato elegível tiver `score` maior que o melhor anterior por pelo menos
   10%, mantenha o commit. A margem evita selecionar ruído, lição do Piloto 2.
8. **Discard**: caso contrário, `git reset --hard HEAD~1`. As linhas no `results.tsv` e no ledger continuam lá.
9. Repita até acabar o orçamento.

**Critério de simplicidade** (como no autoresearch original): com resultados equivalentes, prefira o candidato
com menos `args` fora do padrão. Complexidade gratuita não vale.

## Honestidade dos registros

- Timestamps do ledger vêm de `date -Iseconds`. Nunca invente horários.
- Não declare melhoria maior que o ruído. Se existir `results/<run_controls>/round.json` (`prepare.py controls`)
  ou um teste A/A (`workload_aa.py`), compare com a dispersão entre as cópias idênticas antes de afirmar um efeito.
- Runs `--smoke` (2 min) servem só para testar o fluxo. Não tire conclusões a partir delas.

## Ao final

Pare e escreva `summary_detect.md` com:

- a tabela de todas as rodadas;
- o melhor candidato e o porquê (qual fator acelera o sintoma);
- a evidência de que o mecanismo foi ativado (tamanho de resposta e `db_rows` crescendo) e de que o controle
  interno ficou plano;
- todos os atalhos observados;
- as limitações: janela de 60 min, uma semente por rodada, app e cliente na mesma máquina (pinados em núcleos
  distintos), cenário sem controle positivo.

Não declare aging confirmado. A confirmação é humana: `prepare.py confirm --phase detect` (as quatro rodadas
R0, R0b, R1 e o candidato, 8 h cada) e depois `prepare.py freeze`, que gera a carga congelada para a Campanha 2.
