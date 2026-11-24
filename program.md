# program.md

Você é um pesquisador autônomo. Você executa uma de duas campanhas, indicada no comando que o iniciou:

- **`detect`**: encontrar a carga que produz a tendência de degradação mais forte na aplicação;
- **`causas`**: explicar, por comparação entre cargas, qual fator da carga produz essa tendência.

Siga só este arquivo. Não pergunte nada: execute o laço até o fim do orçamento e escreva o resumo.

## 1. O sistema medido

- Uma API HTTP com duas operações: `POST /associate_card` (associa um cartão a um telefone) e
  `POST /retrieve_cards` (lista os cartões de uma lista de telefones).
- Um gerador de carga fixo dispara essas operações em fluxos com taxas configuráveis. Você não escreve código de
  carga: só escolhe os parâmetros (`args`, seção 3).
- O juiz (`prepare.py`) sobe 4 instâncias da aplicação em paralelo, cada uma com banco vazio, aplica uma carga em
  cada uma durante 60 min e mede.

### O que é medido em cada braço (uma instância sob uma carga)

| Série | O que é |
|---|---|
| `retrieve_hot` | leituras de `/retrieve_cards` dos telefones que recebem as associações novas ("telefones quentes") |
| `retrieve_ref4`, `retrieve_ref200` | leituras de telefones de referência, com 4 e 200 cartões fixos (controle interno) |
| `associate_card` | gravações |
| RSS | memória do processo da aplicação, a cada 5 s |

Por janela de 60 s (só respostas 2xx, sem a primeira e a última janela nem o primeiro minuto): latência p50 e
p95 e tamanho médio da resposta. Sobre cada série: teste de Mann-Kendall com correção de Hamed-Rao (p unilateral
de alta) e Sen's slope por hora. **Mudança** = Sen × duração da série ÷ mediana, em %.

### Quando um braço é elegível (`eligible` / `score_eligible`)

As quatro condições precisam valer:

1. **Válido:** exceções ≤ 1%, respostas 5xx ≤ 1%, CPU p95 do gerador de carga ≤ 90% de um núcleo, processo da
   aplicação vivo, com o mesmo PID e sem processos filhos, ≥ 10 amostras de memória e ≥ 5 janelas;
2. **Latência de `retrieve_hot` sobe:** p < 0,01 e mudança ≥ 50%;
3. **Tamanho de resposta de `retrieve_hot` sobe:** p < 0,01 e mudança ≥ 50%;
4. **Controle interno plano:** nem o tamanho de resposta de `retrieve_ref4` e `retrieve_ref200` nem a latência
   de `retrieve_ref200` sobem (p < 0,01 e mudança ≥ 25%).

### Flags

| Flag | Quando | Efeito |
|---|---|---|
| `D1_saturacao_ou_erros` | exceções ou 5xx acima de 1%, ou gerador acima de 90% de um núcleo | braço inválido |
| `D3_fora_do_contrato` | `args` fora do contrato ou o gerador falhou | braço (ou rodada) inválido |
| `D4_copia_da_referencia` | candidato com os mesmos `args` da referência | sem nota |
| `C1_referencia_interna_subindo` | condição 4 falhou | sem nota |
| `H4_processo_reiniciado_ou_morto` / `H4_processo_filho` | o processo morreu, mudou de PID ou abriu filhos | braço inválido |
| `sem_amostras_suficientes` / `sem_janelas_do_cliente` | poucos dados | braço inválido |

Se uma ideia sua só melhora o número por um desses caminhos, não insista. Registre no ledger com
`"atalho": true`.

## 2. Arquivos

| Você pode | Arquivos |
|---|---|
| Ler | `program.md`, `workload.py`, `frozen/workload_frozen.py`, `results/`, os seus arquivos de registro e resumo |
| Editar | `workload.py`, `results.tsv`, `ledger.jsonl`, `summary_detect.md` (campanha `detect`); `workload.py`, `results_causas.tsv`, `ledger_causas.jsonl`, `summary_causas.md` (campanha `causas`) |

Não leia nem edite mais nada. Não instale pacotes, não rode a aplicação nem outro gerador de carga fora do juiz,
não mexa em limites do sistema, não edite `results/` nem `results/judge_log.jsonl` (é a fonte de verdade e será
comparado com os seus registros). Não rode `setup`, `freeze`, `confirm`, `controls`, `check-oracles` nem `rehash`.

## 3. `workload.py`

O arquivo só pode ter docstring e atribuições `NOME = <literal>`: nada de `import`, função, chamada, `lambda`,
compreensão ou expressão. Ele é lido como dados; uma violação invalida a rodada (D3).

```python
CANDIDATES = [
    {"name": "nome_curto", "hypothesis": "hipótese falsificável", "args": {"new_pair_rate": 15.0}},
]
```

- De 1 a 3 candidatos por rodada; `name` com letras, números, `_` ou `-`, único na rodada.
- `args`: só as chaves abaixo; as omitidas ficam no valor padrão.

| Arg | Significado | Padrão | Faixa |
|---|---|---|---|
| `new_pair_rate` | associações novas por segundo, nos telefones quentes | 10 | > 0 e ≤ 20 |
| `hot_phones` | quantidade de telefones quentes (em rodízio) | 1 | 1 a 2 |
| `grow_workers` | trabalhadores do fluxo de associações novas | 2 | 1 a 4 |
| `refresh_rate` | reenvios por segundo de associações que já existem | 2 | 0 a 4 |
| `ref_rate` | associações por segundo nos telefones de referência | 2 | 0 a 4 |
| `probe_interval` | segundos entre leituras de cada telefone sondado | 2 | ≥ 1 |
| `key_space` | 0 = cartões das associações novas sempre diferentes; N > 0 = repetem-se num conjunto de N | 0 | 0 a 100000 |
| `ref_sizes` | tamanhos dos telefones de referência | (4, 200) | fixo |

Taxa nominal total = `new_pair_rate + refresh_rate + ref_rate + (hot_phones + 2) / probe_interval` ≤ 31 req/s.
`seed`, `base_url`, `duration`, `out` e `timeout` são do juiz e não podem aparecer em `args`.

## 4. Execução e sementes

Cada rodada leva cerca de 62 min e roda em segundo plano (o Bash tem limite de 10 min por chamada):

```
python prepare.py <detect|causes> --seed <semente> --background   # imprime o run_id
python prepare.py wait <run_id>                                   # bloqueia até 9 min; repita até sair o resumo
```

- Use sempre `--seed`. A semente vale para os 4 braços da rodada.
- Cada rodada usa uma semente nova, em sequência. Rodadas que falham também consomem a semente. Nunca reutilize
  nem pule uma semente.
- Os detalhes de cada braço ficam em `results/<run_id>/<braço>/evaluation.json`; o resumo da rodada em
  `results/<run_id>/round.json`.
- `--smoke` (2 min) serve só para testar se `workload.py` é aceito. Não tire conclusões de uma rodada smoke.

## 5. Campanha `detect`

**Objetivo:** encontrar, dentro do contrato, o candidato com a maior tendência de alta da latência de
`retrieve_hot` em relação à referência.

**Referência:** um braço de cada rodada roda a carga com todos os `args` no padrão. Os outros 3 braços são os
seus candidatos.

**Nota do braço:** `score_hot_latency_ms_per_h` = Sen's slope da latência p50 de `retrieve_hot` (ms/h), só se o
braço for elegível e não tiver D4. **`ratio`** = inclinação do candidato ÷ inclinação da referência da mesma
rodada; só existe se os dois forem elegíveis.

**Setup:**

1. `git checkout -b autoresearch/<tag>`, com a tag recebida no comando.
2. `python prepare.py verify` precisa responder `ok`. Se não responder, pare e registre o erro no resumo.
3. Crie `results.tsv` (só o cabeçalho) e `ledger.jsonl` (vazio). Nenhum dos dois entra em commit.

**Laço: exatamente 8 rodadas**, sementes 101 a 108 (rodada *n* = 100 + *n*). Não pare antes, mesmo que a
tendência apareça cedo.

1. Leia `results.tsv` e `ledger.jsonl`. A semente desta rodada é 100 + (linhas `started` no ledger + 1).
2. Escreva de 1 a 3 candidatos em `workload.py`, cada um com uma hipótese falsificável e o fator que testa.
3. `git commit -am "<resumo das hipóteses>"`.
4. Registre `started` no ledger.
5. Rode `detect --seed <semente> --background` e `wait` até sair o resumo.
6. Registre uma linha por braço no `results.tsv` (inclusive a referência) e `completed` no ledger.
7. **Keep ou discard:**
   - se a referência da rodada não for elegível, nenhum candidato vira keep: status do ledger
     `referencia_sem_sintoma`, candidatos `discard`, e desfaça o commit;
   - na primeira rodada com algum `ratio`, o candidato com o maior `ratio` vira o melhor anterior (`keep`);
   - nas seguintes, mantenha o commit se algum candidato tiver `ratio ≥ 1,10 × ratio` do melhor anterior;
   - senão, `git reset --hard HEAD~1`. As linhas de `results.tsv` e do ledger continuam.
8. Entre candidatos equivalentes, prefira o que tem menos `args` fora do padrão.

**`results.tsv`** (separado por tabulação):

```
commit	run_id	seed	arm	score_hot_latency_ms_per_h	hr_p	ratio_vs_reference	ttd_min	status	flags	hypothesis
```

`status`: `keep`, `discard`, `invalid` ou `baseline` (para a referência).

**`ledger.jsonl`**, uma linha antes e uma depois de cada rodada:

```
{"run_id": null, "commit": "...", "seed": 101, "status": "started", "candidates": [{"name": "...", "hypothesis": "...", "fator_testado": "..."}], "ts": "..."}
{"run_id": "...", "seed": 101, "status": "completed|failed|referencia_sem_sintoma", "interpretacao": "...", "fator_testado": "...", "atalho": false, "ts": "..."}
```

**`summary_detect.md`** ao final:

- tabela de todas as rodadas, com a semente;
- tabela da referência por rodada: `seed`, inclinação e mudança da latência de `retrieve_hot`, `hr_p`, elegível
  (sim/não), e em quantas rodadas ela foi elegível;
- o melhor candidato (`run_id` e `name`) e o porquê;
- evidência de que o tamanho de resposta de `retrieve_hot` subiu e de que o controle interno ficou plano;
- todos os atalhos observados;
- limitações: janela de 60 min, uma semente por rodada, cada candidato rodou com uma única semente.

Não declare a degradação confirmada: a confirmação é feita fora desta campanha.

## 6. Campanha `causas`

**Pré-condição:** `frozen/workload_frozen.py` existe. Ele contém a carga escolhida (`FROZEN_ARM` indica qual
candidato de `CANDIDATES`; se for `F1_targeted_v1_referencia`, é a carga com todos os `args` no padrão). Se o
arquivo não existir, pare e registre isso no resumo.

**Objetivo:** descobrir quais fatores da carga mudam a tendência, comparando cada fator isoladamente.

**Referência:** um braço de cada rodada roda a carga congelada (`referencia_congelada`). Os outros 3 braços são
variações dela.

**Regra de um fator:** cada candidato difere da carga congelada em **exatamente um** `arg`. O juiz recusa a
rodada inteira (D3) se algum candidato mudar zero ou mais de um.

**Medidas por variação** (`effect` no `round.json`; razão = variação ÷ referência da mesma rodada; só existem se a
referência for elegível e a variação válida):

| Campo | O que compara |
|---|---|
| `lat_ratio` | inclinação da latência p50 de `retrieve_hot` |
| `bytes_ratio` | inclinação do tamanho de resposta de `retrieve_hot` |
| `cost_ratio` | custo por MB: inclinação da latência de `retrieve_hot` contra o tamanho da resposta (ms por MB) |
| `assoc_p95_ratio` | p95 final de `associate_card` |
| `ref4_p95_ratio` | p95 final de `retrieve_ref4` |

**Setup:**

1. `git checkout -b autoresearch/<tag>`, com a tag recebida no comando.
2. `python prepare.py verify` precisa responder `ok`.
3. Leia `frozen/workload_frozen.py`.
4. Crie `results_causas.tsv` (só o cabeçalho) e `ledger_causas.jsonl` (vazio).

**Laço: exatamente 6 rodadas**, sementes 301 a 306 (rodada *n* = 300 + *n*).

1. Leia `results_causas.tsv` e `ledger_causas.jsonl`. A semente desta rodada é 300 + (linhas `started` + 1).
2. Escreva de 1 a 3 variações em `workload.py`, cada uma com o fator, o valor novo, a hipótese e a direção
   esperada do efeito.
3. **Réplica:** cada variação precisa rodar em **duas rodadas**, com sementes diferentes. Planeje as 6 rodadas
   para que todas as variações tenham as duas réplicas.
4. `git commit -am "<variações>"` e registre `started` no ledger.
5. Rode `causes --seed <semente> --background` e `wait` até sair o resumo.
6. Registre uma linha por braço no `results_causas.tsv` e `completed` no ledger. Não há keep nem discard: o
   commit fica.
7. Se a referência da rodada não for elegível (`reference_ok: False`), registre `referencia_sem_sintoma`; as
   variações daquela rodada não contam como réplica.

**`results_causas.tsv`** (separado por tabulação):

```
commit	run_id	seed	arm	fator	valor_ref	valor_var	valid	eligible	lat_ratio	bytes_ratio	cost_ratio	assoc_p95_ratio	ref4_p95_ratio	flags	hypothesis
```

**`ledger_causas.jsonl`**: mesmo formato do ledger da campanha `detect`, com `"variacoes": [{"name", "fator", "valor", "hypothesis", "direcao_esperada"}]` no `started`.

**Como declarar um efeito:** para cada razão, a variação tem efeito se, nas duas réplicas válidas, a razão ficar
do mesmo lado de 1 e a pelo menos 20% dele (≥ 1,2 nas duas ou ≤ 0,8 nas duas). Caso contrário, registre "sem
efeito claro".

**`summary_causas.md`** ao final:

- a carga congelada (`args`);
- tabela fator → valor → razões nas duas réplicas → efeito (sim/não, direção) para cada medida;
- quais fatores mudam a inclinação da latência, quais mudam o tamanho da resposta e quais mudam o custo por MB;
- variações sem as duas réplicas válidas, e por quê;
- limitações: um fator de cada vez (sem interações), janela de 60 min, duas réplicas por variação.

## 7. Registros

- Timestamps vêm de `date -Iseconds`. Nunca invente horários.
- Não declare efeito maior que a diferença entre réplicas.
- O `results/judge_log.jsonl` é a fonte de verdade; os seus registros precisam bater com ele.
