# program.md

Você é um pesquisador autônomo. O sistema chama você **uma vez por tarefa**, e o comando diz qual é:

- **uma rodada da campanha `detect`** (rodada e semente): propor cargas que produzam a tendência de degradação
  mais forte na aplicação;
- **uma rodada da campanha `causas`** (campanha, rodada e semente): explicar, por comparação entre cargas, qual
  fator da carga produz essa tendência;
- **um resumo** (`detect` ou `causas` N).

Siga só este arquivo. Faça só a tarefa pedida e termine. Não pergunte nada.

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
| Editar (`detect`) | `workload.py`, `results.tsv`, `ledger.jsonl`, `summary_detect.md` |
| Editar (`causas`) | `workload.py`, `results_causas.tsv`, `ledger_causas.jsonl`, `summary_causas_<N>.md` |

Os registros e resumos não entram em commit. Se um arquivo de registro não existir, crie-o (o `.tsv` com o
cabeçalho, o `.jsonl` vazio).

Não leia nem edite mais nada. Não instale pacotes, não rode a aplicação nem outro gerador de carga fora do juiz,
não mexa em limites do sistema, não edite `results/` nem `results/judge_log.jsonl` (é a fonte de verdade e será
comparado com os seus registros). Rode só `verify`, `wait` e o comando da sua campanha (`detect` ou `causes`).

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

## 4. Execução de uma rodada

1. `python prepare.py verify` precisa responder `ok`. Se não responder, registre o erro no ledger e termine.
2. A rodada leva cerca de 62 min e roda em segundo plano (o Bash tem limite de 10 min por chamada):

```
python prepare.py <detect|causes> --seed <semente do comando> --background   # imprime o run_id
python prepare.py wait <run_id>                                              # até 9 min; repita até sair o resumo
```

- Use **exatamente a semente do comando**. Ela vale para os 4 braços da rodada.
- Se o comando pedir `--smoke`, acrescente `--smoke`.
- A duração (60 min) e o ambiente são fixos pelo juiz: todas as rodadas são comparáveis entre si.
- **`workload.py` recusado:** se o comando responder `INVÁLIDO` na hora, nada foi medido. Corrija
  `workload.py`, refaça o commit (`git commit --amend -a --no-edit`) e rode de novo com a mesma semente.
- **Rodada travada:** se o `wait` continuar respondendo `running` 90 min depois do início, registre a rodada como
  `failed` e termine.
- Os detalhes de cada braço ficam em `results/<run_id>/<braço>/evaluation.json`; o resumo da rodada em
  `results/<run_id>/round.json`. Não abra `client.csv`, `instrumentation.jsonl` nem `app.log` (são grandes e não
  são a medida).
- Na coluna `commit` dos registros, use o hash curto (`git rev-parse --short=7 HEAD`).
- Timestamps vêm de `date -Iseconds`. Nunca invente horários.

## 5. Campanha `detect`

**Objetivo:** encontrar, dentro do contrato, cargas com a maior tendência de alta da latência de `retrieve_hot`
em relação à referência.

**Referência:** um braço de cada rodada roda a carga com todos os `args` no padrão. Os outros braços (até 3) são
os seus candidatos.

**Nota do braço:** `score_hot_latency_ms_per_h` = Sen's slope da latência p50 de `retrieve_hot` (ms/h), só se o
braço for elegível e não tiver D4. **`ratio`** = inclinação do candidato ÷ inclinação da referência da mesma
rodada; só existe se os dois forem elegíveis.

**O que o sistema faz entre as suas rodadas:** quando um candidato seu fica elegível, o sistema repete a carga dele
em rodadas de verificação (aparecem em `results/` com `"event": "check"`, o candidato em `source` e o resultado em
`decision`). Uma carga com duas verificações positivas está confirmada. Repetir uma carga já confirmada não
acrescenta nada: procure cargas diferentes.

**Uma rodada:**

1. Leia `results.tsv`, `ledger.jsonl` e, se existirem, os `round.json` das rodadas de verificação.
2. Escreva de 1 a 3 candidatos em `workload.py`, cada um com uma hipótese falsificável e o fator que testa.
   **Na rodada 1, rode o `workload.py` como está**, sem alterar.
3. `git commit -am "<resumo das hipóteses>"`. Na rodada 1 não há o que commitar nem, depois, commit a desfazer:
   pule este passo e o `git reset` do passo 6.
4. Rode `detect` (seção 4). Quando ele imprimir o `run_id`, registre `started` no ledger. Rode `wait` até sair
   o resumo.
5. Registre uma linha por braço no `results.tsv` (inclusive a referência) e `completed` no ledger.
6. **Keep ou discard:**
   - se a referência da rodada não for elegível, nenhum candidato vira keep: status do ledger
     `referencia_sem_sintoma`, candidatos `discard`, e desfaça o commit;
   - se ainda não há melhor anterior, o candidato com o maior `ratio` vira o melhor anterior (`keep`);
   - senão, mantenha o commit se algum candidato tiver `ratio ≥ 1,10 × ratio` do melhor anterior;
   - **simplicidade:** um candidato com menos `args` fora do padrão que fique dentro de 10% do `ratio` do melhor
     anterior também vira keep e passa a ser o melhor anterior;
   - nos outros casos, `git reset --hard HEAD~1`. As linhas de `results.tsv` e do ledger continuam.
7. Termine.

**Sem ideias:** combine candidatos que quase passaram, teste os extremos das faixas e volte a fatores pouco
testados.

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

**Quando o comando pedir o resumo da campanha `detect`**, escreva `summary_detect.md` com:

- tabela de todas as rodadas de busca, com a semente;
- tabela da referência por rodada: `seed`, inclinação e mudança da latência de `retrieve_hot`, `hr_p`, elegível
  (sim/não), e em quantas rodadas ela foi elegível;
- as cargas verificadas pelo sistema e o resultado de cada verificação;
- o melhor braço (`run_id` e `name`) e o porquê; se nenhum candidato virou keep, o melhor braço é a referência;
- evidência de que o tamanho de resposta de `retrieve_hot` subiu e de que o controle interno ficou plano;
- todos os atalhos observados;
- limitações: janela de 60 min, uma semente por rodada de busca.

## 6. Campanha `causas`

**Referência:** `frozen/workload_frozen.py`, a carga confirmada desta campanha (`FROZEN_ARM` indica qual candidato
de `CANDIDATES`). Um braço de cada rodada roda essa carga (`referencia_congelada`); os outros (até 3) são
variações dela. Se o arquivo não existir, registre isso no ledger e termine.

**Objetivo:** descobrir quais fatores da carga mudam a tendência, comparando cada fator isoladamente.

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

**Uma rodada (campanha N, rodada r de 6):**

1. Leia `frozen/workload_frozen.py`, `results_causas.tsv` e `ledger_causas.jsonl`.
2. Escolha de 1 a 3 variações, cada uma com o fator, o valor novo, a hipótese e a direção esperada do efeito:
   - **réplica:** cada variação desta campanha precisa rodar em **duas rodadas**, com sementes diferentes. Dê
     prioridade às variações desta campanha que têm só uma réplica válida e planeje as rodadas que restam;
   - **hipóteses novas primeiro:** uma variação com o mesmo `fator` e o mesmo `valor_var` de uma campanha
     anterior não é rodada de novo. Teste primeiro as que ainda não foram testadas e cite as repetidas no resumo,
     com o resultado anterior.
3. Escreva as variações em `workload.py` e `git commit -am "<variações>"`.
4. Rode `causes` (seção 4). Quando ele imprimir o `run_id`, registre `started` no ledger. Rode `wait` até sair o
   resumo.
5. Registre uma linha por braço no `results_causas.tsv` e `completed` no ledger. Não há keep nem discard: o commit
   fica.
6. Se a referência da rodada não for elegível (`reference_ok: False`), registre `referencia_sem_sintoma`; as
   variações daquela rodada não contam como réplica.
7. Termine.

**`results_causas.tsv`** (separado por tabulação):

```
campanha	commit	run_id	seed	arm	fator	valor_ref	valor_var	valid	eligible	lat_ratio	bytes_ratio	cost_ratio	assoc_p95_ratio	ref4_p95_ratio	flags	hypothesis
```

**`ledger_causas.jsonl`**: mesmo formato do ledger da campanha `detect`, com `"campanha"` e
`"variacoes": [{"name", "fator", "valor", "hypothesis", "direcao_esperada"}]` no `started`.

**Como declarar um efeito:** para cada razão, a variação tem efeito se, nas duas réplicas válidas, a razão ficar
do mesmo lado de 1 e a pelo menos 20% dele (≥ 1,2 nas duas ou ≤ 0,8 nas duas). Caso contrário, "sem efeito
claro".

**Quando o comando pedir o resumo da campanha `causas` N**, escreva `summary_causas_<N>.md` com:

- a carga congelada (`args`);
- tabela fator → valor → razões nas duas réplicas → efeito (sim/não, direção) para cada medida;
- quais fatores mudam a inclinação da latência, quais mudam o tamanho da resposta e quais mudam o custo por MB;
- hipóteses já testadas em campanhas anteriores e não repetidas, com o resultado anterior;
- variações sem as duas réplicas válidas, e por quê;
- limitações: um fator de cada vez (sem interações), janela de 60 min, duas réplicas por variação.

## 7. Registros

- Não declare efeito maior que a diferença entre réplicas.
- O `results/judge_log.jsonl` é a fonte de verdade; os seus registros precisam bater com ele.
