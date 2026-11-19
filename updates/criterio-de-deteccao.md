# Critério de detecção e liberação do reparo

A campanha de reparo (campanha 2) só tem sentido se a detecção mostrou o sintoma do finding F1 de forma
robusta. Este documento define, sem ambiguidade, **quando um braço conta como detecção** e **quando o humano
pode liberar o reparo**. Os números vêm do código (`harness/evaluate.py`, `prepare.py`); se o código mudar,
esta nota precisa mudar junto.

## 1. O que é medido

Para cada braço (uma instância da app sob uma carga), o juiz agrega o `client.csv` em **janelas de 60 s**
(5 s no smoke). Só entram respostas 2xx, e caem a primeira janela parcial, a última e o aquecimento (60 s).
Por janela, calcula:

- `retrieve_hot`: latência p50 e tamanho médio da resposta (o **sintoma**). Com vários telefones quentes
  (`retrieve_hot1`, `retrieve_hot2`…), vale o de maior inclinação de latência. No R0, que não tem telefone
  quente, vale a única leitura (`retrieve_cards`);
- `retrieve_ref4` e `retrieve_ref200`: latência p50 e tamanho, para os telefones com 4 e 200 cartões fixos (o
  **controle interno**).

Sobre cada série: **Mann-Kendall com a modificação de Hamed-Rao** (corrige autocorrelação, a mesma biblioteca
do `analyze_run.py` do Cesar) e **Sen's slope** por hora. A "mudança implícita" é
`Sen × duração da série / mediana`, em %.

O RSS do processo (psutil, a cada 5 s, sem os primeiros `min(300 s, 10%)`) é medido com MK + Sen e reportado.

## 2. Um braço é DETECÇÃO POSITIVA (`score_eligible = True`) quando as quatro condições valem

| # | Condição | Regra exata | Constante |
|---|---|---|---|
| 1 | **Braço válido** | sem D1 (exceções > 1%, 5xx > 1% ou CPU p95 do gerador > 90% de um núcleo), sem D3 (`args` fora do contrato ou script com erro), sem H4 (processo morreu, mudou de PID ou teve filhos), com ≥ 10 amostras de psutil e ≥ 5 janelas do cliente | `MAX_EXCEPTION_RATE`, `MAX_SERVER_ERROR_RATE`, `MAX_LOADGEN_CPU` |
| 2 | **Sintoma: latência sobe** | latência p50 de `retrieve_hot`: p unilateral de alta (Hamed-Rao) **< 0,01** **e** mudança implícita **≥ 50%** | `ALPHA`, `MIN_LATENCY_CHANGE_PCT` |
| 3 | **Mecanismo ativado: resposta cresce** | tamanho médio da resposta de `retrieve_hot`: mesmo teste, p < 0,01 e mudança ≥ 50% | idem |
| 4 | **Controle interno plano** | nenhum destes sinais tem p < 0,01 **e** mudança ≥ 25%: tamanho de resposta de `retrieve_ref4`, tamanho de `retrieve_ref200`, latência p50 de `retrieve_ref200`. Se tiver, a flag é `C1_referencia_interna_subindo`: é deriva global (disco, page cache, máquina), não crescimento por linha. **A latência do `ref4` não entra**: no R1 ela sobe pela interferência do próprio aging (8 h do Cesar: +14,8%, p = 0,0002); fica como diagnóstico `ref4_diag` e, onde há R0b na rodada (`controls`, `confirm`), como `ref4_vs_R0b` (se o R0b também sobe, é deriva global) | `MAX_REF_CHANGE_PCT`, `C1_SIGNALS` |

Na detecção, um braço também não pontua com **D4** (`args` iguais aos da referência: é só ruído) nem com **D2**
(que está **desligada no F1**, porque o crescimento de linhas é o próprio finding).

**Nota do braço:** `score_hot_latency_ms_per_h` = Sen's slope da latência p50 de `retrieve_hot`, em ms/h. Só
existe se o braço for elegível. O RSS **não** entra no critério de detecção; ele é registrado como evidência
secundária (`rss_upward_significant`: MK p < 0,01 e Sen ≥ 5 MB/h).

## 3. Um controle é PLANO (`flat = True`)

Usado em `prepare.py controls` para R0 e R0b:

`flat = válido E não (latência sobe) E não (resposta cresce) E não (RSS sobe)`

com as mesmas regras e limiares das condições 1–3 acima e do RSS. **Os controles negativos usam exatamente os
limiares que pontuam os candidatos:** se um controle "detectar", o critério é frouxo demais e nenhuma detecção vale.

## 4. Liberação do reparo (protocolo humano)

O reparo é liberado quando **todas** as etapas abaixo dão certo, nesta ordem:

| Etapa | Comando | Passa se |
|---|---|---|
| A. Controles de 60 min | `prepare.py controls --with-r1` | R0 e R0b com `flat = True` (`ok=True`) **e** R1 com `score_eligible = True` |
| B. Campanha 1 | `./run_campaign.sh detect <tag>` | existe um braço `keep` no `results.tsv`, conferido contra `results/judge_log.jsonl` (ou a própria referência, se nenhum candidato a superou por ≥ 10% no `ratio`). Cada rodada usa uma semente própria (101–108); ver "Semente por rodada e keep" abaixo |
| C. Congelar o candidato | `prepare.py freeze <run_id> <arm>` | o braço é o `keep` da etapa B. O congelamento ainda **não libera o reparo**: ele existe porque `confirm` lê a carga de `frozen/workload_frozen.py` |
| D. Confirmação de 8 h, **duas execuções** | `prepare.py confirm --phase detect --seed 201` e depois `--seed 202` | **as duas** passam: em cada uma, **R0 e R0b planos** (regra da seção 3) **e** R1 e o candidato congelado com `score_eligible = True`. Ver "Confirmação: duas execuções" abaixo |

Só depois de D o humano roda `prepare.py repair-init` e a campanha 2.

### Confirmação: duas execuções, as duas precisam passar (etapa D)

- **Duas execuções de `confirm --phase detect`**, com sementes distintas (201 e 202; nenhuma das sementes da
  campanha 1). Cada execução sobe as quatro rodadas (R0, R0b, R1 e o candidato) em paralelo, com banco limpo, e
  as quatro usam a **mesma semente** daquela execução.
- **Mesma duração da execução que se confirma:** a confirmação é de 8 h, então as duas execuções têm 8 h. Só a
  semente muda entre elas; a carga e a duração ficam fixas. O critério de subida (mudança ≥ 50%) depende do tamanho
  da janela, por isso execuções de durações diferentes não são comparáveis.
- **Unanimidade, não maioria:** o aging é confirmado só se **as duas** passarem. O efeito do F1 é grande (no
  experimento de 8 h do Cesar, a latência de `retrieve_hot` foi de ~10 ms para ~1,3 s); se for aging real, deve
  aparecer sempre. Uma execução negativa não é absorvida por maioria: ela indica um problema a investigar.
- **Inválida ≠ negativa:**
  - **inválida** é um problema de infraestrutura: `valid = False` em algum braço (D1, D3, H4, `sem_amostras`, app
    caiu). Ela **não conta**: repete-se com a próxima semente (203, 204, …), sem reutilizar nenhuma;
  - **negativa** é uma execução válida em que R0 ou R0b subiram, ou R1 ou o candidato não ficaram elegíveis. Com
    uma negativa, **a detecção não está confirmada**.
- Custo: cerca de 16 h de máquina (duas execuções de 8 h, cada uma com os quatro braços em paralelo).

### Semente por rodada e keep (etapa B)

- **Semente:** rodada *n* da campanha 1 usa `--seed 100+n` (101 … 108). Rodadas que falham também consomem a
  semente; nenhuma é reutilizada. A semente vale para os 4 braços, então a referência R1 (4º slot) é, a cada
  rodada, uma réplica com semente independente. O `summary_detect.md` traz a tabela da R1 por rodada (semente,
  `hot_lat_slope`, `hot_lat_change`, `hr_p`, elegível): é a evidência de que o sintoma do F1 se repete com
  sementes independentes, a conferir antes de liberar o reparo.
- **Keep:** o score bruto (ms/h) não é comparado entre rodadas. Compara-se `ratio_vs_reference` = inclinação do
  candidato / inclinação da referência da mesma rodada (`ratio=` na saída do juiz). Um candidato elegível vira
  keep se `ratio ≥ 1,10 × ratio` do melhor anterior; na primeira rodada, o maior `ratio` elegível define o
  melhor anterior.
- **Referência sem sintoma:** se a R1 da rodada não for elegível, o `ratio` de todos sai `NA`; nenhum candidato
  daquela rodada vira keep e o ledger registra `referencia_sem_sintoma`.

**Se D falhar** (uma das duas execuções válidas é negativa): não comece o reparo. A carga congelada em C não vale;
remova `frozen/workload_frozen.py` e rode `prepare.py rehash` antes de qualquer outro passo. Registre o resultado; é
um achado (detecção não confirmada), não um erro a contornar.

### Dentro de cada rodada de reparo

Mesmo com o reparo liberado, uma rodada só pontua candidatos se o **original (`instrumented`) mostrar o sintoma
naquela rodada** (`score_eligible = True` sob a carga congelada). Senão, todos os candidatos recebem
`original_sem_tendencia_nesta_rodada` e ficam sem nota: não dá para medir redução de uma tendência que não apareceu.

Para candidatos que passaram nos portões (H1–H5, replay) e nas travas pós-medição (H6, H7):

- `score_reduction = 1 − Sen(latência retrieve_hot do candidato) / Sen(do original)`;
- `fixed = True` se (a) a redução é ≥ `FIXED_MIN_REDUCTION` (0,30, provisório) **ou** a latência do candidato deixa
  de satisfazer a condição 2, **e** (b) a **interferência** não piora: p95 de `associate_card` e de `retrieve_ref4`
  do candidato ≤ p95 do original × (1 + `FIXED_P95_TOL`) (0,10, provisório), na mesma rodada e semente;
- o tamanho de resposta de `retrieve_hot` **não** entra (a spec não deixa mudar), e não se exige zerar a inclinação
  (parte dela cresce por design). Os dois parâmetros **não estão calibrados**: conta e plano em
  `program_repair.md` ("Métrica"). O reparo do Cesar (~43% em 8 h) passa no critério de inclinação.

## O que o código impõe e o que não impõe

| Etapa | Imposto pelo código? |
|---|---|
| Condições 1–4 e a nota | Sim (`harness/evaluate.py`, `prepare.py:cmd_detect`) |
| `flat` dos controles | Sim, em `controls`. **Em `confirm` não**: a saída de `confirm --phase detect` mostra as métricas, mas o humano confere a regra da seção 3 |
| Etapas A–D | **Não.** `freeze` aceita qualquer `run_id` e braço, e `repair` só exige que `frozen/workload_frozen.py` exista |
| Duas confirmações, ambas aprovadas (etapa D) | **Não.** `confirm` roda uma execução por chamada e não sabe das anteriores; o humano roda as duas e confere as duas antes de `repair-init` |
| Semente por rodada (101–108) e keep pela razão contra a referência da rodada | **Não.** É protocolo do agente (`program_detect.md`). O juiz só usa o `--seed` recebido (padrão 1), grava-o em `round.json` e calcula o `ratio_vs_reference`; não confere a sequência de sementes nem decide keep/discard |
| Original com sintoma em cada rodada de reparo | Sim (`prepare.py:score_repair`) |

A ordem é **congelar (C) e depois confirmar (D)** porque o `confirm` atual lê a carga do candidato de
`frozen/workload_frozen.py`. Fazer o `repair-init` recusar começar sem duas confirmações aprovadas (controles
planos e braços elegíveis) é uma mudança pequena no juiz e ainda não foi feita.

## Limiares e o que os sustenta

| Limiar | Valor | De onde vem |
|---|---|---|
| `ALPHA` | 0,01 | mantido do piloto anterior (mais estrito que o 0,05 do `analyze_run.py` do Cesar) |
| `MIN_LATENCY_CHANGE_PCT` | 50% | valor de partida. Pela calibração do Cesar (`F1_targeted_v1.md`), a latência vai de ~2–5 ms a ~50 ms em 1 h, uma mudança muito maior |
| `MAX_REF_CHANGE_PCT` | 25% | valor de partida |
| `MIN_EFFECT_MB_PER_H` (RSS) | 5 MB/h | A/A da versão Node (20/10); **precisa ser recalibrado** para a app FastAPI |

Os três primeiros só serão considerados calibrados depois de a rodada de 60 min de R0/R0b/R1 e a confirmação de
8 h mostrarem os controles planos e o R1 elegível com folga.
