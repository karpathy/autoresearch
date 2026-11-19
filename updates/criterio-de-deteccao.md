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
| 4 | **Controle interno plano** | nem `retrieve_ref4` nem `retrieve_ref200` têm latência ou tamanho com p < 0,01 **e** mudança ≥ 25%. Se tiverem, a flag é `C1_referencia_interna_subindo`: é deriva global (disco, page cache, máquina), não crescimento por linha | `MAX_REF_CHANGE_PCT` |

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
| C. Confirmação de 8 h | `prepare.py confirm --phase detect --seed <semente>` | **R0 e R0b planos** (regra da seção 3) **e** R1 e o candidato escolhido com `score_eligible = True`, todos na mesma rodada de 8 h. Cada execução de `confirm` usa a sua própria semente; repetir não é obrigatório, mas, se repetir, use sementes distintas |
| D. Congelar | `prepare.py freeze <run_id> <arm>` | o braço congelado é o mesmo que passou em C |

Só depois de D o humano roda `prepare.py repair-init` e a campanha 2.

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

**Se C falhar** (um controle sobe, ou o candidato não se sustenta em 8 h): não congele e não comece o reparo.
Registre o resultado; é um achado (detecção não confirmada), não um erro a contornar.

### Dentro de cada rodada de reparo

Mesmo com o reparo liberado, uma rodada só pontua candidatos se o **original (`instrumented`) mostrar o sintoma
naquela rodada** (`score_eligible = True` sob a carga congelada). Senão, todos os candidatos recebem
`original_sem_tendencia_nesta_rodada` e ficam sem nota: não dá para medir redução de uma tendência que não apareceu.

Para candidatos que passaram nos portões (H1–H5, replay) e nas travas pós-medição (H6, H7):

- `score_reduction = 1 − Sen(latência retrieve_hot do candidato) / Sen(do original)`;
- `fixed = True` se a latência do candidato **deixa de satisfazer a condição 2** **ou** se a redução é ≥ 0,8.

## O que o código impõe e o que não impõe

| Etapa | Imposto pelo código? |
|---|---|
| Condições 1–4 e a nota | Sim (`harness/evaluate.py`, `prepare.py:cmd_detect`) |
| `flat` dos controles | Sim, em `controls`. **Em `confirm` não**: a saída de `confirm --phase detect` mostra as métricas, mas o humano confere a regra da seção 3 |
| Etapas A–D | **Não.** `freeze` aceita qualquer `run_id` e braço, e `repair` só exige que `frozen/workload_frozen.py` exista |
| Semente por rodada (101–108) e keep pela razão contra a referência da rodada | **Não.** É protocolo do agente (`program_detect.md`). O juiz só usa o `--seed` recebido (padrão 1), grava-o em `round.json` e calcula o `ratio_vs_reference`; não confere a sequência de sementes nem decide keep/discard |
| Original com sintoma em cada rodada de reparo | Sim (`prepare.py:score_repair`) |

Fazer o `freeze` recusar um braço sem um `confirm --phase detect` aprovado (controles planos e braço elegível)
é uma mudança pequena e ainda não foi feita.

## Limiares e o que os sustenta

| Limiar | Valor | De onde vem |
|---|---|---|
| `ALPHA` | 0,01 | mantido do piloto anterior (mais estrito que o 0,05 do `analyze_run.py` do Cesar) |
| `MIN_LATENCY_CHANGE_PCT` | 50% | valor de partida. Pela calibração do Cesar (`F1_targeted_v1.md`), a latência vai de ~2–5 ms a ~50 ms em 1 h, uma mudança muito maior |
| `MAX_REF_CHANGE_PCT` | 25% | valor de partida |
| `MIN_EFFECT_MB_PER_H` (RSS) | 5 MB/h | A/A da versão Node (20/10); **precisa ser recalibrado** para a app FastAPI |

Os três primeiros só serão considerados calibrados depois de a rodada de 60 min de R0/R0b/R1 e a confirmação de
8 h mostrarem os controles planos e o R1 elegível com folga.
