# Auditoria de evidências e estrutura (23/11)

Auditoria só de leitura, feita com as skills `auditor-de-evidencias` e `educador-baseado-em-fontes`. Nenhum
arquivo do juiz, da app ou das cargas foi alterado; esta nota é o único arquivo novo.

**Escopo.** Lidos a fundo: `README.md`, `docs/DESENHO-EXPERIMENTAL.md`, `updates/criterio-de-deteccao.md`,
`harness/stats.py`; lidos por busca: `harness/evaluate.py`, `harness/instance.py`, `harness/load.py`,
`workloads/F1_targeted_v1.py`. **Não checados:** `program.md`, `orchestrator.py`, testes, e nenhum dado (ainda
não existe rodada real). Citações externas não foram verificadas. A checagem não foi independente.

## Veredito curto

O desenho é **bem estruturado e acima da média** em controle de viés do agente: juiz fixo com hash, controles
negativos pareados (R0, R0b), controle interno (ref4/ref200), réplicas com unanimidade, separação inválida ×
negativa, taxonomia de atalhos e limitações declaradas com honestidade. Os problemas abaixo não derrubam o
desenho, mas um revisor de aging atacaria primeiro o **P1** e o **P2**.

## Problemas maiores

**P1 — Conceitual: o critério de detecção mede crescimento por volume, não aging separado da carga.**
O F1 é, por definição, custo O(n) nos cartões acumulados. A detecção exige latência **e** tamanho de resposta de
`retrieve_hot` subindo (`evaluate.py:181-182`). Como o tamanho cresce porque a carga insere linhas, a condição 3
é quase garantida pela própria carga, e a condição 2 segue dela. Isso é "acúmulo de estado que degrada o
serviço" (válido), mas não é a mesma coisa que "o sistema piora com o tempo de operação à mesma carga".
- *Conclusão permitida:* "a carga ativa o acúmulo do F1 e a latência de leitura cresce com o estado acumulado".
- *Não permitida:* "aging confirmado" sem qualificar o tipo.
- *Para subir de degrau:* (a) latência normalizada pelo tamanho (o `cost_ratio`/custo por MB já existe na
  `causas`; usar também na detecção: ms por MB ou por cartão subindo = degradação além do volume); (b) fase de
  pausa da inserção (grow=0) no fim da rodada: se a latência fica no patamar alto, é persistente; (c) definir no
  texto, com fonte, que tipo de aging o estudo assume (acúmulo de recurso/estado).

**P2 — Instrumento: os 4 braços e os 4 geradores rodam na mesma máquina.**
Há `taskset` para CPU (`instance.py:93`), mas disco, page cache, barramento de memória e SQLite em disco são
compartilhados. O braço com aging (leituras grandes, travamentos de ~150 ms) pode contaminar R0/R0b na mesma
rodada, ou o contrário. Já está em Pendências, mas é ameaça à validade dos controles, não detalhe.
- *Correção:* uma rodada A/A (4× R0b em paralelo) e uma com R1 sozinho versus R1 com vizinhos, para medir a
  interferência entre braços; registrar iowait e frequência de CPU.

**P3 — O gerador depende do estado.** O `Pacer` (`F1_targeted_v1.py:42-63`) descarta slots perdidos quando o
servidor atrasa: a taxa realizada cai justamente quando o sistema degrada (menos linhas inseridas, menos
sondas). O gerador é malha fechada com ritmo fixo (workers fixos, docstring do script); a carga deixa de ser controlada. *Correção:* reportar por braço a taxa
realizada/planejada de cada fluxo e invalidar (ou marcar) abaixo de ~90%.

**P4 — Limiares sem calibração e mudança implícita dependente da duração.**
`sen_change_pct = Sen × duração / mediana` (`stats.py`) cresce com a duração; 50%/25%/20% são de partida.
Comparar com as 8 h do Cesar exige a mesma métrica na mesma duração. Já reconhecido no texto; falta o plano.

## Problemas menores

- **Fallback para MK original** quando Hamed-Rao degenera (`stats.py`, `original_hr_degenerado`): o MK sem
  correção infla a significância sob autocorrelação. Está registrado em `mk_method`, mas o relatório deveria
  contar quantas decisões dependem desse fallback.
- **Docstring de `stats.py`** diz "só biblioteca padrão, sem scipy", mas `trend_hr` usa numpy e pymannkendall.
  A referência "Santos, Andrade, Natella, arXiv:2510.24188" ficou **NÃO VERIFICADA** (conferir o PDF).
- **Seleção do melhor de até 24 candidatos** (8 rodadas × 3): maldição do vencedor. A verificação em 2 rodadas
  mitiga; reportar também quantos candidatos foram elegíveis no total.
- **Causas:** razão de duas inclinações ruidosas é instável quando a referência é pequena; 2 réplicas e corte de
  20% sem intervalo. Reportar as razões das duas réplicas e um IC (bootstrap das janelas).
- **Réplicas sequenciais na mesma máquina:** a semente só muda números de cartão; a variação real é da
  execução. Reportar a dispersão das inclinações de R1 entre todas as rodadas (já são ~10 réplicas grátis).
- **RSS:** psutil `rss` inclui páginas de arquivo; `uss` já é coletado (`instance.py:119`) e é mais adequado
  como evidência de memória.
- **"Sem controle positivo"**: na prática R1 *é* um controle positivo para o efeito forte do F1 (Fase 0 exige
  R1 elegível). O que falta é controle positivo de **efeito fraco**, para medir sensibilidade.

## Estrutura e didática dos documentos

- Pontos fortes: separação clara por público (agente, projeto, notas), tabela "de onde vem cada decisão",
  RQs ligadas a fases.
- `criterio-de-deteccao.md` ainda descreve reparo e etapas humanas; o aviso de histórico está no topo, mas o
  corpo confunde quem chega novo. Sugestão: mover para `updates/historico/` ou cortar o que não vale mais.
- Faltam: um **glossário** (aging, ativação, elegível, plano, braço, rodada, D1–H4) e a **definição operacional
  de aging com referência** na literatura. O corpus do repositório não tem PDFs de referência, então não deu
  para ancorar definições em fonte; isso é pendência, não erro.

## Perguntas que um revisor faria

1. O que distingue aqui "aging" de "complexidade linear esperada da consulta sem LIMIT"?
2. A latência por cartão (ou por MB) muda ao longo da rodada?
3. Os controles ficam planos quando rodam sozinhos e quando rodam ao lado de R1?
4. Qual a dispersão do efeito de R1 entre rodadas?
5. Como o agente se compara a uma busca aleatória ou em grade no mesmo contrato (baseline para a RQ1)?

## Pendências desta auditoria

Ler `program.md` e `orchestrator.py`; auditar dados assim que houver a primeira rodada real (rodar
`auditor.py serie/coleta/operacoes` no `client.csv` e no CSV do psutil); conferir arXiv:2510.24188.
