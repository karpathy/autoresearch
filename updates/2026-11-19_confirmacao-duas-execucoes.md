# 2026-11-19 — Confirmação de 8 h em duas execuções, as duas precisam passar

**Objetivo:** só liberar o reparo depois de confirmar o comportamento do F1 com sementes independentes, como
combinado com o Ermeson (e descrito na resposta ao Fumio e ao Artur).

**Antes:** a confirmação de 8 h (`confirm --phase detect`) era uma execução só; repetir era opcional. A
documentação mandava confirmar e depois congelar, mas o `confirm` lê a carga de `frozen/workload_frozen.py`, então
essa ordem não rodava.

**Agora:**

| Ponto | Regra |
|---|---|
| Ordem | `freeze` do candidato (etapa C) → `confirm --phase detect` duas vezes (etapa D) → só então `repair-init` |
| Execuções | **duas**, com sementes distintas: `--seed 201` e `--seed 202` (fora da faixa 101–108 da campanha 1) |
| Em cada execução | R0, R0b, R1 e o candidato em paralelo, banco limpo, a **mesma semente** para os quatro |
| Duração | igual à da execução que se confirma: 8 h nas duas. Só a semente muda; carga e duração ficam fixas |
| Decisão | **unanimidade**: o aging é confirmado só se as duas passarem (R0 e R0b planos, R1 e candidato elegíveis) |
| Inválida | problema de infraestrutura (`valid = False`: D1, D3, H4, `sem_amostras`, app caiu): não conta, repete-se com a próxima semente (203, …) |
| Negativa | execução válida em que um controle subiu ou R1/candidato não ficaram elegíveis: detecção **não confirmada**, sem reparo; remover `frozen/workload_frozen.py` e rodar `rehash` |

Custo: cerca de 16 h de máquina.

## Por quê

- **Reprodutibilidade:** a crítica do Fumio e do Artur foi a falta de confirmação entre sementes. "Passou em 2 de
  3" admitiria que o resultado não se reproduziu.
- **Efeito grande:** no experimento de 8 h do Cesar (`docs/verdicts/credit-card-fastapi-R1-8h.md` no repo dele,
  semente 42), a latência de `retrieve_hot` foi de ~10 ms para ~1,3 s, e R0/R0b ficaram planos. Se é aging real,
  deve passar sempre; uma falha merece investigação, não ser absorvida por maioria.
- **Mesma duração:** a mudança implícita (`Sen × duração / mediana`, critério ≥ 50%), o aquecimento descartado e
  o número de janelas dependem da duração; só execuções de mesma duração são comparáveis.
- **O que a semente muda:** nos scripts do Cesar, só os números dos cartões gerados e quais pares o fluxo
  `refresh` reenvia; taxas, sondagem e crescimento do banco são os mesmos. No R0 (`F1_control`) a semente não muda
  nada. Por isso a variação entre execuções vem tanto da semente quanto da própria execução (máquina, momento).

## Arquivos

| Arquivo | Mudança |
|---|---|
| `updates/criterio-de-deteccao.md` | Etapas C (congelar) e D (duas confirmações) na seção 4; subseção "Confirmação: duas execuções"; "Se D falhar"; linha nova em "O que o código impõe" |
| `README.md` | Passo 6 com `freeze` e as duas chamadas de `confirm`; seção "Duração e confirmação" |
| `program_detect.md` | "Ao final": confirmação humana em duas execuções; texto da primeira rodada do keep corrigido (os candidatos do `workload.py` base rodam normalmente) |
| `updates/contexto.md`, `updates/README.md` | Decisão registrada; estado e pendências atualizados |

Nada em `prepare.py`, `harness/`, `oracle/`, `workloads/`, `findings/` ou `apps/` mudou: `confirm` já aceita
`--seed` e aplica a mesma semente aos quatro braços. Não houve rehash.

## O que ainda falta

- **O juiz não impõe:** `confirm` roda uma execução por chamada e não sabe das anteriores; nada impede
  `repair-init` antes das duas confirmações. Fazer o `repair-init` recusar começar sem duas confirmações aprovadas
  é uma mudança pequena no juiz, ainda não feita.
- **A confirmação do reparo** (`confirm --phase repair`) continua com uma execução; esta nota trata só da
  detecção.
