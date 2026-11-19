# Metodologia: AutoResearch-Aging no cenário `credit-card-fastapi` / F1

Material para o fluxograma. Descreve o que roda, em que ordem, o que cada braço, portão e flag faz, e quais
decisões liberam a etapa seguinte. Os números vêm do código (`prepare.py`, `harness/`); se o código mudar, este
documento precisa mudar junto. Regras detalhadas: `program_detect.md`, `program_repair.md` e
`updates/criterio-de-deteccao.md`.

## 1. Visão geral

O **método** é o do autoresearch: um agente (Claude Code) propõe, um **juiz fixo** mede, e um laço mantém ou
descarta a proposta. O **alvo e o cenário** são os do Cesar: app `credit-card-fastapi` (BaxBench), finding
`bax-fastapi-credit-card-F1`, rodadas R0, R0b, R1 e R2 do `campaign.yaml`. Nenhum bug é injetado.

**O finding F1:** cada `POST /associate_card` com par novo grava uma linha em `associations`, que nada apaga;
`POST /retrieve_cards` lê todas as linhas do telefone, sem `LIMIT`, e materializa tudo.

- **Ativação (exigida pela spec, não é aging):** as linhas e o arquivo do banco crescem (`db_rows`).
- **Aging (misto):** (1) custo alto por cartão em cada leitura do telefone quente; (2) pico de memória ao
  materializar respostas grandes; (3) **interferência**: o processo trava ~150 ms ou mais a cada leitura grande e
  atrasa as outras requisições (`associate_card`, `retrieve_ref4`).

**Integridade do juiz:** `prepare.py`, `harness/*.py`, `oracle/*.py`, `workloads/*.py`, `findings/*.yaml` (e a
carga congelada) têm hash em `.judge/manifest.json`; todo comando confere (`verify`) e aborta se algo mudou (H5).
Os testes escondidos ficam fora do repo (`~/.aging-hidden`).

```mermaid
flowchart TD
    S[Setup humano: setup, check-oracles] --> A[A. Controles 60 min: R0, R0b planos e R1 elegível?]
    A -- não --> X1[Parar: limiares ou ambiente inadequados]
    A -- sim --> B[B. Campanha 1: detecção, 8 rodadas, sementes 101-108]
    B --> C[C. Humano congela o keep: freeze]
    C --> D[D. Confirmação 8 h, duas execuções: sementes 201 e 202]
    D -- alguma negativa --> X2[Detecção não confirmada: sem reparo; remove frozen e rehash]
    D -- inválida --> D
    D -- as duas passam --> E[repair-init]
    E --> F[Campanha 2: reparo, 8 rodadas]
    F --> G[Confirmação 8 h do reparo: R1, R2, base do agente, R0b]
```

## 2. Braços

Cada braço é uma instância da app em cópia limpa, com banco novo, porta própria, `uvicorn app:app` com log de
acesso ligado (como o `python app.py` do Cesar), app pinada na CPU *i* e gerador de carga na CPU 4+*i*. Até 4
braços rodam em paralelo (`SLOTS = 4`).

| Braço | App | Carga | Papel | Esperado |
|---|---|---|---|---|
| **R0** | `instrumented` | `F1_control` (o `.jmx` original: um par fixo, laço fechado) | Controle negativo | Plano |
| **R0b** | `instrumented` | `F1_targeted_v1 --key-space 4` (mesma carga, nada acumula) | Controle negativo pareado | Plano |
| **R1** | `instrumented` | `F1_targeted_v1` padrão | Referência: o sintoma do F1 | Sobe |
| **R2** | `repaired/` do Cesar | carga congelada | Reparo de referência (só na confirmação do reparo) | Inclinação menor |
| **Candidatos de detecção** (até 3) | `instrumented` | variantes do `F1_targeted_v1`: só `args` literais, taxas ≤ 2× os padrões, ≤ 31 req/s | Propostas do agente | Sobem mais cedo/mais forte |
| **Candidatos de reparo** v1..v3 | `candidates/vN` (código editado pelo agente) | carga congelada | Propostas do agente | Inclinação menor, sem interferência |
| **original** | `instrumented` | carga congelada | 4º slot de cada rodada de reparo | Sobe |

**Quem roda com quem:**

| Comando | Braços (em paralelo) | Duração |
|---|---|---|
| `controls --with-r1` | R0, R0b, R1 | 60 min |
| `detect` (rodada da campanha 1) | R1 (4º slot) + até 3 candidatos | 60 min |
| `confirm --phase detect` | R0, R0b, R1, candidato congelado | 8 h |
| `repair` (rodada da campanha 2) | original (4º slot) + v1, v2, v3 | 60 min |
| `confirm --phase repair` | R1 (original), R2 (`repaired/` do Cesar), base do agente, R0b | 8 h |

## 3. Sementes

A semente (`--seed`) vale para todos os braços de uma execução. Nos scripts do Cesar ela muda só os números dos
cartões gerados e quais pares o fluxo `refresh` reenvia; taxas, sondagem e crescimento do banco não mudam.

| Etapa | Sementes | Regra |
|---|---|---|
| Campanha 1 (detecção) | 101, 102, …, 108 (rodada *n* = 100 + *n*) | Uma nova por rodada; falhas também consomem; nunca reutilizar. A R1 vira uma réplica independente a cada rodada |
| Confirmação da detecção | 201 e 202 | Duas execuções de 8 h; inválida repete com 203, 204, … |
| Controles, reparo, confirmação do reparo | `--seed` explícito (padrão do juiz = 1) | Uma semente por execução |

O protocolo de sementes é do agente e do humano (`program_detect.md`, `criterio-de-deteccao.md`); o juiz só usa e
registra a semente recebida (`round.json`, `judge_log.jsonl`).

## 4. Medição (igual em todos os braços)

- **Cliente** (`client.csv` do script do Cesar): janelas de 60 s (5 s no smoke), só respostas 2xx, sem a primeira
  e a última janela nem o aquecimento. Por janela: latência p50/p95 e tamanho médio da resposta de
  `retrieve_hot` (sintoma), `retrieve_ref4`/`retrieve_ref200` (controle interno: telefones com 4 e 200 cartões
  fixos) e `associate_card`.
- **Servidor** (psutil, de fora, a cada 5 s): RSS, descritores, threads, filhos, tamanho do banco.
- **Estatística:** Mann-Kendall com Hamed-Rao (como o `analyze_run.py` do Cesar) + Sen's slope por hora; mudança
  implícita = Sen × duração / mediana. Se o Hamed-Rao degenerar (variância negativa), cai para o MK original.

## 5. Flags da detecção

| Flag | Dispara quando | Efeito |
|---|---|---|
| **D1** saturação ou erros | exceções/timeouts > 1%, 5xx > 1%, ou gerador de carga > 90% de um núcleo (p95) | Braço inválido |
| **D2** estado legítimo | mais de N linhas no banco. **Desligada no F1**: o crescimento de linhas é exigido pela spec (ativação) | Não se aplica |
| **D3** fora do contrato | `args` fora das faixas, código em `workload.py`, ou o script saiu com erro | Braço (ou rodada) inválido |
| **D4** cópia da referência | candidato com `args` iguais aos padrão | Medido, sem nota |
| **C1** deriva global | tamanho de resposta de `ref4` ou `ref200`, ou latência p50 de `ref200`, sobe (p < 0,01 e ≥ 25%) | Medido, sem nota |
| **H4** processo | o processo morreu, mudou de PID ou teve filhos | Braço inválido |

**A latência do `ref4` não entra no C1:** no R1 ela sobe por causa do próprio aging (interferência; 8 h do Cesar:
p50 +14,8%, p95 de 5 para 43 ms). Ela sai como diagnóstico `ref4_diag` e, onde há R0b na rodada (`controls`,
`confirm`), como `ref4_vs_R0b`: se o R0b também sobe, é deriva global; se só o braço sobe, é interferência.

## 6. Critério de detecção

Um braço é **detecção positiva** (`score_eligible`) quando:

1. é **válido** (sem D1, D3, H4; ≥ 10 amostras do psutil e ≥ 5 janelas do cliente);
2. a **latência p50 de `retrieve_hot` sobe**: p unilateral < 0,01 e mudança ≥ 50%;
3. o **tamanho de resposta de `retrieve_hot` sobe** (o mecanismo foi ativado): mesmo teste;
4. o **controle interno está plano** (sem C1).

Nota do braço = Sen da latência p50 de `retrieve_hot` (ms/h). O RSS é evidência secundária (p < 0,01 e ≥ 5 MB/h).
Um **controle é plano** quando é válido e nem a latência, nem o tamanho, nem o RSS sobem (mesmos limiares).

**Keep na campanha 1:** pela razão contra a R1 da mesma rodada (`ratio_vs_reference`), não pelo score bruto:
mantém o candidato se `ratio ≥ 1,10 × ratio` do melhor anterior. R1 sem sintoma → nenhum keep na rodada.

**Liberação do reparo (humano):** A (controles planos, R1 elegível) → B (keep na campanha 1) → C (`freeze`) → D
(duas confirmações de 8 h, sementes 201 e 202; **as duas** com R0 e R0b planos e R1 e candidato elegíveis).

## 7. Portões do reparo (antes de medir)

`harness/gates.py`, nesta ordem; uma falha vira flag e o candidato não é medido.

| Flag | Portão |
|---|---|
| **H5** dependência ou entrypoint | `requirements.txt` e `aging_probe.py` iguais ao original; `app.py` define `app = FastAPI(...)` |
| **H4** processo filho | o código não usa `subprocess`, `multiprocessing`, `workers=`, `os._exit`, `os.kill` |
| **H1** funcionalidade | **teste funcional** do Cesar (`oracle/test_credit_card_service.py`) e **testes escondidos** (validação 400, idempotência, interseção; o agente vê só a contagem de falhas) |
| **H2** persistência | os dados sobrevivem a um reinício; conteúdo do banco idêntico ao do original depois do replay |
| **H3** resposta alterada | replay diferencial de ~1700 requisições no formato do F1: status e corpo idênticos aos do original (paginar ou truncar `/retrieve_cards` cai aqui) |
| `sem_alteracao` | o diff precisa ter pelo menos uma linha |

**Depois da medição de 60 min:**

| Flag | Regra |
|---|---|
| **H6** desempenho | vazão < 90% da do original, ou latência p50 de `associate_card`, `ref4` ou `ref200` > 110% da do original + 1 ms |
| **H7** erros | mais 5xx sob carga que o original (> 1,5× + 0,5 pp) |
| **H4** processo | morreu ou mudou de PID durante a carga |
| `original_sem_tendencia_nesta_rodada` | o original não mostrou o sintoma na rodada: ninguém recebe nota |

## 8. Critério de reparo

- **Nota (keep):** `score_reduction = 1 − Sen(latência de retrieve_hot do candidato) / Sen(do original)`, mesma
  rodada e semente. Keep se for ≥ 0,10 acima da base atual (`promote`).
- **Não entram:** o tamanho de resposta de `retrieve_hot` (a spec não deixa mudar) e o RSS (reportado). Não se
  exige zerar a inclinação (parte dela cresce por design).
- **`fixed`** exige as duas condições:
  1. **inclinação:** redução ≥ `FIXED_MIN_REDUCTION` (ou a latência deixa de subir de forma significativa);
  2. **interferência:** p95 de `associate_card` e de `retrieve_ref4` do candidato ≤ p95 do original ×
     (1 + `FIXED_P95_TOL`), na mesma rodada e semente. O p95 é por janela de 60 s; compara-se a mediana dos p95 na
     última hora (ou na última metade da série, se ela for menor).
- **Parâmetros (provisórios, NÃO calibrados):** `FIXED_MIN_REDUCTION = 0,30` (o `repaired/` do Cesar reduziu
  ~43% em 8 h; 0,43 × 0,7 de folga = 0,30) e `FIXED_P95_TOL = 0,10`. A calibração prevista é uma rodada de reparo
  com o `repaired/` como candidato.
- **Confirmação de 8 h do reparo:** R1, R2, base do agente e R0b sob a carga congelada; reporta a interferência de
  cada braço contra o R1 e o p95 do **R0b como piso** de referência.

## 9. Duração

| Etapa | Duração | Origem |
|---|---|---|
| Rodada do laço (detecção ou reparo) | 60 min (`DEFAULT_MINUTES`) | Artur: sinal em cerca de 1 h |
| Confirmação | 8 h (`CONFIRM_MINUTES`) | `campaign.yaml` do Cesar |
| Orçamento de cada campanha | 8 rodadas (~8,5 h) | `program_*.md` |
| Smoke | 2 min, janelas de 5 s | só valida o fluxo; nunca entra no ranking |

## 10. O que fica de fora

- **Controle positivo:** o cenário não tem; "não achamos aging" não diz nada sobre a sensibilidade do juiz. Só os
  controles negativos R0 e R0b existem.
- **Stall do `ps`:** não existe no Credit Card (só no app Monitor, injetado pelo `aging_probe.py` do Monitor).
  Correto não implementar.
- **Rejuvenescimento** (reinício periódico, workers que reciclam): proibido no reparo (H4); é outro estudo.
- **Máquinas separadas:** cliente e app rodam na mesma máquina, em núcleos distintos (no artigo eram máquinas
  separadas).
- **Imposição pelo código:** a ordem A–D, as sementes e o keep são protocolo (agente e humano); o juiz não os impõe.
