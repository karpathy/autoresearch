# 2026-11-23 — `program.md` só com as instruções do agente, conferido com o autoresearch original

## O que mudou

- **`program.md` único e só para o agente**, como no `karpathy/autoresearch`: instruções diretas de execução e
  regras das campanhas `detect` e `causas`. Sem histórico, nomes de pessoas, mecanismo do finding, resultados
  esperados ou justificativas.
- O **desenho do experimento** (perguntas de pesquisa, fases, parâmetros, custo, limitações) foi para
  [`docs/DESENHO-EXPERIMENTAL.md`](../docs/DESENHO-EXPERIMENTAL.md), para as pessoas do projeto.
- O `program_detect.md` foi absorvido pelo `program.md` e removido.
- O `README.md` foi reescrito para o estado atual.

## Menos viés

| Medida | Como |
|---|---|
| Leitura restrita | o agente lê só `program.md`, `workload.py`, `frozen/`, `results/` e os próprios registros; código da app, card do F1, cargas do Cesar, juiz, testes, `docs/`, `updates/`, `README.md` e o histórico do git ficam negados (`.claude/settings.{detect,causas}.json`) |
| Git sem conteúdo | só `git log --oneline -n 20`; `git log -p`, `git show` e `git diff` negados |
| Textos neutros | `workload.py` sem descrição do mecanismo; hipóteses dos braços fixos ("Referência…", "Controle negativo R0") e mensagens de erro do contrato sem nomes nem pistas |
| Sistema descrito pelo necessário | o `program.md` descreve as duas operações da API, os parâmetros da carga e as medidas, sem dizer o que se espera encontrar |

## Conferência com o `program.md` do karpathy/autoresearch

| Conceito do original | No nosso `program.md` | Situação |
|---|---|---|
| Tag e branch `autoresearch/<tag>` nova | setup das duas campanhas | mantido |
| Ler os arquivos do escopo (README, prepare.py, train.py) | lê só `program.md` e `workload.py` | **diferença deliberada**: evitar viés |
| Verificar o ambiente antes | `prepare.py verify` | mantido |
| `results.tsv` com cabeçalho, sem commit | `results.tsv` / `results_causas.tsv` e ledgers, sem commit | mantido |
| Um único arquivo editado pelo agente | `workload.py`, só literais | mantido |
| Não mexer na avaliação nem instalar pacotes | regras da seção 2 | mantido |
| Orçamento de tempo fixo, rodadas comparáveis | 60 min fixos pelo juiz | **acrescentado** explicitamente |
| Uma métrica, objetivo simples | `ratio` (detect) e razões (causas) | mantido |
| Primeira rodada = baseline sem alterações | rodada 1 roda o `workload.py` como está; a referência roda em toda rodada | **acrescentado** |
| Formato de saída conhecido | resumo do `wait`, `round.json`, `evaluation.json` | mantido |
| Hash curto do commit no TSV | `git rev-parse --short=7 HEAD` | **acrescentado** |
| Keep avança a branch, discard faz `git reset` | detect: igual; causas: sem keep/discard (comparação) | mantido / adaptado |
| Critério de simplicidade (empate com menos complexidade é ganho) | empate dentro de 10% com menos `args` alterados vira keep | **reforçado** (antes era só "prefira") |
| Crash: corrigir o trivial e repetir, senão registrar e seguir | `INVÁLIDO` na hora: corrigir, `--amend` e repetir com a mesma semente; falha durante a rodada: `failed`, próxima semente | **acrescentado** |
| Timeout (rodada longa demais = falha) | `wait` em `running` por mais de 90 min = falha | **acrescentado** |
| Não inundar o contexto com logs | não abrir `client.csv`, `instrumentation.jsonl`, `app.log` | **acrescentado** |
| NEVER STOP: não perguntar, seguir até ser interrompido | não perguntar nada; orçamento fixo (8 e 6 rodadas), sem parar antes | adaptado: orçamento fixo é parte do desenho (decisão com o Cesar: mínimo de rodadas) |
| Sem ideias: pensar mais, combinar quase-acertos, mudanças mais radicais | "combine candidatos que quase passaram, teste os extremos das faixas, volte a fatores pouco testados" | **acrescentado** |

## O que veio do desenho para o `program.md`

Só o que o agente precisa para executar: critério de elegibilidade e limiares, flags, sementes e orçamentos, regra
de keep, regra de um fator, réplicas, medidas da fase de causas e a regra para declarar efeito, formatos dos
registros e o conteúdo dos resumos. Ficaram só no desenho: perguntas de pesquisa, fases humanas (controles,
congelamento, confirmação), custos, origem das decisões e limitações do estudo.

## Detalhe corrigido

Na rodada 1 da `detect` o `workload.py` não muda, então não há commit; o `program.md` agora diz para pular o
commit e o `git reset` nessa rodada (sem isso, um `git reset --hard HEAD~1` apagaria o commit de setup).
