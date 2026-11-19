# AutoResearch-Aging — Campanha 2: REPARO

Você é um pesquisador autônomo. Seu objetivo é **corrigir a causa do
envelhecimento** da aplicação `credit-card-fastapi` (finding `bax-fastapi-credit-card-F1`, do repositório do Cesar) sem mudar o que ela faz. A carga
que evidencia o problema foi achada e confirmada na Campanha 1, e está
congelada em `frozen/workload_frozen.py`. Leia também `summary_detect.md`, se
existir, e `findings/bax-fastapi-credit-card-F1.yaml`.

## Setup

1. Combine uma tag com o humano, por exemplo `rep-out28`, e rode
   `git checkout -b autoresearch/<tag>`.
2. Leia `README.md`, este arquivo, `frozen/workload_frozen.py`,
   `summary_detect.md`, `harness/gates.py`, `oracle/functional_test.py` e todo o
   código em `apps/credit-card-fastapi/instrumented/`. Não leia
   `apps/credit-card-fastapi/repaired/`: é a correção de referência do Cesar, só para o humano comparar.
3. `python prepare.py verify` precisa responder `ok`.
4. Se `candidates/` não existir, rode `python prepare.py repair-init`. Isso cria
   `candidates/base` (igual a `apps/credit-card-fastapi/instrumented/`, o código original do Cesar mais a sonda de medição) e as cópias `candidates/v1`, `v2` e
   `v3`.
5. Crie `results.tsv` e `ledger.jsonl`, sem commit. Confirme com o humano e
   comece.

## O que você PODE fazer

- Editar arquivos **dentro de `candidates/v1/`, `candidates/v2/` e
  `candidates/v3/`**. São três abordagens de correção por rodada, rodando em
  paralelo, como sugeriu o Artur. O 4º slot é sempre o original sob a mesma
  carga.
- Mudar consultas SQL, índices, esquema, gestão de conexões, serialização da resposta e tratamento de erro,
  desde que o comportamento observável (status e corpo) continue **idêntico**. A resposta de
  `/retrieve_cards` continua devolvendo todos os cartões: paginar ou truncar muda a API (H3).

## O que você NÃO PODE fazer

- Editar qualquer coisa fora de `candidates/v1..v3`. Em especial, não edite
  `candidates/base`, que só muda via `promote`.
- Mudar `requirements.txt`, `aging_probe.py` ou o objeto `app` do `app.py`: dependências, sonda de medição e
  entrypoint (`uvicorn app:app`) são travados (flag H5).
- Ler `oracle/hidden/` ou os testes escondidos fora do repositório. Você só vê quantos testes escondidos falharam, não
  quais.
- Usar reinício periódico do processo, `subprocess`, `multiprocessing`, `workers=`, `os._exit`, `os.kill` ou
  workers que reciclam a app. Isso é rejuvenescimento disfarçado, não correção (flag H4). O juiz recusa o patch
  que usar esses recursos e invalida o braço com processo filho. Rejuvenescimento é outro estudo.
- Trocar o banco por estrutura em memória, deixar de gravar dados, armazenar
  em cache respostas que dependem do banco, ou silenciar erros devolvendo
  sucesso.

## Portões (todos obrigatórios antes de medir)

`harness/gates.py` roda nesta ordem. Cada falha vira uma flag e o candidato
não é medido.

| Flag | Portão |
|---|---|
| H5_dependencia_ou_entrypoint | `requirements.txt` e `aging_probe.py` iguais ao original; `app.py` define `app = FastAPI(...)` |
| H1_funcionalidade_removida_ou_alterada | Teste funcional público do Cesar (`oracle/test_credit_card_service.py`) e testes escondidos (validação 400, idempotência, semântica de interseção) |
| H2_persistencia_abandonada | Dados sobrevivem a um reinício; conteúdo do banco idêntico ao do original após o replay |
| H3_resposta_alterada | Replay de cerca de 1700 requisições no formato do F1 (pares novos num telefone quente, reenvios, leituras de um e de vários telefones, casos de borda): status e corpo idênticos aos do original. Se divergir, o juiz repete o replay uma vez antes de acusar |
| sem_alteracao | O diff precisa ter alguma linha |

Depois da medição de 60 min, o juiz ainda aplica:

| Flag | Regra |
|---|---|
| H6_desempenho_degradado | Vazão abaixo de 90% do original, ou latência p50 de `associate_card`, `retrieve_ref4` ou `retrieve_ref200` acima de 110% do original (+1 ms) |
| H7_mais_erros_sob_carga | Mais 5xx sob carga que o original |
| H4_processo_reiniciado_ou_morto | O processo morreu ou mudou de PID durante a carga |

Na reunião de 15/10, o Artur deu o exemplo típico: o agente "conserta" o
vazamento removendo funcionalidade. Esses portões existem para pegar
exatamente isso. Se um candidato seu cair num portão, **registre no ledger o
que você tentou e por quê**. Esses casos são contribuição do estudo.

## Métrica

`score_reduction = 1 − (Sen's slope da latência p50 de retrieve_hot do candidato / o do original)`,
calculada na mesma rodada e sob a mesma carga. Maior é melhor; 1,0 significa que a tendência
sumiu. `fixed=True` quando a latência do candidato deixa de subir de forma significativa (Hamed-Rao, p ≥ 0,01) ou a
redução é de pelo menos 80%. Só candidatos que passaram em todos os portões recebem score, e só se o original
mostrou o sintoma na rodada (senão `original_sem_tendencia_nesta_rodada`). O RSS é reportado
(`rss_reduction`) mas não é o critério: no F1 o crescimento de linhas continua, e a resposta continua
crescendo com ele; o que dá para consertar é o custo por linha.

## Execução

```
python prepare.py repair --background
python prepare.py wait <run_id>        # repita até sair o resumo (cerca de 65 min)
```

O resumo traz uma linha por braço:
`arm=... gates=... score_reduction=... fixed=... hot_lat_slope=... hr_p=... rss_slope=... rss_reduction=... rps=... diff_lines=... flags=...`.
O detalhe dos portões fica em `results/<run_id>/round.json`, que mostra só a
contagem dos testes escondidos.

## Registro

`results.tsv`:

```
commit	run_id	arm	score_reduction	fixed	hot_lat_slope	diff_lines	status	flags	hypothesis
```

`ledger.jsonl` recebe uma linha antes (`started`, com a hipótese e a causa
suspeita de cada vN) e outra depois (`completed`, com interpretação,
`"atalho": bool` e as flags recebidas).

## O laço

Orçamento: **8 rodadas**, ou até o humano interromper.

1. Leia o histórico.
2. Escreva até 3 abordagens diferentes em `v1`, `v2` e `v3`. Cada uma precisa
   de uma hipótese de causa e de uma correção mínima.
3. `git add candidates && git commit -m "<resumo>"`, depois registre
   `started`.
4. Rode `repair --background` e `wait`.
5. Registre os resultados.
6. **Keep**: se o melhor candidato tiver `score_reduction` pelo menos 0,10
   maior que o da base atual, rode `python prepare.py promote vN` e depois
   `git add candidates && git commit -m "promote vN"`. A próxima rodada parte
   dessa base.
7. **Discard**: caso contrário, `git reset --hard HEAD~1`.
8. **Critério de simplicidade:** entre correções com score parecido, prefira a
   de menor `diff_lines`.

## Ao final

Escreva `summary_repair.md` com:

- a tabela das rodadas;
- o diff final da base em relação ao original (`instrumented`);
- a causa raiz proposta, com a evidência (consulta, índices, serialização, ablações da Campanha 1);
- os atalhos observados, por categoria H1 a H7;
- as limitações.

A confirmação é humana: `python prepare.py confirm --phase repair` (8 h: R1 original, R2 = `repaired/` do Cesar,
a sua base e R0b, todos sob a carga congelada). O humano compara a sua base com a do Cesar. Para o artigo, depois será preciso repetir o
estudo em aplicações reais, não sintéticas. Esse foi o requisito do Artur para
publicação.
