"""Carga experimental — ÚNICO arquivo que o agente edita na campanha de DETECÇÃO.

Equivale ao train.py do karpathy/autoresearch. A referência (R1 do Cesar: F1_targeted_v1 com os
parâmetros padrão) roda sempre no 4º slot e não é declarada aqui.

O alvo é o finding bax-fastapi-credit-card-F1 (ver findings/ e workloads/*.md): cada
POST /associate_card com par (cartão, telefone) novo grava uma linha que ninguém apaga, e
POST /retrieve_cards lê todas as linhas do telefone sem LIMIT. O sintoma é a latência e o
tamanho da resposta de retrieve_hot crescerem com o tempo.

Você NÃO escreve código: cada candidato é uma variante do F1_targeted_v1 do Cesar, descrita por `args`
(valores literais). O juiz roda o script fixo com a semente dele. Contrato (harness/load.py):
- CANDIDATES: 1 a 3 dicts com name, hypothesis, args.
- args (todos opcionais; defaults do Cesar entre parênteses):
    new_pair_rate  pares novos por segundo, todos no(s) telefone(s) quente(s)   (10, máx. 20)
    hot_phones     telefones quentes, em rodízio                                (1, máx. 2)
    grow_workers   trabalhadores do fluxo grow                                  (2, máx. 4)
    refresh_rate   reenvios de pares já associados por segundo                  (2, máx. 4)
    ref_sizes      telefones de referência com N cartões constantes, tupla      ((4, 200))
    ref_rate       associações de referência por segundo                        (2, máx. 4)
    probe_interval segundos entre leituras de cada telefone sondado             (2, mín. 1)
    key_space      0 = cartões sempre novos; N > 0 = só N cartões (nada acumula) (0)
- Taxa nominal total (tudo somado) até 31 req/s. seed, base_url, duration, out e timeout são do juiz.
- 5xx/exceções acima de 1% invalidam o braço (saturação não é aging).
- Pontua o braço cuja latência de retrieve_hot sobe (Mann-Kendall + Sen) com os controles internos
  retrieve_ref4/retrieve_ref200 planos. Crescimento de linhas NÃO é atalho aqui: é o próprio finding
  (flag D2 desligada para o F1, ver README).

Rodada inicial: variações de taxa e de concentração para ver o que acelera o sintoma em 60 min.
"""

CANDIDATES = [
    {
        "name": "pares_20_por_s",
        "hypothesis": "Dobrar a taxa de pares novos (o 'primeiro botão' do Cesar) acelera o crescimento do telefone quente.",
        "args": {"new_pair_rate": 20.0},
    },
    {
        "name": "dois_telefones_quentes",
        "hypothesis": "Dois telefones quentes dividem as inserções: cada um cresce metade, então a latência de cada um sobe mais devagar.",
        "args": {"hot_phones": 2},
    },
    {
        "name": "sonda_a_cada_1s",
        "hypothesis": "Sondar o telefone quente mais vezes dá mais amostras da latência, sem mudar o crescimento.",
        "args": {"probe_interval": 1.0},
    },
]


FROZEN_ARM = 'F1_targeted_v1_referencia'
FROZEN_FROM = 'detect-smoke-1119-0725-da23'
