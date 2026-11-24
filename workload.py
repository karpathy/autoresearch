"""Cargas candidatas da rodada. Formato, args e faixas: program.md, seção 3."""

CANDIDATES = [
    {
        "name": "pares_20_por_s",
        "hypothesis": "Variar new_pair_rate muda a tendência da latência de retrieve_hot.",
        "args": {"new_pair_rate": 20.0},
    },
    {
        "name": "dois_telefones_quentes",
        "hypothesis": "Variar hot_phones muda a tendência da latência de retrieve_hot.",
        "args": {"hot_phones": 2},
    },
    {
        "name": "sonda_a_cada_1s",
        "hypothesis": "Variar probe_interval muda a tendência da latência de retrieve_hot.",
        "args": {"probe_interval": 1.0},
    },
]
