"""Candidate workloads for the round. Format, args and ranges: program.md, section 3."""

CANDIDATES = [
    {
        "name": "pares_20_por_s",
        "hypothesis": "Changing new_pair_rate changes the retrieve_hot latency trend.",
        "args": {"new_pair_rate": 20.0},
    },
    {
        "name": "dois_telefones_quentes",
        "hypothesis": "Changing hot_phones changes the retrieve_hot latency trend.",
        "args": {"hot_phones": 2},
    },
    {
        "name": "sonda_a_cada_1s",
        "hypothesis": "Changing probe_interval changes the retrieve_hot latency trend.",
        "args": {"probe_interval": 1.0},
    },
]
