"""Teste A/A: três cópias da referência (args = defaults do Cesar) medem o ruído do juiz.

Uso (humano, uma vez, 60 min): cp workload.py workload_campanha.py && cp workload_aa.py workload.py
  python prepare.py detect && cp workload_campanha.py workload.py
As cópias são marcadas D4 e não pontuam; a comparação entre elas e a referência é o ruído.
"""

CANDIDATES = [
    {"name": "aa1", "hypothesis": "Cópia da referência (ruído).", "args": {}},
    {"name": "aa2", "hypothesis": "Cópia da referência (ruído).", "args": {}},
    {"name": "aa3", "hypothesis": "Cópia da referência (ruído).", "args": {}},
]
