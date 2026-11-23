"""Checagem funcional da aplicação alvo (fixa; o agente não edita).

Sobe uma cópia limpa da app e roda o teste funcional do Cesar (oracle/test_credit_card_service.py).
Usada só pelo `prepare.py check-oracles`, no preparo do experimento: confirma que a app copiada funciona
antes de qualquer medição. Não há portões de reparo nesta versão (o reparo está fora do escopo).
"""

from .instance import Instance
from oracle import functional_test


def functional_check(app_dir, workdir):
    """Lista de falhas do teste funcional (vazia = passou)."""
    inst = Instance(app_dir, workdir)
    inst.prepare()
    fails = []
    try:
        port = inst.start()
        fails = functional_test.run(port)
    except Exception as e:
        fails.append(f"erro ao executar: {e}")
    finally:
        inst.stop()
    return fails
