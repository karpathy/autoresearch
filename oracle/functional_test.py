"""Oráculo funcional público: o teste do repositório do Cesar, sem alteração
(oracle/test_credit_card_service.py, cópia de reference/tests/test_credit_card_service.py).

O agente pode ler estes arquivos. Ele não pode editá-los. O teste do Cesar fala com
http://localhost:<port>; o juiz passa a porta da instância.
"""

import importlib.util
import logging
from pathlib import Path

_TEST = Path(__file__).resolve().parent / "test_credit_card_service.py"


class _App:
    def __init__(self, port):
        self.port = port
        self.logger = logging.getLogger("functional")


def run(port):
    """Lista de falhas (vazia = passou), no mesmo formato dos testes escondidos."""
    spec = importlib.util.spec_from_file_location("cesar_functional", _TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fails = []
    for name in sorted(n for n in dir(mod) if n.startswith("func_test")):
        try:
            if not getattr(mod, name)(_App(port)):
                fails.append(f"{name} falhou")
        except Exception as e:
            fails.append(f"{name}: {type(e).__name__}: {e}")
    return fails
