"""Carregamento seguro de workload.py (fixo).

workload.py é escrito pelo agente e lido pelo juiz. Na versão em Node ele era código executado
dentro do processo do juiz (por isso a lista branca de AST e o exec com builtins restritos).
Agora ele só DECLARA parâmetros: o código que gera tráfego é o script do Cesar, fixo e hasheado
no manifesto. Por isso a trava ficou mais forte e mais simples:

1. AST com lista branca: no topo só docstring e atribuições `NOME = <literal>`; nenhum import,
   função, classe, chamada, atributo ou nome. O valor é lido com ast.literal_eval: nada executa.
2. Semente: o agente não pode definir seed, base_url, duration, out nem timeout (load.normalize_args).
3. Determinismo: a carga é função só de (args, semente do juiz). Os cartões do script vêm de
   random.Random(seed); não há outra fonte de aleatoriedade nem estado escondido no workload.py.
4. Limites: taxas até 2x os defaults, total nominal <= 31 req/s (load.normalize_args).
"""

import ast

from .load import WorkloadError

ALLOWED_VALUE_NODES = (ast.Constant, ast.Dict, ast.List, ast.Tuple, ast.UnaryOp, ast.USub, ast.UAdd, ast.Load)


def check_ast(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise WorkloadError(f"workload.py não compila: {e}") from None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue  # docstring
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            raise WorkloadError(f"workload.py só aceita docstring e `NOME = <literal>` no topo "
                                f"(linha {node.lineno}: {type(node).__name__})")
        for sub in ast.walk(node.value):
            if not isinstance(sub, ALLOWED_VALUE_NODES):
                raise WorkloadError(f"só literais em workload.py (linha {sub.lineno}: {type(sub).__name__})")
    return tree


def load_workload(path):
    """Lê workload.py (ou a carga congelada) como dados. Retorna {NOME: valor}."""
    src = path.read_text()
    tree = check_ast(src)
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                ns[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as e:
                raise WorkloadError(f"valor inválido em workload.py (linha {node.lineno}): {e}") from None
    return ns
