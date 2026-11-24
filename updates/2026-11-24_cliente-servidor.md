# 2026-11-24 — Topologia cliente-servidor

**O que mudou:** a app pode rodar numa segunda máquina, com a carga, o agente e o juiz no cliente (como o
Cesar). Ativado por `AGING_SERVER`; sem a variável, nada muda. Detalhes em `docs/CLIENTE-SERVIDOR.md`.

**Rigor preservado:** critérios, limiares, sementes e orquestração iguais; `evaluate.py` intacto; o servidor
só recebe parâmetros; manifesto conferido nas duas pontas; relógio medido antes e depois de cada braço.

**Verificação:** 26 testes passam (4 novos em `tests/test_remote.py`); manifesto atualizado (24 arquivos).
Sem execução real em duas máquinas ainda.
