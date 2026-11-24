# 2026-11-23 — Instruções do agente em inglês

**O que mudou:** tudo o que o agente lê ou recebe passou para o inglês, sem mudar regras, limiares, orçamentos nem
o fluxo:

- `program.md` (tradução integral, mesma estrutura e mesmas regras);
- `workload.py` (docstring e hipóteses iniciais);
- os prompts que o orquestrador envia ao agente (`prepare.py experiment`);
- as mensagens do juiz que o agente vê: `verify`, `INVALID` (antes `INVÁLIDO`), `wait` e os erros do contrato
  (`harness/load.py`, `harness/workload_check.py`, `prepare.py`);
- as hipóteses neutras dos braços fixos (referência, R0, R0b).

**O que ficou igual, de propósito:** os identificadores produzidos pelo juiz (flags como `D1_saturacao_ou_erros`,
`C1_referencia_interna_subindo`, status `referencia_sem_sintoma`, braço `referencia_congelada`, campos do
`round.json`). Mudar esses nomes alteraria código, testes e dados já gravados; o `program.md` avisa que são
identificadores e devem ser usados como estão.

**Campos dos registros do agente** (definidos só no `program.md`, sem código dependente) passaram para o inglês:
`factor_tested`, `interpretation`, `shortcut`, `campaign`, `factor`, `value_ref`, `value_var`, `variations`,
`expected_direction`.

**Documentos para pessoas** (README, `docs/`, `updates/`) continuam em português.

**Verificação:** 22 testes passam; `rehash` + `verify` ok.
