# 2026-11-23 — Instruções do agente em inglês

**O que mudou:** tudo o que o agente lê ou recebe passou para o inglês, sem mudar regras, limiares, orçamentos nem
o fluxo:

- `program.md` (tradução integral, mesma estrutura e mesmas regras);
- `workload.py` (docstring e hipóteses iniciais);
- os prompts que o orquestrador envia ao agente (`prepare.py experiment`);
- as mensagens do juiz que o agente vê: `verify`, `INVALID` (antes `INVÁLIDO`), `wait` e os erros do contrato
  (`harness/load.py`, `harness/workload_check.py`, `prepare.py`);
- as hipóteses neutras dos braços fixos (referência, R0, R0b).

**Identificadores:** numa segunda etapa, os identificadores produzidos pelo juiz também passaram para o inglês
(por exemplo `D3_out_of_contract`, `D4_copy_of_reference`, `frozen_reference`, decisões `positive` / `negative` /
`invalid`, `no_budget`, campanha `causes` e `.claude/settings.causes.json`), com código, testes e documentos
atualizados juntos. O `prepare.py` teve todos os comentários, docstrings e mensagens traduzidos e os comentários
desnecessários removidos, sem mudar a lógica.

**Campos dos registros do agente** (definidos só no `program.md`, sem código dependente) passaram para o inglês:
`factor_tested`, `interpretation`, `shortcut`, `campaign`, `factor`, `value_ref`, `value_var`, `variations`,
`expected_direction`.

**Documentos para pessoas** (README, `docs/`, `updates/`) continuam em português.

**Verificação:** 22 testes passam; `rehash` + `verify` ok.
