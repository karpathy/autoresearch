# Prováveis ameaças à validade

Documento para as pessoas do projeto (o agente não lê). Lista as ameaças à validade do experimento descrito em
[`DESENHO-EXPERIMENTAL.md`](DESENHO-EXPERIMENTAL.md), na forma usual de estudos empíricos em engenharia de
software (construto, interna, conclusão, externa), com o que já mitiga cada uma e o que ainda falta.

## Validade de construto (medimos aging de fato?)

| Ameaça | Por que importa | O que já mitiga | O que falta |
|---|---|---|---|
| **Janela de 60 min** | Aging é degradação progressiva em execução prolongada; o Cesar mediu 8 h e o artigo, 48 h. Em 60 min a regra de mudança ≥ 50% capta uma tendência de curto prazo, que um revisor pode chamar de degradação dependente de estado | Verificação em 3 rodadas com sementes independentes; controles R0 e R0b planos | Uma execução longa (8 h) da carga confirmada como validação externa, fora do laço |
| **Aging misto no F1** | Parte do crescimento é exigida pela especificação (a resposta traz todos os cartões): é acúmulo legítimo de estado, não defeito | D2 desligada e justificada; custo por MB e interferência medidos na campanha de causas | Separar no texto e nos resultados "acúmulo legítimo" de "aging" (custo por cartão, interferência); hoje o critério de detecção usa latência e tamanho da resposta |
| **Falta de definição operacional citada** | Sem uma definição apoiada na literatura (por exemplo Grottke, Trivedi e colaboradores), "tendência de latência" pode não ser aceita como aging | Critério explícito (4 condições) e alinhado ao do Cesar | Escrever a definição operacional com referências |
| **RSS só como evidência secundária** | A literatura de aging costuma usar memória como indicador principal | RSS medido e registrado em todo braço | Discutir por que a latência é o indicador principal neste finding |

## Validade interna (o efeito vem da carga?)

| Ameaça | O que já mitiga | O que falta |
|---|---|---|
| **Deriva da máquina** (disco, cache, ruído) | Controle interno (C1) e controles negativos na mesma rodada e semente | — |
| **Cliente e app na mesma máquina** | Pinagem em núcleos distintos; CPU do gerador medida (D1); modo cliente-servidor implementado ([`CLIENTE-SERVIDOR.md`](CLIENTE-SERVIDOR.md)) | Rodar de fato em duas máquinas, como o Cesar |
| **Atalhos do agente** (saturar, sair do contrato) | Flags D1–D4, C1, H4; juiz com hash; permissões restritas | — |
| **Viés do agente** (conhecer o mecanismo) | O agente não lê código da app, card do F1, cargas do Cesar nem documentação | Confirmar a restrição com o Cesar |
| **Um fator por vez** na campanha de causas | Duas réplicas por variação | Interações entre fatores não são estudadas |

## Validade de conclusão (a estatística sustenta?)

| Ameaça | O que já mitiga | O que falta |
|---|---|---|
| **Limiares não calibrados** (50%, 25%, 5 MB/h, 20%) | Declarados como provisórios | Calibrar com a Fase 0 e um teste A/A com várias sementes |
| **Múltiplas comparações** (até 24 candidatos com α = 0,01) | Exigência de 3 de 3 rodadas positivas | Justificar ou ajustar no texto |
| **Sem controle positivo** | Controles negativos | Um controle positivo de latência, de taxa conhecida, para medir a sensibilidade |
| **Poucas réplicas** | Sementes independentes, unanimidade | Avaliar a variância entre execuções |

## Validade externa (generaliza?)

| Ameaça | Comentário |
|---|---|
| **Uma aplicação sintética** | Gerada por LLM (BaxBench); o Artur exige apps reais para publicação |
| **Finding já conhecido** | O estudo mede se o agente chega ao aging e o explica, não se descobre aging novo; deixar isso explícito nas perguntas de pesquisa |
| **Uma máquina e um ambiente** | Resultados dependem do hardware (o próprio Cesar viu latência 3× maior que a calibração) |

## Estado da evidência

Nenhuma execução real foi feita nesta branch: o fluxo foi validado com juiz e agente simulados. Todas as
conclusões dependem da primeira execução com a aplicação.
