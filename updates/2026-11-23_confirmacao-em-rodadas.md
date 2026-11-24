# 2026-11-23 — Confirmação dentro da campanha (opção 2) e opção 1 em stand-by

**Status:** desenho fechado e **implementado** ([2026-11-23_orquestrador.md](2026-11-23_orquestrador.md)). Texto original: o código e o `program.md` seguem com a confirmação de
2 × 8 h e com etapas humanas no meio.

## Decisão (Rafael, 23/11)

O autoresearch roda **sem intervenção humana no meio**: o humano só dispara o início e lê o relatório final. A
confirmação do aging acontece **dentro da campanha de detecção**, usando as rodadas seguintes (opção 2).

```
Humano dispara
 └─ Fase 0: controles de 60 min (R0, R0b, R1) ── falha ──► relatório: sem calibração
 └─ Campanha de detecção: até 8 rodadas de busca de 60 min (+ verificação até a rodada 10)
      Rodada n: uma carga dá aging
      Rodadas n+1 e n+2: repetem essa carga (verificação: R1, carga, R0, R0b, mesma semente)
        ├─ as 3 rodadas dão aging ──► aging confirmado + carga congelada
        │     └─ a detecção pausa ──► Campanha de causas (variações de um fator contra a carga confirmada)
        │           └─ a detecção retoma nas rodadas que restam (cada nova carga confirmada ganha a sua campanha de causas)
        └─ alguma não dá aging ──► a campanha volta a procurar nas rodadas que restam
      Fim do orçamento de detecção ──► relatório final (todos os agings confirmados, cargas, causas e hipóteses já cobertas)
Humano lê o relatório
```

**Rodadas de verificação** (seguem a regra do Ermeson de 19/11): a mesma carga, com parâmetros fixos; a app
reiniciada com banco limpo; semente nova em cada rodada; nos 4 slots, **R1, a carga, R0 e R0b com a mesma
semente**. A repetição tem a mesma duração da rodada que se confirma (60 min).

**Onde entram as causas:** logo depois da confirmação, como campanha seguinte, disparada pelo orquestrador (não
pelo humano). A referência é a carga confirmada; o resto segue a nota
[2026-11-23_fase4-causas.md](2026-11-23_fase4-causas.md).

**Substitui:** a Fase 0 manual (que passa a ser automática), o `freeze` manual e a confirmação de 2 × 8 h (`confirm --seed 201/202`).

## Pontos a fechar antes de implementar

1. ~~Ao confirmar, a detecção para?~~ **Decidido (Rafael, 23/11):** ao confirmar, a detecção **pausa**, o
   orquestrador roda a campanha de causas para a carga confirmada e, terminada essa campanha, **a detecção retoma**
   nas rodadas que restam do orçamento. O relatório final sai quando o orçamento de detecção acaba.
   - **(a) Decidido:** toda carga confirmada depois da retomada também ganha a sua campanha de causas.
   - **(b) Decidido:** uma carga parecida com uma já confirmada **conta como aging novo** e pode consumir o resto do
     orçamento. Na campanha de causas dela, o agente testa **primeiro as hipóteses que ainda não foram testadas**;
     hipóteses iguais às de uma campanha de causas anterior não são rodadas de novo, só entram no relatório, com
     referência ao resultado já obtido.
2. **Orçamento — decidido (Rafael, 23/11): sim.** A busca usa até 8 rodadas; se o aging aparecer tarde (rodada 7
   ou 8), as 2 rodadas de verificação podem passar das 8, até a rodada 10.
3. **Controles nas rodadas de verificação — decidido (Rafael, 23/11): sim.** Os 4 slots são R1, a carga, R0 e R0b,
   todos com a mesma semente (regra do Ermeson: "R1 e seus controles com a mesma semente"). R0 e R0b precisam ficar
   planos; R1 e a carga, elegíveis.
4. **Fase 0 — decidido (Rafael, 23/11): continua, automática.** O orquestrador roda os controles de 60 min (R0, R0b
   e R1) antes da rodada 1; se R0 ou R0b subirem, ou R1 não subir, para com "sem calibração" antes de gastar o
   orçamento de detecção.

**Todos os pontos estão decididos.** Falta implementar.

## O que precisa mudar quando for implementado

- **Orquestrador** (código fixo, no manifesto): roda a campanha `detect`, detecta a carga elegível, agenda as 2
  rodadas de verificação, decide confirmado / não confirmado, congela a carga e dispara a campanha `causas`.
- **Rodada de verificação no juiz:** braços R1, carga, R0 e R0b; decisão automática (carga e R1 elegíveis, R0 e
  R0b planos, todos válidos).
- **`program.md`:** a detecção passa a ter "até 8 rodadas"; as rodadas de verificação são do orquestrador, não do
  agente.
- **`docs/DESENHO-EXPERIMENTAL.md`, `README.md`** e a figura.

## Opção 1 em stand-by: três campanhas completas

```
Campanha de detecção 1 (até 8 rodadas) ── aging ──► Campanha 2 ──► Campanha 3
  └─ as 3 confirmam ──► Campanha de causas
```

Cada campanha procura do zero. **Não confirma uma carga** (as campanhas podem chegar a cargas diferentes), mas
mede se **o agente redescobre o aging** de forma consistente: é um resultado sobre o método autoresearch, não sobre
a aplicação. Custo: cerca de 24 h só de detecção.

**Quando retomar:** depois de existir um caso confirmado pela opção 2, como estudo seguinte (reprodutibilidade do
agente), por exemplo para o artigo.
