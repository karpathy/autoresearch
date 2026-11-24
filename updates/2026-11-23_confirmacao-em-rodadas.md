# 2026-11-23 — Confirmação dentro da campanha (opção 2) e opção 1 em stand-by

**Status:** decisão de desenho. **Ainda não implementada**: o código e o `program.md` seguem com a confirmação de
2 × 8 h e com etapas humanas no meio.

## Decisão (Rafael, 23/11)

O autoresearch roda **sem intervenção humana no meio**: o humano só dispara o início e lê o relatório final. A
confirmação do aging acontece **dentro da campanha de detecção**, usando as rodadas seguintes (opção 2).

```
Humano dispara
 └─ Campanha de detecção: até 8 rodadas de 60 min
      Rodada n: uma carga dá aging
      Rodadas n+1 e n+2: repetem essa carga (verificação)
        ├─ as 3 rodadas dão aging ──► aging confirmado + carga congelada
        │     └─ Campanha de causas: variações de um fator contra a carga confirmada ──► relatório final
        └─ alguma não dá aging ──► a campanha volta a procurar nas rodadas que restam
      Fim do orçamento sem confirmar ──► relatório: aging não confirmado
Humano lê o relatório
```

**Rodadas de verificação** (seguem a regra do Ermeson de 19/11): a mesma carga, com parâmetros fixos; a app
reiniciada com banco limpo; semente nova em cada rodada; nos 4 slots, **R1, a carga, R0 e R0b com a mesma
semente**. A repetição tem a mesma duração da rodada que se confirma (60 min).

**Onde entram as causas:** logo depois da confirmação, como campanha seguinte, disparada pelo orquestrador (não
pelo humano). A referência é a carga confirmada; o resto segue a nota
[2026-11-23_fase4-causas.md](2026-11-23_fase4-causas.md).

**Substitui:** a Fase 0 manual, o `freeze` manual e a confirmação de 2 × 8 h (`confirm --seed 201/202`).

## Pontos a fechar antes de implementar

1. **Ao confirmar, a detecção para** e passa para as causas, ou continua até as 8 rodadas procurando uma carga
   mais forte?
2. **Orçamento:** se o aging aparecer tarde (rodada 7 ou 8), permitir que as 2 rodadas de verificação passem das 8
   (até 10). Recomendado.
3. **Controles R0 e R0b** nas rodadas de verificação: recomendado (é o que cumpre "R1 e seus controles com a
   mesma semente").
4. **Fase 0 (controles de 60 min) automática** no início, antes da primeira rodada, ou dispensada porque R0 e R0b
   já entram na verificação.

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
