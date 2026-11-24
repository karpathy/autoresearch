# Topologia cliente-servidor

Documento para as pessoas do projeto (o agente não lê). Sem configuração, tudo continua rodando numa máquina,
como antes. Com `AGING_SERVER` definido, a aplicação roda numa segunda máquina.

## Quem fica em cada máquina

| Máquina cliente | Máquina servidor |
|---|---|
| Orquestrador (`run_experiment.sh` / `prepare.py experiment`) | Agente do servidor (`harness/server_agent.py`) |
| Agente Claude Code (nunca fala com o servidor) | 4 instâncias da app (uvicorn, cópia limpa, banco vazio, CPU fixa) |
| Juiz: contrato, manifesto, avaliação, decisões | Amostragem `psutil` (5 s) e `aging_probe` (10 s) |
| 4 geradores de carga → `http://IP_DO_SERVIDOR:porta` (latência medida aqui) | |

O servidor recebe só parâmetros (rótulo do braço, diretório da app dentro de `apps/`, intervalo de amostragem,
slot). Nunca recebe código do agente.

## Configuração

### Nas duas máquinas
1. Mesmo commit da branch `deteccao-f1` e `python prepare.py setup` (cria o venv e o manifesto igual).
2. O mesmo token (≥ 16 caracteres) em `~/.aging-hidden/server_token`, com `chmod 600`. Por exemplo:
   `mkdir -p ~/.aging-hidden && openssl rand -hex 24 > ~/.aging-hidden/server_token && chmod 600 ~/.aging-hidden/server_token`
   (gere numa máquina e copie para a outra).
3. Relógios sincronizados por NTP (o juiz ainda mede e corrige o offset, mas NTP mantém a deriva pequena).

### No servidor
```
.cache/venv/bin/python -m harness.server_agent --listen 0.0.0.0:8100 --app-ports 8101-8104
```
- Firewall: abrir 8100 (agente) e 8101–8104 (apps) **só para o IP do cliente**.
- O agente se recusa a subir se os arquivos fixos diferirem do manifesto.
- Opcional: `--workdir DIR` (padrão: diretório temporário) e `--token-file`.

### No cliente
```
export AGING_SERVER=IP_DO_SERVIDOR:8100
./run_experiment.sh <tag>          # ou --smoke primeiro
```
Antes de cada rodada o juiz confere se o manifesto do servidor é igual ao do cliente (fora a carga congelada,
que só existe no cliente) e aborta se não for.

## Relógio

Antes e depois de cada braço, o cliente faz 7 consultas a `/time` e usa a de menor RTT (estilo NTP):
`offset = t_servidor − (t_envio + t_retorno)/2`. As amostras do servidor (relativas ao t0 do servidor) são
deslocadas pela média dos dois offsets para ficar relativas ao t0 do cliente, como no modo local. O braço é
recusado se o RTT passar de 200 ms. Offset e RTT ficam em `results/<run>/<braço>/clock.json` e no campo
`clock` do `evaluation.json` (que também registra `topology`).

## O que mudou no código

- `harness/server_agent.py` (novo, entra no manifesto): iniciar, consultar e parar braços; devolve
  `psutil.csv`, `instrumentation.jsonl`, contagens do banco, PIDs vistos, filhos e se o processo estava vivo (H4).
- `harness/remote.py` (novo): `RemoteInstance`, com a mesma interface de `Instance`, e a medição do relógio.
- `harness/instance.py`: aceita `host` e `port` (o servidor sobe as apps em `0.0.0.0` em portas fixas).
- `prepare.py` (`run_arms`): escolhe local ou remoto por `AGING_SERVER`; no remoto, o gerador i fica no
  núcleo i do cliente e a app i no núcleo i do servidor.
- `.claude/settings.*.json`: o agente também não pode usar `env`, `printenv` nem `nc`.
- `harness/evaluate.py` não mudou: o deslocamento do relógio é aplicado antes, nas amostras.
- `tests/test_remote.py`: token, caminho fora de `apps/`, ciclo completo com relógio adiantado 1000 s,
  portas esgotadas.

## Estado

Validado só com testes (HTTP real em localhost, app simulada). Falta a primeira execução real em duas
máquinas; sugestão: começar com `--smoke`.
