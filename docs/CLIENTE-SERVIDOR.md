# Topologia cliente-servidor: como rodar em dois PCs

Documento para as pessoas do projeto (o agente não lê). Explica, passo a passo, como rodar o experimento com a
aplicação num PC (**servidor**) e a carga, o agente e o juiz em outro PC (**cliente**), como o Cesar fez.

Sem configuração nenhuma, tudo continua rodando num PC só, como antes. O modo de dois PCs só é ativado quando a
variável `AGING_SERVER` está definida no cliente.

---

## 1. Visão geral

```
        PC CLIENTE                                   PC SERVIDOR
 ┌──────────────────────────────┐            ┌──────────────────────────────┐
 │ run_experiment.sh            │  porta     │ harness/server_agent.py      │
 │  ├─ orquestrador             │  8100      │  (porta 8100, com token)     │
 │  ├─ agente Claude Code       │ ─────────► │  sobe/para as 4 apps         │
 │  ├─ juiz (valida, avalia,    │  comandos  │  psutil a cada 5 s           │
 │  │   decide)                 │            │  aging_probe a cada 10 s     │
 │  └─ 4 geradores de carga     │ ─────────► │  app 1..4 nas portas         │
 │       (latência medida aqui) │  HTTP da   │  8101, 8102, 8103, 8104      │
 │                              │  carga     │                              │
 └──────────────────────────────┘            └──────────────────────────────┘
```

| PC cliente | PC servidor |
|---|---|
| Orquestrador (`run_experiment.sh`) | Agente do servidor (`harness/server_agent.py`) |
| Agente Claude Code (nunca fala com o servidor) | 4 instâncias da app (cópia limpa, banco vazio, CPU fixa) |
| Juiz: valida a carga, avalia e decide | Amostragem de memória (`psutil`) e instrumentação (`aging_probe`) |
| 4 geradores de carga, que medem a latência | — |
| Precisa do Claude Code logado | Não precisa do Claude Code |

O servidor **nunca recebe código**, só parâmetros: nome do braço, pasta da app dentro de `apps/`, intervalo de
amostragem e slot. Por isso o agente não tem como rodar nada no servidor.

### O que acontece numa rodada
1. O agente (no cliente) escreve a carga e chama o juiz.
2. O juiz valida a carga no cliente e confere se o servidor tem **os mesmos arquivos fixos** (manifesto).
3. O cliente mede a diferença de relógio com o servidor e pede para iniciar os 4 braços.
4. O servidor sobe as 4 apps (portas 8101–8104) e começa a amostrar memória.
5. Os 4 geradores rodam no cliente por 60 min, apontando para `http://IP_DO_SERVIDOR:810x`.
6. O cliente pede para parar. O servidor encerra as apps e devolve memória, instrumentação, banco e PIDs.
7. O cliente mede o relógio de novo e alinha os tempos. Depois o juiz avalia e decide, como no modo local.

---

## 2. Requisitos

**Nos dois PCs**
- Linux ou WSL2 (Windows), Python 3.10+, Git.
- Os dois na **mesma rede local**, de preferência por cabo (Wi-Fi aumenta o ruído da latência).
- O **mesmo commit** deste repositório, na branch `deteccao-f1`.
- O repositório do Cesar clonado ao lado (o `setup` copia a app dele).

**Só no cliente**
- Claude Code instalado e logado.

**Recomendado**
- Pelo menos **4 núcleos** em cada PC: no servidor, uma app por núcleo; no cliente, um gerador por núcleo.
  Com menos, roda, mas sem pinagem de CPU.
- Relógio sincronizado por NTP nos dois. O juiz corrige o offset, mas é bom a deriva ser pequena.

---

## 3. Preparar os dois PCs (repita em cada um)

```bash
git clone https://github.com/Rafaelmos/autoresearch.git autoresearch-deteccao-f1
cd autoresearch-deteccao-f1
git checkout deteccao-f1

git clone https://github.com/chas42/llm-aging-detection-agents ../llm-aging-detection-agents
python3 prepare.py setup --src ../llm-aging-detection-agents
python3 prepare.py check-oracles
```

Confira que os dois estão no mesmo commit e que o juiz está íntegro:

```bash
git rev-parse --short HEAD        # tem que dar o mesmo valor nos dois PCs
python3 prepare.py verify         # "verify: ok ..."
```

> Se algum arquivo do juiz mudar num PC e não no outro, o cliente aborta com
> `server judge differs from the client judge (manifest mismatch)`. Solução: deixar os dois no mesmo commit.

---

## 4. Criar o token (senha compartilhada)

Gere o token **num** dos PCs:

```bash
mkdir -p ~/.aging-hidden
openssl rand -hex 24 > ~/.aging-hidden/server_token
chmod 600 ~/.aging-hidden/server_token
cat ~/.aging-hidden/server_token
```

Copie o mesmo valor para o **outro** PC, no mesmo caminho:

```bash
mkdir -p ~/.aging-hidden
echo "COLE_O_TOKEN_AQUI" > ~/.aging-hidden/server_token
chmod 600 ~/.aging-hidden/server_token
```

O token precisa ter pelo menos 16 caracteres. Uma chamada ao servidor sem ele recebe `401 unauthorized`.
A pasta `~/.aging-hidden` já é bloqueada para o agente Claude.

---

## 5. Descobrir o IP do servidor

No PC servidor:

```bash
hostname -I          # Linux: o primeiro endereço, por exemplo 192.168.0.20
```

**Se o servidor for WSL2 no Windows**, o IP do WSL não é visível na rede. Há duas opções:

- **Opção A (recomendada, Windows 11):** ativar a rede espelhada. Crie ou edite
  `C:\Users\<usuario>\.wslconfig`:
  ```ini
  [wsl2]
  networkingMode=mirrored
  ```
  Depois rode `wsl --shutdown` no PowerShell e abra o WSL de novo. O WSL passa a usar o IP do Windows
  (veja com `ipconfig` no PowerShell).

- **Opção B:** redirecionar as portas do Windows para o WSL. Rode no PowerShell como administrador e repita
  sempre que o IP do WSL mudar, ou seja, a cada reinício:
  ```powershell
  $wsl = (wsl hostname -I).Split(" ")[0]
  foreach ($p in 8100..8104) {
    netsh interface portproxy add v4tov4 listenport=$p listenaddress=0.0.0.0 connectport=$p connectaddress=$wsl
  }
  ```
  O IP a usar no cliente é o do **Windows** (`ipconfig`).

---

## 6. Abrir o firewall no servidor (só para o IP do cliente)

Descubra o IP do cliente com `hostname -I` no cliente. Depois, no servidor:

**Linux com ufw**
```bash
sudo ufw allow from IP_DO_CLIENTE to any port 8100:8104 proto tcp
```

**Windows (servidor em WSL2)**, PowerShell como administrador:
```powershell
New-NetFirewallRule -DisplayName "aging-server" -Direction Inbound -Protocol TCP `
  -LocalPort 8100-8104 -RemoteAddress IP_DO_CLIENTE -Action Allow
```

Não abra essas portas para a internet: a app é de teste e não tem proteção.

---

## 7. Ligar o servidor

No PC servidor, dentro do repositório:

```bash
.cache/venv/bin/python -m harness.server_agent --listen 0.0.0.0:8100 --app-ports 8101-8104
```

Saída esperada:
```
server agent on 0.0.0.0:8100, app ports 8101-8104, workdir /tmp/aging-server-xxxx
```

Opções:
- `--app-ports 8101-8104`: as portas das 4 apps. Precisam ser 4 e estar liberadas no firewall.
- `--workdir DIR`: onde ficam as cópias das apps durante a rodada (padrão: pasta temporária).
- `--token-file CAMINHO`: outro arquivo de token (padrão: `~/.aging-hidden/server_token`).

O servidor **se recusa a subir** se algum arquivo fixo diferir do manifesto.

O experimento completo leva horas. Para o servidor continuar ligado depois de fechar o terminal:
```bash
nohup .cache/venv/bin/python -m harness.server_agent --listen 0.0.0.0:8100 --app-ports 8101-8104 \
  > server_agent.log 2>&1 &
```
Para desligar: `pkill -f harness.server_agent`.

---

## 8. Testar a conexão (no cliente)

```bash
TOKEN=$(cat ~/.aging-hidden/server_token)
curl -s -H "X-Token: $TOKEN" http://IP_DO_SERVIDOR:8100/time
curl -s -H "X-Token: $TOKEN" http://IP_DO_SERVIDOR:8100/manifest | head -c 200; echo
```

Como ler as respostas:
- `{"t": 1764...}` na primeira chamada: conexão e token ok.
- `"ok": true` na segunda: o juiz do servidor está íntegro.
- `401`: o token é diferente nos dois PCs.
- Travou ou deu `Connection refused`: servidor desligado, IP errado, firewall ou rede do WSL (seções 5 a 7).

---

## 9. Rodar o experimento (no cliente)

Primeiro um teste rápido, com rodadas de 2 min. Ele valida o fluxo, mas não mostra tendência:

```bash
export AGING_SERVER=IP_DO_SERVIDOR:8100
./run_experiment.sh teste-2pcs --smoke
```

Se tudo correr bem, o experimento completo:

```bash
export AGING_SERVER=IP_DO_SERVIDOR:8100
nohup ./run_experiment.sh exp-2pcs > exp-2pcs.log 2>&1 &
```

A variável `AGING_SERVER` precisa estar definida **no mesmo terminal** que dispara o experimento. Sem ela, o
experimento roda todo no cliente, em modo local.

---

## 10. Como conferir que rodou em dois PCs

Em cada braço, o arquivo `results/<rodada>/<braço>/evaluation.json` traz:
- `"topology": "client-server"` (no modo local, `"local"`);
- `"clock"`, com o offset e o RTT medidos antes e depois do braço.

O offset e o RTT também ficam em `results/<rodada>/<braço>/clock.json`. Numa rede local, o esperado é um RTT
de poucos milissegundos e um offset estável entre o "antes" e o "depois".

---

## 11. Relógio: como os tempos são alinhados

A memória é medida no servidor, com o relógio dele, e a latência no cliente, com o relógio do cliente. Para
juntar as duas séries:

1. Antes de cada braço, o cliente faz 7 consultas a `/time` e usa a de **menor RTT**:
   `offset = t_servidor − (t_envio + t_retorno) / 2`.
2. Depois do braço, mede de novo e usa a média dos dois offsets.
3. Desloca as amostras do servidor para ficarem relativas ao início da carga no cliente, como no modo local.
4. Se o RTT passar de **200 ms**, o braço é recusado: a rede está ruim demais para medir latência.

O `evaluate.py` não mudou: recebe as amostras já alinhadas.

---

## 12. Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `server judge differs from the client judge` | Commits diferentes ou arquivo do juiz alterado | Mesmo commit nos dois; `python3 prepare.py verify` em cada um |
| `401 unauthorized` | Token diferente nos dois PCs | Copiar o mesmo `server_token` para os dois |
| `Connection refused` ou timeout | Servidor desligado, IP, firewall ou rede do WSL | Seções 5 a 8 |
| `RTT to server too high` | Rede lenta (Wi-Fi, VPN) | Usar cabo e a mesma rede local |
| `no free application port` | Braços de uma rodada interrompida ainda ativos | Reiniciar o `server_agent` |
| `arm X already running` | Idem | Reiniciar o `server_agent` |
| Braço inválido com `error:` | A app não subiu no servidor | Ver o `server_agent.log` e o `app.log` na pasta `--workdir` |
| Servidor não sobe: `token too short` | Token com menos de 16 caracteres | Gerar de novo (seção 4) |

---

## 13. Segurança

- O agente Claude roda só no cliente e não consegue ler o token, porque a pasta `~/.aging-hidden` é bloqueada.
  Ele também não pode usar `curl`, `wget`, `nc`, `env` nem `printenv`.
- O servidor só aceita pastas de app dentro de `apps/` e nomes de braço simples. Nunca executa código vindo do
  cliente.
- As portas devem ficar abertas só para o IP do cliente.

---

## 14. O que mudou no código (referência)

- `harness/server_agent.py` (novo, no manifesto): inicia, consulta e para braços. Devolve `psutil.csv`,
  `instrumentation.jsonl`, contagens do banco, PIDs vistos, filhos e se o processo estava vivo (H4).
- `harness/remote.py` (novo, no manifesto): `RemoteInstance` (mesma interface de `Instance`) e a medição do relógio.
- `harness/instance.py`: aceita `host` e `port`. O servidor sobe as apps em `0.0.0.0` e em portas fixas.
- `prepare.py` (`run_arms`): escolhe local ou remoto por `AGING_SERVER`. No remoto, o gerador i fica no núcleo i
  do cliente e a app i no núcleo i do servidor.
- `.claude/settings.*.json`: o agente também não pode usar `env`, `printenv` nem `nc`.
- `tests/test_remote.py`: testa token, pasta fora de `apps/`, ciclo completo com relógio adiantado em 1000 s e
  portas esgotadas.

## Estado

Validado só com testes (HTTP real em localhost, app simulada). Falta a primeira execução real em dois PCs;
comece pelo `--smoke` (seção 9).
