"""Ciclo de vida de uma instância da aplicação (fixo; o agente não edita).

Cada instância roda numa cópia limpa do código da app FastAPI do Cesar (apps/credit-card-fastapi),
com banco novo (db.sqlite3 no diretório da cópia), porta própria e a instrumentação do próprio
Cesar (aging_probe.py: AGING_INSTRUMENT=1 e AGING_INSTRUMENT_LOG). A memória e os processos
filhos vêm do psutil, de fora da app.

Diferença para o Cesar: ele sobe `python app.py` na porta fixa 5000 (app.yaml); aqui são 4
instâncias em paralelo, então sobe-se o mesmo `app:app` pelo uvicorn em porta livre, com o mesmo nível de
log (info) e o log de acesso ligado, como o uvicorn.run() do app.py.
"""

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".cache" / "venv" / "bin" / "python"
DB_NAME = "db.sqlite3"  # DB_PATH relativo no app.py: fica no cwd da cópia
SNAPSHOT_S = 10         # intervalo padrão do Cesar para a instrumentação


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def copy_writable(src, dst, ignore=None):
    """copytree que devolve a escrita ao dono. run_campaign.sh faz chmod -R a-w em apps/,
    e copytree copia o modo: sem isto o app falha com EACCES ao criar o banco."""
    shutil.copytree(src, dst, ignore=ignore, copy_function=shutil.copyfile)
    for d, _, files in os.walk(dst):
        os.chmod(d, os.stat(d).st_mode | 0o700)
        for f in files:
            fp = os.path.join(d, f)
            if not os.path.islink(fp):
                os.chmod(fp, os.stat(fp).st_mode | 0o600)
    return Path(dst)


def copy_app(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    copy_writable(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", DB_NAME + "*", ".git"))
    return dst


class Instance:
    def __init__(self, app_src, workdir, sample_s=5.0, instrument=True, cpus=None):
        self.app_src = Path(app_src)
        self.workdir = Path(workdir)
        self.code_dir = self.workdir / "app"
        self.port = None
        self.proc = None
        self.psutil_out = self.workdir / "psutil.csv"
        self.instr_out = self.workdir / "instrumentation.jsonl"
        self.log_out = self.workdir / "app.log"
        self.sample_s = sample_s
        self.instrument = instrument
        self.cpus = cpus  # taskset do servidor (como --app-cpus do Cesar); None = sem pinagem
        self._stop = threading.Event()
        self._sampler = None
        self.pids_seen = set()
        self.max_children = 0  # filhos do servidor: rejuvenescimento escondido (H4)
        self.t0 = None

    def prepare(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        copy_app(self.app_src, self.code_dir)

    def start(self, timeout=30):
        self.port = free_port()
        env = dict(os.environ)
        env.update({"APP_SECRET": "aging-autoresearch", "PYTHONUNBUFFERED": "1"})
        if self.instrument:
            env.update({"AGING_INSTRUMENT": "1", "AGING_INSTRUMENT_LOG": str(self.instr_out),
                        "AGING_SNAPSHOT_INTERVAL": str(SNAPSHOT_S)})
        # Paridade com o Cesar: ele roda `python app.py`, que chama uvicorn.run(app, host=..., port=5000) com os
        # padrões do uvicorn (log_level="info", access_log=True): uma linha de log de acesso por requisição.
        # Aqui o mesmo app:app e os mesmos níveis de log, explícitos; só host e porta mudam (4 instâncias).
        cmd = [str(VENV_PY), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(self.port),
               "--log-level", "info", "--access-log"]
        if self.cpus and shutil.which("taskset"):
            cmd = ["taskset", "-c", str(self.cpus)] + cmd
        log = open(self.log_out, "ab")
        self.proc = subprocess.Popen(cmd, cwd=self.code_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"aplicação encerrou ao iniciar (ver {self.log_out})")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("aplicação não abriu a porta a tempo")
        self.t0 = time.time()
        self.pids_seen.add(self.proc.pid)
        return self.port

    def start_sampler(self):
        self._stop.clear()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()

    def _sample_loop(self):
        with open(self.psutil_out, "w") as f:
            f.write("t,pid,rss,vms,uss,num_fds,num_threads,cpu_percent,children,db_bytes\n")
            try:
                p = psutil.Process(self.proc.pid)
                p.cpu_percent(None)
            except psutil.Error:
                return
            while not self._stop.wait(self.sample_s):
                try:
                    with p.oneshot():
                        mi = p.memory_full_info()
                        ch = len(p.children(recursive=True))
                        self.max_children = max(self.max_children, ch)
                        db = 0
                        for suf in ("", "-wal"):  # o -wal pode sumir entre o exists() e o stat() (checkpoint)
                            try:
                                db += (self.code_dir / (DB_NAME + suf)).stat().st_size
                            except OSError:
                                pass
                        f.write(f"{time.time()-self.t0:.2f},{p.pid},{mi.rss},{mi.vms},{mi.uss},{p.num_fds()},"
                                f"{p.num_threads()},{p.cpu_percent(None)},{ch},{db}\n")
                    f.flush()
                except psutil.Error:
                    f.write(f"{time.time()-self.t0:.2f},DEAD,,,,,,,,\n")
                    return
                except OSError:
                    continue  # amostra perdida, a próxima compensa

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        self._stop.set()
        if self._sampler:
            self._sampler.join(timeout=10)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()  # SIGTERM: o uvicorn encerra limpo e a instrumentação escreve o snapshot final
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def db_state(self, full=False):
        """Conteúdo lógico do banco (sem ids nem timestamps), para comparar versões.
        full=False devolve só as contagens (barato mesmo com centenas de milhares de linhas)."""
        db = self.code_dir / DB_NAME
        if not db.exists():
            return None
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            state = {"tables": sorted(tables),
                     "counts": {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}}
            if full:
                try:
                    state["associations"] = sorted(
                        f"{c}|{p}" for c, p in con.execute("SELECT credit_card, phone FROM associations"))
                except sqlite3.Error:
                    state["associations"] = None
            return state
        finally:
            con.close()

    def probe_rows(self):
        """Amostras do psutil como dicts (t em s desde o início); o que o evaluate lê."""
        rows = []
        if not self.psutil_out.exists():
            return rows
        for line in self.psutil_out.read_text().splitlines()[1:]:
            c = line.split(",")
            if len(c) < 10 or c[1] == "DEAD":
                continue
            rows.append({"t": float(c[0]), "pid": int(c[1]), "rss": int(c[2]), "vms": int(c[3]), "uss": int(c[4]),
                         "handles": int(c[5]), "threads": int(c[6]), "cpu": float(c[7]), "children": int(c[8]),
                         "db_bytes": int(c[9])})
        return rows

    def instrumentation(self):
        """Snapshots da instrumentação do Cesar: {"ts", "metrics"}."""
        out = []
        if self.instr_out.exists():
            for line in self.instr_out.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "snapshot":
                    out.append(rec)
        return out
