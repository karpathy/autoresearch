#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / "workflows"
RUNS_DIR = WORKFLOWS_DIR / "runs"
RESULTS_TSV = REPO_ROOT / "results.tsv"
RUN_LOG = REPO_ROOT / "run.log"
VAL_RE = re.compile(r"^val_bpb:\s*([0-9]+\.[0-9]+)", re.MULTILINE)
VRAM_RE = re.compile(r"^peak_vram_mb:\s*([0-9]+\.[0-9]+)", re.MULTILINE)

TOP_STAGES = ["setup", "baseline", "loop"]
LOOP_STAGES = ["propose", "apply", "commit", "train", "triage", "record", "decide"]


def sh(cmd: list[str], check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=str(cwd or REPO_ROOT))


def now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_dirs() -> None:
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def branch_slug(branch: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", branch).strip("-").lower()


def current_branch() -> str:
    return sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def short_head() -> str:
    return sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def pick_base_branch() -> str:
    for candidate in ("master", "main"):
        if sh(["git", "show-ref", "--verify", f"refs/heads/{candidate}"], check=False).returncode == 0:
            return candidate
    return current_branch()


def branch_exists(branch: str) -> bool:
    return sh(["git", "show-ref", "--verify", f"refs/heads/{branch}"], check=False).returncode == 0


def suggest_tag() -> str:
    return dt.datetime.utcnow().strftime("%b%d").lower().replace("0", "")


def ensure_autoresearch_branch(branch_arg: str | None) -> str:
    if branch_arg:
        if not branch_exists(branch_arg):
            base = pick_base_branch()
            sh(["git", "checkout", base])
            sh(["git", "checkout", "-b", branch_arg])
        else:
            sh(["git", "checkout", branch_arg])
        return branch_arg

    cur = current_branch()
    if cur.startswith("autoresearch/"):
        return cur

    base = pick_base_branch()
    tag = suggest_tag()
    branch = f"autoresearch/{tag}"
    i = 1
    while branch_exists(branch):
        i += 1
        branch = f"autoresearch/{tag}-{i}"
    sh(["git", "checkout", base])
    sh(["git", "checkout", "-b", branch])
    return branch


def next_run_id(branch: str) -> str:
    slug = branch_slug(branch)
    prefix = f"{slug}-r"
    max_n = 0
    for p in RUNS_DIR.glob(f"{prefix}*"):
        m = re.match(rf"^{re.escape(prefix)}(\d{{3}})$", p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{prefix}{max_n + 1:03d}"


def latest_run_id_for_branch(branch: str) -> str | None:
    slug = branch_slug(branch)
    prefix = f"{slug}-r"
    candidates: list[tuple[int, str]] = []
    for p in RUNS_DIR.glob(f"{prefix}*"):
        m = re.match(rf"^{re.escape(prefix)}(\d{{3}})$", p.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), p.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def run_paths(run_id: str) -> dict[str, Path]:
    run_dir = RUNS_DIR / run_id
    return {
        "run_dir": run_dir,
        "state": run_dir / "state.json",
        "history": run_dir / "history.jsonl",
        "runner_log": run_dir / "runner.log",
    }


def load_state(run_id: str) -> dict[str, Any]:
    p = run_paths(run_id)["state"]
    if not p.exists():
        raise RuntimeError(f"state not found for run_id={run_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    p = run_paths(state["run_id"])["state"]
    p.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    p.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def log_event(run_id: str, event: dict[str, Any]) -> None:
    p = run_paths(run_id)["history"]
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": now_iso(), **event}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    append_runner_log(run_id, "EVENT", json.dumps(event, sort_keys=True))


def append_runner_log(run_id: str, level: str, message: str) -> None:
    p = run_paths(run_id)["runner_log"]
    p.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_iso()}] [{level}] {message}"
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def parse_stage_list(text: str | None, allowed: list[str]) -> list[str] | None:
    if not text:
        return None
    out = []
    for part in text.split(","):
        s = part.strip().lower()
        if not s:
            continue
        if s not in allowed:
            raise RuntimeError(f"invalid stage '{s}'. allowed={allowed}")
        out.append(s)
    return out or None


def compute_top_selection(only: list[str] | None, from_stage: str | None, to_stage: str | None) -> list[str]:
    if only:
        selected = [s for s in TOP_STAGES if s in only]
        if any(s in LOOP_STAGES for s in only) and "loop" not in selected:
            selected.append("loop")
        return selected
    if from_stage or to_stage:
        a = TOP_STAGES.index(from_stage) if from_stage else 0
        b = TOP_STAGES.index(to_stage) if to_stage else len(TOP_STAGES) - 1
        if a > b:
            raise RuntimeError("from-stage must be <= to-stage")
        return TOP_STAGES[a : b + 1]
    return TOP_STAGES[:]


def ensure_results_header() -> None:
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")


def parse_log_metrics(log_text: str) -> tuple[float | None, float | None]:
    val_m = VAL_RE.search(log_text)
    vram_m = VRAM_RE.search(log_text)
    return (float(val_m.group(1)) if val_m else None, float(vram_m.group(1)) if vram_m else None)


def parse_run_result_from_log(log_path: Path, return_code: int | None, timed_out: bool) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    val, vram = parse_log_metrics(text)
    status = "timeout" if timed_out else ("success" if val is not None else "crash")
    return {
        "status": status,
        "return_code": return_code,
        "val_bpb": val,
        "peak_vram_mb": vram,
        "memory_gb": round((vram or 0.0) / 1024.0, 1),
        "log_tail": "\n".join(text.splitlines()[-50:]),
        "log_path": str(log_path),
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
        time.sleep(1.0)
        if _pid_alive(pid):
            os.killpg(pid, signal.SIGKILL)
    except Exception:
        pass


def start_training_job(run_id: str, job_name: str, log_path: Path, timeout_seconds: int = 600) -> dict[str, Any]:
    run_dir = run_paths(run_id)["run_dir"]
    rc_file = run_dir / f"{job_name}.rc"
    cmd = (
        f"cd {json.dumps(str(REPO_ROOT))} && "
        f"uv run train.py > {json.dumps(str(log_path))} 2>&1; "
        f"echo $? > {json.dumps(str(rc_file))}"
    )
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    started = now_iso()
    deadline = (dt.datetime.utcnow() + dt.timedelta(seconds=timeout_seconds)).isoformat(timespec="seconds") + "Z"
    return {
        "job_name": job_name,
        "pid": proc.pid,
        "log_path": str(log_path),
        "rc_file": str(rc_file),
        "started_at": started,
        "deadline_at": deadline,
        "timeout_seconds": timeout_seconds,
    }


def poll_training_job(job: dict[str, Any]) -> dict[str, Any]:
    pid = int(job["pid"])
    rc_file = Path(str(job["rc_file"]))
    log_path = Path(str(job["log_path"]))
    deadline = dt.datetime.fromisoformat(str(job["deadline_at"]).replace("Z", ""))

    if dt.datetime.utcnow() > deadline and _pid_alive(pid):
        _kill_process_group(pid)
        result = parse_run_result_from_log(log_path, return_code=None, timed_out=True)
        result["training_seconds"] = float(job.get("timeout_seconds", 600))
        return {"done": True, "result": result}

    if rc_file.exists():
        rc_text = rc_file.read_text(encoding="utf-8", errors="replace").strip()
        rc = int(rc_text) if rc_text and rc_text.lstrip("-").isdigit() else None
        result = parse_run_result_from_log(log_path, return_code=rc, timed_out=False)
        try:
            started = dt.datetime.fromisoformat(str(job["started_at"]).replace("Z", ""))
            result["training_seconds"] = round((dt.datetime.utcnow() - started).total_seconds(), 1)
        except Exception:
            result["training_seconds"] = None
        return {"done": True, "result": result}

    if _pid_alive(pid):
        return {"done": False, "state": "running"}

    result = parse_run_result_from_log(log_path, return_code=None, timed_out=False)
    result["training_seconds"] = None
    return {"done": True, "result": result}


def run_training(timeout_seconds: int = 600) -> dict[str, Any]:
    started = dt.datetime.utcnow()
    with RUN_LOG.open("w", encoding="utf-8") as f:
        try:
            proc = subprocess.run(["uv", "run", "train.py"], cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=timeout_seconds)
            rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = None
            timed_out = True
    text = RUN_LOG.read_text(encoding="utf-8", errors="replace") if RUN_LOG.exists() else ""
    val, vram = parse_log_metrics(text)
    status = "timeout" if timed_out else ("success" if val is not None else "crash")
    elapsed = (dt.datetime.utcnow() - started).total_seconds()
    return {
        "status": status,
        "return_code": rc,
        "val_bpb": val,
        "peak_vram_mb": vram,
        "memory_gb": round((vram or 0.0) / 1024.0, 1),
        "training_seconds": round(elapsed, 1),
        "log_tail": "\n".join(text.splitlines()[-50:]),
        "log_path": str(RUN_LOG),
    }


def read_results_rows() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []
    with RESULTS_TSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_results_rows(rows: list[dict[str, str]]) -> None:
    with RESULTS_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["commit", "val_bpb", "memory_gb", "status", "description"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_result_row(commit: str, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
    with RESULTS_TSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow([commit, f"{val_bpb:.6f}", f"{memory_gb:.1f}", status, description[:200]])


def update_last_result_status(new_status: str) -> None:
    rows = read_results_rows()
    if not rows:
        return
    rows[-1]["status"] = new_status
    write_results_rows(rows)


def extract_text_events(stream: str) -> str:
    texts: list[str] = []
    for line in stream.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "text":
            part = obj.get("part", {})
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return "\n".join(texts).strip()


def extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[:-1])
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        s = min([p for p in [cleaned.find("{"), cleaned.find("[")] if p != -1], default=-1)
        e = max([p for p in [cleaned.rfind("}"), cleaned.rfind("]")] if p != -1], default=-1)
        if s == -1 or e == -1 or e < s:
            raise
        return json.loads(cleaned[s : e + 1])


def run_stochastic_json(prompt: str, trace_file: Path | None = None) -> dict[str, Any]:
    cmd = ["opencode", "run", "--format", "json", prompt]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            json.dumps(
                {
                    "command": cmd,
                    "return_code": proc.returncode,
                    "stderr": proc.stderr,
                    "stdout": proc.stdout,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise RuntimeError(f"opencode failed: {proc.stderr.strip()}")
    text = extract_text_events(proc.stdout)
    if not text:
        raise RuntimeError("no text response found in opencode event stream")
    payload = extract_json_payload(text)
    if not isinstance(payload, dict):
        raise RuntimeError("stochastic payload must be a JSON object")
    return payload


def default_proposal(state: dict[str, Any], iteration: int) -> dict[str, Any]:
    return {
        "status": "ok",
        "description": f"iteration {iteration}: tune optimization hyperparameters conservatively",
        "change_plan": "Make one small, safe training-hyperparameter adjustment in train.py.",
        "commit_description": f"experiment {iteration}: small optimizer/hyperparameter tweak",
    }


def run_setup(state: dict[str, Any], auto_prepare: bool) -> None:
    append_runner_log(state["run_id"], "INFO", "stage=setup start")
    in_scope = ["README.md", "prepare.py", "train.py", "program.md", "pyproject.toml"]
    missing = [f for f in in_scope if not (REPO_ROOT / f).exists()]
    data_dir = Path.home() / ".cache" / "autoresearch" / "data"
    tok_path = Path.home() / ".cache" / "autoresearch" / "tokenizer" / "tokenizer.pkl"
    ensure_results_header()
    state["setup_done"] = True
    data_ready = data_dir.exists() and any(data_dir.glob("*.parquet"))
    tokenizer_ready = tok_path.exists()

    if auto_prepare and (not data_ready or not tokenizer_ready):
        append_runner_log(
            state["run_id"],
            "INFO",
            "stage=setup auto_prepare start command='uv run prepare.py'",
        )
        prepare_log = run_paths(state["run_id"])["run_dir"] / "prepare.log"
        with prepare_log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                ["uv", "run", "prepare.py"],
                cwd=str(REPO_ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        append_runner_log(
            state["run_id"],
            "INFO",
            f"stage=setup auto_prepare done return_code={proc.returncode} log={prepare_log}",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"auto-prepare failed with return code {proc.returncode}. See {prepare_log}")
        data_ready = data_dir.exists() and any(data_dir.glob("*.parquet"))
        tokenizer_ready = tok_path.exists()

    state["setup"] = {
        "missing_files": missing,
        "data_ready": data_ready,
        "tokenizer_ready": tokenizer_ready,
    }
    state["setup_ready"] = (
        len(missing) == 0
        and bool(state["setup"]["data_ready"])
        and bool(state["setup"]["tokenizer_ready"])
    )
    log_event(state["run_id"], {"type": "stage", "stage": "setup", "ok": True, "details": state["setup"]})
    append_runner_log(
        state["run_id"],
        "INFO",
        f"stage=setup done missing_files={len(missing)} data_ready={state['setup']['data_ready']} tokenizer_ready={state['setup']['tokenizer_ready']}",
    )


def ensure_setup_ready(state: dict[str, Any]) -> None:
    setup = state.get("setup") or {}
    missing = setup.get("missing_files") or []
    data_ready = bool(setup.get("data_ready"))
    tokenizer_ready = bool(setup.get("tokenizer_ready"))
    if missing or not data_ready or not tokenizer_ready:
        msg = (
            "setup preconditions not met: "
            f"missing_files={missing}, data_ready={data_ready}, tokenizer_ready={tokenizer_ready}. "
            "Run 'uv run prepare.py' in this environment, then resume."
        )
        append_runner_log(state["run_id"], "ERROR", msg)
        raise RuntimeError(msg)


def run_baseline(state: dict[str, Any]) -> None:
    append_runner_log(state["run_id"], "INFO", "stage=baseline start training")
    result = run_training(timeout_seconds=600)
    commit = short_head()
    if result["status"] == "success":
        append_result_row(commit, float(result["val_bpb"]), float(result["memory_gb"]), "keep", "baseline")
        state["best_val_bpb"] = float(result["val_bpb"])
        state["kept_commit"] = commit
    else:
        append_result_row(commit, 0.0, 0.0, "crash", "baseline")
    state["baseline_done"] = True
    state["baseline"] = result
    log_event(state["run_id"], {"type": "stage", "stage": "baseline", "ok": True, "details": result})
    append_runner_log(
        state["run_id"],
        "INFO",
        f"stage=baseline done status={result['status']} val_bpb={result['val_bpb']} peak_vram_mb={result['peak_vram_mb']} training_seconds={result['training_seconds']}",
    )


def run_baseline_maybe_background(state: dict[str, Any], background_train: bool) -> bool:
    run_id = state["run_id"]
    if state.get("baseline_done"):
        return True

    existing_job = state.get("baseline_job")
    if existing_job:
        poll = poll_training_job(existing_job)
        if not poll.get("done"):
            append_runner_log(run_id, "INFO", "stage=baseline waiting training_in_progress=true")
            return False
        result = poll["result"]
        commit = short_head()
        if result["status"] == "success":
            append_result_row(commit, float(result["val_bpb"]), float(result["memory_gb"]), "keep", "baseline")
            state["best_val_bpb"] = float(result["val_bpb"])
            state["kept_commit"] = commit
        else:
            append_result_row(commit, 0.0, 0.0, "crash", "baseline")
        state["baseline_done"] = True
        state["baseline"] = result
        state["baseline_job"] = None
        log_event(run_id, {"type": "stage", "stage": "baseline", "ok": True, "details": result})
        append_runner_log(
            run_id,
            "INFO",
            f"stage=baseline done status={result['status']} val_bpb={result['val_bpb']} peak_vram_mb={result['peak_vram_mb']} training_seconds={result.get('training_seconds')}",
        )
        save_state(state)
        return True

    if not background_train:
        run_baseline(state)
        return True

    append_runner_log(run_id, "INFO", "stage=baseline start background_training=true")
    job = start_training_job(run_id, "baseline_train", RUN_LOG, timeout_seconds=600)
    state["baseline_job"] = job
    save_state(state)
    log_event(run_id, {"type": "stage", "stage": "baseline", "ok": True, "details": {"started": True, "background": True, "pid": job["pid"]}})
    append_runner_log(run_id, "INFO", f"stage=baseline background_started pid={job['pid']} log_path={job['log_path']}")
    return False


def run_loop_iteration(
    state: dict[str, Any],
    iteration: int,
    loop_stage_subset: list[str],
    stochastic: bool,
    background_train: bool,
) -> bool:
    run_id = state["run_id"]
    run_dir = run_paths(run_id)["run_dir"]
    iter_dir = run_dir / "iterations" / f"{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    append_runner_log(
        run_id,
        "INFO",
        f"loop iteration={iteration} start loop_stage_subset={','.join(loop_stage_subset)} stochastic={stochastic}",
    )

    in_progress = state.get("in_progress")
    if in_progress and int(in_progress.get("iteration", -1)) == iteration:
        iter_state = in_progress
    else:
        iter_state = {
            "iteration": iteration,
            "base_commit": short_head(),
            "stages_done": [],
            "proposal": None,
            "apply": None,
            "candidate_commit": None,
            "train": None,
            "triage": None,
            "recorded": False,
            "decision": None,
        }
        state["in_progress"] = iter_state
        save_state(state)

    def mark_done(stage: str) -> None:
        if stage not in iter_state["stages_done"]:
            iter_state["stages_done"].append(stage)
        (iter_dir / "iteration_state.json").write_text(json.dumps(iter_state, indent=2, sort_keys=True), encoding="utf-8")
        save_state(state)

    for stage in LOOP_STAGES:
        if stage not in loop_stage_subset:
            continue
        if stage in iter_state["stages_done"]:
            append_runner_log(run_id, "INFO", f"loop iteration={iteration} stage={stage} skip reason=already_done")
            continue

        append_runner_log(run_id, "INFO", f"loop iteration={iteration} stage={stage} start")

        if stage == "propose":
            if stochastic:
                tail = ""
                if RESULTS_TSV.exists():
                    tail = "\n".join(RESULTS_TSV.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                prompt = (
                    "You are running autoresearch experiments. Propose ONE next experiment for train.py only. "
                    "Keep it simple unless clear gain is expected. Return JSON object with keys: "
                    "status, description, change_plan, commit_description. status should be ok or need_input.\n\n"
                    f"Current best val_bpb: {state.get('best_val_bpb')}\n"
                    f"Recent results:\n{tail}\n"
                )
                proposal = run_stochastic_json(prompt, trace_file=iter_dir / "propose_opencode.json")
            else:
                proposal = default_proposal(state, iteration)
            iter_state["proposal"] = proposal
            log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "propose", "proposal": proposal})
            append_runner_log(
                run_id,
                "INFO",
                f"loop iteration={iteration} stage=propose done status={proposal.get('status')} description={str(proposal.get('description', ''))[:120]}",
            )
            mark_done(stage)

        elif stage == "apply":
            proposal = iter_state.get("proposal") or default_proposal(state, iteration)
            if stochastic and proposal.get("status") == "ok":
                prompt = (
                    "Apply this experiment to train.py ONLY. Make direct code edits in the repo. "
                    "Do not touch prepare.py or dependencies. Return JSON object with keys: status, summary. "
                    "status must be applied or failed.\n\n"
                    f"Experiment proposal:\n{json.dumps(proposal, indent=2)}"
                )
                apply_res = run_stochastic_json(prompt, trace_file=iter_dir / "apply_opencode.json")
            else:
                apply_res = {"status": "applied", "summary": "manual/default proposal path"}
            iter_state["apply"] = apply_res
            log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "apply", "apply": apply_res})
            append_runner_log(
                run_id,
                "INFO",
                f"loop iteration={iteration} stage=apply done status={apply_res.get('status')} summary={str(apply_res.get('summary', ''))[:120]}",
            )
            mark_done(stage)

        elif stage == "commit":
            sh(["git", "add", "train.py"], check=False)
            changed = sh(["git", "diff", "--cached", "--quiet"], check=False).returncode != 0
            if changed:
                subject = (iter_state.get("proposal") or {}).get("commit_description") or f"experiment {iteration}"
                sh(["git", "commit", "-m", str(subject)[:120]])
            iter_state["candidate_commit"] = short_head()
            log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "commit", "commit": iter_state["candidate_commit"], "changed": changed})
            append_runner_log(
                run_id,
                "INFO",
                f"loop iteration={iteration} stage=commit done commit={iter_state['candidate_commit']} changed={changed}",
            )
            mark_done(stage)

        elif stage == "train":
            active_job = iter_state.get("train_job")
            if active_job:
                poll = poll_training_job(active_job)
                if not poll.get("done"):
                    append_runner_log(run_id, "INFO", f"loop iteration={iteration} stage=train waiting pid={active_job.get('pid')}")
                    save_state(state)
                    return False
                train_res = poll["result"]
                iter_state["train"] = train_res
                iter_state["train_job"] = None
                log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "train", "train": train_res})
                append_runner_log(
                    run_id,
                    "INFO",
                    f"loop iteration={iteration} stage=train done status={train_res['status']} val_bpb={train_res['val_bpb']} peak_vram_mb={train_res['peak_vram_mb']} training_seconds={train_res.get('training_seconds')}",
                )
                mark_done(stage)
            elif background_train:
                job = start_training_job(run_id, f"iter_{iteration:04d}_train", RUN_LOG, timeout_seconds=600)
                iter_state["train_job"] = job
                log_event(
                    run_id,
                    {
                        "type": "loop",
                        "iteration": iteration,
                        "stage": "train",
                        "train": {"started": True, "background": True, "pid": job["pid"]},
                    },
                )
                append_runner_log(
                    run_id,
                    "INFO",
                    f"loop iteration={iteration} stage=train background_started pid={job['pid']} log_path={job['log_path']}",
                )
                save_state(state)
                return False
            else:
                train_res = run_training(timeout_seconds=600)
                iter_state["train"] = train_res
                log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "train", "train": train_res})
                append_runner_log(
                    run_id,
                    "INFO",
                    f"loop iteration={iteration} stage=train done status={train_res['status']} val_bpb={train_res['val_bpb']} peak_vram_mb={train_res['peak_vram_mb']} training_seconds={train_res['training_seconds']}",
                )
                mark_done(stage)

        elif stage == "triage":
            train_res = iter_state.get("train") or {"status": "crash", "log_tail": ""}
            if train_res.get("status") == "success":
                triage = {"action": "proceed_no_crash", "reason": "metrics found"}
            elif stochastic:
                prompt = (
                    "Candidate training run did not succeed. Choose action as one of: proceed_no_crash, fix_and_rerun, mark_crash_and_discard. "
                    "Return JSON object with keys: action, reason.\n\n"
                    f"Run status: {train_res.get('status')}\n"
                    f"Log tail:\n{train_res.get('log_tail', '')}"
                )
                triage = run_stochastic_json(prompt, trace_file=iter_dir / "triage_opencode.json")
            else:
                triage = {"action": "mark_crash_and_discard", "reason": "non-success run"}
            iter_state["triage"] = triage
            log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "triage", "triage": triage})
            append_runner_log(
                run_id,
                "INFO",
                f"loop iteration={iteration} stage=triage done action={triage.get('action')} reason={str(triage.get('reason', ''))[:120]}",
            )
            mark_done(stage)

        elif stage == "record":
            train_res = iter_state.get("train") or {"status": "crash", "val_bpb": None, "memory_gb": 0.0}
            commit = iter_state.get("candidate_commit") or short_head()
            desc = ((iter_state.get("proposal") or {}).get("description") or "experiment")[:200]
            if train_res.get("status") == "success":
                append_result_row(commit, float(train_res["val_bpb"]), float(train_res["memory_gb"]), "pending", desc)
            else:
                append_result_row(commit, 0.0, 0.0, "crash", desc)
            iter_state["recorded"] = True
            log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "record", "status": train_res.get("status")})
            append_runner_log(
                run_id,
                "INFO",
                f"loop iteration={iteration} stage=record done status={train_res.get('status')} commit={commit}",
            )
            mark_done(stage)

        elif stage == "decide":
            train_res = iter_state.get("train") or {"status": "crash"}
            base_commit = iter_state.get("base_commit")
            decision = "discard"
            if train_res.get("status") == "success":
                val = float(train_res["val_bpb"])
                best = state.get("best_val_bpb")
                if best is None or val < float(best):
                    decision = "keep"
                    state["best_val_bpb"] = val
                    state["kept_commit"] = iter_state.get("candidate_commit")
                else:
                    decision = "discard"
            else:
                decision = "crash"

            if decision == "keep":
                update_last_result_status("keep")
            elif decision == "discard":
                update_last_result_status("discard")
                if base_commit:
                    sh(["git", "reset", "--hard", base_commit])
            else:
                update_last_result_status("crash")
                if base_commit:
                    sh(["git", "reset", "--hard", base_commit])

            iter_state["decision"] = decision
            log_event(run_id, {"type": "loop", "iteration": iteration, "stage": "decide", "decision": decision})
            append_runner_log(
                run_id,
                "INFO",
                f"loop iteration={iteration} stage=decide done decision={decision} best_val_bpb={state.get('best_val_bpb')} kept_commit={state.get('kept_commit')}",
            )
            mark_done(stage)

    iter_state["completed"] = True
    state["iterations_completed"] = max(int(state.get("iterations_completed", 0)), iteration)
    state["in_progress"] = None
    save_state(state)
    append_runner_log(run_id, "INFO", f"loop iteration={iteration} complete")
    return True


def run_selected(
    state: dict[str, Any],
    top_selection: list[str],
    loop_count: int,
    loop_only: list[str],
    stochastic: bool,
    auto_prepare: bool,
    background_train: bool,
) -> None:
    append_runner_log(
        state["run_id"],
        "INFO",
        f"run_selected top={','.join(top_selection)} loops={loop_count} loop_only={','.join(loop_only)} stochastic={stochastic}",
    )
    if "setup" in top_selection and not state.get("setup_done"):
        run_setup(state, auto_prepare=auto_prepare)
        save_state(state)
    elif "setup" in top_selection:
        append_runner_log(state["run_id"], "INFO", "stage=setup skip reason=already_done")

    if ("baseline" in top_selection or "loop" in top_selection) and not state.get("setup_done"):
        append_runner_log(state["run_id"], "INFO", "stage=setup auto-run reason=required_for_baseline_or_loop")
        run_setup(state, auto_prepare=auto_prepare)
        save_state(state)

    if "baseline" in top_selection or "loop" in top_selection:
        ensure_setup_ready(state)

    if "baseline" in top_selection and not state.get("baseline_done"):
        done = run_baseline_maybe_background(state, background_train=background_train)
        if not done:
            append_runner_log(state["run_id"], "INFO", "stage=baseline pending background completion")
            return
        save_state(state)
    elif "baseline" in top_selection:
        append_runner_log(state["run_id"], "INFO", "stage=baseline skip reason=already_done")

    if "loop" in top_selection and loop_count > 0:
        start_it = int(state.get("iterations_completed", 0)) + 1
        if state.get("in_progress"):
            start_it = int(state["in_progress"]["iteration"])
            append_runner_log(state["run_id"], "INFO", f"resuming partial iteration={start_it}")
        completed = 0
        current = start_it
        while completed < loop_count:
            finished = run_loop_iteration(state, current, loop_only, stochastic, background_train)
            if not finished:
                append_runner_log(state["run_id"], "INFO", f"loop iteration={current} pending background completion")
                return
            completed += 1
            current = int(state.get("iterations_completed", 0)) + 1
    elif "loop" in top_selection:
        append_runner_log(state["run_id"], "INFO", "stage=loop skip reason=loops=0")


def create_initial_state(run_id: str, branch: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "branch": branch,
        "repo_root": str(REPO_ROOT),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "setup_done": False,
        "baseline_done": False,
        "baseline_job": None,
        "iterations_completed": 0,
        "best_val_bpb": None,
        "kept_commit": None,
        "in_progress": None,
        "status": "running",
    }


def print_status(state: dict[str, Any]) -> None:
    paths = run_paths(state["run_id"])
    out = {
        "run_id": state["run_id"],
        "branch": state["branch"],
        "setup_done": state.get("setup_done"),
        "baseline_done": state.get("baseline_done"),
        "iterations_completed": state.get("iterations_completed"),
        "best_val_bpb": state.get("best_val_bpb"),
        "kept_commit": state.get("kept_commit"),
        "in_progress": state.get("in_progress"),
        "baseline_job": state.get("baseline_job"),
        "state_path": str(paths["state"]),
        "history_path": str(paths["history"]),
        "runner_log_path": str(paths["runner_log"]),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autoresearch loop runner (no DAG dependency).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run-id", help="Run id. If omitted, auto-generated for start or latest for resume.")
        sp.add_argument("--loops", type=int, default=0, help="Number of loop iterations to execute.")
        sp.add_argument("--only", help="Comma list: setup,baseline,loop or loop sub-stages.")
        sp.add_argument("--from-stage", choices=TOP_STAGES)
        sp.add_argument("--to-stage", choices=TOP_STAGES)
        sp.add_argument("--loop-only", help="Comma list of loop stages: propose,apply,commit,train,triage,record,decide")
        sp.add_argument("--no-stochastic", action="store_true", help="Disable opencode stochastic stages and use deterministic fallbacks.")
        sp.add_argument(
            "--auto-prepare",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run 'uv run prepare.py' during setup when cache/tokenizer are missing (default: true).",
        )
        sp.add_argument(
            "--background-train",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run train stage in background and resume/poll later (default: true).",
        )

    s = sub.add_parser("start", help="Start a new run")
    s.add_argument("--branch", help="Optional branch name (e.g. autoresearch/mar10).")
    add_common(s)

    r = sub.add_parser("resume", help="Resume an existing run")
    add_common(r)

    st = sub.add_parser("status", help="Show status of a run")
    st.add_argument("--run-id", help="Run id. If omitted, latest for current branch.")

    return p


def resolve_resume_run_id(given: str | None) -> str:
    if given:
        return given
    branch = current_branch()
    rid = latest_run_id_for_branch(branch)
    if not rid:
        raise RuntimeError("no run found for current branch; pass --run-id")
    return rid


def main() -> int:
    ensure_dirs()
    args = build_parser().parse_args()

    if args.cmd == "status":
        run_id = args.run_id or resolve_resume_run_id(None)
        state = load_state(run_id)
        append_runner_log(run_id, "INFO", "command=status")
        print_status(state)
        return 0

    only = parse_stage_list(args.only, TOP_STAGES + LOOP_STAGES)
    loop_only = parse_stage_list(args.loop_only, LOOP_STAGES) or LOOP_STAGES[:]
    top_selection = compute_top_selection(only, args.from_stage, args.to_stage)
    stochastic = not args.no_stochastic
    auto_prepare = bool(args.auto_prepare)
    background_train = bool(args.background_train)

    if args.cmd == "start":
        branch = ensure_autoresearch_branch(args.branch)
        run_id = args.run_id or next_run_id(branch)
        paths = run_paths(run_id)
        if paths["state"].exists():
            raise RuntimeError(f"run_id already exists: {run_id}")
        state = create_initial_state(run_id, branch)
        save_state(state)
        append_runner_log(run_id, "INFO", f"command=start branch={branch} run_id={run_id}")
        append_runner_log(
            run_id,
            "INFO",
            f"config only={args.only} from_stage={args.from_stage} to_stage={args.to_stage} loop_only={args.loop_only} loops={args.loops} stochastic={stochastic}",
        )
        append_runner_log(run_id, "INFO", f"config auto_prepare={auto_prepare} background_train={background_train}")
        log_event(run_id, {"type": "run", "action": "start", "branch": branch})
        run_selected(state, top_selection, max(0, args.loops), loop_only, stochastic, auto_prepare, background_train)
        save_state(state)
        append_runner_log(run_id, "INFO", "command=start complete")
        print_status(state)
        return 0

    if args.cmd == "resume":
        run_id = resolve_resume_run_id(args.run_id)
        state = load_state(run_id)
        append_runner_log(run_id, "INFO", f"command=resume run_id={run_id}")
        append_runner_log(
            run_id,
            "INFO",
            f"config only={args.only} from_stage={args.from_stage} to_stage={args.to_stage} loop_only={args.loop_only} loops={args.loops} stochastic={stochastic}",
        )
        append_runner_log(run_id, "INFO", f"config auto_prepare={auto_prepare} background_train={background_train}")
        log_event(run_id, {"type": "run", "action": "resume"})
        run_selected(state, top_selection, max(0, args.loops), loop_only, stochastic, auto_prepare, background_train)
        save_state(state)
        append_runner_log(run_id, "INFO", "command=resume complete")
        print_status(state)
        return 0

    raise RuntimeError("unsupported command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
