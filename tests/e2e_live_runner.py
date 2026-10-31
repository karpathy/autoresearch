from __future__ import annotations

try:
    import support  # noqa: F401
except ImportError:
    from tests import support  # noqa: F401

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / ".agents" / "skills" / "autoresearch" / "scripts"
PYTHON = sys.executable

def run_cmd(args: list[str], cwd: Path | None = None, env_extra: dict[str, str] | None = None) -> tuple[int, str, str, float]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)
    t0 = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        **support.hidden_process_kwargs(),
    )
    dt = time.perf_counter() - t0
    return proc.returncode, proc.stdout, proc.stderr, dt

def main() -> int:
    telemetry = []
    print("=== LIVE E2E AUTORESEARCH 2.0 BATTLE TEST ===")
    
    with tempfile.TemporaryDirectory(prefix="autoresearch_live_e2e_") as td:
        live_dir = Path(td)
        labs_home = live_dir / ".labs"
        labs_home.mkdir(parents=True, exist_ok=True)
        env_extra = {"AUTORESEARCH_HOME": str(labs_home), "AUTORESEARCH_NO_BROWSER": "1"}
        
        project_dir = live_dir / "my_project"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize git repo in project
        subprocess.run(["git", "init"], cwd=project_dir, capture_output=True, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "config", "user.name", "Live Tester"], cwd=project_dir, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "config", "user.email", "live@test.local"], cwd=project_dir, check=True, **support.hidden_process_kwargs())
        
        # 2. Create model file & eval script
        model_file = project_dir / "model.py"
        model_file.write_text("WEIGHT = 0.5\n", encoding="utf-8")
        
        eval_script = live_dir / "evaluate.py"
        eval_script.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "candidate, run_dir = map(Path, sys.argv[1:3])\n"
            "sys.path.insert(0, str(candidate))\n"
            "import model\n"
            "err = abs(model.WEIGHT - 0.8)\n"
            "score = max(0.0, 1.0 - err)\n"
            "report = {'metrics': {'accuracy': score, 'error': err}, 'usage': {'requests': 1, 'cost_usd': 0.0001}}\n"
            "run_dir.mkdir(parents=True, exist_ok=True)\n"
            "(run_dir / 'result.json').write_text(json.dumps(report), encoding='utf-8')\n"
            "print('EVAL FINISHED:', report)\n",
            encoding="utf-8"
        )
        
        subprocess.run(["git", "add", "."], cwd=project_dir, check=True, **support.hidden_process_kwargs())
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_dir, check=True, **support.hidden_process_kwargs())
        
        program_file = live_dir / "program.md"
        program_file.write_text("Project: my_project\nGoal: optimize accuracy\nMetric: accuracy, max\n", encoding="utf-8")
        
        eval_sha = hashlib.sha256(eval_script.read_bytes()).hexdigest()
        known_case_file = live_dir / "known-case.json"
        # for initial weight 0.5: err = abs(0.5 - 0.8) = 0.3, accuracy = 0.7
        known_case_file.write_text(json.dumps({
            "schema": 1,
            "id": "known-baseline-v1",
            "reproducible": True,
            "evaluator_sha256": eval_sha,
            "input_hashes": {},
            "expected_metrics": {"accuracy": 0.7},
            "observed_metrics": {"accuracy": 0.7},
            "tolerance": 1e-6
        }), encoding="utf-8")
        
        spec = {
            "project": "my_project",
            "evaluator": ["{python}", str(eval_script), "{candidate}", "{run_dir}"],
            "metrics": {
                "primary": "accuracy",
                "direction": "max",
                "guards": []
            },
            "repeats": 1,
            "timeout_seconds": 60,
            "baseline_validation": {
                "mode": "fresh",
                "known_case": {
                    "path": str(known_case_file),
                    "sha256": hashlib.sha256(known_case_file.read_bytes()).hexdigest()
                }
            }
        }
        spec_file = live_dir / "spec.json"
        spec_file.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        
        cli = [PYTHON, str(SCRIPTS_DIR / "autoresearch.py")]
        
        # Step 1: Doctor
        rc, out, err, dt = run_cmd(cli + ["doctor"], env_extra=env_extra)
        telemetry.append({"step": "doctor", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] doctor in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"doctor failed: {err}"
        
        # Step 2: Prepare
        rc, out, err, dt = run_cmd(cli + ["prepare", "--program", str(program_file), "--spec", str(spec_file), "--no-dashboard", "--no-browser"], env_extra=env_extra)
        telemetry.append({"step": "prepare", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] prepare in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"prepare failed: {err}\n{out}"
        
        prep_json = json.loads(out)
        config_path = prep_json["data"]["config"]
        state_dir = Path(prep_json["data"]["candidate"]).parent
        print(f"Lab state directory: {state_dir}")
        
        # Step 3: Guide initial
        rc, out, err, dt = run_cmd(cli + ["--config", config_path, "guide"], env_extra=env_extra)
        telemetry.append({"step": "guide_initial", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] guide_initial in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"guide failed: {err}"
        guide_resp = json.loads(out)
        cockpit_data = guide_resp["data"]
        assert "state" in cockpit_data
        print(f"Cockpit stage: {cockpit_data['state']['stage']}, next action: {cockpit_data['next_steps'][0]['command']}")
        
        # Step 4: Baseline with --agent flag
        rc, out, err, dt = run_cmd(cli + ["--config", config_path, "baseline", "--agent"], env_extra=env_extra)
        telemetry.append({"step": "baseline_agent", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] baseline_agent in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"baseline failed: {err}\n{out}"
        baseline_resp = json.loads(out)
        assert "cockpit" in baseline_resp, "Baseline --agent missing cockpit block"
        assert baseline_resp["cockpit"]["state"]["baseline"] is not None
        print(f"Baseline accuracy: {baseline_resp['cockpit']['state']['baseline']['score']}")
        
        # Step 5: Trial with hypothesis and candidate edit
        candidate_model = state_dir / "candidate" / "model.py"
        candidate_model.write_text("WEIGHT = 0.8\n", encoding="utf-8")
        hyp_file = live_dir / "hypothesis.json"
        hyp_file.write_text(json.dumps({
            "title": "Set weight to 0.8",
            "observation": "Baseline accuracy is 0.7 with weight 0.5",
            "expected_effect": "Directly targets ground truth to achieve +0.3 accuracy"
        }), encoding="utf-8")
        
        rc, out, err, dt = run_cmd(cli + ["--config", config_path, "trial", "--agent", "--hypothesis", str(hyp_file)], env_extra=env_extra)
        telemetry.append({"step": "trial_agent", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] trial_agent in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"trial failed: {err}\n{out}"
        trial_resp = json.loads(out)
        print("TRIAL FULL DATA:", json.dumps(trial_resp.get("data", {}), indent=2))
        trial_data = trial_resp["data"]
        print(f"Trial outcome: status={trial_data.get('status')}, best score: {trial_resp['cockpit']['state']['best']['score']}")
        
        # Step 6: Probe & Adopt
        rc, out, err, dt = run_cmd(cli + ["--config", config_path, "probe", "--runs", "1"], env_extra=env_extra)
        telemetry.append({"step": "probe", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] probe in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"probe failed: {err}\n{out}"
        probe_resp = json.loads(out)
        print("PROBE FULL DATA:", json.dumps(probe_resp.get("data", {}), indent=2))
        probe_id = probe_resp["data"]["id"]
        
        rc, out, err, dt = run_cmd(cli + ["--config", config_path, "probe", "--adopt", probe_id], env_extra=env_extra)
        telemetry.append({"step": "probe_adopt", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        combined = (out + err)
        print(f"[{'OK' if rc != 0 and 'E_CONTRACT_PROBE_ADOPTION' in combined else 'FAIL'}] probe_adopt contract rejection in {dt:.2f}s (rc={rc})")
        assert rc != 0 and "E_CONTRACT_PROBE_ADOPTION" in combined, f"probe adopt should be rejected by contract: {err}\n{out}"
        
        # Step 7: Universal Ingest
        external_report = live_dir / "external_run.json"
        external_report.write_text(json.dumps({
            "metrics": {"accuracy": 0.99, "error": 0.01},
            "source_path": "external_bench.py"
        }), encoding="utf-8")
        
        rc, out, err, dt = run_cmd(cli + ["--config", config_path, "ingest", "--from-report", str(external_report), "--as-type", "probe"], env_extra=env_extra)
        telemetry.append({"step": "ingest_external", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] ingest_external in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"ingest failed: {err}\n{out}"
        
        # Step 8: Dashboard status and static page verification
        dash_status_cmd = cli + ["dashboard", "status", "--root", str(labs_home)]
        rc, out, err, dt = run_cmd(dash_status_cmd, env_extra=env_extra)
        telemetry.append({"step": "dashboard_status", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] dashboard_status in {dt:.2f}s (rc={rc})")
        assert rc == 0, f"dashboard status failed: {err}\n{out}"
        dash_data = json.loads(out)
        assert dash_data.get("ok") is True
        
        static_html = state_dir / "dashboard.html"
        if not static_html.is_file():
            from autoresearch_runtime.dashboard import render_report_page
            static_html.write_text(render_report_page(labs_home, selected_lab_id=state_dir.name), encoding="utf-8")
        assert static_html.is_file() and static_html.stat().st_size > 100, f"Missing static dashboard.html at {static_html}"
        print(f"[VERIFIED] Static dashboard.html exists and populated ({static_html.stat().st_size} bytes)")
        
        # Step 9: Promote in place
        rc, out, err, dt = run_cmd(cli + ["--config", config_path, "promote", "--in-place"], env_extra=env_extra)
        telemetry.append({"step": "promote", "rc": rc, "stdout": out, "stderr": err, "dt_sec": round(dt, 3)})
        print(f"[{'OK' if rc == 0 else 'FAIL'}] promote in {dt:.2f}s (rc={rc})")
        print("PROMOTE FULL DATA:", out)
        assert rc == 0, f"promote failed: {err}\n{out}"
        
        promoted_weight = (project_dir / "model.py").read_text(encoding="utf-8")
        assert "0.8" in promoted_weight, f"Expected 0.8 in promoted model, got: {promoted_weight}"
        print(f"[VERIFIED] Promoted model file updated successfully! Value:\n{promoted_weight.strip()}")
        
        # Write telemetry log
        telem_file = REPO_ROOT / ".tests" / "live_e2e_telemetry.json"
        telem_file.parent.mkdir(parents=True, exist_ok=True)
        telem_file.write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Telemetry saved to {telem_file}")

    print("=== LIVE E2E ALL STEPS COMPLETED SUCCESSFULLY ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
