"""Experiment runner for the crew system.

Executes individual GPU experiments:
1. Modifies train.py with test parameters
2. Runs training and captures metrics
3. Records results to TSV and git
4. Always restores train.py from backup
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import subprocess
import shutil
import json
import ast
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentParams:
    """What to test in this experiment."""
    parameters: dict  # {"learning_rate": 0.005, "warmup_steps": 100, ...}
    rationale: str    # Why we're testing this
    index: int        # Experiment number within task


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    index: int                              # Experiment number
    parameters: dict                        # What was tested
    success: bool                           # Did it complete?
    metric_value: Optional[float]           # val_bpb (None if failed)
    train_loss: Optional[float]             # Final training loss
    peak_memory_gb: Optional[float]         # Peak GPU memory used
    training_seconds: Optional[float]       # Wall-clock training time
    commit_hash: Optional[str]              # Git commit if created
    error: Optional[str]                    # Error message if failed
    log_path: Path                          # Path to run.log
    timestamp: datetime                     # When experiment ran


class ExperimentRunner:
    """Executes GPU experiments with train.py modifications."""

    def __init__(self, config, results_dir: Path = None):
        """Initialize runner.

        Args:
            config: System configuration object
            results_dir: Directory for experiment results (default: data/experiments)
        """
        self.config = config
        self.train_file = Path(getattr(config, 'train_file', 'train.py'))
        self.results_dir = results_dir or Path(getattr(config, 'results_dir', 'data/experiments'))
        self.results_tsv = self.results_dir / "results.tsv"
        self.experiment_counter = self._load_counter()

        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(self, params: ExperimentParams, task_id: int) -> ExperimentResult:
        """Execute one experiment.

        Returns result regardless of success or failure.

        Args:
            params: ExperimentParams with parameters and rationale
            task_id: Which task this experiment belongs to

        Returns: ExperimentResult with all details
        """
        exp_num = self.experiment_counter
        self.experiment_counter += 1
        exp_dir = self.results_dir / f"exp_{exp_num:04d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc)

        # 1. BACKUP
        backup_path = self.train_file.with_suffix('.py.backup')
        if not self.train_file.exists():
            return self._failed_result(
                exp_num, params, timestamp, exp_dir,
                f"train.py not found at {self.train_file}"
            )

        shutil.copy2(self.train_file, backup_path)
        original_content = self.train_file.read_text()

        try:
            # 2. MODIFY
            modified_content = self._apply_modifications(
                original_content, params.parameters
            )

            # Validate syntax
            if not self._validate_python(modified_content):
                return self._failed_result(
                    exp_num, params, timestamp, exp_dir,
                    "Syntax error in modified train.py"
                )

            self.train_file.write_text(modified_content)
            (exp_dir / "train.py").write_text(modified_content)

            # 3. RUN
            log_path = exp_dir / "run.log"
            result = self._execute_training(log_path)

            if not result["success"]:
                return self._failed_result(
                    exp_num, params, timestamp, exp_dir,
                    result.get("error", "Training failed")
                )

            # 4. PARSE
            metrics = self._parse_metrics(log_path)

            if metrics is None:
                return self._failed_result(
                    exp_num, params, timestamp, exp_dir,
                    "Could not parse metrics from run.log"
                )

            # 5. RECORD
            commit_hash = self._git_commit(exp_num, params.parameters)

            # Save metrics
            (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            # Append to results.tsv
            self._append_results_tsv(
                exp_num, task_id, params, metrics, commit_hash
            )

            return ExperimentResult(
                index=exp_num,
                parameters=params.parameters,
                success=True,
                metric_value=metrics.get("val_bpb"),
                train_loss=metrics.get("train_loss"),
                peak_memory_gb=metrics.get("peak_memory_gb"),
                training_seconds=metrics.get("training_seconds"),
                commit_hash=commit_hash,
                error=None,
                log_path=log_path,
                timestamp=timestamp,
            )

        finally:
            # 6. RESTORE (always, even on crash)
            if backup_path.exists():
                shutil.copy2(backup_path, self.train_file)
                backup_path.unlink(missing_ok=True)
                logger.debug(f"Restored train.py from backup")

    def _apply_modifications(self, content: str, parameters: dict) -> str:
        """Apply parameter modifications to train.py content.

        Uses regex to find and replace parameter values.
        Looks for patterns like: PARAM_NAME = value

        Args:
            content: Original train.py content
            parameters: Dict of {param_name: new_value}

        Returns: Modified content

        Raises: ValueError if a parameter couldn't be found
        """
        result = content

        for param_name, value in parameters.items():
            # Try to find PARAM_NAME = ... pattern
            param_upper = param_name.upper()
            pattern = rf"({param_upper}\s*=\s*)([^\n]+)"
            match = re.search(pattern, result)

            if match:
                old_text = match.group(0)
                new_text = f"{match.group(1)}{value}"
                result = result.replace(old_text, new_text, 1)
                logger.debug(f"Modified {param_name}: {old_text[:60]} → {new_text[:60]}")
            else:
                logger.warning(f"Could not find parameter {param_name} in train.py")

        return result

    def _validate_python(self, content: str) -> bool:
        """Validate that content is syntactically valid Python.

        Args:
            content: Python code to validate

        Returns: True if valid, False if syntax error
        """
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in modified train.py: {e}")
            return False

    def _execute_training(self, log_path: Path) -> dict:
        """Run train.py and capture output.

        Args:
            log_path: Path where to save run.log

        Returns: Dict with {success: bool, error: str|None}
        """
        timeout = getattr(self.config, 'time_budget_seconds', 300) + 60

        try:
            with open(log_path, 'w') as log_file:
                proc = subprocess.run(
                    ["python", str(self.train_file)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    cwd=Path.cwd(),
                )

            if proc.returncode != 0:
                return {
                    "success": False,
                    "error": f"train.py exited with code {proc.returncode}"
                }

            return {"success": True}

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Training exceeded timeout ({timeout}s)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}"
            }

    def _parse_metrics(self, log_path: Path) -> Optional[dict]:
        """Parse training metrics from run.log.

        Looks for standard output patterns from train.py:
        - val_bpb: X.XXX
        - loss: Y.YYY
        - peak memory: Z.Z GB
        - training time: T.T seconds

        Args:
            log_path: Path to run.log

        Returns: Dict with {val_bpb, train_loss, peak_memory_gb, training_seconds},
                 or None if val_bpb not found
        """
        content = log_path.read_text()
        metrics = {}

        # Parse val_bpb (required metric)
        match = re.search(r'val[_\s]bpb[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if match:
            metrics["val_bpb"] = float(match.group(1))
        else:
            logger.error("Could not find val_bpb in run.log")
            return None

        # Parse training loss (optional)
        losses = re.findall(r'loss[:\s]+([0-9.]+)', content)
        if losses:
            metrics["train_loss"] = float(losses[-1])

        # Parse peak memory (optional)
        mem_match = re.search(r'peak[_\s]memory[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if mem_match:
            metrics["peak_memory_gb"] = float(mem_match.group(1))

        # Parse training time (optional)
        time_match = re.search(r'training[_\s]time[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if time_match:
            metrics["training_seconds"] = float(time_match.group(1))

        return metrics

    def _git_commit(self, exp_num: int, parameters: dict) -> Optional[str]:
        """Commit the modified train.py to git.

        Args:
            exp_num: Experiment number
            parameters: Parameters that were tested

        Returns: Git commit hash (first 7 chars), or None if commit failed
        """
        if not getattr(self.config, 'git_commit_each', False):
            return None

        try:
            # Create commit message
            param_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
            msg = f"exp_{exp_num:04d}: {param_str}"

            subprocess.run(
                ["git", "add", str(self.train_file)],
                capture_output=True,
                check=True,
                cwd=Path.cwd(),
            )

            subprocess.run(
                ["git", "commit", "-m", msg],
                capture_output=True,
                check=True,
                cwd=Path.cwd(),
            )

            # Get the commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                check=True,
                text=True,
                cwd=Path.cwd(),
            )

            commit_hash = result.stdout.strip()[:7]
            logger.debug(f"Created git commit {commit_hash}")
            return commit_hash

        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not create git commit: {e}")
            return None

    def _append_results_tsv(self, exp_num: int, task_id: int,
                            params: ExperimentParams, metrics: dict,
                            commit_hash: Optional[str]):
        """Append experiment result to results.tsv.

        Args:
            exp_num: Experiment number
            task_id: Which task this belongs to
            params: ExperimentParams
            metrics: Parsed metrics dict
            commit_hash: Git commit if created
        """
        # Create header if file doesn't exist
        if not self.results_tsv.exists():
            header = "experiment\ttask\tcommit\tval_bpb\ttrain_loss\tmemory_gb\tseconds\tparameters\ttimestamp\n"
            self.results_tsv.write_text(header)

        # Format parameters as JSON
        param_str = json.dumps(params.parameters, separators=(',', ':'))

        # Create row
        row = (
            f"exp_{exp_num:04d}\t{task_id}\t{commit_hash or 'none'}\t"
            f"{metrics.get('val_bpb', 'NaN')}\t{metrics.get('train_loss', 'NaN')}\t"
            f"{metrics.get('peak_memory_gb', 'NaN')}\t{metrics.get('training_seconds', 'NaN')}\t"
            f"{param_str}\t{datetime.now(timezone.utc).isoformat()}\n"
        )

        # Append to file
        with open(self.results_tsv, 'a') as f:
            f.write(row)

        logger.debug(f"Appended results to {self.results_tsv}")

    def _failed_result(self, exp_num: int, params: ExperimentParams,
                       timestamp: datetime, exp_dir: Path,
                       error: str) -> ExperimentResult:
        """Create a failed experiment result.

        Args:
            exp_num: Experiment number
            params: ExperimentParams
            timestamp: When the experiment started
            exp_dir: Experiment directory
            error: Error message

        Returns: ExperimentResult with success=False
        """
        # Log error to file
        (exp_dir / "error.txt").write_text(error)
        logger.error(f"Experiment {exp_num} failed: {error}")

        return ExperimentResult(
            index=exp_num,
            parameters=params.parameters,
            success=False,
            metric_value=None,
            train_loss=None,
            peak_memory_gb=None,
            training_seconds=None,
            commit_hash=None,
            error=error,
            log_path=exp_dir / "run.log",
            timestamp=timestamp,
        )

    def _load_counter(self) -> int:
        """Load the experiment counter from disk.

        Returns: Next experiment number
        """
        counter_file = self.results_dir / ".counter"
        if counter_file.exists():
            try:
                return int(counter_file.read_text().strip())
            except ValueError:
                return 1
        return 1

    def _save_counter(self):
        """Persist the experiment counter."""
        counter_file = self.results_dir / ".counter"
        counter_file.write_text(str(self.experiment_counter))
