"""Daemon process for the autonomous crew system.

The daemon is always-on. It:
1. Loads configuration and state
2. Starts background threads (heartbeat, triggers, webhooks)
3. Runs the main loop (scheduler → runner → brain)
4. Handles graceful shutdown
5. Recovers from crashes
"""

import os
import sys
import time
import signal
import socket
import threading
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import yaml

from crew.scheduler import Scheduler, Task
from crew.runner import ExperimentRunner, ExperimentParams
from crew.brain import CrewBrain

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Loader
# ============================================================================

class Config:
    """System configuration from YAML."""

    DEFAULTS = {
        "crew": {
            "name": "Crew",
            "personality": "balanced",
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "api_key": "",
        },
        "gpu": {
            "device": "cuda:0",
            "max_memory_gb": None,
            "temperature_throttle_c": 85.0,
            "temperature_shutdown_c": 95.0,
        },
        "experiments": {
            "time_budget_seconds": 300,
            "git_commit_each": False,
            "keep_checkpoints": "best_only",
            "train_file": "train.py",
            "results_dir": "data/experiments",
        },
        "tasks": {
            "auto_follow_ups": True,
            "study_enabled": True,
            "study_max_experiments": 10,
        },
        "daemon": {
            "log_level": "info",
            "heartbeat_interval_seconds": 30,
            "graceful_shutdown_timeout_seconds": 60,
        },
        "swarm": {
            "enabled": False,
            "max_agents": 1,
            "roles": {
                # Core agents
                "researcher": 1,
                "teacher": 1,
                "critic": 0,
                "distiller": 0,
                # Research agents
                "scientist": 0,
                # Creative agents
                "writer": 0,
                "editor": 0,
                # Software agents
                "code_reviewer": 0,
                # Quality agents
                "consistency": 0,
                # Business agents
                "project_manager": 0,
                "strategy": 0,
                # Security agents
                "security": 0,
            },
        },
        "hardware": {
            "profile": None,  # Auto-detect if not set
        },
    }

    def __init__(self, config_path: Path = None):
        """Load configuration from YAML."""
        self.data = self.DEFAULTS.copy()

        if config_path is None:
            config_path = Path("data/config.yaml")

        if config_path.exists():
            try:
                user_config = yaml.safe_load(config_path.read_text())
                if user_config:
                    self._deep_merge(self.data, user_config)
                    logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading config: {e}")

        # Convert to object attributes
        for section, values in self.data.items():
            setattr(self, section, type('obj', (object,), values)())

    def _deep_merge(self, base, override):
        """Recursively merge override into base."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# ============================================================================
# Crew Daemon
# ============================================================================

class CrewDaemon:
    """The always-on autonomous crew process."""

    def __init__(self, config_path: Path = None, foreground: bool = False):
        """Initialize the daemon."""
        self.config = Config(config_path)
        self.foreground = foreground
        self.shutdown_requested = False
        self.threads = []

        # Core components
        self.scheduler = Scheduler(Path("data/tasks"))
        self.runner = ExperimentRunner(self.config.experiments)
        self.brain = CrewBrain(self.config.llm)

        # State
        self.state_file = Path("data/crew/state.yaml")
        self.state = self._load_state()
        self.current_task: Optional[Task] = None
        self.mode = "starting"

        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def start(self):
        """Start the daemon process."""
        logger.info("Starting crew daemon...")

        # Validate environment
        self._validate_environment()

        # Start background threads
        self._start_heartbeat_thread()
        self._start_command_listener_thread()

        logger.info("Crew daemon running. Mode: working")

        try:
            self.main_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted")
        except Exception as e:
            logger.error(f"Daemon error: {e}", exc_info=True)
        finally:
            self.shutdown()

    def main_loop(self):
        """Main event loop. Runs forever."""
        while not self.shutdown_requested:
            try:
                # 1. Process captain commands (from CLI)
                # (Handled by command listener thread)

                # 2. Check task board
                next_task = self.scheduler.get_next_task()

                if next_task:
                    # Work on task
                    self.set_mode("working")
                    self.work_on_task(next_task)

                    # Review results
                    self.set_mode("reviewing")
                    self.review_results(next_task)

                else:
                    # No tasks - check if we should maintain or study
                    if self.needs_maintenance():
                        self.set_mode("maintaining")
                        self.run_maintenance()
                    elif self.config.tasks.study_enabled:
                        self.set_mode("studying")
                        self.run_study_session()
                    else:
                        self.set_mode("idle")
                        time.sleep(60)  # Check again in 1 minute

                # 3. Persist state
                self.persist_state()

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)  # Brief pause before retry

    def work_on_task(self, task: Task):
        """Execute a task by running experiments.

        Args:
            task: Task to work on
        """
        logger.info(f"Starting task #{task.id}: {task.title}")

        # Activate task
        self.scheduler.activate_task(task)
        self.current_task = task

        # Plan experiments
        if not task.experiment:
            logger.warning(f"Task #{task.id} has no experiment spec")
            return

        plan = self.brain.plan_experiments(task, knowledge=[])
        logger.info(f"Planning {len(plan.experiments)} experiments")

        # Run experiments
        results = []
        for exp in plan.experiments:
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping experiments")
                break

            params = ExperimentParams(
                parameters=exp["parameters"],
                rationale=exp.get("rationale", ""),
                index=len(results),
            )

            logger.info(f"Running experiment {len(results)+1}/{len(plan.experiments)}: {params.parameters}")

            result = self.runner.run_experiment(params, task.id)
            results.append(result)

            if result.success:
                logger.info(f"  ✓ Success: val_bpb={result.metric_value}")
            else:
                logger.error(f"  ✗ Failed: {result.error}")

        # Task is done
        if results:
            self.scheduler.complete_task(
                task,
                {
                    "experiments_run": len(results),
                    "successful": sum(1 for r in results if r.success),
                }
            )
            logger.info(f"Task #{task.id} completed")
        else:
            self.scheduler.pause_task(task)
            logger.warning(f"Task #{task.id} paused (no experiments run)")

        self.current_task = None

    def review_results(self, task: Task):
        """Review task results and extract insights.

        Args:
            task: Completed task
        """
        logger.debug(f"Reviewing results for task #{task.id}")

        # Load experiment results from task
        if not task.results or task.results.get("experiments_run", 0) == 0:
            logger.debug(f"Task #{task.id} has no experiment results to review")
            return

        # Parse results directory for this task
        results_dir = Path("data/experiments")
        task_results = []

        # Find experiment directories for this task (naive: just take most recent)
        # In a full implementation, would track experiment IDs per task

        logger.info(f"Task #{task.id} review: {task.results.get('successful', 0)}/{task.results.get('experiments_run', 0)} successful")

    def run_maintenance(self):
        """Run maintenance tasks (archive, organize, cleanup)."""
        logger.info("Running maintenance")

        # Archive old experiment directories (older than 30 days)
        exp_dir = Path("data/experiments")
        if exp_dir.exists():
            import time
            now = time.time()
            cutoff = now - (30 * 86400)  # 30 days ago

            archived = 0
            for exp_folder in sorted(exp_dir.glob("exp_*")):
                if exp_folder.stat().st_mtime < cutoff:
                    # Move to archived directory
                    archive_dir = exp_dir / "archived"
                    archive_dir.mkdir(exist_ok=True)
                    try:
                        shutil.move(str(exp_folder), str(archive_dir / exp_folder.name))
                        archived += 1
                    except Exception as e:
                        logger.warning(f"Could not archive {exp_folder}: {e}")

            if archived > 0:
                logger.info(f"Archived {archived} experiment folders")

        # Clean up temporary files
        cleanup_patterns = ["*.tmp", "*.log~", "*~"]
        for pattern in cleanup_patterns:
            for f in Path("data").glob(f"**/{pattern}"):
                try:
                    f.unlink()
                except Exception as e:
                    logger.debug(f"Could not clean {f}: {e}")

        logger.info("Maintenance completed")
        self.set_mode("idle")

    def run_study_session(self):
        """Self-directed study when task board is empty."""
        logger.info("Entering study mode")

        # Get completed tasks for context
        completed = self.scheduler.get_completed(limit=10)

        # Ask brain what to study
        study_topic = self.brain.decide_study_topic(knowledge=[], recent_tasks=completed)

        if study_topic:
            logger.info(f"Study topic: {study_topic.title}")
            logger.info(f"Reason: {study_topic.reason}")

            # Create a temporary study task (don't add to board permanently)
            num_exps = study_topic.experiment_plan.get('num_experiments', 3)

            # Run a few quick experiments to explore the topic
            for i in range(min(num_exps, self.config.tasks.study_max_experiments or 10)):
                if self.shutdown_requested:
                    break

                params = {
                    "study_seed": i,
                    "learning_rate": 0.001 * (2 ** (i % 3)),
                }

                from crew.runner import ExperimentParams
                exp_params = ExperimentParams(
                    parameters=params,
                    rationale=f"Study exploration {i+1}: {study_topic.title}",
                    index=i,
                )

                logger.info(f"Running study experiment {i+1}/{num_exps}")
                result = self.runner.run_experiment(exp_params, task_id=-1)  # -1 = study, not tied to task

                if result.success:
                    logger.info(f"  ✓ Study exp: {result.parameters}")
                else:
                    logger.debug(f"  - Study exp failed: {result.error}")

            logger.info(f"Study session completed: {study_topic.title}")
        else:
            logger.info("No study topic determined")

        self.set_mode("idle")

    def needs_maintenance(self) -> bool:
        """Check if maintenance is needed.

        Checks:
        - Disk space (warn if <10% free)
        - Results directory size (warn if >50GB)
        - Logs file size (warn if >1GB)
        """
        import shutil

        # Check disk space
        try:
            usage = shutil.disk_usage("data")
            percent_free = (usage.free / usage.total) * 100
            if percent_free < 10:
                logger.warning(f"Low disk space: {percent_free:.1f}% free")
                return True
        except Exception as e:
            logger.debug(f"Could not check disk space: {e}")

        # Check results directory size
        exp_dir = Path("data/experiments")
        if exp_dir.exists():
            try:
                total_size = sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file())
                size_gb = total_size / (1024**3)
                if size_gb > 50:
                    logger.warning(f"Large experiments directory: {size_gb:.1f}GB")
                    return True
            except Exception as e:
                logger.debug(f"Could not check results size: {e}")

        # Check if we have many old experiment directories (monthly check)
        if exp_dir.exists():
            exp_count = len(list(exp_dir.glob("exp_*")))
            if exp_count > 1000:
                logger.info(f"Many experiments ({exp_count}), may want to archive")
                return True

        return False

    def set_mode(self, mode: str):
        """Change operational mode."""
        if self.mode != mode:
            logger.info(f"Mode: {self.mode} → {mode}")
            self.mode = mode

    def persist_state(self):
        """Save state to disk for crash recovery."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state_data = {
            "mode": self.mode,
            "current_task": self.current_task.id if self.current_task else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.write_text(yaml.dump(state_data))

    def _load_state(self) -> dict:
        """Load persisted state from disk."""
        if self.state_file.exists():
            try:
                return yaml.safe_load(self.state_file.read_text()) or {}
            except Exception as e:
                logger.warning(f"Error loading state: {e}")
        return {}

    def _validate_environment(self):
        """Check that system is ready."""
        logger.info("Validating environment...")

        # Create data directory
        Path("data").mkdir(exist_ok=True)

        # Check GPU (if configured)
        if self.config.gpu.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("CUDA device configured but GPU not available")
            except ImportError:
                logger.warning("PyTorch not installed, cannot check GPU")

    def _handle_signal(self, signum, frame):
        """Handle SIGTERM and SIGINT."""
        logger.info(f"Received signal {signum}")
        self.shutdown_requested = True

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.set_mode("shutting_down")

        # Stop background threads
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)

        # Persist final state
        self.persist_state()

        logger.info("Crew daemon stopped")

    # ========================================================================
    # Background Threads
    # ========================================================================

    def _start_heartbeat_thread(self):
        """Start background thread for GPU monitoring."""
        def heartbeat():
            while not self.shutdown_requested:
                try:
                    # TODO: Read GPU stats via nvidia-smi
                    # Update self.state["gpu_stats"]
                    # Check thresholds
                    pass
                except Exception as e:
                    logger.debug(f"Heartbeat error: {e}")

                time.sleep(self.config.daemon.heartbeat_interval_seconds)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.debug("Started heartbeat thread")

    def _start_command_listener_thread(self):
        """Start background thread for CLI command socket."""
        def listener():
            socket_path = Path("data/crew.sock")
            socket_path.unlink(missing_ok=True)

            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.bind(str(socket_path))
                sock.listen(1)
                logger.debug(f"Command listener on {socket_path}")

                while not self.shutdown_requested:
                    try:
                        sock.settimeout(1)
                        connection, _ = sock.accept()
                        self._handle_command(connection)
                        connection.close()
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.debug(f"Command handler error: {e}")

            except Exception as e:
                logger.warning(f"Command listener error: {e}")
            finally:
                socket_path.unlink(missing_ok=True)

        thread = threading.Thread(target=listener, daemon=True)
        thread.start()
        self.threads.append(thread)
        logger.debug("Started command listener thread")

    def _handle_command(self, connection):
        """Handle a CLI command.

        Args:
            connection: Socket connection from CLI
        """
        try:
            # Receive command (simple text protocol)
            cmd = connection.recv(1024).decode().strip()

            if not cmd:
                return

            response = self._execute_command(cmd)
            connection.sendall(response.encode())

        except Exception as e:
            logger.error(f"Command error: {e}")

    def _execute_command(self, cmd: str) -> str:
        """Execute a captain command and return response.

        Args:
            cmd: Command string (JSON format from CLI)

        Returns: Response JSON
        """
        import json

        # Parse JSON command from CLI
        try:
            msg = json.loads(cmd)
            command = msg.get("command")
            args = msg.get("args", {})
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "error": "Invalid JSON"})

        # Dispatch commands
        if command == "status":
            return json.dumps({
                "status": "ok",
                "data": {
                    "mode": self.mode,
                    "current_task": self.current_task.id if self.current_task else None,
                    "queued_tasks": len(self.scheduler.get_queued()),
                }
            })

        elif command == "add":
            title = args.get("title")
            priority = args.get("priority", self.scheduler.default_priority("captain_order"))
            description = args.get("description", "")

            task = self.scheduler.add_task(
                title,
                task_type="captain_order",
                priority=priority,
                description=description,
            )
            if task:
                return json.dumps({
                    "status": "ok",
                    "data": {"task_id": task.id, "title": task.title}
                })
            else:
                return json.dumps({
                    "status": "error",
                    "error": "Duplicate task"
                })

        elif command == "board":
            board = self.scheduler.get_board()
            return json.dumps({
                "status": "ok",
                "data": {
                    "tasks": [
                        {
                            "id": t.id,
                            "title": t.title,
                            "priority": t.priority,
                            "status": t.status,
                            "type": t.type,
                        }
                        for t in board
                    ]
                }
            })

        elif command == "show":
            task_id = args.get("task_id")
            task = self.scheduler.get_task(task_id)
            if task:
                return json.dumps({
                    "status": "ok",
                    "data": {
                        "task": {
                            "id": task.id,
                            "title": task.title,
                            "type": task.type,
                            "priority": task.priority,
                            "status": task.status,
                            "description": task.description,
                            "created_at": task.created_at.isoformat() if task.created_at else None,
                            "started_at": task.started_at.isoformat() if task.started_at else None,
                            "experiment": task.experiment,
                            "hints": task.hints,
                            "results": task.results,
                        }
                    }
                })
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Task {task_id} not found"
                })

        elif command == "add_hint":
            task_id = args.get("task_id")
            hint_text = args.get("text")
            task = self.scheduler.get_task(task_id)
            if task:
                if not task.hints:
                    task.hints = []
                task.hints.append({
                    "text": hint_text,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self.scheduler.save_task(task)
                return json.dumps({
                    "status": "ok",
                    "data": {"message": "Hint added"}
                })
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Task {task_id} not found"
                })

        elif command == "pause_task":
            task_id = args.get("task_id")
            task = self.scheduler.get_task(task_id)
            if task:
                self.scheduler.pause_task(task)
                return json.dumps({
                    "status": "ok",
                    "data": {"message": "Task paused"}
                })
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Task {task_id} not found"
                })

        elif command == "resume_task":
            task_id = args.get("task_id")
            task = self.scheduler.get_task(task_id)
            if task:
                self.scheduler.resume_task(task)
                return json.dumps({
                    "status": "ok",
                    "data": {"message": "Task resumed"}
                })
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Task {task_id} not found"
                })

        elif command == "cancel_task":
            task_id = args.get("task_id")
            task = self.scheduler.get_task(task_id)
            if task:
                self.scheduler.cancel_task(task)
                return json.dumps({
                    "status": "ok",
                    "data": {"message": "Task cancelled"}
                })
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Task {task_id} not found"
                })

        elif command == "set_priority":
            task_id = args.get("task_id")
            priority = args.get("priority")
            task = self.scheduler.get_task(task_id)
            if task:
                self.scheduler.set_priority(task, priority)
                return json.dumps({
                    "status": "ok",
                    "data": {"message": f"Priority set to {priority}"}
                })
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Task {task_id} not found"
                })

        elif command == "get_completed":
            limit = args.get("limit", 10)
            completed = self.scheduler.get_completed(limit=limit)
            return json.dumps({
                "status": "ok",
                "data": {
                    "tasks": [
                        {
                            "id": t.id,
                            "title": t.title,
                            "status": t.status,
                            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                            "results": t.results,
                        }
                        for t in completed
                    ]
                }
            })

        elif command == "get_findings":
            return json.dumps({
                "status": "ok",
                "data": {"findings": []}
            })

        elif command == "get_study_status":
            return json.dumps({
                "status": "ok",
                "data": {
                    "mode": self.mode,
                    "studying": self.mode == "studying",
                }
            })

        elif command == "get_metrics":
            return json.dumps({
                "status": "ok",
                "data": {
                    "mode": self.mode,
                    "queued_tasks": len(self.scheduler.get_queued()),
                    "completed_tasks": len(self.scheduler.get_completed()),
                }
            })

        elif command == "stop":
            self.shutdown_requested = True
            return json.dumps({
                "status": "ok",
                "data": {"message": "Shutting down"}
            })

        else:
            return json.dumps({
                "status": "error",
                "error": f"Unknown command: {command}"
            })


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Entry point for `crew start` command."""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous research crew daemon")
    parser.add_argument("--config", type=Path, default=None, help="Config file path")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--swarm", action="store_true", help="Force swarm mode")

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = Config(args.config)

    # Swarm mode: multi-agent pool + message bus
    swarm_cfg = getattr(config, "swarm", None)
    swarm_enabled = args.swarm or (swarm_cfg and getattr(swarm_cfg, "enabled", False))
    max_agents = getattr(swarm_cfg, "max_agents", 1) if swarm_cfg else 1

    if swarm_enabled and max_agents > 1:
        _start_swarm(config)
    else:
        daemon = CrewDaemon(args.config, args.foreground)
        daemon.start()


def _start_swarm(config: "Config"):
    """Start multi-agent swarm mode."""
    from crew.messaging.bus import get_bus
    from crew.agents.pool import AgentPool, register_agent_class
    from crew.agents.researcher import ResearcherAgent
    from crew.agents.teacher import TeacherAgent
    from crew.agents.critic import CriticAgent
    from crew.agents.distiller import DistillerAgent
    from crew.agents.scientist import ScientistAgent
    from crew.agents.code_reviewer import CodeReviewerAgent
    from crew.agents.editor import EditorAgent
    from crew.agents.writer import WriterAgent
    from crew.agents.project_manager import ProjectManagerAgent
    from crew.agents.consistency import ConsistencyAgent
    from crew.agents.security import SecurityAgent
    from crew.agents.strategy import StrategyAgent
    from crew.hardware.detector import HardwareDetector

    # Register concrete agent classes (core)
    register_agent_class("researcher", ResearcherAgent)
    register_agent_class("teacher", TeacherAgent)
    register_agent_class("critic", CriticAgent)
    register_agent_class("distiller", DistillerAgent)

    # Register concrete agent classes (research & analysis)
    register_agent_class("scientist", ScientistAgent)

    # Register concrete agent classes (creative & content)
    register_agent_class("writer", WriterAgent)
    register_agent_class("editor", EditorAgent)

    # Register concrete agent classes (software & engineering)
    register_agent_class("code_reviewer", CodeReviewerAgent)

    # Register concrete agent classes (quality & consistency)
    register_agent_class("consistency", ConsistencyAgent)

    # Register concrete agent classes (business & strategy)
    register_agent_class("project_manager", ProjectManagerAgent)
    register_agent_class("strategy", StrategyAgent)

    # Register concrete agent classes (security)
    register_agent_class("security", SecurityAgent)

    # Detect hardware profile
    hw_config = getattr(config, "hardware", None)
    hw_override = getattr(hw_config, "profile", None) if hw_config else None
    detector = HardwareDetector()
    profile_name, profile_info = detector.detect(force=False)
    if hw_override:
        profile_name = hw_override
        profile_info = detector.PROFILES.get(hw_override, profile_info)

    logger.info(f"Swarm mode: hardware profile={profile_name}, max_agents={profile_info.get('max_agents', 2)}")

    # Build swarm config dict
    swarm_cfg = {}
    if hasattr(config, "swarm"):
        for attr in dir(config.swarm):
            if not attr.startswith("_"):
                swarm_cfg[attr] = getattr(config.swarm, attr)

    # Start agent pool
    bus = get_bus()
    pool = AgentPool(swarm_cfg, bus=bus, hardware_profile=profile_name)
    pool.start()

    logger.info("Swarm started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(30)
            stats = pool.status()
            logger.info(
                f"Swarm status: {stats['total_agents']} agents, "
                f"queue={stats.get('queue', {})}"
            )
    except KeyboardInterrupt:
        logger.info("Swarm shutting down...")
        pool.stop()


if __name__ == "__main__":
    main()
