"""Captain's CLI for the autonomous crew system.

Communicates with the running daemon via Unix socket.
Provides 30+ commands for task management, monitoring, and control.
"""

import sys
import json
import socket
import argparse
import textwrap
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

SOCKET_PATH = Path("data/crew.sock")


class CrewClient:
    """Client for communicating with the daemon via Unix socket."""

    def __init__(self, socket_path: Path = SOCKET_PATH):
        """Initialize socket client."""
        self.socket_path = socket_path

    def send_command(self, command: str, args: Dict[str, Any] = None) -> Optional[Dict]:
        """Send a command to the daemon and get response.

        Args:
            command: Command name (e.g., "status")
            args: Command arguments dict

        Returns: Response data dict, or None if daemon not running
        """
        if args is None:
            args = {}

        if not self.socket_path.exists():
            return None

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(str(self.socket_path))

            # Send command as JSON
            request = json.dumps({"command": command, "args": args})
            sock.sendall(request.encode())

            # Receive response
            response_data = b""
            while True:
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response_data += chunk
                except socket.timeout:
                    break

            sock.close()

            if response_data:
                response = json.loads(response_data.decode())
                return response.get("data")

            return None

        except (ConnectionRefusedError, FileNotFoundError):
            return None
        except Exception as e:
            print(f"Error communicating with daemon: {e}", file=sys.stderr)
            return None


class CLI:
    """Captain CLI interface."""

    def __init__(self):
        """Initialize CLI."""
        self.client = CrewClient()

    def daemon_running(self) -> bool:
        """Check if daemon is running."""
        return SOCKET_PATH.exists()

    def print_box(self, title: str, content: str, width: int = 50):
        """Print a bordered box."""
        print("┌" + "─" * (width - 2) + "┐")
        if title:
            print(f"│ {title:<{width-4}} │")
            print("├" + "─" * (width - 2) + "┤")
        for line in content.split("\n"):
            print(f"│ {line:<{width-4}} │")
        print("└" + "─" * (width - 2) + "┘")

    def print_table(self, rows: list, headers: list, widths: list):
        """Print a formatted table."""
        # Header
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(header_line)
        print("─" * len(header_line))

        # Rows
        for row in rows:
            row_line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            print(row_line)

    # ========================================================================
    # Lifecycle Commands
    # ========================================================================

    def cmd_start(self, args):
        """Start the daemon."""
        if self.daemon_running():
            print("Crew is already running.")
            return

        import subprocess
        cmd = ["python", "-m", "crew.daemon"]
        if args.config:
            cmd.extend(["--config", str(args.config)])
        if args.foreground:
            cmd.append("--foreground")
        if args.verbose:
            cmd.append("--verbose")

        try:
            if args.foreground:
                subprocess.run(cmd)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("Crew daemon started in background.")
        except Exception as e:
            print(f"Error starting daemon: {e}", file=sys.stderr)

    def cmd_stop(self, args):
        """Stop the daemon."""
        if not self.daemon_running():
            print("Crew is not running.")
            return

        data = self.client.send_command("stop")
        if data:
            print("Crew daemon stopping...")
        else:
            print("Error communicating with daemon.", file=sys.stderr)

    def cmd_restart(self, args):
        """Restart the daemon."""
        self.cmd_stop(args)
        import time
        time.sleep(1)
        self.cmd_start(args)

    def cmd_status(self, args):
        """Show crew status."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("status")
        if not data:
            print("Error getting status from daemon.", file=sys.stderr)
            return

        mode = data.get("mode", "unknown")
        current_task = data.get("current_task")
        queued = data.get("queued_tasks", 0)

        content = f"""Mode: {mode.upper()}
Queued: {queued} tasks
Current: Task #{current_task if current_task else 'None'}"""

        if args.json:
            print(json.dumps(data, indent=2))
        else:
            self.print_box("CREW STATUS", content, width=45)

    # ========================================================================
    # Task Management
    # ========================================================================

    def cmd_add(self, args):
        """Add a new task."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        priority = args.priority
        if args.urgent:
            priority = 1

        task_args = {
            "title": args.title,
            "priority": priority,
        }
        if args.description:
            task_args["description"] = args.description

        data = self.client.send_command("add", task_args)
        if data:
            task_id = data.get("task_id")
            print(f"Added task #{task_id}: {args.title}")
        else:
            print("Error adding task.", file=sys.stderr)

    def cmd_board(self, args):
        """Show the task board."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("board")
        if not data:
            print("Error getting board.", file=sys.stderr)
            return

        tasks = data.get("tasks", [])

        if not tasks:
            print("Task board is empty.")
            return

        # Print table
        headers = ["#", "Pri", "Status", "Type", "Title"]
        widths = [3, 3, 8, 12, 40]

        self.print_table(
            [
                [
                    t["id"],
                    t.get("priority", "-"),
                    t["status"][:8].upper(),
                    t.get("type", "-")[:12],
                    t["title"][:40],
                ]
                for t in tasks
            ],
            headers,
            widths,
        )

        print(f"\n{len(tasks)} task(s) on board")

    def cmd_show(self, args):
        """Show details of a task."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("show", {"task_id": args.task_id})
        if not data:
            print(f"Task #{args.task_id} not found.", file=sys.stderr)
            return

        task = data.get("task")
        if not task:
            print(f"Task #{args.task_id} not found.", file=sys.stderr)
            return

        print(f"\nTask #{task['id']}: {task['title']}")
        print("=" * 60)
        print(f"Type:     {task['type']}")
        print(f"Priority: {task['priority']}")
        print(f"Status:   {task['status'].upper()}")
        print(f"Created:  {task['created_at']}")

        if task.get("description"):
            print(f"\nDescription:\n  {task['description']}")

        if task.get("experiment"):
            print(f"\nExperiment:")
            for key, value in task["experiment"].items():
                print(f"  {key}: {value}")

        if task.get("hints"):
            print(f"\nHints:")
            for hint in task["hints"]:
                print(f"  - {hint.get('text', '')}")

    def cmd_hint(self, args):
        """Add a hint to a task."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command(
            "add_hint",
            {"task_id": args.task_id, "text": args.hint_text},
        )
        if data:
            print(f"Added hint to task #{args.task_id}")
        else:
            print("Error adding hint.", file=sys.stderr)

    def cmd_pause(self, args):
        """Pause a task."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("pause_task", {"task_id": args.task_id})
        if data:
            print(f"Paused task #{args.task_id}")
        else:
            print("Error pausing task.", file=sys.stderr)

    def cmd_resume(self, args):
        """Resume a paused task."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("resume_task", {"task_id": args.task_id})
        if data:
            print(f"Resumed task #{args.task_id}")
        else:
            print("Error resuming task.", file=sys.stderr)

    def cmd_cancel(self, args):
        """Cancel a task."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("cancel_task", {"task_id": args.task_id})
        if data:
            print(f"Cancelled task #{args.task_id}")
        else:
            print("Error cancelling task.", file=sys.stderr)

    def cmd_priority(self, args):
        """Set task priority."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command(
            "set_priority",
            {"task_id": args.task_id, "priority": args.priority},
        )
        if data:
            print(f"Set task #{args.task_id} priority to {args.priority}")
        else:
            print("Error setting priority.", file=sys.stderr)

    def cmd_log(self, args):
        """Show recent completed tasks."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        limit = args.limit or 10
        data = self.client.send_command("get_completed", {"limit": limit})
        if not data:
            print("Error getting logs.", file=sys.stderr)
            return

        tasks = data.get("tasks", [])
        if not tasks:
            print("No completed tasks.")
            return

        print(f"\nRecent Completed Tasks (last {limit})")
        print("=" * 60)

        for task in tasks:
            completed_at = task.get("completed_at", "unknown")
            print(f"\n#{task['id']}: {task['title']}")
            print(f"  Completed: {completed_at}")
            if task.get("results"):
                print(f"  Results: {task['results']}")

    # ========================================================================
    # Monitoring Commands
    # ========================================================================

    def cmd_findings(self, args):
        """Show recent findings."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        limit = args.limit or 10
        data = self.client.send_command("get_findings", {"limit": limit})
        if not data:
            print("No findings available.", file=sys.stderr)
            return

        findings = data.get("findings", [])
        if not findings:
            print("No findings yet.")
            return

        print(f"\nFindings (last {limit})")
        print("=" * 60)

        for finding in findings:
            print(
                f"\n[{finding.get('timestamp', 'unknown')}] "
                f"Task #{finding.get('task_id', '?')}"
            )
            print(f"  {finding.get('summary', '')}")
            print(f"  Confidence: {finding.get('confidence', 'unknown')}")

    def cmd_knowledge(self, args):
        """Browse the knowledge base."""
        print("Knowledge base commands not yet implemented.")

    def cmd_study(self, args):
        """Show study queue or current study topic."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("get_study_status")
        if not data:
            print("Error getting study status.", file=sys.stderr)
            return

        print("Study Status")
        print("=" * 60)
        print(json.dumps(data, indent=2))

    def cmd_gpu(self, args):
        """Show GPU stats."""
        print("GPU stats command not yet implemented.")

    def cmd_metrics(self, args):
        """Show lifetime metrics."""
        if not self.daemon_running():
            print("Crew is not running. Start with: crew start")
            return

        data = self.client.send_command("get_metrics")
        if not data:
            print("Error getting metrics.", file=sys.stderr)
            return

        print("Crew Metrics")
        print("=" * 60)
        print(json.dumps(data, indent=2))

    def cmd_notifications(self, args):
        """Show notifications."""
        print("Notifications command not yet implemented.")

    # ========================================================================
    # Config Commands
    # ========================================================================

    def cmd_config(self, args):
        """Show or edit configuration."""
        config_path = Path("data/config.yaml")

        if args.action == "show" or args.action is None:
            if config_path.exists():
                print(f"Configuration ({config_path}):")
                print(config_path.read_text())
            else:
                print(f"No config file at {config_path}")

        elif args.action == "edit":
            import subprocess

            editor = os.environ.get("EDITOR", "nano")
            subprocess.run([editor, str(config_path)])

        elif args.action == "set":
            if not config_path.exists():
                print(f"Config file not found: {config_path}")
                return

            import yaml

            config = yaml.safe_load(config_path.read_text()) or {}

            # Simple set: key=value format
            keys = args.key.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Try to parse value as JSON, otherwise string
            try:
                current[keys[-1]] = json.loads(args.value)
            except:
                current[keys[-1]] = args.value

            config_path.write_text(yaml.dump(config))
            print(f"Set {args.key} = {args.value}")

    def cmd_triggers(self, args):
        """Manage triggers."""
        print("Triggers command not yet implemented.")


def main():
    """Main entry point for crew CLI."""
    import os

    cli = CLI()

    parser = argparse.ArgumentParser(
        description="Autonomous research crew command-line interface",
        prog="crew",
    )

    # Global options
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Lifecycle commands
    start_parser = subparsers.add_parser("start", help="Start the daemon")
    start_parser.add_argument("--foreground", action="store_true")
    start_parser.add_argument("--verbose", action="store_true")
    start_parser.add_argument("--config", type=Path)

    stop_parser = subparsers.add_parser("stop", help="Stop the daemon")

    restart_parser = subparsers.add_parser("restart", help="Restart the daemon")

    status_parser = subparsers.add_parser("status", help="Show crew status")

    # Task management
    add_parser = subparsers.add_parser("add", help="Add a task")
    add_parser.add_argument("title", help="Task title")
    add_parser.add_argument("--priority", type=int, default=2)
    add_parser.add_argument("--urgent", action="store_true")
    add_parser.add_argument("--description", type=str)

    board_parser = subparsers.add_parser("board", help="Show task board")

    show_parser = subparsers.add_parser("show", help="Show task details")
    show_parser.add_argument("task_id", type=int)

    hint_parser = subparsers.add_parser("hint", help="Add a hint to a task")
    hint_parser.add_argument("task_id", type=int)
    hint_parser.add_argument("hint_text", type=str)

    pause_parser = subparsers.add_parser("pause", help="Pause a task")
    pause_parser.add_argument("task_id", type=int)

    resume_parser = subparsers.add_parser("resume", help="Resume a task")
    resume_parser.add_argument("task_id", type=int)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a task")
    cancel_parser.add_argument("task_id", type=int)

    priority_parser = subparsers.add_parser("priority", help="Set task priority")
    priority_parser.add_argument("task_id", type=int)
    priority_parser.add_argument("priority", type=int)

    log_parser = subparsers.add_parser("log", help="Show recent completed tasks")
    log_parser.add_argument("--limit", type=int, default=10)

    # Monitoring
    findings_parser = subparsers.add_parser("findings", help="Show findings")
    findings_parser.add_argument("--limit", type=int, default=10)

    knowledge_parser = subparsers.add_parser("knowledge", help="Browse knowledge base")

    study_parser = subparsers.add_parser("study", help="Show study status")

    gpu_parser = subparsers.add_parser("gpu", help="Show GPU stats")

    metrics_parser = subparsers.add_parser("metrics", help="Show metrics")

    notifications_parser = subparsers.add_parser("notifications", help="Show notifications")

    # Config
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "action",
        nargs="?",
        choices=["show", "set", "edit"],
        default="show",
    )
    config_parser.add_argument("key", nargs="?")
    config_parser.add_argument("value", nargs="?")

    triggers_parser = subparsers.add_parser("triggers", help="Manage triggers")

    args = parser.parse_args()

    # Dispatch to command handler
    if args.command == "start":
        cli.cmd_start(args)
    elif args.command == "stop":
        cli.cmd_stop(args)
    elif args.command == "restart":
        cli.cmd_restart(args)
    elif args.command == "status":
        cli.cmd_status(args)
    elif args.command == "add":
        cli.cmd_add(args)
    elif args.command == "board":
        cli.cmd_board(args)
    elif args.command == "show":
        cli.cmd_show(args)
    elif args.command == "hint":
        cli.cmd_hint(args)
    elif args.command == "pause":
        cli.cmd_pause(args)
    elif args.command == "resume":
        cli.cmd_resume(args)
    elif args.command == "cancel":
        cli.cmd_cancel(args)
    elif args.command == "priority":
        cli.cmd_priority(args)
    elif args.command == "log":
        cli.cmd_log(args)
    elif args.command == "findings":
        cli.cmd_findings(args)
    elif args.command == "knowledge":
        cli.cmd_knowledge(args)
    elif args.command == "study":
        cli.cmd_study(args)
    elif args.command == "gpu":
        cli.cmd_gpu(args)
    elif args.command == "metrics":
        cli.cmd_metrics(args)
    elif args.command == "notifications":
        cli.cmd_notifications(args)
    elif args.command == "config":
        cli.cmd_config(args)
    elif args.command == "triggers":
        cli.cmd_triggers(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
