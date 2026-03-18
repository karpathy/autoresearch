"""Task scheduler for the crew system.

The scheduler manages the task board and decides what to work on next.
It's intentionally simple: priority ordering with creation time as tiebreaker.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A unit of work on the task board.

    Loaded from/saved to YAML files in data/tasks/{id}.yaml
    See schemas/task.yaml for complete schema.
    """
    # Required fields
    id: int
    title: str
    type: str  # captain_order, triggered, follow_up, maintenance, study
    priority: int
    status: str  # queued, active, paused, completed, failed, cancelled

    # Optional fields with defaults
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Source tracking
    source: dict = field(default_factory=lambda: {"creator": "captain", "trigger_id": None, "parent_task_id": None, "reason": ""})

    # Experiment spec
    experiment: Optional[dict] = None

    # Hints and notes
    hints: List[dict] = field(default_factory=list)
    notes: List[dict] = field(default_factory=list)

    # Results (populated when task completes)
    results: Optional[dict] = None

    # Resource tracking
    gpu_seconds_used: float = 0.0
    api_tokens_used: int = 0

    # Tags
    tags: List[str] = field(default_factory=list)

    def to_dict(self):
        """Convert to dict for YAML serialization."""
        data = asdict(self)
        # Convert datetime to ISO format strings
        for key in ['created_at', 'started_at', 'paused_at', 'completed_at']:
            if data[key] is not None:
                if isinstance(data[key], datetime):
                    data[key] = data[key].isoformat()
        return data

    @staticmethod
    def from_dict(data):
        """Create Task from dict (after YAML deserialization)."""
        # Convert ISO format strings back to datetime
        for key in ['created_at', 'started_at', 'paused_at', 'completed_at']:
            if key in data and data[key] is not None and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        return Task(**data)


class Scheduler:
    """Manages the task board and schedules work."""

    def __init__(self, tasks_dir: Path = None):
        """Initialize scheduler.

        Args:
            tasks_dir: Path to tasks directory (default: data/tasks)
        """
        if tasks_dir is None:
            tasks_dir = Path("data/tasks")

        self.tasks_dir = tasks_dir
        self.tasks: List[Task] = []
        self.next_id: int = 1

        # Create directory if it doesn't exist
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        (self.tasks_dir / "completed").mkdir(exist_ok=True)

        self.load_tasks()

    def load_tasks(self):
        """Load all tasks from YAML files."""
        self.tasks = []

        # Load from main directory (skip ACTIVE symlink)
        for f in self.tasks_dir.glob("*.yaml"):
            if f.name == "ACTIVE":
                continue
            try:
                task_data = yaml.safe_load(f.read_text())
                if task_data:
                    self.tasks.append(Task.from_dict(task_data))
            except Exception as e:
                logger.error(f"Error loading task {f}: {e}")

        # Load from completed directory
        for f in (self.tasks_dir / "completed").glob("*.yaml"):
            try:
                task_data = yaml.safe_load(f.read_text())
                if task_data:
                    self.tasks.append(Task.from_dict(task_data))
            except Exception as e:
                logger.error(f"Error loading completed task {f}: {e}")

        # Update next_id
        if self.tasks:
            self.next_id = max(t.id for t in self.tasks) + 1

    def get_next_task(self) -> Optional[Task]:
        """Return the highest-priority queued task.

        Returns: Task with lowest priority number (1=highest), or None if empty.
        """
        queued = [t for t in self.tasks if t.status == "queued"]

        if not queued:
            return None

        # Sort by priority (ascending), then creation time (oldest first)
        queued.sort(key=lambda t: (t.priority, t.created_at))
        return queued[0]

    def default_priority(self, task_type: str) -> int:
        """Get default priority for a task type."""
        return {
            "captain_order": 2,
            "triggered": 4,
            "follow_up": 5,
            "maintenance": 7,
            "study": 9,
        }.get(task_type, 5)

    def add_task(self, title: str, task_type: str = "captain_order",
                 priority: Optional[int] = None, description: str = "",
                 source: Optional[dict] = None, experiment: Optional[dict] = None,
                 tags: Optional[List[str]] = None, **kwargs) -> Optional[Task]:
        """Create a new task and add to board.

        Args:
            title: Task title
            task_type: Task type (captain_order, triggered, follow_up, maintenance, study)
            priority: Priority (1-10). If None, uses default for task_type.
            description: Detailed description
            source: Source info {creator, trigger_id, parent_task_id, reason}
            experiment: Experiment spec if this task runs experiments
            tags: Tags for categorization

        Returns: Created Task, or None if duplicate.
        """
        # Check for duplicates
        if self.is_duplicate(title, task_type):
            logger.info(f"Skipped duplicate task: {title}")
            return None

        # Set priority
        if priority is None:
            priority = self.default_priority(task_type)

        # Create task
        task = Task(
            id=self.next_id,
            title=title,
            type=task_type,
            priority=priority,
            status="queued",
            description=description,
            source=source or {"creator": "captain", "trigger_id": None, "parent_task_id": None, "reason": ""},
            experiment=experiment,
            tags=tags or [],
        )

        self.next_id += 1
        self.save_task(task)
        self.tasks.append(task)

        # Prune if needed
        self.prune_if_needed()

        logger.info(f"Created task #{task.id}: {title}")
        return task

    def activate_task(self, task: Task):
        """Move task from queued to active."""
        task.status = "active"
        task.started_at = datetime.now(timezone.utc)
        self.save_task(task)
        self._update_active_symlink(task)
        logger.info(f"Activated task #{task.id}")

    def complete_task(self, task: Task, results: Optional[dict] = None):
        """Mark task as completed."""
        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        if results:
            task.results = results

        # Move to completed directory
        src = self.tasks_dir / f"{task.id}.yaml"
        dst = self.tasks_dir / "completed" / f"{task.id}.yaml"

        self.save_task(task, dst)
        if src.exists():
            src.unlink()

        # Remove ACTIVE symlink
        self._clear_active_symlink()

        logger.info(f"Completed task #{task.id}")

    def pause_task(self, task: Task):
        """Pause task (will be resumed later)."""
        task.status = "paused"
        task.paused_at = datetime.now(timezone.utc)
        self.save_task(task)
        self._clear_active_symlink()
        logger.info(f"Paused task #{task.id}")

    def resume_task(self, task: Task):
        """Resume a paused task."""
        task.status = "queued"
        task.paused_at = None
        self.save_task(task)
        logger.info(f"Resumed task #{task.id}")

    def cancel_task(self, task: Task):
        """Cancel a task."""
        task.status = "cancelled"
        self.complete_task(task, {"summary": "Cancelled by captain"})
        logger.info(f"Cancelled task #{task.id}")

    def set_priority(self, task: Task, new_priority: int):
        """Change a task's priority."""
        old_priority = task.priority
        task.priority = max(1, min(10, new_priority))  # Clamp to 1-10
        self.save_task(task)
        logger.info(f"Task #{task.id} priority changed: {old_priority} → {task.priority}")

    def should_preempt(self, current_priority: int, new_priority: int) -> bool:
        """Check if new task should preempt current work.

        Preempt if:
        - New task is priority 1 (urgent), or
        - New task is at least 3 levels higher priority
        """
        if new_priority == 1:
            return True
        return (current_priority - new_priority) >= 3

    def is_duplicate(self, title: str, task_type: str) -> bool:
        """Check if this task already exists on the board."""
        normalized = title.strip().lower()
        for t in self.tasks:
            if t.status in ("queued", "active"):
                if t.title.strip().lower() == normalized and t.type == task_type:
                    return True
        return False

    def prune_if_needed(self, max_depth: int = 50):
        """Remove lowest-priority tasks if board is too full.

        Never prunes: active, paused, or captain_order tasks.
        Priority order for pruning: study > maintenance > follow_up > triggered
        """
        queued = [t for t in self.tasks if t.status == "queued"]

        if len(queued) <= max_depth:
            return

        # Sort by priority descending (lowest priority first = candidates for pruning)
        candidates = [t for t in queued if t.type != "captain_order"]
        candidates.sort(key=lambda t: (-t.priority, t.created_at))

        while len([t for t in self.tasks if t.status == "queued"]) > max_depth:
            if not candidates:
                break

            victim = candidates.pop(0)
            victim.status = "cancelled"
            self.complete_task(victim, {"summary": "Pruned: board full"})
            logger.info(f"Pruned task #{victim.id}")

    def save_task(self, task: Task, path: Optional[Path] = None):
        """Persist task to YAML file."""
        if path is None:
            path = self.tasks_dir / f"{task.id}.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(task.to_dict(), default_flow_style=False, sort_keys=False))

    def _update_active_symlink(self, task: Task):
        """Update the ACTIVE symlink to point to this task."""
        active_link = self.tasks_dir / "ACTIVE"
        active_link.unlink(missing_ok=True)
        try:
            active_link.symlink_to(f"{task.id}.yaml")
        except OSError as e:
            logger.warning(f"Could not create ACTIVE symlink: {e}")

    def _clear_active_symlink(self):
        """Remove the ACTIVE symlink."""
        active_link = self.tasks_dir / "ACTIVE"
        active_link.unlink(missing_ok=True)

    # --- Board queries ---

    def get_board(self) -> List[Task]:
        """Get all non-completed tasks, sorted by priority."""
        board = [t for t in self.tasks if t.status in ("queued", "active", "paused")]
        board.sort(key=lambda t: (t.priority, t.created_at))
        return board

    def get_active(self) -> Optional[Task]:
        """Get the currently active task."""
        for t in self.tasks:
            if t.status == "active":
                return t
        return None

    def get_task(self, task_id: int) -> Optional[Task]:
        """Get a task by ID."""
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def get_queued(self) -> List[Task]:
        """Get all queued tasks, sorted by priority."""
        queued = [t for t in self.tasks if t.status == "queued"]
        queued.sort(key=lambda t: (t.priority, t.created_at))
        return queued

    def get_completed(self, limit: int = 20) -> List[Task]:
        """Get recently completed tasks."""
        completed = [t for t in self.tasks if t.status == "completed"]
        completed.sort(key=lambda t: t.completed_at or t.created_at, reverse=True)
        return completed[:limit]

    def get_paused(self) -> List[Task]:
        """Get all paused tasks."""
        paused = [t for t in self.tasks if t.status == "paused"]
        paused.sort(key=lambda t: t.paused_at or t.created_at, reverse=True)
        return paused

    def get_by_tag(self, tag: str) -> List[Task]:
        """Get all tasks with a specific tag."""
        matching = [t for t in self.tasks if tag in t.tags]
        matching.sort(key=lambda t: (t.priority, t.created_at))
        return matching
