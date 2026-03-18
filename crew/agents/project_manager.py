"""ProjectManagerAgent — Task tracking and project planning.

Specializes in:
- Task decomposition
- Timeline management
- Dependency tracking
- Progress monitoring
"""

from typing import Optional, List
from crew.agents.base import BaseAgent
from crew.messaging.bus import Message
import logging

logger = logging.getLogger(__name__)


class ProjectManagerAgent(BaseAgent):
    """Project planning and task management."""

    ROLE = "project_manager"
    DEFAULT_PRIORITY = 3

    def get_capabilities(self) -> List[str]:
        return [
            "task_decomposition",
            "timeline_planning",
            "dependency_tracking",
            "progress_monitoring",
            "risk_assessment",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == "task_request":
            action = message.payload.get("action", "plan")
            if action == "plan":
                return self._create_plan(message)
            elif action == "decompose":
                return self._decompose_task(message)
            elif action == "assess_risks":
                return self._assess_risks(message)
        return None

    def _create_plan(self, message: Message) -> Optional[Message]:
        """Create project plan."""
        project = message.payload.get("project", "Unknown Project")
        objectives = message.payload.get("objectives", [])
        duration = message.payload.get("duration", "3 months")

        plan = {
            "project": project,
            "timeline": duration,
            "phases": [
                {
                    "phase": "Planning",
                    "duration": "1 week",
                    "tasks": ["Define scope", "Identify resources", "Set milestones"],
                },
                {
                    "phase": "Execution",
                    "duration": "2 weeks",
                    "tasks": ["Implement core", "Build components", "Integrate"],
                },
                {
                    "phase": "Testing",
                    "duration": "1 week",
                    "tasks": ["Test functionality", "Fix issues", "Optimize"],
                },
                {
                    "phase": "Delivery",
                    "duration": "3 days",
                    "tasks": ["Final review", "Deploy", "Documentation"],
                },
            ],
        }

        logger.info(f"Created plan for {project} ({duration})")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "plan": plan,
                "project": project,
                "total_tasks": sum(len(p["tasks"]) for p in plan["phases"]),
                "phases": len(plan["phases"]),
            },
        )

    def _decompose_task(self, message: Message) -> Optional[Message]:
        """Break down a large task into subtasks."""
        task = message.payload.get("task", "Unknown Task")
        complexity = message.payload.get("complexity", "medium")

        subtasks = []
        if "research" in task.lower():
            subtasks = ["Literature review", "Data collection", "Analysis", "Synthesis"]
        elif "build" in task.lower() or "develop" in task.lower():
            subtasks = ["Design", "Implementation", "Testing", "Documentation"]
        elif "report" in task.lower():
            subtasks = ["Gather data", "Analyze", "Write", "Edit", "Format"]
        else:
            subtasks = ["Planning", "Execution", "Review", "Completion"]

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "task": task,
                "subtasks": subtasks,
                "task_count": len(subtasks),
                "complexity": complexity,
            },
        )

    def _assess_risks(self, message: Message) -> Optional[Message]:
        """Identify and assess project risks."""
        project = message.payload.get("project", "Project")
        duration = message.payload.get("duration", "3 months")

        risks = [
            {
                "risk": "Resource constraints",
                "probability": "medium",
                "impact": "high",
                "mitigation": "Allocate buffer time",
            },
            {
                "risk": "Scope creep",
                "probability": "high",
                "impact": "high",
                "mitigation": "Strict change control",
            },
            {
                "risk": "Technical complexity",
                "probability": "medium",
                "impact": "medium",
                "mitigation": "Prototype early",
            },
            {
                "risk": "Timeline pressure",
                "probability": "medium",
                "impact": "medium",
                "mitigation": "Regular monitoring",
            },
        ]

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "project": project,
                "risks": risks,
                "high_risk_count": sum(1 for r in risks if r["impact"] == "high"),
                "mitigation_strategies": len(risks),
            },
        )

    def idle_work(self):
        import time
        time.sleep(15)
