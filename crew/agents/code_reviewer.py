"""CodeReviewerAgent — Code analysis and review automation.

Specializes in:
- Static code analysis
- Best practice detection
- Security patterns
- Performance antipatterns
"""

from typing import Optional, List, Dict, Any
from crew.agents.base import BaseAgent
from crew.messaging.bus import Message
import logging

logger = logging.getLogger(__name__)


class CodeReviewerAgent(BaseAgent):
    """Automated code review and quality checking."""

    ROLE = "code_reviewer"
    DEFAULT_PRIORITY = 5

    def get_capabilities(self) -> List[str]:
        return [
            "static_analysis",
            "style_checking",
            "complexity_analysis",
            "security_patterns",
            "performance_hints",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == "task_request":
            action = message.payload.get("action", "review")
            if action == "review":
                return self._review_code(message)
            elif action == "analyze_complexity":
                return self._analyze_complexity(message)
        return None

    def _review_code(self, message: Message) -> Optional[Message]:
        """Analyze code and return review comments."""
        code = message.payload.get("code", "")
        language = message.payload.get("language", "python")
        standards = message.payload.get("standards", {})

        issues = []
        suggestions = []

        # Heuristic checks
        if len(code.split('\n')) > 100:
            suggestions.append({
                "type": "complexity",
                "severity": "medium",
                "message": "Function is long (>100 lines). Consider breaking it up.",
            })

        if code.count("TODO") > 0:
            issues.append({
                "type": "incomplete",
                "severity": "low",
                "message": f"Found {code.count('TODO')} TODO markers",
            })

        if "pass" in code and code.count("def ") > code.count("pass"):
            issues.append({
                "type": "incomplete",
                "severity": "medium",
                "message": "Some functions may be stubs (contain only 'pass')",
            })

        logger.info(f"Code review: {len(issues)} issues, {len(suggestions)} suggestions")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "issues": issues,
                "suggestions": suggestions,
                "language": language,
                "overall_quality": "good" if len(issues) < 3 else "needs_work",
            },
        )

    def _analyze_complexity(self, message: Message) -> Optional[Message]:
        """Analyze cyclomatic complexity and other metrics."""
        code = message.payload.get("code", "")

        # Simple heuristics
        nesting_level = max(len(line) - len(line.lstrip()) for line in code.split('\n')) // 4
        if_count = code.count("if ")
        loop_count = code.count("for ") + code.count("while ")

        complexity_score = if_count + loop_count + (nesting_level * 2)

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "complexity_score": min(complexity_score, 100),
                "nesting_level": nesting_level,
                "conditionals": if_count,
                "loops": loop_count,
                "assessment": "high" if complexity_score > 20 else "moderate" if complexity_score > 10 else "low",
            },
        )

    def idle_work(self):
        import time
        time.sleep(15)
