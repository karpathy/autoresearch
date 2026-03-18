"""SecurityAgent — Threat analysis and security assessment.

Specializes in:
- Vulnerability scanning
- Attack vector identification
- Risk scoring
- Mitigation recommendation
"""

from typing import Optional, List
from crew.agents.base import BaseAgent
from crew.messaging.bus import Message
import logging

logger = logging.getLogger(__name__)


class SecurityAgent(BaseAgent):
    """Security threat analysis and assessment."""

    ROLE = "security"
    DEFAULT_PRIORITY = 3

    def get_capabilities(self) -> List[str]:
        return [
            "vulnerability_scanning",
            "threat_modeling",
            "risk_assessment",
            "mitigation_planning",
            "security_review",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == "task_request":
            action = message.payload.get("action", "scan")
            if action == "scan":
                return self._scan_threats(message)
            elif action == "assess_risk":
                return self._assess_risk(message)
        return None

    def _scan_threats(self, message: Message) -> Optional[Message]:
        """Scan for security threats."""
        code = message.payload.get("code", "")
        threat_model = message.payload.get("threat_model", "owasp")

        threats = []

        # Heuristic checks
        if "eval(" in code or "exec(" in code:
            threats.append({
                "type": "code_injection",
                "cwe": "CWE-95",
                "severity": "critical",
                "line": code.find("eval("),
                "message": "Dangerous eval/exec usage detected",
                "mitigation": "Use safer alternatives or strict input validation",
            })

        if "SELECT" in code.upper() and "WHERE" in code.upper() and "'" in code:
            threats.append({
                "type": "sql_injection",
                "cwe": "CWE-89",
                "severity": "critical",
                "message": "Potential SQL injection vulnerability",
                "mitigation": "Use parameterized queries",
            })

        if "password" in code.lower() and "plain" in code.lower():
            threats.append({
                "type": "hardcoded_credentials",
                "cwe": "CWE-798",
                "severity": "high",
                "message": "Potential hardcoded credentials",
                "mitigation": "Use environment variables or secrets management",
            })

        if code.count("try") < code.count("except") / 3:
            threats.append({
                "type": "unhandled_exceptions",
                "severity": "medium",
                "message": "Limited exception handling",
                "mitigation": "Add proper error handling",
            })

        logger.info(f"Security scan: {len(threats)} threats found")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "threats": threats,
                "critical_count": sum(1 for t in threats if t["severity"] == "critical"),
                "threat_model": threat_model,
                "overall_risk": "high" if sum(1 for t in threats if t["severity"] in ["critical", "high"]) > 0 else "medium" if len(threats) > 0 else "low",
            },
        )

    def _assess_risk(self, message: Message) -> Optional[Message]:
        """Assess risk level."""
        likelihood = message.payload.get("likelihood", "medium")
        impact = message.payload.get("impact", "medium")

        likelihood_score = {"low": 1, "medium": 2, "high": 3}.get(likelihood, 2)
        impact_score = {"low": 1, "medium": 2, "high": 3}.get(impact, 2)

        risk_score = likelihood_score * impact_score

        risk_level = "critical" if risk_score >= 6 else "high" if risk_score >= 4 else "medium" if risk_score >= 2 else "low"

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "risk_score": risk_score,
                "risk_level": risk_level,
                "likelihood": likelihood,
                "impact": impact,
                "requires_action": risk_score >= 4,
            },
        )

    def idle_work(self):
        import time
        time.sleep(15)
