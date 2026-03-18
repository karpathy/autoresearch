"""ConsistencyAgent — Rule enforcement and constraint checking.

Specializes in:
- Rule validation
- Contradiction detection
- Coherence checking
- Version consistency
"""

from typing import Optional, List, Dict
from crew.agents.base import BaseAgent
from crew.messaging.bus import Message
import logging

logger = logging.getLogger(__name__)


class ConsistencyAgent(BaseAgent):
    """Enforces consistency rules and constraints."""

    ROLE = "consistency"
    DEFAULT_PRIORITY = 5

    def get_capabilities(self) -> List[str]:
        return [
            "rule_validation",
            "contradiction_detection",
            "coherence_checking",
            "version_consistency",
            "constraint_enforcement",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == "task_request":
            action = message.payload.get("action", "check")
            if action == "check":
                return self._check_consistency(message)
            elif action == "validate_rules":
                return self._validate_rules(message)
        return None

    def _check_consistency(self, message: Message) -> Optional[Message]:
        """Check for contradictions and inconsistencies."""
        entries = message.payload.get("entries", [])
        rules = message.payload.get("rules", {})

        violations = []
        warnings = []

        # Check internal consistency
        for i, entry in enumerate(entries):
            if isinstance(entry, dict):
                # Check for contradictory fields
                if entry.get("confidence") == "high" and entry.get("evidence_count", 0) == 0:
                    violations.append({
                        "entry": i,
                        "type": "confidence_without_evidence",
                        "severity": "high",
                        "message": "High confidence but no supporting evidence",
                    })

                # Check for missing required fields
                required = rules.get("required_fields", [])
                for field in required:
                    if field not in entry:
                        warnings.append({
                            "entry": i,
                            "type": "missing_field",
                            "field": field,
                            "severity": "medium",
                        })

        # Cross-entry consistency
        if len(entries) > 1:
            for i in range(len(entries) - 1):
                e1 = entries[i]
                e2 = entries[i + 1]
                if e1.get("timestamp") and e2.get("timestamp"):
                    if e1["timestamp"] > e2["timestamp"]:
                        warnings.append({
                            "entries": [i, i + 1],
                            "type": "temporal_inconsistency",
                            "message": "Timestamps out of order",
                        })

        logger.info(f"Consistency check: {len(violations)} violations, {len(warnings)} warnings")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "violations": violations,
                "warnings": warnings,
                "entries_checked": len(entries),
                "consistency_score": max(0, 100 - len(violations) * 20 - len(warnings) * 5),
            },
        )

    def _validate_rules(self, message: Message) -> Optional[Message]:
        """Validate content against rules."""
        content = message.payload.get("content", {})
        rules = message.payload.get("rules", {})

        violations = []

        for rule_name, rule_spec in rules.items():
            if rule_name == "max_length" and "text" in content:
                if len(content["text"]) > rule_spec:
                    violations.append({
                        "rule": rule_name,
                        "violation": f"Text length {len(content['text'])} > {rule_spec}",
                    })

            elif rule_name == "required_keywords":
                text = content.get("text", "").lower()
                for keyword in rule_spec:
                    if keyword.lower() not in text:
                        violations.append({
                            "rule": rule_name,
                            "violation": f"Missing required keyword: {keyword}",
                        })

            elif rule_name == "min_confidence" and "confidence" in content:
                if content["confidence"] < rule_spec:
                    violations.append({
                        "rule": rule_name,
                        "violation": f"Confidence {content['confidence']} < {rule_spec}",
                    })

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "violations": violations,
                "rules_checked": len(rules),
                "passes_validation": len(violations) == 0,
            },
        )

    def idle_work(self):
        import time
        time.sleep(15)
