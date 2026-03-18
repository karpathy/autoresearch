"""ScientistAgent — Literature review and hypothesis generation.

Specializes in:
- Paper scanning and citation tracking
- Hypothesis generation from gaps
- Literature synthesis
- Research question formation
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any

from crew.agents.base import BaseAgent
from crew.messaging.bus import Message

logger = logging.getLogger(__name__)


class ScientistAgent(BaseAgent):
    """Conducts literature review and generates research hypotheses."""

    ROLE = "scientist"
    DEFAULT_PRIORITY = 4

    def get_capabilities(self) -> List[str]:
        return [
            "literature_review",
            "hypothesis_generation",
            "gap_analysis",
            "citation_tracking",
            "research_question_formation",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        """Handle literature review and hypothesis tasks."""
        if message.type == "task_request":
            action = message.payload.get("action", "review")
            if action == "review":
                return self._literature_review(message)
            elif action == "hypothesize":
                return self._generate_hypotheses(message)
            elif action == "synthesize":
                return self._synthesize_findings(message)
        return None

    def idle_work(self):
        """Background: scan for emerging research gaps."""
        import time
        if self.messages_processed % 20 == 0 and self.messages_processed > 0:
            self.logger.debug("Running idle gap analysis")
        time.sleep(15)

    # ========================================================================
    # Literature Review
    # ========================================================================

    def _literature_review(self, message: Message) -> Optional[Message]:
        """Conduct structured literature review."""
        payload = message.payload
        domain = payload.get("domain", "unknown")
        keywords = payload.get("keywords", [])
        time_window = payload.get("time_window", "5_years")

        # Simulate paper analysis
        papers = self._mock_papers(domain, keywords)

        summary = f"Literature Review: {domain}\n"
        summary += f"Time window: {time_window}\n"
        summary += f"Papers analyzed: {len(papers)}\n\n"

        consensus = {}
        gaps = []

        for i, paper in enumerate(papers[:5], 1):
            summary += f"{i}. {paper['title']}\n"
            summary += f"   Finding: {paper['finding']}\n"
            summary += f"   Confidence: {paper['confidence']}\n\n"

            key = paper["finding"][:30]
            consensus[key] = consensus.get(key, 0) + 1

        # Identify gaps
        for finding, count in consensus.items():
            if count == 1:
                gaps.append(f"Underexplored: {finding}")

        self.logger.info(f"Literature review: {len(papers)} papers, {len(gaps)} gaps")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            priority=message.priority,
            payload={
                "description": f"Literature review on {domain}",
                "summary": summary,
                "papers_analyzed": len(papers),
                "gaps_identified": gaps,
                "consensus": consensus,
                "domain": domain,
            },
        )

    def _generate_hypotheses(self, message: Message) -> Optional[Message]:
        """Generate testable hypotheses from research questions."""
        payload = message.payload
        question = payload.get("question", "")
        domain = payload.get("domain", "")
        constraints = payload.get("constraints", {})

        hypotheses = []

        # H1: Direct effect
        hypotheses.append({
            "hypothesis": f"{question} → positive effect",
            "type": "directional",
            "testable": True,
            "feasibility": "high",
        })

        # H2: Moderated effect
        if constraints.get("context"):
            hypotheses.append({
                "hypothesis": f"{question} → effect (moderated by {constraints['context']})",
                "type": "moderated",
                "testable": True,
                "feasibility": "medium",
            })

        # H3: Null hypothesis
        hypotheses.append({
            "hypothesis": f"No relationship between {question.split()[0]} and outcome",
            "type": "null",
            "testable": True,
            "feasibility": "high",
        })

        self.logger.info(f"Generated {len(hypotheses)} hypotheses for: {question}")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            priority=message.priority,
            payload={
                "description": f"Hypotheses for: {question}",
                "hypotheses": hypotheses,
                "research_question": question,
                "testable_count": sum(1 for h in hypotheses if h["testable"]),
            },
        )

    def _synthesize_findings(self, message: Message) -> Optional[Message]:
        """Synthesize multiple research findings into narrative."""
        payload = message.payload
        findings = payload.get("findings", [])
        theme = payload.get("theme", "literature synthesis")

        synthesis = f"# Synthesis: {theme}\n\n"
        synthesis += f"Analyzing {len(findings)} findings...\n\n"

        # Group findings by confidence
        high_conf = [f for f in findings if f.get("confidence", 0) > 0.7]
        medium_conf = [f for f in findings if 0.4 < f.get("confidence", 0) <= 0.7]
        low_conf = [f for f in findings if f.get("confidence", 0) <= 0.4]

        synthesis += f"## High Confidence ({len(high_conf)})\n"
        for f in high_conf[:3]:
            synthesis += f"- {f.get('insight', 'Finding')}\n"

        synthesis += f"\n## Medium Confidence ({len(medium_conf)})\n"
        for f in medium_conf[:3]:
            synthesis += f"- {f.get('insight', 'Finding')}\n"

        synthesis += f"\n## Low Confidence / Outliers ({len(low_conf)})\n"
        synthesis += "- Further investigation needed\n"

        self.logger.info(f"Synthesized {len(findings)} findings")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            priority=message.priority,
            payload={
                "description": f"Synthesis of {len(findings)} findings",
                "synthesis": synthesis,
                "high_confidence": len(high_conf),
                "medium_confidence": len(medium_conf),
                "low_confidence": len(low_conf),
            },
        )

    # ========================================================================
    # Helpers
    # ========================================================================

    def _mock_papers(self, domain: str, keywords: List[str]) -> List[Dict[str, Any]]:
        """Mock paper retrieval (would call PubMed/arXiv in production)."""
        return [
            {
                "title": f"Paper on {domain} with keywords {keywords[0] if keywords else 'general'}",
                "year": 2024,
                "finding": "Novel finding about the domain",
                "confidence": 0.85,
                "citations": 42,
            },
            {
                "title": f"Comparative study in {domain}",
                "year": 2023,
                "finding": "Effect sizes vary by context",
                "confidence": 0.72,
                "citations": 28,
            },
            {
                "title": f"Systematic review of {domain}",
                "year": 2022,
                "finding": "Consensus emerging on mechanism",
                "confidence": 0.91,
                "citations": 156,
            },
            {
                "title": f"Exploratory work in {domain}",
                "year": 2024,
                "finding": "Unexpected result that contradicts prior work",
                "confidence": 0.55,
                "citations": 3,
            },
            {
                "title": f"Meta-analysis of {domain}",
                "year": 2023,
                "finding": "Publication bias detected",
                "confidence": 0.68,
                "citations": 89,
            },
        ]
