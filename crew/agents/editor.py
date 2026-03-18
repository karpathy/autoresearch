"""EditorAgent — Content refinement and style consistency.

Specializes in:
- Grammar and style checking
- Tone consistency
- Readability improvement
- Citation formatting
"""

from typing import Optional, List
from crew.agents.base import BaseAgent
from crew.messaging.bus import Message
import logging

logger = logging.getLogger(__name__)


class EditorAgent(BaseAgent):
    """Content refinement and editing."""

    ROLE = "editor"
    DEFAULT_PRIORITY = 6

    def get_capabilities(self) -> List[str]:
        return [
            "grammar_checking",
            "style_consistency",
            "readability_improvement",
            "tone_analysis",
            "citation_formatting",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == "task_request":
            action = message.payload.get("action", "edit")
            if action == "edit":
                return self._edit_content(message)
            elif action == "analyze_tone":
                return self._analyze_tone(message)
        return None

    def _edit_content(self, message: Message) -> Optional[Message]:
        """Edit and refine content."""
        text = message.payload.get("text", "")
        style_guide = message.payload.get("style_guide", "academic")

        issues = []
        suggestions = []

        # Heuristic checks
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 0 and len(paragraphs[0].split()) > 50:
            issues.append({
                "type": "readability",
                "severity": "low",
                "line": 1,
                "message": "Opening paragraph is dense. Break it up.",
            })

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            suggestions.append({
                "type": "style",
                "message": f"Found {len(long_sentences)} sentences >25 words. Consider breaking them up.",
            })

        # Passive voice detection (simple heuristic)
        passive_markers = text.count(" was ") + text.count(" were ") + text.count(" been ")
        if passive_markers > 5:
            suggestions.append({
                "type": "style",
                "message": f"High passive voice usage ({passive_markers}). Use active voice more.",
            })

        # Repetition detection
        words = text.lower().split()
        if words:
            word_freq = {}
            for word in words:
                if len(word) > 5:
                    word_freq[word] = word_freq.get(word, 0) + 1
            repeated = [w for w, c in word_freq.items() if c > 5]
            if repeated:
                suggestions.append({
                    "type": "style",
                    "message": f"Words repeated frequently: {', '.join(repeated[:3])}",
                })

        logger.info(f"Edited content: {len(issues)} issues, {len(suggestions)} suggestions")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "issues": issues,
                "suggestions": suggestions,
                "style_guide": style_guide,
                "readability_score": max(0, 100 - len(issues) * 10),
            },
        )

    def _analyze_tone(self, message: Message) -> Optional[Message]:
        """Analyze tone and register of text."""
        text = message.payload.get("text", "")
        target_tone = message.payload.get("target_tone", "neutral")

        # Simple heuristic tone analysis
        exclamations = text.count("!")
        questions = text.count("?")
        caps = sum(1 for c in text if c.isupper())

        formal_words = sum(1 for word in text.lower().split() if word in [
            "furthermore", "moreover", "hence", "thus", "consequently"
        ])
        casual_words = sum(1 for word in text.lower().split() if word in [
            "yeah", "gonna", "kinda", "like", "basically"
        ])

        if formal_words > casual_words:
            detected_tone = "formal"
        elif casual_words > formal_words:
            detected_tone = "casual"
        else:
            detected_tone = "neutral"

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "detected_tone": detected_tone,
                "target_tone": target_tone,
                "matches_target": detected_tone == target_tone or target_tone == "neutral",
                "energy_level": "high" if (exclamations + questions) > 5 else "low",
                "formal_markers": formal_words,
                "casual_markers": casual_words,
            },
        )

    def idle_work(self):
        import time
        time.sleep(15)
