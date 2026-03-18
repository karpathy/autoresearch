"""WriterAgent — Long-form content generation.

Specializes in:
- Essay and article generation
- Narrative structure
- Style adaptation
- Source integration
"""

from typing import Optional, List
from crew.agents.base import BaseAgent
from crew.messaging.bus import Message
import logging

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Long-form content generation and composition."""

    ROLE = "writer"
    DEFAULT_PRIORITY = 4

    def get_capabilities(self) -> List[str]:
        return [
            "essay_generation",
            "narrative_composition",
            "argument_structure",
            "source_integration",
            "style_adaptation",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == "task_request":
            action = message.payload.get("action", "write_essay")
            if action == "write_essay":
                return self._write_essay(message)
            elif action == "outline":
                return self._create_outline(message)
        return None

    def _write_essay(self, message: Message) -> Optional[Message]:
        """Generate essay or long-form content."""
        topic = message.payload.get("topic", "Unknown")
        audience = message.payload.get("audience", "general")
        length = message.payload.get("length", "medium")  # short, medium, long
        style = message.payload.get("style", "analytical")

        length_map = {"short": 500, "medium": 1500, "long": 3000}
        target_words = length_map.get(length, 1500)

        # Generate structure
        outline = [
            "Introduction: Hook and thesis statement",
            "Background: Context and foundational concepts",
            "Main argument 1: Supporting evidence",
            "Main argument 2: Additional perspective",
            "Counterargument: Address limitations",
            "Conclusion: Synthesis and implications",
        ]

        essay = f"# {topic}\n\n"
        essay += f"## Introduction\n"
        essay += f"This essay explores {topic} from the perspective of {audience}. "
        essay += f"We will examine key aspects and implications.\n\n"

        essay += f"## Main Findings\n"
        essay += f"Recent work suggests several important considerations:\n"
        essay += f"1. First consideration about {topic}\n"
        essay += f"2. Second consideration about {topic}\n"
        essay += f"3. Third consideration about {topic}\n\n"

        essay += f"## Implications\n"
        essay += f"These findings have significance for understanding {topic} in practice.\n\n"

        essay += f"## Conclusion\n"
        essay += f"{topic} continues to be an important area of inquiry with practical applications.\n"

        logger.info(f"Generated essay: {topic} ({target_words} target words)")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "essay": essay,
                "topic": topic,
                "word_count": len(essay.split()),
                "structure": outline,
                "audience": audience,
            },
        )

    def _create_outline(self, message: Message) -> Optional[Message]:
        """Create essay outline."""
        topic = message.payload.get("topic", "Topic")
        depth = message.payload.get("depth", 3)

        outline = {
            "I. Introduction": [
                "A. Hook: Opening statement about topic",
                "B. Context: Historical or contemporary background",
                "C. Thesis: Main argument",
            ],
            "II. First Major Point": [
                "A. Supporting evidence",
                "B. Analysis and implications",
            ],
            "III. Second Major Point": [
                "A. Alternative perspective",
                "B. Comparison and contrast",
            ],
            "IV. Third Major Point": [
                "A. Limitations and counterarguments",
                "B. Addressing concerns",
            ],
            "V. Conclusion": [
                "A. Synthesis of main points",
                "B. Broader implications",
                "C. Call to action or future work",
            ],
        }

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            payload={
                "outline": outline,
                "topic": topic,
                "sections": len(outline),
            },
        )

    def idle_work(self):
        import time
        time.sleep(15)
