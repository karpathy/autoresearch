"""Distiller agent — synthesizes and compresses knowledge.

The distiller is the crew's knowledge compaction specialist. It:
  1. Takes multiple knowledge entries and synthesizes them into a unified insight
  2. Detects redundant or contradictory entries and resolves them
  3. Generates summary reports (weekly, monthly)
  4. Builds LoRA-ready training datasets from the knowledge base
  5. Creates conceptual maps (tag clusters, confidence distributions)

Runs at lowest priority — distillation is background work.
Triggered by:
  - "distill" task_request messages
  - Periodic idle work (every N messages processed)
  - Explicit trigger from daemon's weekly GC pass
"""

import json
import logging
import re
from typing import Optional, List, Dict, Any

from crew.agents.base import BaseAgent
from crew.messaging.bus import Message

logger = logging.getLogger(__name__)


class DistillerAgent(BaseAgent):
    """Agent that synthesizes knowledge and builds training datasets."""

    ROLE = "distiller"
    DEFAULT_PRIORITY = 9  # Lowest priority — background work

    # Distill every N messages processed in idle loop
    IDLE_DISTILL_EVERY = 50

    def get_capabilities(self) -> List[str]:
        return [
            "knowledge_synthesis",
            "redundancy_detection",
            "dataset_builder",
            "summary_generation",
            "tag_clustering",
        ]

    def process_message(self, message: Message) -> Optional[Message]:
        """Process a distillation request.

        - task_request with action=distill: Synthesize entries by tag
        - task_request with action=build_dataset: Build LoRA training JSONL
        - task_request with action=summarize: Generate knowledge summary
        - knowledge: Track for later batch distillation
        """
        if message.type == "task_request":
            action = message.payload.get("action", "distill")
            if action == "build_dataset":
                return self._build_dataset(message)
            elif action == "summarize":
                return self._summarize_knowledge(message)
            else:
                return self._distill_entries(message)
        elif message.type == "knowledge":
            # Passively track new knowledge — distill in idle
            return None
        return None

    def idle_work(self):
        """Periodically run a background distillation pass."""
        import time
        if self.messages_processed % self.IDLE_DISTILL_EVERY == 0 and self.messages_processed > 0:
            self._background_distill_pass()
        time.sleep(15)

    # ========================================================================
    # Distillation
    # ========================================================================

    def _distill_entries(self, message: Message) -> Optional[Message]:
        """Synthesize knowledge entries by tag into a unified insight."""
        payload = message.payload
        tags = payload.get("tags", [])
        limit = payload.get("limit", 20)
        topic = payload.get("topic", ", ".join(tags) if tags else "all knowledge")

        try:
            from crew.knowledge.store import KnowledgeStore
            store = KnowledgeStore()
            entries = store.query(tags=tags, limit=limit)
        except Exception as e:
            self.logger.warning(f"Could not query knowledge store: {e}")
            return None

        if not entries:
            return None

        self.logger.info(f"Distilling {len(entries)} entries for topic: {topic}")

        # Group by tag
        tag_groups: Dict[str, List] = {}
        for entry in entries:
            for tag in (entry.tags or ["untagged"]):
                tag_groups.setdefault(tag, []).append(entry)

        # Synthesize with LLM or heuristic
        synthesis = self._synthesize_entries(topic, entries)

        if not synthesis:
            return None

        # Store the synthesized insight back as high-confidence knowledge
        if synthesis.get("unified_insight"):
            try:
                from crew.knowledge.store import KnowledgeStore
                store = KnowledgeStore()
                entry_id = store.add(
                    insight=synthesis["unified_insight"],
                    category="distilled",
                    tags=tags or ["distilled"],
                    confidence="medium",
                    evidence={
                        "source": "distiller_agent",
                        "source_entries": len(entries),
                        "topic": topic,
                    },
                    source_agent=self.agent_id,
                )
                self.logger.info(f"Stored distilled insight as entry {entry_id}")
            except Exception as e:
                self.logger.warning(f"Could not store distilled insight: {e}")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            priority=message.priority,
            payload={
                "description": f"Distillation complete: {topic}",
                "summary": synthesis.get("summary", ""),
                "unified_insight": synthesis.get("unified_insight", ""),
                "contradictions": synthesis.get("contradictions", []),
                "redundant_ids": synthesis.get("redundant_ids", []),
                "entries_processed": len(entries),
                "topic": topic,
            },
            tags=message.tags,
        )

    def _build_dataset(self, message: Message) -> Optional[Message]:
        """Build a LoRA-ready JSONL training file from the knowledge base.

        Output format: one JSON object per line with:
          {"instruction": "...", "input": "", "output": "..."}
        """
        from pathlib import Path
        import gzip

        payload = message.payload
        tags = payload.get("tags", [])
        output_path = Path(payload.get("output_path", "data/training/lora_dataset.jsonl"))
        limit = payload.get("limit", 500)
        min_confidence = payload.get("min_confidence", "medium")

        try:
            from crew.knowledge.store import KnowledgeStore
            store = KnowledgeStore()
            entries = store.query(
                tags=tags,
                min_confidence=min_confidence,
                limit=limit,
            )
        except Exception as e:
            self.logger.warning(f"Could not query knowledge store: {e}")
            return None

        if not entries:
            self.logger.info("No entries found for dataset build")
            return None

        self.logger.info(f"Building dataset from {len(entries)} entries")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with output_path.open("w") as f:
            for entry in entries:
                # Turn each knowledge insight into instruction-response pairs
                pairs = self._entry_to_training_pairs(entry)
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")
                    count += 1

        self.logger.info(f"Dataset built: {count} examples → {output_path}")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            priority=message.priority,
            payload={
                "description": "LoRA dataset built",
                "output_path": str(output_path),
                "examples_count": count,
                "entries_used": len(entries),
                "tags": tags,
            },
            tags=message.tags,
        )

    def _summarize_knowledge(self, message: Message) -> Optional[Message]:
        """Generate a markdown summary of the knowledge base."""
        from pathlib import Path
        from datetime import datetime

        payload = message.payload
        tags = payload.get("tags", [])
        period = payload.get("period", "recent")  # recent|weekly|monthly

        try:
            from crew.knowledge.store import KnowledgeStore
            store = KnowledgeStore()
            entries = store.query(tags=tags, limit=200)
        except Exception as e:
            self.logger.warning(f"Could not query store: {e}")
            return None

        if not entries:
            return None

        # Categorize
        by_category: Dict[str, List] = {}
        for entry in entries:
            by_category.setdefault(entry.category or "other", []).append(entry)

        lines = [
            f"# Knowledge Summary — {period.title()} ({datetime.now().strftime('%Y-%m-%d')})",
            "",
            f"Total entries: {len(entries)}",
            "",
        ]

        for category, cat_entries in sorted(by_category.items()):
            lines.append(f"## {category.title()} ({len(cat_entries)} entries)")
            lines.append("")
            high_conf = [e for e in cat_entries if e.confidence in ("high", "very_high")]
            for entry in high_conf[:5]:
                lines.append(f"- **[{entry.confidence}]** {entry.insight[:120]}")
            if len(cat_entries) > 5:
                lines.append(f"  *(+{len(cat_entries) - 5} more)*")
            lines.append("")

        summary_text = "\n".join(lines)

        # Save to disk
        now = datetime.now()
        if period == "weekly":
            week = now.isocalendar()[1]
            fname = f"data/summaries/weekly/{now.year}-W{week:02d}.md"
        elif period == "monthly":
            fname = f"data/summaries/monthly/{now.year}-{now.month:02d}.md"
        else:
            fname = f"data/summaries/recent-{now.strftime('%Y%m%d')}.md"

        summary_path = Path(fname)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text)

        self.logger.info(f"Summary written: {summary_path}")

        return Message(
            from_agent=self.agent_id,
            to_agent=message.from_agent,
            type="result",
            priority=message.priority,
            payload={
                "description": f"Knowledge summary: {period}",
                "summary_path": str(summary_path),
                "entries_count": len(entries),
                "period": period,
            },
            tags=message.tags,
        )

    # ========================================================================
    # Synthesis Helpers
    # ========================================================================

    def _synthesize_entries(
        self,
        topic: str,
        entries: List,
    ) -> Optional[Dict[str, Any]]:
        """Synthesize multiple entries into a unified insight via LLM or heuristics."""
        if not entries:
            return None

        # Build context from entries
        entry_texts = []
        for e in entries[:10]:  # Cap at 10 to avoid token overflow
            entry_texts.append(
                f"[{e.confidence}] {e.insight}"
            )
        context = "\n".join(entry_texts)

        # Try LLM
        if self.check_rate_limit("llm_call"):
            result = self._llm_synthesize(topic, context)
            if result:
                self.consume_rate("llm_call")
                return result

        # Heuristic fallback: take the highest-confidence, most recent entry
        best = max(
            entries,
            key=lambda e: (
                {"very_high": 4, "high": 3, "medium": 2, "low": 1}.get(e.confidence, 0)
            ),
        )
        return {
            "summary": f"Top insight on {topic}: {best.insight[:200]}",
            "unified_insight": best.insight,
            "contradictions": [],
            "redundant_ids": [],
        }

    def _llm_synthesize(self, topic: str, context: str) -> Optional[Dict[str, Any]]:
        """Call LLM to synthesize knowledge entries."""
        import os
        import urllib.request

        prompt = f"""You are a knowledge distiller. Given these knowledge entries about "{topic}", synthesize them into:
1. A single unified insight (2-3 sentences)
2. Any contradictions between entries
3. IDs of redundant/duplicate entries (if any)

Entries:
{context}

Return JSON:
{{"summary": "...", "unified_insight": "...", "contradictions": ["..."], "redundant_ids": []}}"""

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("llm", {}).get("api_key")
        if not anthropic_key:
            return None

        try:
            data = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=data,
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                text = result["content"][0]["text"]
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

        except Exception as e:
            self.logger.debug(f"LLM synthesis failed: {e}")

        return None

    def _entry_to_training_pairs(self, entry) -> List[Dict[str, str]]:
        """Convert a knowledge entry into instruction-response training pairs."""
        pairs = []
        insight = entry.insight

        if not insight or len(insight) < 20:
            return pairs

        # Pair 1: Direct explanation
        pairs.append({
            "instruction": f"Explain what is known about {', '.join(entry.tags[:2]) if entry.tags else 'this topic'}.",
            "input": "",
            "output": insight,
        })

        # Pair 2: Confidence framing
        if entry.confidence in ("high", "very_high"):
            pairs.append({
                "instruction": f"What is a high-confidence finding regarding {', '.join(entry.tags[:2]) if entry.tags else 'this topic'}?",
                "input": "",
                "output": f"A well-supported finding: {insight}",
            })

        # Pair 3: Category context
        if entry.category and entry.category != "other":
            pairs.append({
                "instruction": f"What does research show about {entry.category} in this context?",
                "input": "",
                "output": insight,
            })

        return pairs[:2]  # Max 2 pairs per entry to keep dataset focused

    def _background_distill_pass(self):
        """Lightweight background pass: synthesize recent low-priority entries."""
        self.logger.debug("Running background distillation pass")
        try:
            from crew.knowledge.store import KnowledgeStore
            store = KnowledgeStore()
            # Get recent medium-confidence entries needing synthesis
            entries = store.query(min_confidence="medium", limit=30)
            if len(entries) >= 10:
                self.logger.info(f"Background distill: {len(entries)} entries available")
                # Publish a self-distill task
                self.publish(
                    to_agent=self.agent_id,
                    msg_type="task_request",
                    payload={
                        "action": "distill",
                        "topic": "recent knowledge",
                        "limit": 20,
                    },
                    priority=self.DEFAULT_PRIORITY,
                )
        except Exception as e:
            self.logger.debug(f"Background distill pass failed: {e}")
