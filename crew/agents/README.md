# Agent Ecosystem

AutoCrew's multi-agent system includes 30+ specialized agents that collaborate through the message bus. Each agent has a specific role, focuses on particular message types, and contributes to the knowledge base.

---

## Agent Registry (30+ Agents)

### 🔍 Research & Analysis (6 agents)

| Agent | Role | Capabilities | Input | Output |
|-------|------|--------------|-------|--------|
| **Researcher** | Web exploration & synthesis | Web search, URL fetch, LLM synthesis, heuristic fallback | Research task, topic query | Findings, sources, confidence score |
| **ScientistAgent** | Literature review & hypotheses | Paper scanning, citation tracking, hypothesis generation | Research domain, keywords | Structured literature summary, open questions |
| **StatisticianAgent** | Data analysis & validation | Statistical testing, outlier detection, significance analysis | Dataset, analysis type | Validated statistics, p-values, effect sizes |
| **FactCheckerAgent** | Claim verification | Cross-reference validation, source checking, contradiction detection | Claim, context | Verification result, supporting/contradicting evidence |
| **DataValidatorAgent** | Quality assessment | Schema validation, completeness check, anomaly detection | Dataset, schema | Quality report, anomalies, recommendations |
| **ArchivistAgent** | Document organization | Indexing, tagging, version control, metadata extraction | Documents, taxonomy | Organized archive, cross-references, search index |

### ✍️ Creative & Content (7 agents)

| Agent | Role | Capabilities | Input | Output |
|-------|------|--------------|-------|--------|
| **WriterAgent** | Long-form content | Essay generation, narrative flow, style adaptation | Topic, target audience, length | Polished prose, citations, outline |
| **EditorAgent** | Content refinement | Grammar/style checking, tone consistency, readability | Raw text, style guide | Edited version, suggestions, feedback |
| **StorytellerAgent** | Narrative generation | Plot branching, character arcs, pacing | Story premise, constraints | Story outline, scene descriptions |
| **WorldBuilderAgent** | Setting/lore creation | Consistency checking, map generation, rule synthesis | World concept, constraints | World bible, timelines, faction charts |
| **CharacterDevAgent** | Character consistency | Personality tracking, arc development, dialogue authenticity | Character bio, interactions | Character sheet, behavior model, voice pattern |
| **DialogueAgent** | Conversation generation | Character voice, natural language, subtext | Characters, scene context | Dialogue with stage directions |
| **TranslatorAgent** | Multi-language support | Translation, localization, cultural adaptation | Text, target language | Translated version, notes on idioms |

### 📚 Education & Learning (5 agents)

| Agent | Role | Capabilities | Input | Output |
|-------|------|--------------|-------|--------|
| **TeacherAgent** | Training data generation | Q&A pairs, instruction-response, example generation | Knowledge, topic | Training dataset (JSONL), exercises |
| **TutorAgent** | Concept explanation | Level-appropriate teaching, prerequisite tracking | Concept, learner level | Explanation, examples, analogies |
| **MentorAgent** | Personalized guidance | Progress tracking, gap analysis, path recommendation | Learner profile, goal | Personalized curriculum, feedback |
| **AssessmentAgent** | Knowledge evaluation | Quiz generation, grading, skill identification | Learning objectives | Assessment, score, skill gaps |
| **CurriculumAgent** | Learning path design | Prerequisite mapping, difficulty progression, pacing | Domain, target level | Learning roadmap, milestones |

### 💻 Software & Engineering (5 agents)

| Agent | Role | Capabilities | Input | Output |
|-------|------|--------------|-------|--------|
| **CodeReviewerAgent** | Code analysis | Static analysis, pattern detection, best practices | Code, standards | Review comments, suggestions, severity |
| **ArchitectAgent** | System design | Architecture evaluation, scalability analysis, tradeoff analysis | Requirements, constraints | Architecture diagram, decisions, rationale |
| **SecurityAgent** | Threat analysis | Vulnerability scanning, attack vector identification, risk scoring | Code/design, threat model | Security report, vulnerabilities, mitigations |
| **PerformanceAgent** | Optimization detection | Bottleneck identification, benchmark analysis, profiling | Metrics, code | Optimization opportunities, estimated impact |
| **APIDocumentationAgent** | API documentation | Doc generation, example creation, schema validation | API, code | API docs, examples, schema |

### 🎯 Business & Strategy (4 agents)

| Agent | Role | Capabilities | Input | Output |
|-------|------|--------------|-------|--------|
| **ProjectManagerAgent** | Task tracking | Task decomposition, timeline management, dependency tracking | Project, goals | Task board, timeline, risk assessment |
| **StrategyAgent** | Strategic planning | Scenario analysis, goal decomposition, decision analysis | Business context, constraints | Strategic plan, options analysis, roadmap |
| **CompetitorIntelligenceAgent** | Market monitoring | Competitor tracking, trend analysis, opportunity identification | Market, competitors | Market report, trends, opportunities |
| **SynthesisAgent** | Report generation | Data compilation, narrative synthesis, recommendation formulation | Data, audience | Executive summary, insights, recommendations |

### ✅ Quality & Consistency (5 agents)

| Agent | Role | Capabilities | Input | Output |
|-------|------|--------------|-------|--------|
| **CriticAgent** | Quality checking | Fact-checking, logic validation, consistency review | Claims, evidence | Quality score, issues, confidence adjustment |
| **DistillerAgent** | Knowledge synthesis | Entry synthesis, redundancy detection, summary generation | Multiple entries, topic | Unified insight, summary, LoRA dataset |
| **ConsistencyAgent** | Rule enforcement | Constraint checking, contradiction detection, coherence validation | Rules, knowledge entries | Violation report, suggestions |
| **LinkerAgent** | Cross-reference creation | Relationship identification, knowledge graph construction | Knowledge entries | Links, relationship types, citation map |
| **BiasDetectorAgent** | Fairness analysis | Bias identification, representation check, fairness scoring | Text, protected attributes | Bias report, examples, recommendations |
| **RedundancyAgent** | Duplicate detection | Content similarity analysis, deduplication | Knowledge base, similarity threshold | Duplicate groups, merge suggestions |

### 🔧 Utilities & Processing (3 agents)

| Agent | Role | Capabilities | Input | Output |
|-------|------|--------------|-------|--------|
| **FormatterAgent** | Output formatting | Format conversion, style application, template rendering | Data, format spec | Formatted output |
| **AggregatorAgent** | Information compilation | Data aggregation, summary creation, trend analysis | Multiple sources | Aggregated view, statistics |
| **IndexerAgent** | Cross-referencing | Index creation, search optimization, back-linking | Documents, schema | Searchable index, metadata |

---

## Quick Reference by Use Case

### 🎓 **Personal Learning**
Use agents for adaptive education:
- **TutorAgent** (concept explanation)
- **TeacherAgent** (generates practice problems)
- **MentorAgent** (personalized path)
- **AssessmentAgent** (progress evaluation)
- **CurriculumAgent** (learning roadmap)

### 📖 **Creative Writing & World-Building**
Use agents for storytelling projects:
- **WriterAgent** (content generation)
- **EditorAgent** (refinement)
- **StorytellerAgent** (narrative flow)
- **WorldBuilderAgent** (lore consistency)
- **CharacterDevAgent** (character arcs)
- **DialogueAgent** (conversations)
- **ConsistencyAgent** (rule enforcement)

### 🔬 **Research & Academic**
Use agents for knowledge discovery:
- **Researcher** (web search & synthesis)
- **ScientistAgent** (literature review)
- **StatisticianAgent** (data validation)
- **FactCheckerAgent** (verification)
- **DataValidatorAgent** (quality check)
- **ArchivistAgent** (documentation)
- **LinkerAgent** (knowledge graph)

### 💼 **Business & Strategy**
Use agents for planning and analysis:
- **StrategyAgent** (strategic planning)
- **ProjectManagerAgent** (execution)
- **CompetitorIntelligenceAgent** (market analysis)
- **SynthesisAgent** (reporting)
- **DataValidatorAgent** (metrics validation)

### 💻 **Software Development**
Use agents for code quality:
- **CodeReviewerAgent** (review automation)
- **ArchitectAgent** (design validation)
- **SecurityAgent** (threat analysis)
- **PerformanceAgent** (optimization)
- **APIDocumentationAgent** (docs generation)

### 🎮 **Game Development & Narrative**
Use agents for dynamic games:
- **StorytellerAgent** (plot generation)
- **CharacterDevAgent** (NPC behavior)
- **DialogueAgent** (conversations)
- **WorldBuilderAgent** (game world)
- **ConsistencyAgent** (lore checking)
- **ProjectManagerAgent** (milestone tracking)

---

## Agent Architecture

### Base Class: `BaseAgent`

All agents inherit from `BaseAgent` and implement:

```python
class BaseAgent(ABC):
    ROLE: str                          # "researcher", "teacher", etc.
    DEFAULT_PRIORITY: int              # 1-10 (9=background, 1=critical)

    def get_capabilities(self) -> List[str]:
        """List of capabilities: web_search, llm_query, qa_generation, etc."""

    def process_message(self, message: Message) -> Optional[Message]:
        """Main handler: receives message, returns result or None"""

    def idle_work(self):
        """Background work when no messages (e.g., periodic tasks)"""
```

### Message Protocol

All agents communicate via standardized messages:

```python
Message(
    from_agent="agent_1",
    to_agent="researcher_2",              # agent_id | "any_role" | "broadcast"
    type="task_request",                  # task_request | result | challenge | knowledge
    priority=5,                           # 1-10
    payload={
        "action": "research",
        "topic": "neural scaling",
        "temperature": 0.7,
    }
)
```

### Message Types

| Type | Sender | Receiver | Purpose |
|------|--------|----------|---------|
| `task_request` | Scheduler/CLI | Any agent | "Do this work" |
| `result` | Any | Any | "Here's what I found" |
| `challenge` | Critic | Any | "This claim is weak" |
| `knowledge` | Any | KnowledgeStore | "Store this insight" |
| `quality_rating` | Critic | Teacher | "This training data scores X" |

---

## Creating a New Agent

### 1. Create the file
```bash
touch crew/agents/myagent.py
```

### 2. Implement the class
```python
from crew.agents.base import BaseAgent
from crew.messaging.bus import Message

class MyAgent(BaseAgent):
    ROLE = "myagent"
    DEFAULT_PRIORITY = 5

    def get_capabilities(self) -> List[str]:
        return ["my_capability_1", "my_capability_2"]

    def process_message(self, message: Message) -> Optional[Message]:
        if message.type == "task_request":
            action = message.payload.get("action")
            if action == "do_thing":
                result = self._do_thing(message.payload)
                return Message(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    type="result",
                    payload=result
                )
        return None

    def _do_thing(self, payload):
        # Your logic here
        return {"status": "done", "data": ...}

    def idle_work(self):
        # Optional background work
        pass
```

### 3. Register the agent
In `crew/daemon.py` `_start_swarm()`:
```python
from crew.agents.myagent import MyAgent
register_agent_class("myagent", MyAgent)
```

### 4. Configure composition
In `data/config.yaml`:
```yaml
swarm:
  roles:
    myagent: 1  # Number of instances
```

---

## Rate Limiting & Resources

Agents have built-in rate limiting per resource:

```python
# Check before using
if agent.check_rate_limit("llm_call"):
    result = llm.call(...)
    agent.consume_rate("llm_call", count=1)
else:
    # Use heuristic fallback
    result = heuristic_approach()
```

**Default limits (per hour):**
- `llm_call`: 100
- `web_search`: 50
- `db_query`: 1000
- `api_call`: 200

---

## Agent Coordination Patterns

### Sequential (Chain)
```
Task → Agent1 → Agent2 → Agent3 → Result
```
Example: Researcher → Critic → Distiller

### Parallel (Fan-out)
```
        ↓ Agent1
Task →  ↓ Agent2  → Aggregator
        ↓ Agent3
```
Example: Multiple researchers → aggregate results

### Feedback Loop
```
Agent1 ↔ Agent2 ↔ Agent3
```
Example: Writer ↔ Editor ↔ Critic (iterative refinement)

### Broadcast
```
Task → Broadcast → [Any Agent Listening] → Result
```
Example: New knowledge published to all agents

---

## Performance Characteristics

| Agent | Latency | Cost | Complexity |
|-------|---------|------|-----------|
| Writer | 2-5s | High (LLM) | High |
| Critic | 100ms | Low | Medium |
| CodeReviewer | 500ms | Medium | Medium |
| Researcher | 3-10s | High (Web) | High |
| Formatter | 10ms | Very Low | Low |
| Teacher | 1-2s | Medium | Medium |

---

## Testing an Agent

```python
from crew.agents.myagent import MyAgent
from crew.messaging.bus import get_bus, Message

agent = MyAgent(agent_id="test_1", bus=get_bus())

msg = Message(
    from_agent="test",
    to_agent="test_1",
    type="task_request",
    payload={"action": "do_thing", "param": "value"}
)

result = agent.process_message(msg)
print(result.payload)
```

---

## Monitoring Agents

```bash
# See agent status
crew agents status

# Check specific agent logs
tail -f data/logs/agent_researcher_1.log

# Monitor message queue depth
crew cf status  # Shows queue depths per agent type
```

---

## Future Agents (Research)

Planned agents under development:
- **EmbeddingAgent** - Vector generation and semantic search
- **SpeechAgent** - Audio transcription and generation
- **VisionAgent** - Image analysis and generation
- **SimulatorAgent** - Monte Carlo simulation and forecasting
- **DevilsAdvocateAgent** - Automated counterargument generation
- **MediatorAgent** - Conflict resolution and consensus building

---

## License

MIT - All agents follow the same local-first, fallback-friendly philosophy.
