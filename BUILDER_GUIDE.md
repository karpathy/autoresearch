# AUTOCLAW SUPER-APP: Builder Guide for Haiku 4.6

## Who This Is For

You are Haiku 4.6, tasked with implementing the AutoClaw Super-App.
This guide tells you exactly what to build, in what order, and how.

**Read the schema files first.** They are your blueprints.
This guide is your construction manual.

---

## Codebase Orientation

```
autoclaw/
├── crew/                        # ALL YOUR WORK GOES HERE
│   ├── agents/
│   │   ├── base.py              # BaseAgent — you'll add properties to this
│   │   ├── researcher.py        # Example agent — study this pattern
│   │   ├── critic.py            # You'll extend this for feedback
│   │   ├── pool.py              # Agent pool manager
│   │   └── ...
│   ├── messaging/
│   │   └── bus.py               # MessageBus — SQLite-backed pub/sub
│   ├── knowledge/
│   │   ├── store.py             # KnowledgeStore — hot/warm/cold tiers
│   │   └── lifecycle.py         # GC, scoring, tier promotion
│   ├── hardware/
│   │   └── detector.py          # Hardware detection + profiles
│   ├── brain.py                 # LLM client + decision engine
│   ├── runner.py                # GPU experiment execution
│   ├── scheduler.py             # Task board management
│   ├── daemon.py                # Main entry point
│   ├── cli.py                   # CLI interface
│   │
│   │  # NEW DIRECTORIES YOU CREATE:
│   ├── crdt/                    # Phase A
│   ├── memory/                  # Phase B
│   ├── router/                  # Phase C
│   ├── geometry/                # Phase D
│   ├── gateway/                 # Phase E
│   └── ui/                      # Phase F
│
├── schemas/                     # YOUR BLUEPRINTS
│   ├── phase_a_crdt.yaml
│   ├── phase_b_memory.yaml
│   ├── phase_c_router.yaml
│   ├── phase_d_geometry.yaml
│   ├── phase_e_gateway.yaml
│   ├── phase_f_reactive_ui.yaml
│   └── SUPERAPP_MASTER_PLAN.md
│
├── data/                        # Runtime data (SQLite DBs, state files)
└── config/                      # YAML configs
```

## Key Patterns to Follow

Study these files before writing code. Your code must match their style:

1. **crew/agents/base.py** — How agents are structured (ABC, lifecycle, message helpers)
2. **crew/messaging/bus.py** — How SQLite is used (threading lock, WAL mode, `_get_conn`)
3. **crew/knowledge/store.py** — How tiers work (hot dict, warm SQLite, cold gzip)
4. **crew/knowledge/lifecycle.py** — How GC works (scoring, demotion, summaries)
5. **crew/scheduler.py** — How dataclasses map to YAML files
6. **crew/brain.py** — How LLM calls work (provider abstraction)

### Style Rules

- **Threading:** Always use `threading.Lock()` for shared state
- **SQLite:** Always use `PRAGMA journal_mode=WAL` and `timeout=10`
- **Logging:** Use `logging.getLogger(__name__)` per module
- **Dataclasses:** Use `@dataclass` with `to_dict()` and `from_dict()` methods
- **Errors:** Catch, log, and continue. Never crash the agent loop.
- **Config:** Read from `self.config` dict passed at init
- **Paths:** Use `pathlib.Path`, create directories with `mkdir(parents=True, exist_ok=True)`
- **JSON in SQLite:** Use `json.dumps()` / `json.loads()` for complex fields
- **Timestamps:** Always ISO 8601 UTC: `datetime.now(timezone.utc).isoformat()`
- **Type hints:** Full type hints on public methods. `Optional`, `List`, `Dict` from typing.

---

## PHASE A: CRDT Knowledge Fabric

**Schema:** `schemas/phase_a_crdt.yaml`
**New directory:** `crew/crdt/`
**Time estimate:** This is your foundation. Get it right.

### Files to Create

#### 1. `crew/crdt/__init__.py`
```python
"""CRDT-based distributed knowledge operations."""
```

#### 2. `crew/crdt/operations.py`

Create `CRDTOperation` dataclass exactly as specified in schema.
Create merge functions for each op_type.

Key merge function signatures:
```python
def merge_confidence_votes(votes: Dict[str, str]) -> str:
    """Majority vote with high-confidence tie-breaking."""

def merge_score_deltas(deltas: List[float], base_score: float) -> float:
    """Sum deltas, clamp to [0.0, 1.0]."""

def merge_tags(added: Set[str], removed: Set[str]) -> List[str]:
    """Set difference: added - removed."""
```

#### 3. `crew/crdt/layer.py`

`CRDTKnowledgeLayer` class:
- Wraps `KnowledgeStore`
- Provides `emit_operation(op)` → writes to crdt_operations table
- Provides `get_merged_state(entry_id)` → reads/computes from crdt_state
- Provides `converge()` → materializes unmerged ops into crdt_state

Follow this pattern:
```python
class CRDTKnowledgeLayer:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._lock = threading.Lock()

    def emit_operation(self, op: CRDTOperation) -> int:
        """Write operation to crdt_operations table. Returns op ID."""
        # Use store's DB connection (same warm.db)

    def get_merged_state(self, entry_id: int) -> Optional[CRDTState]:
        """Get materialized state for entry."""

    def converge(self) -> int:
        """Merge all unmerged operations. Returns count merged."""
```

#### 4. `crew/crdt/sync.py`

Periodic convergence background thread:
```python
class CRDTSyncWorker:
    def __init__(self, layer: CRDTKnowledgeLayer, interval_minutes: int = 5):
        ...
    def start(self):
        """Start background convergence thread."""
    def stop(self):
        """Stop background thread."""
```

### Integration Points

**In `crew/knowledge/store.py`:**
- Add CRDT table creation to `_init_db()` method
- Import and use in the `__init__` if crdt is available

**In `crew/agents/base.py`:**
- Add CRDT convenience methods (see schema `integration.base_agent`)
- Methods should fail gracefully if CRDT layer not initialized

**In `crew/knowledge/lifecycle.py`:**
- Add `merge_crdt_operations` step to `run_gc_pass()`

### Testing

Run schema tests in `schemas/phase_a_crdt.yaml → tests` section.
Core test: apply operations in different orders, assert same final state.

### Done When

- [ ] `CRDTOperation` and `CRDTState` dataclasses work
- [ ] `crdt_operations` and `crdt_state` tables created in warm.db
- [ ] `emit_operation()` writes to DB
- [ ] `converge()` materializes ops into state
- [ ] Merge functions are commutative (test with shuffled ops)
- [ ] `BaseAgent.crdt_vote_confidence()` works
- [ ] `KnowledgeStore.query()` uses merged values when available
- [ ] Existing tests still pass (nothing broken)

---

## PHASE B: Cognitive Memory

**Schema:** `schemas/phase_b_memory.yaml`
**New directory:** `crew/memory/`

### Files to Create

1. `crew/memory/__init__.py`
2. `crew/memory/working.py` — `WorkingMemory` (RAM-only, 7-item limit)
3. `crew/memory/episodic.py` — `EpisodicMemory` (SQLite, forgetting curve)
4. `crew/memory/semantic.py` — `SemanticMemory` (SQLite, concept graph)
5. `crew/memory/procedural.py` — `ProceduralMemory` (SQLite, strategies)
6. `crew/memory/consolidation.py` — `ConsolidationEngine` (promotion rules)
7. `crew/memory/manager.py` — `CognitiveMemory` (facade over all 4 tiers)

### Key Design Decisions

**Each agent gets its own memory.db** at `data/agents/{agent_id}/memory.db`.
This is intentional — agent memory is private, not shared.
(Sharing happens via CRDT when agents publish to knowledge store.)

**WorkingMemory is RAM only.** No SQLite. Use `OrderedDict` like hot tier.

**Forgetting curve:** `retention = e^(-age_hours / (strength × 720))`
At strength=1.0, 50% retention after 30 days (720 hours).

### Integration Points

**In `crew/agents/base.py`:**
- Add `self.memory` property (lazy-initialized `CognitiveMemory`)
- In `_handle_message()`, after processing: record episodic memory
- Before processing: check procedural memory for applicable strategies

### Done When

- [ ] All 4 memory classes work independently
- [ ] Working memory respects 7-item capacity
- [ ] Episodic memory follows forgetting curve
- [ ] Consolidation promotes episodic → semantic after 3 similar events
- [ ] Procedural memory tracks execution success rate
- [ ] `BaseAgent.memory` property works
- [ ] Memory persists across agent restart (except working memory)
- [ ] Existing tests still pass

---

## PHASE C: Escalation Router

**Schema:** `schemas/phase_c_router.yaml`
**New directory:** `crew/router/`

### Files to Create

1. `crew/router/__init__.py`
2. `crew/router/classifier.py` — `TaskClassifier` (complexity/stakes/novelty/urgency)
3. `crew/router/router.py` — `EscalationRouter` (route + call + record)
4. `crew/router/tiers.py` — `TierConfig` + model configs
5. `crew/router/feedback.py` — `RoutingFeedback` (learn from outcomes)

### Key Design Decisions

**Router is a global singleton** like `get_bus()`. Use same pattern:
```python
_default_router: Optional[EscalationRouter] = None
_router_lock = threading.Lock()

def get_router() -> EscalationRouter:
    global _default_router
    with _router_lock:
        if _default_router is None:
            _default_router = EscalationRouter()
    return _default_router
```

**Classification uses heuristics, not ML.** Keep it simple:
- Token count → complexity
- Priority number → stakes
- Episodic memory match → novelty (inverse)
- Expiry proximity → urgency

**Budget enforcement is critical.** Track cost per tier per day in routing_decisions table.

### Integration Points

**In `crew/brain.py`:**
- `LLMClient.chat()` calls through router when available
- Fallback: direct LLM call if router not initialized

**In `crew/agents/base.py`:**
- Add `self.llm_call(prompt, context)` method that uses router
- Replace direct `self.bus` LLM calls in agent subclasses

### Done When

- [ ] Classification produces tier_score for any message
- [ ] Bot/Brain/Human tier selection works
- [ ] LLM calls go through router
- [ ] Cost tracked per call
- [ ] Budget enforcement throttles when over limit
- [ ] Router learns from failures (demote/promote patterns)
- [ ] `routing_decisions` table populated
- [ ] Existing LLM calls work (fallback without router)

---

## PHASE D: Geometric Coordinator

**Schema:** `schemas/phase_d_geometry.yaml`
**New directory:** `crew/geometry/`

### Files to Create

1. `crew/geometry/__init__.py`
2. `crew/geometry/position.py` — `DodecetPosition` (3-char hex encoding)
3. `crew/geometry/coordinator.py` — `GeometricCoordinator` (main class)
4. `crew/geometry/kdtree.py` — Simple KD-tree for 3D spatial queries
5. `crew/geometry/flow.py` — Position drift logic

### Key Design Decisions

**Don't use scipy.** Build a simple KD-tree yourself. With < 100 agents,
even a linear scan is fast enough. KD-tree is for correctness, not performance.

**Dodecet encoding is just 3 hex nibbles.** Keep it simple:
```python
@dataclass
class DodecetPosition:
    x: int  # 0-15
    y: int  # 0-15
    z: int  # 0-15

    @property
    def hex(self) -> str:
        return f"{self.x:X}{self.y:X}{self.z:X}"

    @staticmethod
    def from_hex(s: str) -> "DodecetPosition":
        return DodecetPosition(int(s[0], 16), int(s[1], 16), int(s[2], 16))

    def distance_to(self, other: "DodecetPosition") -> float:
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)
```

**Position drift is 1 unit per outcome.** Simple and predictable.
Max 3 units drift per day prevents oscillation.

### Integration Points

**In `crew/messaging/bus.py`:**
- Optional `spatial` parameter on `receive()` method

**In `crew/agents/base.py`:**
- `self.position` property returns dodecet position
- Position updates after each task completion

### Done When

- [ ] DodecetPosition encodes/decodes correctly
- [ ] Distance calculation is correct
- [ ] KD-tree finds nearest agents
- [ ] Agents initialized at role-default positions
- [ ] Drift moves agents toward successful tasks
- [ ] Drift capped at max_per_day
- [ ] Fallback to role-based routing when spatial fails
- [ ] `agent_positions` table populated

---

## PHASE E: Interface Gateway

**Schema:** `schemas/phase_e_gateway.yaml`
**New directory:** `crew/gateway/`

### Files to Create

1. `crew/gateway/__init__.py`
2. `crew/gateway/session.py` — `SessionManager`
3. `crew/gateway/channels.py` — `CLIChannel` (start here)
4. `crew/gateway/feedback.py` — `FeedbackProcessor`
5. `crew/gateway/server.py` — `InterfaceGateway` (orchestrator)

### Key Design Decisions

**Start with CLI channel only.** No WebSocket yet. The gateway publishes
`user_query` messages to the MessageBus and subscribes to `channel_response`.
CLI is just stdin/stdout wrapping the gateway protocol.

**Session preferences are learned gradually.** Don't ask users for preferences.
Observe their behavior and adjust. This is the self-improving aspect.

### Done When

- [ ] Sessions create and persist in sessions.db
- [ ] CLI channel accepts user queries
- [ ] Queries route through agents and return responses
- [ ] Feedback signals captured (thumbs up/down via CLI commands)
- [ ] Feedback updates CRDT confidence votes
- [ ] Session preferences adjust based on feedback patterns
- [ ] Gateway is optional (daemon works without it)

---

## PHASE F: Reactive Spreadsheet UI

**Schema:** `schemas/phase_f_reactive_ui.yaml`
**New directory:** `crew/ui/`

### Files to Create

1. `crew/ui/__init__.py`
2. `crew/ui/grid.py` — `ReactiveGrid`
3. `crew/ui/cells.py` — Cell types + cascade logic
4. `crew/ui/renderer.py` — ASCII terminal renderer

### Key Design Decisions

**Terminal ASCII first.** Use raw ANSI escape codes for color.
No curses, no rich library. Keep dependencies minimal.

**Grid reads from existing SQLite databases.** It doesn't have its own data.
It's a viewer, not a data store. (Cell change log is the exception — for audit.)

### Done When

- [ ] ASCII knowledge grid renders in terminal
- [ ] Grid auto-refreshes every 5 seconds
- [ ] Color-coded confidence levels
- [ ] Cell changes logged to cell_change_log
- [ ] Keyboard navigation works
- [ ] Grid shows agent status row at bottom

---

## General Rules

### Ordering is Critical

Build A → B → C → D → E → F. Each phase depends on the previous.
Do NOT skip ahead. Do NOT start Phase C before Phase A tests pass.

### Every Phase Must Be Backward Compatible

After implementing Phase A, all existing tests must still pass.
After Phase B, all Phase A tests + existing tests must pass.
And so on. Never break what already works.

### Testing Pattern

For each phase:
1. Write the code
2. Run the tests listed in the schema's `tests:` section
3. Run existing tests (`python -m pytest` if available, or manual testing)
4. Verify backward compatibility
5. Commit with clear message

### Commit Pattern

One commit per phase:
```
git commit -m "Phase A: Add CRDT knowledge fabric

- crew/crdt/operations.py: CRDTOperation dataclass + merge functions
- crew/crdt/layer.py: CRDTKnowledgeLayer wrapping KnowledgeStore
- crew/crdt/sync.py: periodic convergence worker
- Integration: BaseAgent CRDT methods, KnowledgeStore merged queries
- Tests: commutativity, convergence, backward compat"
```

### When Stuck

1. Re-read the schema file for the current phase
2. Study the existing code pattern it's closest to
3. Start with the simplest possible implementation
4. Make it work first, then make it elegant

### What NOT To Do

- Don't add external dependencies (no pip install)
- Don't create new database files (add tables to existing DBs)
- Don't modify existing method signatures (add new methods instead)
- Don't delete existing code (wrap or extend it)
- Don't optimize prematurely (simple is better)
- Don't skip tests

---

## File Dependency Graph

```
Phase A:
  crew/crdt/operations.py    → (no deps)
  crew/crdt/layer.py         → crew/knowledge/store.py
  crew/crdt/sync.py          → crew/crdt/layer.py

Phase B:
  crew/memory/working.py     → (no deps)
  crew/memory/episodic.py    → (no deps, own SQLite)
  crew/memory/semantic.py    → (no deps, own SQLite)
  crew/memory/procedural.py  → (no deps, own SQLite)
  crew/memory/consolidation.py → episodic, semantic, procedural
  crew/memory/manager.py     → all memory modules

Phase C:
  crew/router/tiers.py       → (no deps)
  crew/router/classifier.py  → crew/memory/manager.py (for novelty check)
  crew/router/router.py      → classifier, tiers, crew/brain.py
  crew/router/feedback.py    → crew/memory/episodic.py

Phase D:
  crew/geometry/position.py  → (no deps)
  crew/geometry/kdtree.py    → position.py
  crew/geometry/coordinator.py → kdtree.py, crew/messaging/bus.py
  crew/geometry/flow.py      → coordinator.py

Phase E:
  crew/gateway/session.py    → (no deps, own SQLite)
  crew/gateway/channels.py   → session.py
  crew/gateway/feedback.py   → crew/crdt/layer.py, crew/memory/manager.py
  crew/gateway/server.py     → channels, feedback, crew/messaging/bus.py

Phase F:
  crew/ui/cells.py           → (no deps)
  crew/ui/grid.py            → cells.py, reads from all existing DBs
  crew/ui/renderer.py        → grid.py
```

---

## Final Checklist

When all 6 phases are complete:

- [ ] Phase A: CRDT operations commutative, convergence works
- [ ] Phase B: Agents learn from experience (episodic → semantic → procedural)
- [ ] Phase C: LLM calls route to optimal tier, costs tracked, learning works
- [ ] Phase D: Agents positioned in space, spatial routing works, drift happens
- [ ] Phase E: Users can query via CLI, feedback improves knowledge
- [ ] Phase F: ASCII grid shows live system state
- [ ] All existing tests pass
- [ ] Each phase's schema tests pass
- [ ] No new external dependencies added
- [ ] No existing method signatures changed

**You're building a self-improving intelligence platform that runs on any hardware,
learns from every interaction, and costs 15x less to operate. Make it clean,
make it simple, make it work.**
