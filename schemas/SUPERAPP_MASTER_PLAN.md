# AUTOCLAW SUPER-APP: Master Architecture Plan

## The Vision

AutoCrew today is a solid multi-agent knowledge system for ML experiments.
The SuperInstance ecosystem has 6 proven intelligent systems sitting in separate repos.

**The killer app:** Merge them into a single self-improving intelligence platform
that runs on any hardware, learns from every interaction, costs 15x less to operate,
scales to unlimited agents, and is accessible from any messaging platform.

**Nobody expects a GPU experiment runner to become a general-purpose intelligence
platform.** That's the head-turn.

---

## HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AUTOCLAW SUPER-APP                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  CLAW GATEWAY │    │  WEB UI      │    │  CLI / DAEMON        │  │
│  │  (20+ channels)│    │ (Spreadsheet) │    │  (current entry)     │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                    │                        │              │
│  ═══════╪════════════════════╪════════════════════════╪═══════════  │
│  │      ▼                    ▼                        ▼          │  │
│  │              LAYER 6: INTERFACE GATEWAY                       │  │
│  │    Claw sessions, channel routing, feedback capture           │  │
│  ═══════════════════════════════════════════════════════════════  │
│         │                                                           │
│  ═══════╪═══════════════════════════════════════════════════════  │
│  │      ▼                                                        │  │
│  │              LAYER 5: ESCALATION ROUTER                       │  │
│  │    Task → Complexity Analysis → Tier Selection (Bot/Brain/    │  │
│  │    Human) → Cost tracking → Feedback learning                 │  │
│  ═══════════════════════════════════════════════════════════════  │
│         │                                                           │
│  ═══════╪═══════════════════════════════════════════════════════  │
│  │      ▼                                                        │  │
│  │              LAYER 4: GEOMETRIC COORDINATOR                   │  │
│  │    Agent positions → Dodecet encoding → KD-tree queries →     │  │
│  │    Spatial routing → Ricci flow self-organization             │  │
│  ═══════════════════════════════════════════════════════════════  │
│         │                                                           │
│  ═══════╪═══════════════════════════════════════════════════════  │
│  │      ▼                                                        │  │
│  │              LAYER 3: COGNITIVE MEMORY                        │  │
│  │    Working → Episodic → Semantic → Procedural                 │  │
│  │    Consolidation engine, forgetting curves, spaced repetition │  │
│  ═══════════════════════════════════════════════════════════════  │
│         │                                                           │
│  ═══════╪═══════════════════════════════════════════════════════  │
│  │      ▼                                                        │  │
│  │              LAYER 2: CRDT KNOWLEDGE FABRIC                   │  │
│  │    Local replicas → Commutative ops → Auto-convergence →     │  │
│  │    Peer sync → No central bottleneck                          │  │
│  ═══════════════════════════════════════════════════════════════  │
│         │                                                           │
│  ═══════╪═══════════════════════════════════════════════════════  │
│  │      ▼                                                        │  │
│  │              LAYER 1: CORE (existing AutoCrew)                │  │
│  │    BaseAgent → MessageBus → KnowledgeStore → Scheduler →     │  │
│  │    CrewBrain → HardwareDetector → ExperimentRunner            │  │
│  ═══════════════════════════════════════════════════════════════  │
│         │                                                           │
│  ═══════╪═══════════════════════════════════════════════════════  │
│  │      ▼                                                        │  │
│  │              LAYER 0: HARDWARE SUBSTRATE                      │  │
│  │    GPU Detection → Profile Selection → CUDA/CPU routing →    │  │
│  │    Memory management → Thermal monitoring                     │  │
│  ═══════════════════════════════════════════════════════════════  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## INSTALLATION ORDER

Each layer wraps the one below it. No layer breaks existing functionality.
Every layer has a fallback to the layer below.

```
Build Order (each phase adds one layer):

Phase A: CRDT Knowledge Fabric         [crew/crdt/]
Phase B: Cognitive Memory              [crew/memory/]
Phase C: Escalation Router             [crew/router/]
Phase D: Geometric Coordinator         [crew/geometry/]
Phase E: Interface Gateway (Claw)      [crew/gateway/]
Phase F: Reactive Spreadsheet UI       [crew/ui/]
```

**Why this order:**
- A (CRDT) goes first because it touches the knowledge store directly — the heart of everything
- B (Memory) builds on CRDT convergence for episodic/semantic sync
- C (Router) needs memory to learn routing patterns over time
- D (Geometry) needs router to understand agent tiers
- E (Gateway) needs all layers below for multi-channel intelligence
- F (UI) is visualization of everything below

---

## SCHEMA DESIGN PRINCIPLES

1. **Additive Only:** New schemas extend existing ones. Never break `schemas/*.yaml`.
2. **Optional Fields:** All new fields have defaults. Old code works without them.
3. **Same DB:** New tables in same SQLite files. No new database files.
4. **Same Patterns:** Follow existing `_row_to_entry`, `_get_conn`, `_lock` patterns.
5. **Haiku-Friendly:** Every schema has inline examples and implementation notes.

---

## What Follows

The schemas below are organized by installation phase. Each schema file includes:
- Data structure definition
- SQLite table schema
- Python dataclass
- Integration points with existing code
- Implementation notes for Haiku 4.6
- Examples

Build them in order. Test each phase before starting the next.
