# Jaxson — Architecture

Status: **living document — kept up to date at all times.** Every PR that changes
structure must update this file in the same PR.

## 1. Guiding ideas

1. **The memory graph *is* the agent.** Behavior, mood, and initiative emerge from a
   knowledge graph and the relationship-state variables that memories mutate. The LLM
   is a language surface over that state, not the seat of personality.
2. **Separation of core from shell.** All decision logic lives in plain Rust crates
   with no GUI dependency, so the same core can later drive a hardware bot. The GUI
   (egui) is a thin presentation/IO shell.
3. **Local & private by construction.** No module is permitted to open a network
   socket for inference, memory, or telemetry.

## 2. Layered view

```
┌──────────────────────────────────────────────────────────────────┐
│  Presentation shell  (egui, macOS)                                 │
│  • FaceView (eyes/nose/mouth, egui Painter 2D)                     │
│  • ChatView (text I/O)                                             │
│  • MemoryInspectorView                                             │
│  • (v0.2) Voice I/O surface, Parental-control UI                   │
└───────────────▲───────────────────────────────┬──────────────────┘
                │ observes (mood, transcript)    │ user input
┌───────────────┴───────────────────────────────▼──────────────────┐
│  Orchestration  (jaxson-agent)                                     │
│  Conversation loop: input → retrieve → prompt → generate →         │
│  safety-filter → extract memories → update state → emit response   │
└───┬───────────┬────────────┬───────────┬───────────┬──────────────┘
    │           │            │           │           │
┌───▼─────┐ ┌──▼──────┐  ┌────▼─────┐ ┌───▼────┐  ┌───▼──────┐
│ LLM     │ │ Memory  │  │ Affect   │ │ Safety │  │ Perception│
│llama.cpp│ │ Graph + │  │ Engine   │ │ Guard  │  │ STT (v0.2)│
│ (Metal) │ │ Vector  │  │ mood vec │ │ (v0.2) │  │ TTS (v0.2)│
└─────────┘ │ + State │  └──────────┘ └────────┘  └───────────┘
           │ machine │
           └────┬────┘
        ┌───────▼────────┐
        │ Persistence    │
        │ SQLite (enc.)  │
        │ + vector index │
        └────────────────┘
```

## 3. Cargo workspace / crate map

The non-UI core is a set of independently testable crates in a Cargo workspace. The
macOS app crate (added in v0.1) depends on them.

| Crate | Responsibility | UI-free? | Status |
| ----- | -------------- | -------- | ------ |
| `jaxson-core` | Shared value types: `MoodVector`, `Emotion`, `RelationshipState`, IDs, errors | ✅ | **seeded** |
| `jaxson-memory` | Graph nodes/edges, vector index, retrieval, memory extraction, state mutation | ✅ | backlog |
| `jaxson-affect` | Affect engine: graph-state + sentiment → `MoodVector` + dominant `Emotion` | ✅ | backlog |
| `jaxson-llm` | `llama.cpp` (Metal) bindings: model load, prompt assembly, streaming generation | ✅ (Metal) | backlog |
| `jaxson-safety` | Content filtering, topic guardrails, output sanitization | ✅ | backlog (v0.2) |
| `jaxson-perception` | whisper.cpp STT + local TTS | ✅ | backlog (v0.2) |
| `jaxson-agent` | Orchestration: wires crates into the conversation loop | ✅ | backlog |
| `jaxson-app` | egui shell: FaceView, ChatView, MemoryInspector | ❌ | backlog (v0.1) |

Only `jaxson-llm` and `jaxson-perception` touch heavy/native deps (`llama.cpp`,
Metal, whisper.cpp); the rest are pure Rust to keep mutation testing fast and
meaningful.

## 4. The memory state machine (core design)

### 4.1 Graph
- **Nodes**: `Memory` items — facts, people, events, preferences, episodes. Each has
  `id`, `kind`, `content`, `createdAt`, `provenance`, `confidence`, and an embedding.
- **Edges**: typed, directed, weighted relations (`likes`, `dislikes`, `knows`,
  `relatedTo`, `happenedOn`, `causes`). Weights decay/strengthen over time.

### 4.2 State variables
Derived, persisted scalars that summarize the relationship and gate behavior:
- `trust ∈ [0,1]`, `familiarity ∈ [0,1]`, per-topic `affinity ∈ [-1,1]`, and a
  current `MoodVector`.
- Memories and interactions emit **events** that mutate these variables through
  clamped, well-tested transition functions. This is the "state machine" — state is
  a pure function of the event history, making it deterministic and testable.

### 4.3 Behavior gating (examples)
- `familiarity` low → orchestrator biases toward asking onboarding questions.
- `trust` below a threshold → sensitive topics stay locked.
- `affinity` per topic → influences what Jaxson brings up and the baseline mood.

### 4.4 Retrieval
Hybrid: vector similarity over node embeddings **+** graph traversal from the active
focus node, merged and ranked, injected into the LLM prompt.

## 5. Affect engine

Reads (a) current relationship-state variables, (b) sentiment of the latest
exchange, (c) recent mood, and produces a continuous `MoodVector`
(valence/arousal) plus a snapped dominant `Emotion`. Output is smoothed over time so
the face transitions naturally. **Decoupled from LLM wording** (FR-E4) for a
consistent personality. The face view is a pure egui rendering of this mood signal
plus idle micro-motions.

## 6. Conversation loop (orchestration)

```
user input
  → jaxson-safety: pre-filter (v0.2)
  → jaxson-memory: retrieve(context)       // graph + vector
  → build prompt (persona + state + retrieved memories + history)
  → jaxson-llm: generate (streaming, Metal via llama.cpp)
  → jaxson-safety: post-filter (v0.2)
  → jaxson-memory: extract(new_facts) → graph + state mutation
  → jaxson-affect: update() → MoodVector
  → emit (text/voice + mood) to shell
  → log structured trace (NFR-4)
```

## 7. Persistence

- SQLite file in the app's sandbox container, **encrypted at rest** (SQLCipher or
  app-level encryption — decided at v0.1). Holds nodes, edges, state, and history.
- Vector index alongside (start simple: in-memory + persisted vectors; revisit a
  dedicated index if scale requires).
- No memory data is ever written outside the user's container; never committed to
  git (enforced by `.gitignore`).

## 8. Privacy & security touchpoints

See `docs/PRIVACY-SECURITY.md`. Architecture-level guarantees: no network egress for
core functions; encryption at rest; parental-control auth boundary; model output is
treated as untrusted and sanitized before any privileged use.

## 9. Decisions log (ADR-lite)

| ID | Decision | Rationale |
| -- | -------- | --------- |
| A1 | Rust core split from GUI shell | Portability to hardware bot; fast, UI-free mutation testing |
| A2 | Memory graph as state machine | Owner's core vision; deterministic, testable, explainable behavior |
| A3 | Affect engine decoupled from LLM | Consistent personality independent of token-level wording |
| A4 | `llama.cpp` (Metal) for inference | Proven on-device Metal inference; portable to the hardware bot |
| A5 | SQLite encrypted at rest | Simple, embeddable, private; good fit for a single-owner store |
| A6 | **Rust instead of Swift** (supersedes initial choice) | Best-in-class mutation testing (`cargo-mutants`), zero heavy-toolchain friction (no Xcode), portability to the Linux/embedded hardware-bot endgame, and owner's existing Rust fluency. Trade-off: more wiring for LLM/face vs Swift's MLX/SwiftUI, but those live in the replaceable shell layer. Decided one PR in, before any feature code. |

_Add an entry here in the same PR whenever a structural decision changes._
