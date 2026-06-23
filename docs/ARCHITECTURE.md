# Jaxson — Architecture

Status: **living document — kept up to date at all times.** Every PR that changes
structure must update this file in the same PR.

## 1. Guiding ideas

1. **The memory graph *is* the agent.** Behavior, mood, and initiative emerge from a
   knowledge graph and the relationship-state variables that memories mutate. The LLM
   is a language surface over that state, not the seat of personality.
2. **Separation of core from shell.** All decision logic lives in plain Swift
   modules with no SwiftUI dependency, so the same core can later drive a hardware
   bot. SwiftUI is a thin presentation/IO shell.
3. **Local & private by construction.** No module is permitted to open a network
   socket for inference, memory, or telemetry.

## 2. Layered view

```
┌──────────────────────────────────────────────────────────────────┐
│  Presentation shell  (SwiftUI, macOS)                              │
│  • FaceView (eyes/nose/mouth, Metal/Canvas)                        │
│  • ChatView (text I/O)                                             │
│  • MemoryInspectorView                                             │
│  • (v0.2) Voice I/O surface, Parental-control UI                   │
└───────────────▲───────────────────────────────┬──────────────────┘
                │ observes (mood, transcript)    │ user input
┌───────────────┴───────────────────────────────▼──────────────────┐
│  Orchestration  (JaxsonAgent)                                      │
│  Conversation loop: input → retrieve → prompt → generate →         │
│  safety-filter → extract memories → update state → emit response   │
└───┬───────────┬────────────┬───────────┬───────────┬──────────────┘
    │           │            │           │           │
┌───▼───┐  ┌────▼────┐  ┌────▼─────┐ ┌───▼────┐  ┌───▼──────┐
│ LLM   │  │ Memory  │  │ Affect   │ │ Safety │  │ Perception│
│ MLX   │  │ Graph + │  │ Engine   │ │ Guard  │  │ STT (v0.2)│
│ engine│  │ Vector  │  │ mood vec │ │ (v0.2) │  │ TTS (v0.2)│
└───────┘  │ + State │  └──────────┘ └────────┘  └───────────┘
           │ machine │
           └────┬────┘
        ┌───────▼────────┐
        │ Persistence    │
        │ SQLite (enc.)  │
        │ + vector index │
        └────────────────┘
```

## 3. Swift package / module map

The non-UI core is a SwiftPM package (`JaxsonKit`) of independently testable
libraries. The macOS app (added in v0.1) depends on it.

| Module | Responsibility | UI-free? | Status |
| ------ | -------------- | -------- | ------ |
| `JaxsonCore` | Shared value types: `MoodVector`, `Emotion`, `RelationshipState`, IDs, errors | ✅ | **seeded** |
| `JaxsonMemory` | Graph nodes/edges, vector index, retrieval, memory extraction, state mutation | ✅ | backlog |
| `JaxsonAffect` | Affect engine: graph-state + sentiment → `MoodVector` + dominant `Emotion` | ✅ | backlog |
| `JaxsonLLM` | MLX-Swift wrapper: model load, prompt assembly, streaming generation | ✅ (Metal) | backlog |
| `JaxsonSafety` | Content filtering, topic guardrails, output sanitization | ✅ | backlog (v0.2) |
| `JaxsonPerception` | whisper.cpp STT + local TTS | ✅ | backlog (v0.2) |
| `JaxsonAgent` | Orchestration: wires modules into the conversation loop | ✅ | backlog |
| `JaxsonApp` (Xcode) | SwiftUI shell: FaceView, ChatView, MemoryInspector | ❌ | backlog (v0.1) |

Only `JaxsonLLM` and `JaxsonPerception` touch heavy/native deps (MLX, Metal,
whisper.cpp); the rest are pure Swift to keep mutation testing fast and meaningful.

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
consistent personality. The FaceView is a pure rendering of this mood signal plus
idle micro-motions.

## 6. Conversation loop (orchestration)

```
user input
  → JaxsonSafety.preFilter (v0.2)
  → JaxsonMemory.retrieve(context)         // graph + vector
  → build prompt (persona + state + retrieved memories + history)
  → JaxsonLLM.generate (streaming, Metal)
  → JaxsonSafety.postFilter (v0.2)
  → JaxsonMemory.extract(newFacts) → graph + state mutation
  → JaxsonAffect.update() → MoodVector
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
| A1 | Swift core split from SwiftUI shell | Portability to hardware bot; fast, UI-free mutation testing |
| A2 | Memory graph as state machine | Owner's core vision; deterministic, testable, explainable behavior |
| A3 | Affect engine decoupled from LLM | Consistent personality independent of token-level wording |
| A4 | MLX-Swift for inference | Native Metal on Apple Silicon, Swift-native, on-device |
| A5 | SQLite encrypted at rest | Simple, embeddable, private; good fit for a single-owner store |

_Add an entry here in the same PR whenever a structural decision changes._
