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
| `jaxson-memory` | Memory graph (typed/weighted nodes + edges), `MemoryStore` trait + in-memory store; encrypted SQLite (SQLCipher) persistence behind the `sqlite` feature | ✅ (pure) / SQLCipher (feature) | **built (F1.2)** |
| `jaxson-affect` | Affect engine: graph-state + sentiment → `MoodVector` + dominant `Emotion` | ✅ | backlog |
| `jaxson-llm` | Chat messages, prompt/chat-template assembly, decode config, `TextGenerator` trait; `llama.cpp`+Metal backend behind the `llama` feature | ✅ (pure) / Metal (feature) | **built (F1.1)** |
| `jaxson-extract` | Turn conversation turns into memory nodes/edges: extraction prompt + JSON parsing, over `dyn TextGenerator` | ✅ | **built (F1.3)** |
| `jaxson-safety` | Content filtering, topic guardrails, output sanitization | ✅ | backlog (v0.2) |
| `jaxson-perception` | whisper.cpp STT + local TTS | ✅ | backlog (v0.2) |
| `jaxson-agent` | Orchestration: wires crates into the conversation loop | ✅ | backlog |
| `jaxson-app` | egui shell: FaceView, ChatView, MemoryInspector | ❌ | backlog (v0.1) |

Native/heavy deps are always isolated behind cargo features: `jaxson-llm`'s `llama`
(`llama.cpp` + Metal), `jaxson-memory`'s `sqlite` (SQLCipher), and `jaxson-perception`
(whisper.cpp, v0.2). Default builds are pure Rust, so mutation testing stays fast and
meaningful and the rest of the workspace builds without a C toolchain.

**`jaxson-llm` design.** The crate is split so the heavy dep is isolated: the *pure*
layer (`Message`/`Role`, `GenerationConfig`, chat-template `prompt` assembly, the
`TextGenerator` trait, and a deterministic `MockGenerator`) is fully unit- and
mutation-tested and always compiles. The native `LlamaGenerator` lives behind the
`llama` cargo feature (`llama-cpp-2` → `llama.cpp` with Metal offload), so default
builds, tests, CI, and the rest of the workspace never need cmake or a model. The
orchestrator depends on `dyn TextGenerator`, so the mock and the real model are
interchangeable.

**`jaxson-memory` design.** Same split. The pure layer — `MemoryNode` (kind,
content, provenance, confidence, optional embedding), typed/weighted `Edge`
(`strengthened`/`decayed`), the `MemoryGraph`, and the `MemoryStore` trait with an
`InMemoryStore` — is fully mutation-tested and always compiles. The `MemoryGraph` is
the authoritative, validated model; a `MemoryStore` gives it **snapshot** durability
(`save`/`load`). The encrypted on-disk `SqliteStore` (SQLCipher) lives behind the
`sqlite` feature, so default builds need no C deps. The agent depends on
`dyn MemoryStore`, so in-memory and encrypted-SQLite are interchangeable.

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
  → jaxson-extract: extract(new_facts) → jaxson-memory graph + state mutation
  → jaxson-affect: update() → MoodVector
  → emit (text/voice + mood) to shell
  → log structured trace (NFR-4)
```

## 7. Persistence

- SQLite file in the app's sandbox container, **encrypted at rest via SQLCipher**
  (decided at v0.1; `jaxson-memory`'s `sqlite` feature, `rusqlite` + vendored
  SQLCipher). The key comes from the macOS Keychain; opening with the wrong key
  fails. Holds nodes, edges, and (later) state and history.
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
| A7 | SQLCipher (not app-level encryption) for at-rest, resolving A5 | A real encrypted, queryable DB (`rusqlite` + vendored SQLCipher) beats encrypting a plaintext-SQLite file at the app layer; wrong-key access fails. Feature-gated (`sqlite`) so the pure graph layer stays C-dep-free. |

_Add an entry here in the same PR whenever a structural decision changes._
