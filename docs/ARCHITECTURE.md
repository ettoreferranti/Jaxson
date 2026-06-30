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
│  • FaceView (eyes + mouth, egui Painter 2D)                        │
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
| `jaxson-core` | Shared value types + state machine: `MoodVector`, `Emotion`, `RelationshipState`, per-topic `TopicAffinities` (F1.5), IDs, errors, and `scrub::redact` (masks PII in log strings, F2.6) | ✅ | **seeded (F1.5, F2.6)** |
| `jaxson-memory` | Memory graph (typed/weighted nodes + edges), hybrid retrieval (cosine + graph spread), `MemoryStore` trait + in-memory store; encrypted SQLite (SQLCipher) persistence behind the `sqlite` feature | ✅ (pure) / SQLCipher (feature) | **built (F1.2, F1.4)** |
| `jaxson-affect` | Affect engine: relationship state + (lexicon) sentiment → target `MoodVector`; smoothing via the state machine | ✅ | **built (F1.6)** |
| `jaxson-llm` | Chat messages, prompt/chat-template assembly, decode config, `TextGenerator` trait; `llama.cpp`+Metal backend (`LlamaGenerator` + `LlamaEmbedder`, sharing one loaded model) behind the `llama` feature | ✅ (pure) / Metal (feature) | **built (F1.1, F1.4b)** |
| `jaxson-extract` | Turn conversation turns into memory nodes/edges: extraction prompt + JSON parsing, over `dyn TextGenerator` | ✅ | **built (F1.3)** |
| `jaxson-safety` | Safety governance (FR-S1/S2/S3): a `SafetyFilter` screens text against severity-ordered `Category`s at a configurable `Strictness`, returning `Allow`/`Block(category)` with an in-character `deflection`; plus the parental-control `PasscodeHash` (salted, iterated SHA-256 — no plaintext) that gates strictness. Deterministic lexicon stand-in (swappable for an LLM classifier), 100% mutation-graded | ✅ | **built (F2.4 filter, F2.5 passcode)** |
| `jaxson-perception` | Speech, both directions: `SpeechToText` + `TextToSpeech` seams + pure audio utilities (downmix, RMS, silence-trim, loudness `envelope` for lip-sync) + `speakable_text`/`split_sentences` + deterministic `MockStt`/`MockTts`; whisper.cpp (Metal) STT behind the `whisper` feature and Piper neural TTS (ONNX, cross-platform) behind the `piper` feature | ✅ (pure) / whisper + piper (features) | **STT (F2.1), TTS (F2.2) built** |
| `jaxson-agent` | Orchestration: the per-turn conversation loop (retrieve → prompt → reply → extract → state), with an `Embedder` seam (`HashEmbedder` stand-in), familiarity-gated proactive curiosity, a memory-aware session `opening_greeting` (welcomes a returning user by name instead of re-asking it), a `respond_streaming_with_reply` `on_reply` hook (fires the cleaned reply before the slower extraction pass, so a UI can speak immediately), a `jaxson-safety` post-filter on every reply (blocked output → safe deflection, FR-S1) with parent-tunable `set_safety_strictness` (FR-S3), and structured `tracing` instrumentation | ✅ | **built (F1.7, F1.11, F1.12, F2.4, F2.5)** |
| `jaxson-face` | Pure face geometry (`mood` + time → eye/mouth shapes, blink, gaze) layered with an `Activity` (`face_with`: lip-sync mouth while speaking, attentive eyes while listening) **+ a software rasterizer** to a B/W `Bitmap` — no GUI | ✅ | **built (F1.8a, F2.3)** |
| `jaxson-app` | egui shell: animated face + chat view (streaming, non-blocking generation on a worker thread; wrapping/selectable transcript; clear-chat); model + embedding pickers; push-to-talk mic (cpal → whisper) behind its `whisper` feature; spoken replies (Piper TTS synthesized on the worker thread → `rodio` playback) behind its `piper` feature, with the face lip-syncing to playback (a `SpeechAnimator` maps playback time → mouth level) and showing a listening cue while the mic records; a passcode-gated **parent panel** (FR-S3): tune guardrail strictness + review memories (inspector + Export JSON now hidden behind the gate), persisted to `parental.json`; Keychain-keyed encrypted persistence behind its `sqlite` feature; installs the `tracing` log sink. Excluded from the workspace (native GUI; run on macOS) | ❌ | **built (F1.8b, F1.9, F1.9b, F1.10, F1.12, F1.13, F2.1b, F2.2b, F2.3, F2.5)** |

Native/heavy deps are always isolated behind cargo features: `jaxson-llm`'s `llama`
(`llama.cpp` + Metal), `jaxson-memory`'s `sqlite` (SQLCipher), and `jaxson-perception`'s
`whisper` (whisper.cpp + Metal) and `piper` (Piper TTS over ONNX Runtime + espeak-ng).
Default builds are pure Rust, so mutation testing stays fast and meaningful and the rest
of the workspace builds without a C toolchain.

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
Derived scalars that summarize the relationship and gate behavior:
- `trust ∈ [0,1]`, `familiarity ∈ [0,1]` and a current `MoodVector` on
  `RelationshipState`, plus per-topic `affinity ∈ [-1,1]` in `TopicAffinities`
  (F1.5 — kept separate because it's a heap map, so `RelationshipState` stays `Copy`).
- Memories and interactions emit **events** that mutate these variables through
  clamped, well-tested transition functions. This is the "state machine" — state is
  a pure function of the event history, making it deterministic and testable. Affinity
  reinforcement uses the same diminishing-returns-toward-the-bound shape as trust.

### 4.3 Behavior gating (examples)
- `familiarity` low → orchestrator biases toward asking onboarding questions
  (`jaxson-agent::curiosity`, F1.11): a getting-to-know-you curriculum (Person →
  Preference → Event → Fact) aims questions at gaps in the graph, so Jaxson asks about
  what it doesn't yet know and stops once a topic is answered. Three tiers: onboarding
  leads every turn, acquainted gently nudges remaining gaps, familiar-with-no-gaps just
  converses.
- `trust` below a threshold → sensitive topics stay locked.
- `affinity` per topic (F1.5) → a strongly-liked topic (≥ 0.5) is surfaced in the system
  prompt so Jaxson eagerly brings it up. Affinity is nudged each turn by the sentiment of
  what the user says: learned **preferences** and any already-known topic named in the
  input. (Per-session for now — like trust/familiarity, not yet persisted.)

### 4.4 Retrieval (`retrieve`, F1.4)
Hybrid and pure: **cosine similarity** over node embeddings seeds the relevant nodes,
then relevance **spreads along weighted edges** (max-product relaxation, `graph_decay`
per hop up to `max_hops`) so associated memories — even ones without embeddings —
surface too. Results are ranked by score (ties broken by id for determinism) and
capped at `top_k`, then injected into the LLM prompt. The query embedding is an input;
turning text into one is the **embedder's** job (F1.4b): `HashEmbedder` (deterministic
stand-in) by default, or the model's real semantic embeddings (`jaxson-llm`'s
`LlamaEmbedder`, mean-pooled + L2-normalized) once a model is loaded. cosine tolerates
empty/mismatched vectors (returns 0), so switching embedders degrades gracefully rather
than crashing.

## 5. Affect engine (`jaxson-affect`, F1.6)

`AffectEngine::target_mood` reads (a) relationship-state variables (trust/familiarity
add a warmth baseline) and (b) the sentiment of the latest exchange, producing a target
`MoodVector` (valence/arousal); `dominant_emotion()` snaps it to a discrete `Emotion`
for the face. Smoothing toward the target reuses `MoodVector::blended` — the
orchestrator applies it through the state machine's `MoodObserved` event, keeping
`RelationshipState` the single source of truth. Sentiment comes from a deterministic
lexicon ([`analyze`]) — **decoupled from LLM wording** (FR-E4) — to be upgraded later.
The face view (F1.8) will be a pure egui rendering of this mood signal plus idle
micro-motions.

**`jaxson-agent` design.** `Agent::respond` runs one turn end-to-end and is the
integrator: it owns the persona, [`RelationshipState`], the `MemoryGraph`, and history,
and takes the model (`dyn TextGenerator`) and an `Embedder` per call so the same agent
runs on mock or real backends. State drives behavior (low familiarity injects an
onboarding hint into the system prompt). Persistence is the caller's job (load via
`with_graph`, save `graph()` through a `MemoryStore`). The `Embedder` trait has a
deterministic `HashEmbedder` stand-in until the real model embedder (F1.4b). Each turn
it also reads sentiment of the user's input and applies the affect engine's target mood
to the state machine (`MoodObserved`), so `mood()` reflects the conversation.

## 6. Conversation loop (orchestration)

```
user input
  → jaxson-safety: pre-filter (v0.2)
  → jaxson-memory: retrieve(context)       // graph + vector
  → build prompt (persona + state + retrieved memories + history)
  → jaxson-llm: generate (streaming, Metal via llama.cpp)
  → jaxson-safety: post-filter — block unsafe output → deflection (F2.4 ✓)
  → jaxson-extract: extract(new_facts) → jaxson-memory graph + state mutation
  → jaxson-affect: update() → MoodVector
  → emit (text/voice + mood) to shell
  → log structured trace (NFR-4)
```

## 7. Persistence

- SQLite file in the app's data dir (`~/Library/Application Support/Jaxson` on
  macOS), **encrypted at rest via SQLCipher** (decided at v0.1; `jaxson-memory`'s
  `sqlite` feature, `rusqlite` + vendored SQLCipher). The key comes from the macOS
  Keychain (generated on first run via the `keyring` crate, fetched thereafter);
  opening with the wrong key fails. Holds nodes, edges, and (later) state and history.
- **Wired into the app** behind `jaxson-app`'s own `sqlite` feature: the graph loads
  on launch and saves after every turn and every memory-inspector edit/delete. A
  "Export JSON" button dumps the (otherwise encrypted) graph to a readable file in the
  same dir. Without the feature the app runs ephemerally (memory lost on quit).
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
