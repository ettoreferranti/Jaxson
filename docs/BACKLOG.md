# Jaxson — Feature Backlog

Status: **living document.** Prioritized top-to-bottom within each milestone. Each
item gets its own feature branch + PR. Checkboxes track completion.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Milestone v0.0 — Foundation (this PR)

- [x] **F0.1** Repo conventions: README, `.gitignore`, license.
- [x] **F0.2** Requirements doc from the requirements-engineering session.
- [x] **F0.3** Architecture doc (kept up to date going forward).
- [x] **F0.4** This backlog.
- [x] **F0.5** Development workflow doc (branching, PRs, mutation testing, CI).
- [x] **F0.6** Privacy & security doc + threat model.
- [x] **F0.7** Buildable Cargo workspace with `jaxson-core` seeded (`MoodVector`,
  `Emotion`, `RelationshipState`) and unit tests.
- [x] **F0.9** Mutation testing set up (`cargo-mutants`, `.cargo/mutants.toml`):
  `jaxson-core` at 70/70 viable mutants caught (100%).
- [x] **F0.8** CI: GitHub Actions (`cargo build`/`test`/`fmt`/`clippy` + `cargo mutants`
  on core crates + forbidden-file guard). Blocks merge on surviving mutants.
- [x] **F0.10** Runnable `state_machine_demo` example to visualize core behavior.

## Milestone v0.1 — Talking face + memory (text-first)

- [x] **F1.1** `jaxson-llm`: chat messages, chat-template prompt assembly, decode
  config, `TextGenerator` trait + deterministic mock, and a `llama.cpp` (Metal)
  backend behind the `llama` feature (loads a GGUF, streaming generation). Pure layer
  fully tested + mutation-graded; native backend compile-verified.
- [x] **F1.1b** Wire the real model in: app `llama` feature selects `LlamaGenerator`
  from `JAXSON_MODEL` (else demo brain); headless `llama_chat` smoke-test example; agent
  extraction made non-fatal (real models emit imperfect JSON). Compile-verified on macOS;
  run by the owner with a model.
- [ ] **F1.1c** Benchmark latency on a Mac with a real 7–8B quantized GGUF (NFR-3):
  first-token < 1.5 s, interactive generation. Tune model/quantization (resolve OQ-2).
- [x] **F1.2** `jaxson-memory` graph store: typed/weighted nodes + edges, `MemoryStore`
  trait + in-memory store (pure, mutation-graded), and encrypted-at-rest SQLite
  (SQLCipher) persistence behind the `sqlite` feature (round-trip + wrong-key tests).
- [x] **F1.3** Memory extraction (`jaxson-extract`): turn conversation turns into
  memory nodes/edges with provenance + confidence, via an LLM emitting JSON over the
  `TextGenerator` seam. Pure prompt+parse layer fully mutation-graded.
- [x] **F1.4** Hybrid retrieval (`jaxson-memory::retrieve`): cosine-similarity seeds +
  weighted graph spread (decay per hop), ranked top-k. Query embedding is an input;
  text→embedding wiring deferred to model integration.
- [ ] **F1.4b** Embedder: produce embedding vectors from text via the local model
  (`llama.cpp` embeddings), to populate node embeddings and embed queries for F1.4.
- [ ] **F1.5** State machine (extend `jaxson-core`): per-topic affinity + richer
  transitions with clamped functions (heavy unit + mutation tests).
- [x] **F1.6** `jaxson-affect` engine v1: relationship state + lexicon sentiment →
  target `MoodVector`, smoothed via the state machine; wired into the agent so mood
  moves with the conversation (Neutral→Happy on warm input). Lexicon analyzer is a
  stand-in for a richer/LLM analyzer later.
- [x] **F1.7** `jaxson-agent` orchestration loop: per-turn retrieve → prompt (persona +
  state-gated hints + memories + history) → reply → extract+embed into graph → advance
  state. `Embedder` seam with deterministic `HashEmbedder`; mock-driven end-to-end demo.
  (Mood currently from `RelationshipState`; richer affect is F1.6.)
- [x] **F1.8a** `jaxson-face`: pure face geometry (`mood` + time → eye openness, mouth
  curve/openness, blink, gaze; mutation-graded) **plus a software rasterizer** to a B/W
  `Bitmap`, validated headlessly by property tests + ASCII inspection (`raster_demo`).
- [x] **F1.8b** `jaxson-app` egui shell: window showing the animated face (the
  `jaxson-face` bitmap, refreshed each frame) above a chat box, wired to a mock-backed
  agent — the face reacts live to the sentiment of typed input. Excluded from the
  workspace/CI (native GUI); run on macOS. Reply text is canned until a real model.
- [ ] **F1.9** Chat view text I/O.
- [x] **F1.10** Memory inspector: a window to browse / search / edit / delete memories
  (deleting a node also drops its edges). `MemoryGraph::search` + `remove_edge` (pure,
  tested); `Agent::graph_mut` for curation; egui inspector in the app.
- [x] **F1.11** Proactive question-asking gated by `familiarity` (`jaxson-agent::curiosity`,
  pure + mutation-graded): a getting-to-know-you curriculum (Person → Preference → Event →
  Fact) targets questions at gaps in the graph, so Jaxson asks about what it *doesn't* yet
  know and stops re-asking once answered. Three tiers — onboarding leads every turn,
  acquainted gently nudges remaining gaps, familiar-with-no-gaps just converses.
- [ ] **F1.12** Structured local logging across the loop (NFR-4).
- [x] **F1.13** Persistence wired into the app (behind `jaxson-app`'s `sqlite` feature):
  load the graph on launch via `Agent::with_graph`, save after every turn and every
  inspector edit/delete. Encryption key generated/fetched from the macOS Keychain
  (`keyring`); DB lives in the app data dir. "Export JSON" button for a readable dump of
  the encrypted graph. Degrades to an ephemeral session (never fatal) if persistence is
  off or the Keychain is unavailable.

## Milestone v0.2 — Voice + safety

- [ ] **F2.1** `jaxson-perception` STT via whisper-rs / whisper.cpp (local).
- [ ] **F2.2** Local TTS with a child-friendly voice (resolve OQ-1).
- [ ] **F2.3** Voice-driven face: lip/mouth sync to TTS, listening cues in the eyes.
- [ ] **F2.4** `jaxson-safety`: output content filter + topic guardrails (FR-S1/S2).
- [ ] **F2.5** Parental-control mode (authenticated): review history/memories, tune
  guardrail strictness (FR-S3, resolve OQ-3).
- [ ] **F2.6** Privacy hardening: encryption-at-rest verification, log scrubbing.

## Milestone v0.3 — Depth & polish

- [ ] **F3.1** Long-term memory consolidation (decay/strengthen edges, summarize).
- [ ] **F3.2** Initiative engine: Jaxson starts conversations based on state/time.
- [ ] **F3.3** Richer emotion set + expressive transitions; personality tuning.
- [ ] **F3.4** Onboarding flow (first-run "getting to know you").
- [ ] **F3.5** Performance pass: model/quantization benchmarking (resolve OQ-2).

## Epic (long horizon) — Hardware bot

- [ ] **E1** Extract core into a headless runtime runnable off-Mac.
- [ ] **E2** Embodied expression: features moving "around the body" (FR-E5).
- [ ] **E3** Sensors/actuators abstraction; perception beyond mic.

---

_When you complete an item, check it off in the PR that delivers it and move any
follow-ups into the appropriate milestone._
