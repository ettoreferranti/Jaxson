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
- [x] **F1.1c** Latency benchmark (`latency_bench` example): measures time-to-first-token
  and throughput vs the NFR-3 target. On an M4 Pro, `llama3.1:8b` = 192 ms TTFT / 47.6 tok/s
  and `qwen3` = 141 ms / 44.4 tok/s — both **~10× under** the 1.5 s target with interactive
  throughput. Resolves OQ-2 (an ~8B Q4 model clears NFR-3 easily; default to a non-reasoning
  one so the full reply streams immediately).
- [x] **F1.2** `jaxson-memory` graph store: typed/weighted nodes + edges, `MemoryStore`
  trait + in-memory store (pure, mutation-graded), and encrypted-at-rest SQLite
  (SQLCipher) persistence behind the `sqlite` feature (round-trip + wrong-key tests).
- [x] **F1.3** Memory extraction (`jaxson-extract`): turn conversation turns into
  memory nodes/edges with provenance + confidence, via an LLM emitting JSON over the
  `TextGenerator` seam. Pure prompt+parse layer fully mutation-graded.
- [x] **F1.4** Hybrid retrieval (`jaxson-memory::retrieve`): cosine-similarity seeds +
  weighted graph spread (decay per hop), ranked top-k. Query embedding is an input;
  text→embedding wiring deferred to model integration.
- [x] **F1.4b** Embedder: real semantic embeddings from the local model — `LlamaEmbedder`
  (mean-pooled, L2-normalized `llama.cpp` embeddings) sharing the chat model's weights via
  a shared backend + `Arc<LlamaModel>` (loaded once). Adapted to the agent's `Embedder`
  seam in the app, replacing `HashEmbedder` when a model is loaded; embedding errors
  degrade to an empty vector. The embedding model is **independently selectable** — an
  `embed` dropdown / `$JAXSON_EMBED_MODEL` picks a separate model (e.g. `nomic-embed-text`)
  or reuses the chat model's weights ("same as chat", no extra load). `embed_probe` example
  verifies related text scores higher than unrelated. Populates node embeddings + query
  embeddings for F1.4 retrieval.
- [x] **F1.5** State machine extended (`jaxson-core::TopicAffinities`): per-topic affinity
  in `[-1,1]` with clamped, diminishing-returns reinforcement + decay + favorite query
  (pure, 100% mutation-graded). Wired into the agent — learned preferences and re-mentioned
  topics are nudged by each turn's sentiment, and a strongly-liked topic gets surfaced in
  the system prompt so Jaxson brings it up (agent stays 0-missed).
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
- [x] **F1.9** Chat view text I/O: wrapping, **selectable** transcript (read/copy long
  replies) with a styled speaker tag; full-width input with a hint and a Send button that's
  disabled while empty; "🧹 Clear chat" that resets the visible chat **and** the model's
  short-term context (`Agent::clear_history`) while keeping long-term memory. Agent also
  gained `respond_streaming` (live token callback, tested + 0-missed) — the seam a future
  non-blocking UI builds on.
- [x] **F1.9b** Non-blocking / streaming chat UI: generation runs on a background worker
  thread (agent + model + embedder bundled into a `Send` `Brain` that's handed to the
  worker and back), so the window stays live — the face keeps animating and the reply
  streams in via `Agent::respond_streaming` instead of beach-balling. Controls (send,
  pickers, mic, inspector) are disabled for the brief in-flight window. Compile + clippy
  clean on all feature sets; runtime verified on macOS.
- [x] **F1.10** Memory inspector: a window to browse / search / edit / delete memories
  (deleting a node also drops its edges). `MemoryGraph::search` + `remove_edge` (pure,
  tested); `Agent::graph_mut` for curation; egui inspector in the app.
- [x] **F1.11** Proactive question-asking gated by `familiarity` (`jaxson-agent::curiosity`,
  pure + mutation-graded): a getting-to-know-you curriculum (Person → Preference → Event →
  Fact) targets questions at gaps in the graph, so Jaxson asks about what it *doesn't* yet
  know and stops re-asking once answered. Three tiers — onboarding leads every turn,
  acquainted gently nudges remaining gaps, familiar-with-no-gaps just converses.
- [x] **F1.12** Structured local logging across the loop (NFR-4): `tracing` events from
  the agent (per-turn span, retrieval/learn counts, relationship-state transitions, and —
  the key win — previously-silent extraction failures) plus the app (turn timing, model
  loads, persistence). The app installs a stderr + daily rolling-file sink in the data dir
  (`jaxson.log`), filtered by `JAXSON_LOG` (default `info`). Raw user text is kept out of
  fields (privacy); logs stay on-device and are git-ignored. Agent stays 0-missed mutants.
- [x] **F1.13** Persistence wired into the app (behind `jaxson-app`'s `sqlite` feature):
  load the graph on launch via `Agent::with_graph`, save after every turn and every
  inspector edit/delete. Encryption key generated/fetched from the macOS Keychain
  (`keyring`); DB lives in the app data dir. "Export JSON" button for a readable dump of
  the encrypted graph. Degrades to an ephemeral session (never fatal) if persistence is
  off or the Keychain is unavailable.

## Milestone v0.2 — Voice + safety

- [x] **F2.1** `jaxson-perception` STT: pure `SpeechToText` seam + `Transcript` + audio
  utilities (mono downmix, RMS/silence for push-to-talk, silence-trim) + deterministic
  `MockStt` (mutation-graded), and a whisper.cpp (Metal) backend behind the `whisper`
  feature with a `whisper_transcribe` example. Verified end-to-end (ggml-tiny.en on a `say`
  clip → correct transcript). Live mic capture + push-to-talk UI is the follow-up (F2.1b).
- [x] **F2.1b** Microphone capture + push-to-talk in the app (behind `jaxson-app`'s
  `whisper` feature): a 🎤 button records via `cpal`, then stop → downmix to mono →
  `Audio::resample_to(16 kHz)` (new pure, mutation-graded resampler) → `SpeechToText` →
  the transcript is sent as the user's turn. STT model from `$JAXSON_WHISPER_MODEL`.
  Compile-verified both feature sets; the live audio path is run on macOS by the owner.
- [x] **F2.2** Local TTS (`jaxson-perception`), mirroring the STT split: pure `TextToSpeech`
  seam + `speakable_text` (strips `*action*` cues so they aren't read aloud) + deterministic
  `MockTts` (mutation-graded), and a **Piper** neural-TTS backend (`piper-rs`: VITS over ONNX
  Runtime + espeak-ng) behind the `piper` feature with a `piper_speak` example. Resolves OQ-1
  in favor of Piper (cross-platform, portable to the future hardware bot; the portable seam
  leaves room for a macOS `say` backend later). Spoken-replies wiring into the app is the
  follow-up **F2.2b**.
- [x] **F2.2b** Speak replies in the app (behind `jaxson-app`'s `piper` feature): the reply
  is synthesized on the **generation worker thread** (the `TextToSpeech` is bundled into the
  `Brain` alongside the model/embedder, F1.9b-style, so the window stays responsive) and
  played on the UI thread via `rodio` (resamples + maps to the output device). Synthesis is
  **per-sentence** (pure, mutation-graded `split_sentences`): each sentence is spoken as it's
  ready, so playback starts after the first short sentence instead of the whole paragraph,
  with a natural pause at each boundary. Pace is slowed for a calmer, kid-friendly delivery
  via Piper's `length_scale` (`PiperTts::with_length_scale`; default 1.2, override with
  `$JAXSON_PIPER_LENGTH_SCALE`). Voice from `$JAXSON_PIPER_VOICE`; degrades to silent (never
  fatal) with no voice or output device. Compile + clippy verified for default / `piper` /
  `whisper,piper`; live audio run on macOS by the owner.
- [x] **F2.3** Voice-driven face: the mouth lip-syncs to the playing reply and the eyes
  show a listening cue. Pure, mutation-graded layer: `jaxson-face`'s `Activity`
  (`Idle`/`Speaking{level}`/`Listening`) + `face_with` (speaking drives mouth openness from
  `level`; listening widens + steadies the eyes and perks the ears), and
  `jaxson-perception`'s peak-normalized `Audio::envelope`. App side (behind `piper`): a
  `SpeechAnimator` maps playback time → current loudness from each spoken sentence's
  envelope to drive the mouth; the mic-recording state (`whisper`) shows the listening cue
  (priority over speaking). Compile + clippy verified for default / `piper` / `whisper` /
  `whisper,piper`; live animation run on macOS by the owner.
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
