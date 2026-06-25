# Jaxson — Requirements

Status: **living document.** Captured in the initial requirements-engineering
session and updated as the product evolves. Decisions marked ✅ are confirmed by the
product owner.

## 1. Vision

Jaxson is a privacy-first virtual companion that runs locally on macOS. It is
inspired by the B-bot from *Ron's Gone Wrong*, but inverts the film's premise: where
B-bots are pre-loaded with scraped personal data, **Jaxson begins knowing nothing**
and earns its understanding of the user through conversation. The companion is
embodied by a deliberately minimal animated face (two eyes and a mouth) that
communicates emotion expressively.

The long-term goal is a physical bot; the near-term goal is a polished macOS app.

## 2. Confirmed decisions

| # | Decision | Choice |
| - | -------- | ------ |
| D1 | Primary stack | ✅ Rust (switched from Swift — see ADR A6 in ARCHITECTURE.md) |
| D1a | Face / UI | ✅ Rust GUI via [`egui`](https://github.com/emilk/egui) (custom 2D painting for the face; single binary, portable toward embedded) |
| D2 | Local LLM runtime | ✅ `llama.cpp` with Metal, via Rust bindings |
| D3 | Target hardware | ✅ Apple Silicon, 32 GB+ unified memory |
| D4 | Default model size | ✅ 7–8B params, quantized GGUF (e.g. Llama-3.1-8B-Instruct / Qwen2.5-7B) |
| D5 | Interaction modality | ✅ Voice + text (local STT via whisper.cpp, local TTS); text fallback always present |
| D6 | Memory capture style | ✅ Organic & automatic, but fully reviewable/editable/deletable via an inspector |
| D7 | Memory core model | ✅ Knowledge graph **as a state machine** — nodes/edges plus state variables (trust, familiarity, mood) that memories mutate and that gate behavior |
| D8 | Expression driver | ✅ Separate **affect engine** — computes a mood vector from graph state + conversation sentiment, drives the face independently of the LLM's wording |
| D9 | Primary audience | ✅ Kid-friendly / film-accurate — requires safety guardrails, content filtering, parental controls |
| D10 | v0.1 milestone | ✅ "Talking face + memory": text chat with local LLM, animated face, organic memory capture + inspector |

## 3. Functional requirements

### 3.1 Conversation
- **FR-C1** Jaxson holds a natural multi-turn conversation using a local LLM. No
  network calls for inference.
- **FR-C2** Text input/output in v0.1; voice in/out (STT + TTS) from v0.2.
- **FR-C3** Conversation is grounded in retrieved memories relevant to the current
  topic (graph + vector retrieval).

### 3.2 Memory & learning
- **FR-M1** Jaxson proactively asks appropriate, age-appropriate questions to learn
  about the user, paced so it does not feel like an interrogation.
- **FR-M2** Memories are extracted automatically from conversation and stored as
  graph nodes with typed edges (e.g. *likes*, *knows*, *happened-on*).
- **FR-M3** Each memory records provenance (when/how learned) and a confidence.
- **FR-M4** A **memory inspector** lets the user browse, search, edit, and delete any
  memory and any edge. Deletion is real and propagates to derived state.
- **FR-M5** Memories mutate **relationship state variables** (trust, familiarity,
  and per-topic affinities). State is persisted.
- **FR-M6** State variables **gate behavior**: e.g. low familiarity → more questions;
  higher trust → more personal topics unlocked.

### 3.3 Expression / face
- **FR-E1** A minimal face renders two eyes and a mouth. (The nose was dropped as a
  design choice during F1.8 — a cleaner, B-bot-style look.)
- **FR-E2** A dedicated affect engine produces a continuous **mood vector**
  (valence/arousal) plus a discrete dominant emotion; the face animates from it.
- **FR-E3** Idle micro-motions (blinking, gaze drift, subtle breathing) keep the
  face alive between turns.
- **FR-E4** Expression is driven by internal state + sentiment, **not** by parsing
  the LLM's chosen words, so personality stays consistent.
- **FR-E5** (Future / hardware) Eyes and features may move beyond the face region —
  "around the body" — when embodied. The software face is designed so this extends
  naturally.

### 3.4 Safety (audience = kids)
- **FR-S1** All LLM output passes a safety/content filter before display.
- **FR-S2** Topic guardrails block age-inappropriate content and unsafe advice.
- **FR-S3** Parental controls: a separate, authenticated mode to review memories,
  conversation history, and adjust guardrail strictness.
- **FR-S4** Jaxson never solicits sensitive personal data it does not need (privacy
  by design), and never sends data off-device.

## 4. Non-functional requirements

- **NFR-1 Privacy:** 100% on-device. No telemetry, no cloud inference, no analytics
  beacons. Memory store encrypted at rest. See `docs/PRIVACY-SECURITY.md`.
- **NFR-2 Security:** Threat-modeled; parental-control auth; least-privilege file
  access; no unsanitized execution of model output.
- **NFR-3 Performance:** First-token latency target < 1.5 s and interactive
  generation on a 32 GB Apple Silicon Mac with a 7–8B quantized model, while leaving
  headroom for STT/TTS.
- **NFR-4 Observability:** Extensive structured local logging (decisions, state
  transitions, retrievals) to support debugging and tuning. Logs stay on-device and
  are scrubbed of raw sensitive content where feasible.
- **NFR-5 Testability:** Core logic lives in pure, deterministic Rust crates with
  high unit-test coverage, validated by **mutation testing** (see DEVELOPMENT.md).
- **NFR-6 Portability:** The non-UI core (memory, affect, safety, orchestration) is
  isolated from the GUI so it can later run on a hardware bot.

## 5. Out of scope (for now)

- Cloud sync / multi-device memory.
- Photorealistic avatar or 3D rendering.
- Multi-user accounts on one install (single owner per install in v1).
- The physical bot hardware (tracked as a long-horizon epic).

## 6. Open questions

- OQ-1: Which TTS engine gives the best on-device child-friendly voice from Rust?
  (Candidates: Piper via bindings, a small local neural TTS, or shelling out to macOS
  `say`/AVSpeech via FFI.) — to revisit at v0.2.
- OQ-2: ~~Exact quantization (4-bit vs 8-bit) and model pick — benchmark at v0.1.~~
  **Resolved (F1.1c):** an ~8B 4-bit (Q4) GGUF comfortably clears NFR-3 on Apple Silicon.
  Benchmarked on an M4 Pro (`latency_bench`): `llama3.1:8b` first-token 192 ms / 47.6 tok/s,
  `qwen3` 141 ms / 44.4 tok/s — both ~10× under the 1.5 s target with headroom for STT/TTS.
  Default pick: a non-reasoning ~8B Q4 model (e.g. `llama3.1:8b`) so the whole reply streams
  immediately; reasoning models (qwen3) are just as fast raw but spend tokens "thinking".
- OQ-3: Parental-control authentication mechanism (passcode vs. Touch ID). — v0.2.
