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
- [x] **F0.7** Buildable `JaxsonKit` SwiftPM package with `JaxsonCore` seeded
  (`MoodVector`, `Emotion`, `RelationshipState`) and unit tests.
- [ ] **F0.8** CI: GitHub Actions running `swift build` + `swift test` on macOS.
- [ ] **F0.9** Mutation-testing harness wired into CI (see DEVELOPMENT.md).

## Milestone v0.1 — Talking face + memory (text-first)

- [ ] **F1.1** `JaxsonLLM`: MLX-Swift integration — load a 7–8B quantized model,
  streaming generation, prompt assembly. Benchmark latency (NFR-3).
- [ ] **F1.2** `JaxsonMemory` graph store: nodes/edges + SQLite persistence
  (encrypted at rest).
- [ ] **F1.3** Memory extraction: turn conversation turns into memory nodes/edges
  with provenance + confidence.
- [ ] **F1.4** Vector retrieval + graph traversal hybrid retrieval.
- [ ] **F1.5** State machine: event-driven mutation of trust/familiarity/affinity
  with clamped transition functions (heavy unit + mutation tests).
- [ ] **F1.6** `JaxsonAffect` engine v1: state + sentiment → `MoodVector`.
- [ ] **F1.7** `JaxsonAgent` orchestration loop wiring the above.
- [ ] **F1.8** `JaxsonApp` SwiftUI shell + `FaceView` (animated eyes/nose/mouth from
  mood) + idle micro-motions (blink, gaze drift).
- [ ] **F1.9** `ChatView` text I/O.
- [ ] **F1.10** Memory inspector: browse / search / edit / delete memories & edges.
- [ ] **F1.11** Proactive question-asking behavior gated by `familiarity`.
- [ ] **F1.12** Structured local logging across the loop (NFR-4).

## Milestone v0.2 — Voice + safety

- [ ] **F2.1** `JaxsonPerception` STT via whisper.cpp (local).
- [ ] **F2.2** Local TTS with a child-friendly voice (resolve OQ-1).
- [ ] **F2.3** Voice-driven face: lip/mouth sync to TTS, listening cues in the eyes.
- [ ] **F2.4** `JaxsonSafety`: output content filter + topic guardrails (FR-S1/S2).
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
