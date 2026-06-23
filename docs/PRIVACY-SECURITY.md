# Jaxson — Privacy & Security

Jaxson is a companion that, by design, learns intimate details about its user — and
the intended audience includes children. Privacy and security are therefore
first-class product requirements, not afterthoughts.

## 1. Privacy principles

1. **On-device only.** All LLM inference, memory storage, retrieval, STT, and TTS run
   locally. There is **no cloud inference and no telemetry**. The app functions fully
   offline.
2. **No data egress.** Core modules must never open a network connection for
   inference, memory, or analytics. (Model *downloads* are an explicit, user-initiated,
   one-time setup step — see §4.)
3. **User ownership.** Every memory is inspectable, editable, and truly deletable via
   the memory inspector (FR-M4). Deletion propagates to derived state.
4. **Data minimization.** Jaxson does not solicit sensitive data it doesn't need
   (FR-S4). It asks before forming memories about clearly sensitive topics.
5. **Transparency.** The user can always see what Jaxson knows and why (provenance on
   every memory).

## 2. Security model

- **Encryption at rest.** The memory/state store is encrypted on disk (SQLCipher or
  app-level encryption; final choice at v0.1). Keys live in the macOS Keychain.
- **Sandboxing.** The app runs sandboxed with least-privilege file access — only its
  own container and explicitly chosen model files.
- **Untrusted model output.** LLM output is never executed/evaluated and is sanitized
  before any privileged use; it is also passed through the safety filter (v0.2).
- **Parental-control boundary.** Reviewing history/memories and changing guardrail
  strictness requires authentication (passcode or Touch ID — OQ-3). A child session
  cannot weaken its own guardrails.
- **No secrets in git.** Enforced by `.gitignore` and a CI secret scan; user data,
  model weights, logs, and `.env` files are never committed.

## 3. Threat model (initial)

| Threat | Mitigation |
| ------ | ---------- |
| Local attacker reads memory DB | Encryption at rest; Keychain-held key |
| Malicious/poisoned model output (prompt injection, unsafe content) | Output sanitization; safety/content filter (v0.2); no execution of output |
| Child accessing inappropriate content | Topic guardrails + content filter (v0.2) |
| Child disabling their own safety controls | Parental-control auth boundary |
| Data exfiltration via the app | No network egress in core; sandbox; review of any new dependency for phone-home behavior |
| Sensitive data leaking into logs | Structured logs scrubbed of raw sensitive content; logs git-ignored and local |
| Supply-chain risk in dependencies | Pin versions (`Cargo.lock`); review llama.cpp/whisper/SQLite deps; minimal dependency surface; `cargo audit` in CI |

## 4. Model acquisition

The local model is downloaded once, explicitly, by the user during setup (this is the
only sanctioned network activity, and it is not inference). Downloaded weights live
under `Models/` and are git-ignored. Integrity of downloaded weights should be
verified (checksum) before first use.

## 5. Review cadence

This document is revisited at each milestone boundary and whenever a new dependency,
data flow, or external interface is introduced.
