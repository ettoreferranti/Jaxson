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

- **Encryption at rest.** The memory/state store is encrypted on disk with
  **SQLCipher** (`jaxson-memory`'s `sqlite` feature; chosen at v0.1 — see ADR A7).
  Opening with the wrong key fails. Keys live in the macOS Keychain. _(Dev-only escape
  hatch: `$JAXSON_DB_KEY` supplies the key directly and bypasses the Keychain, so an
  unsigned dev build doesn't re-prompt every launch. The DB stays encrypted, but a key in
  the environment is weaker than the Keychain — never set it for a real install.)_
- **Sandboxing.** The app runs sandboxed with least-privilege file access — only its
  own container and explicitly chosen model files.
- **Untrusted model output.** LLM output is never executed/evaluated and is sanitized
  before any privileged use; it is also passed through the safety filter (v0.2).
- **Parental-control boundary.** Reviewing memories and changing guardrail strictness
  requires a **parent passcode** (F2.5; OQ-3 resolved in favor of a passcode over Touch ID
  for portability). It's stored only as a salted, iterated SHA-256 hash
  (`jaxson-safety::PasscodeHash`) in `parental.json` — never plaintext — so a child session
  cannot weaken its own guardrails. _Threat model: this gate guards against the child using
  the device, not a determined attacker — a short kid-set passcode can't withstand offline
  brute force regardless of hashing; the OS account and the encrypted memory DB are the real
  perimeter. Touch ID may wrap it as an optional macOS unlock later._
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
