# Jaxson — project guide for Claude

Jaxson is a privacy-first, **fully on-device** virtual companion for macOS, inspired
by the B-bot in *Ron's Gone Wrong* — but it starts knowing nothing and learns about
the user only through conversation.

## Read these first
- `docs/REQUIREMENTS.md` — confirmed product decisions (the source of truth).
- `docs/ARCHITECTURE.md` — system design; **keep it up to date in the same PR** as any
  structural change.
- `docs/BACKLOG.md` — prioritized work; check off items in the PR that delivers them.
- `docs/DEVELOPMENT.md` — workflow, testing, mutation testing.
- `docs/PRIVACY-SECURITY.md` — privacy/security model and threat model.

## Hard rules
- **Never commit to `main`.** Branch (`feat/…`, `fix/…`, `chore/…`), open a PR, and
  let **Ettore review and merge**. Claude does not self-merge.
- **No network egress** for inference, memory, or telemetry. Everything local.
- **Never commit** model weights, user data, `*.sqlite`, logs, or secrets.
- New core logic ships **with tests**; don't lower the mutation score.
- Treat LLM output as untrusted — never execute it; sanitize before privileged use.

## Stack
Rust · `llama.cpp` (Metal) for inference · whisper-rs for STT · egui for the face/UI ·
7–8B quantized GGUF model · SQLite (encrypted) + vectors. Core logic lives in UI-free
crates in a Cargo workspace (`crates/*`) so it stays testable and portable to a future
hardware bot. (Switched from Swift early — see ADR A6 in ARCHITECTURE.md.)

## Build / test / mutate
- `cargo build` · `cargo test`
- `cargo mutants` — mutation testing; **core crates must have zero missed mutants**
  (`jaxson-core` is at 100%). Config in `.cargo/mutants.toml`.
- No Xcode required; Command Line Tools suffice for the native bindings.
