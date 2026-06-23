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
Swift + SwiftUI · MLX-Swift (Metal) · 7–8B quantized model · SQLite (encrypted) +
vectors. Core logic lives in the UI-free `JaxsonKit` SwiftPM package so it stays
testable and portable to a future hardware bot.

## Note on this environment
Full Xcode may not be installed (only Command Line Tools). `swift build` works;
`swift test` needs Xcode. To verify pure-Swift core logic without Xcode, compile the
sources with a `main.swift` harness via `swiftc` and run it.
