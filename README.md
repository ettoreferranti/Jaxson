# Jaxson

A privacy-first **virtual companion** that runs entirely on your Mac. Jaxson talks
with you, learns about you by asking and observing, and expresses itself through a
deliberately simple animated face — two eyes, a nose, and a mouth — inspired by the
B-bot from the film *Ron's Gone Wrong*.

Unlike the film's B-bots, which are pre-loaded with scraped personal data, **Jaxson
starts knowing nothing about you**. It builds an understanding organically through
conversation, and every memory it forms is stored locally, inspectable, and
deletable by you. That "defect" — Ron's defect — is the whole point.

## Principles

- **Local-first & private.** All inference and all memory stay on-device. Nothing
  leaves the machine. See [`docs/PRIVACY-SECURITY.md`](docs/PRIVACY-SECURITY.md).
- **Native & fast.** Rust core with Metal-accelerated local LLM inference via
  `llama.cpp` bindings on Apple Silicon. Chosen over Swift for best-in-class mutation
  testing, a frictionless toolchain, and a clean path onto a future hardware bot.
- **Memory is the agent.** A knowledge graph of memories — and the relationship
  state (trust, familiarity, mood) those memories mutate — *is* the core algorithm
  that drives Jaxson's behavior and expression.
- **Expressive, not photorealistic.** A minimal face conveys emotion, driven by a
  dedicated affect engine rather than by the words the LLM happens to choose.
- **Software first, hardware later.** v1 is a macOS app. If it earns its keep, the
  same Rust core ports onto a physical bot.

## Status

🚧 **Pre-v0.1 — foundation.** This repo currently contains requirements,
architecture, the feature backlog, and a tested core skeleton. See the docs below.

## Documentation

| Doc | Purpose |
| --- | --- |
| [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) | Product requirements captured in the requirements-engineering session |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture — kept up to date at all times |
| [`docs/BACKLOG.md`](docs/BACKLOG.md) | Prioritized feature backlog |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Workflow: branching, PRs, testing & mutation testing, CI |
| [`docs/PRIVACY-SECURITY.md`](docs/PRIVACY-SECURITY.md) | Privacy & security model and threat model |

## Building

```bash
cargo build              # builds the workspace
cargo test               # runs the unit test suite
cargo mutants            # runs mutation testing on the core (see docs/DEVELOPMENT.md)

# See the relationship state machine evolve over a scripted conversation:
cargo run --example state_machine_demo -p jaxson-core

# See LLM prompt assembly + streaming (deterministic mock backend, no model needed):
cargo run --example chat_demo -p jaxson-llm
```

The native `llama.cpp` + Metal backend is behind a cargo feature
(`cargo build -p jaxson-llm --features llama`); it needs `cmake` and a local GGUF
model. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

The macOS app crate arrives with v0.1 (see the backlog).

## License

See [`LICENSE`](LICENSE).
