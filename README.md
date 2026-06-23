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
- **Native & fast.** Swift + SwiftUI with [MLX-Swift](https://github.com/ml-explore/mlx-swift)
  for Metal-accelerated local LLM inference on Apple Silicon.
- **Memory is the agent.** A knowledge graph of memories — and the relationship
  state (trust, familiarity, mood) those memories mutate — *is* the core algorithm
  that drives Jaxson's behavior and expression.
- **Expressive, not photorealistic.** A minimal face conveys emotion, driven by a
  dedicated affect engine rather than by the words the LLM happens to choose.
- **Software first, hardware later.** v1 is a macOS app. If it earns its keep, the
  same Swift core ports onto a physical bot.

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
swift build      # builds the core engine package
swift test       # runs the unit test suite
```

The macOS app target arrives with v0.1 (see the backlog).

## License

See [`LICENSE`](LICENSE).
