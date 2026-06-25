# Jaxson

A privacy-first **virtual companion** that runs entirely on your Mac. Jaxson talks
with you, learns about you by asking and observing, and expresses itself through a
deliberately simple animated face — two eyes and a mouth — inspired by the
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

🚧 **v0.1 in progress.** The full companion brain runs end-to-end headless
(conversation → retrieve → reply → learn → remember → mood), and a macOS app shows the
animated face + chat. Still mock-driven — wiring the real local model (`--features
llama`) and voice are next. See the docs below.

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

# See a conversation turned into memory-graph nodes and edges (mock backend):
cargo run --example extract_demo -p jaxson-extract

# See the whole conversation loop: retrieve → reply → learn → remember (mock backend):
cargo run --example agent_demo -p jaxson-agent

# See the face rendered as ASCII for several moods (no GUI):
cargo run --example raster_demo -p jaxson-face
```

### Run the app (macOS)

```bash
# Opens a window with Jaxson's animated face + a chat box, and a "Memories" inspector
# to browse / search / edit / delete what it has learned. The face reacts live to the
# sentiment of what you type. (Excluded from the workspace; build it directly.)
cargo run --manifest-path crates/jaxson-app/Cargo.toml
```

### Use the real local model (macOS, Apple Silicon)

By default the app uses a mock "demo brain" (canned replies; the face still reacts to
your sentiment). To run Jaxson on a real local LLM:

1. Install `cmake` (`brew install cmake`) — needed to build `llama.cpp`.
2. Download a 7–8B instruct **GGUF** (e.g. Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct,
   `Q4_K_M`). It stays on your machine and is never committed.
3. Smoke-test the model headlessly first:
   ```bash
   cargo run -p jaxson-llm --example llama_chat --features llama -- /path/to/model.gguf "Hi!"
   ```
4. Run the app with the real brain — `JAXSON_MODEL` takes an installed Ollama model
   **name** (template auto-selected) or a path to a `.gguf`:
   ```bash
   JAXSON_MODEL=llama3.1 JAXSON_EMBED_MODEL=nomic-embed-text \
     cargo run --manifest-path crates/jaxson-app/Cargo.toml --features sqlite,llama
   ```
   `JAXSON_EMBED_MODEL` (a model name) optionally embeds with a separate model; omit it to
   reuse the chat model. When you pass `JAXSON_MODEL` as a **path**, also set
   `JAXSON_TEMPLATE=llama3` for Llama-3 models (default ChatML); resolving by name handles
   this for you. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

**Using Ollama as a model manager:** if you already have models in Ollama, just build
with `--features llama` — the app shows **dropdowns of your installed Ollama models** for
both the chat model and the embedding model (it reads `~/.ollama` directly; Ollama models
are GGUF, no conversion needed). Pick one to load it; no need to hunt down blob paths.

The macOS app crate arrives with v0.1 (see the backlog).

## License

See [`LICENSE`](LICENSE).
