# Jaxson — Development Workflow

This project follows the practices we standardized on previous work.

## Prerequisites

- **Rust** (stable, via `rustup`) and **`cargo-mutants`**
  (`cargo install cargo-mutants --locked`). No Xcode needed.
- macOS **Command Line Tools** (`xcode-select --install`) + **`cmake`**
  (`brew install cmake`) — only needed to build the `llama` feature (native
  `llama.cpp`). Full Xcode is **not** required, and neither is cmake for the default
  build/test/mutants.
- Apple Silicon Mac (32 GB+ recommended) for running the local model.

## The `llama` feature (native backend)

`jaxson-llm` builds with no native deps by default and uses a deterministic
`MockGenerator`. The real model is behind the `llama` cargo feature:

```bash
# Build/run the native llama.cpp + Metal backend (needs cmake + a GGUF model):
cargo build -p jaxson-llm --features llama
```

It loads a local GGUF model at runtime (e.g. a 7–8B instruct model, Q4_K_M). Model
weights are downloaded by the user, never committed (see PRIVACY-SECURITY.md §4), and
are git-ignored. CI does **not** build this feature — it stays on the pure layer.

## The `sqlite` feature (encrypted persistence)

`jaxson-memory` builds with no native deps by default (pure graph + `InMemoryStore`).
The encrypted on-disk `SqliteStore` (SQLCipher) is behind the `sqlite` feature:

```bash
# Builds vendored SQLCipher + OpenSSL (needs a C toolchain + perl):
cargo test -p jaxson-memory --features sqlite
```

Unlike the model, this is fully testable, so **CI does build and test it** (the
`sqlite` job). The DB is encrypted at rest; the key is supplied at `open()` (from the
Keychain in the app), and opening with the wrong key fails. `*.jaxsondb`/`*.sqlite`
are git-ignored.

## Branching & PRs

- **`main` is protected and always green.** No direct commits to `main`.
- Every change lands on a **feature branch**, named by type:
  - `feat/…` new features · `fix/…` bug fixes · `chore/…` tooling/docs ·
    `refactor/…` · `test/…`
- Open a **Pull Request** for every branch. **The product owner (Ettore) reviews and
  merges every PR.** Claude does not self-merge.
- Keep PRs small and focused — ideally one backlog item per PR.
- Every PR must, in the same PR:
  - update `docs/ARCHITECTURE.md` if structure changed,
  - update `docs/BACKLOG.md` checkboxes,
  - include tests for new logic.

## Testing

- Core logic lives in pure, deterministic Rust crates (no GUI) so it is fast and
  meaningful to test.
- `cargo test` must pass before requesting review.
- Aim for high coverage on `jaxson-core`, `jaxson-memory`, `jaxson-affect`, and the
  state-machine transition functions especially — these encode the agent's behavior.

### Mutation testing

Line/branch coverage proves code *ran*, not that tests *assert* the right thing.
We use **mutation testing** to grade test quality: the tool injects small faults
(mutants) into the code and checks that some test fails. Surviving mutants reveal
weak assertions.

- Tooling: [**`cargo-mutants`**](https://mutants.rs/). Config in `.cargo/mutants.toml`.
- Run locally:
  ```bash
  cargo mutants        # mutate + run the suite, report surviving mutants
  ```
- **Target: zero missed mutants on the behavioral core** (state machine, affect,
  memory extraction). `jaxson-core` is at **70/70 viable mutants caught (100%)** as
  of the foundation PR. New core logic must not introduce surviving mutants.
- "Unviable" mutants (ones that don't compile) are reported separately and are fine.
- **Excluded** (`exclude_globs` in `.cargo/mutants.toml`): the feature-gated native
  backends (`llama.rs`, `sqlite.rs`) and the face **rasterizer** (`raster.rs`). These
  are rendering/FFI glue verified by their own tests, on-device runs, or visual
  inspection — not pure behavioral logic. Everything else must have zero survivors.
- Wired into CI (backlog **F0.9**) — surviving mutants in core crates block merge.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every PR and on pushes to `main`,
in three jobs:
1. **fmt · clippy · build · test** — `cargo fmt --check`, `cargo clippy --all-targets
   -D warnings`, `cargo build`, `cargo test`.
2. **mutation testing** — `cargo mutants --package jaxson-core`; a surviving viable
   mutant fails the job and blocks merge.
3. **guard** — fails if any model weights, user data, or secrets are tracked.

Runs on `ubuntu-latest` while the workspace is pure Rust (faster, ~10× cheaper than
macOS minutes). Switch to `macos-latest` once native Metal deps (`llama.cpp`,
whisper) arrive in v0.1/v0.2.

## Logging

We log a lot, on purpose (NFR-4): decisions, state transitions, retrievals, timings —
all **structured** and **local**. Logs never leave the device and are scrubbed of raw
sensitive content where feasible. Logs are git-ignored.

## Privacy & security in the workflow

- Never commit model weights, `*.gguf`/`*.safetensors`, user data, `*.sqlite`, or
  `.env` (enforced by `.gitignore`).
- Treat LLM output as untrusted: never execute or eval it; sanitize before any
  privileged use.
- See `docs/PRIVACY-SECURITY.md` for the full model.

## Commit messages

- Imperative mood, concise subject, body explains *why*.
- Reference the backlog item (e.g. `F1.5`) where applicable.
