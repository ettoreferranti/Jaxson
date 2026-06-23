# Jaxson — Development Workflow

This project follows the practices we standardized on previous work.

## Prerequisites

- **Rust** (stable, via `rustup`) and **`cargo-mutants`**
  (`cargo install cargo-mutants --locked`). No Xcode needed.
- macOS **Command Line Tools** (`xcode-select --install`) — provides the C toolchain
  used to build the `llama.cpp`/whisper.cpp bindings (arrives in v0.1/v0.2). Full
  Xcode is **not** required.
- Apple Silicon Mac (32 GB+ recommended) for running the local model.

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
- Wired into CI (backlog **F0.9**) — surviving mutants in core crates block merge.

## CI (backlog F0.8)

GitHub Actions on macOS runners:
1. `cargo build`
2. `cargo test`
3. `cargo fmt --check` and `cargo clippy -D warnings`
4. `cargo mutants` on core crates (once F0.9 lands)
5. Secret scan / ensure no model weights or user data are committed.

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
