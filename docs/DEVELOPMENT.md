# Jaxson — Development Workflow

This project follows the practices we standardized on previous work.

## Prerequisites

- **Full Xcode** (not just the Command Line Tools). The SwiftUI app, MLX/Metal, and
  the XCTest-based `swift test` suite all require it. With only the Command Line
  Tools installed, `swift build` works but `swift test` reports `no such module
  'XCTest'`. Install Xcode and run `sudo xcode-select -s /Applications/Xcode.app`.
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

- Core logic lives in pure, deterministic Swift modules (no SwiftUI) so it is fast
  and meaningful to test.
- `swift test` must pass before requesting review.
- Aim for high coverage on `JaxsonCore`, `JaxsonMemory`, `JaxsonAffect`, and the
  state-machine transition functions especially — these encode the agent's behavior.

### Mutation testing

Line/branch coverage proves code *ran*, not that tests *assert* the right thing.
We use **mutation testing** to grade test quality: the tool injects small faults
(mutants) into the code and checks that some test fails. Surviving mutants reveal
weak assertions.

- Tooling: [**muter**](https://github.com/muter-mutation-testing/muter) for Swift.
- Run locally:
  ```bash
  muter run            # mutate + run the suite, report surviving mutants
  ```
- Target: a high mutation score on the behavioral core (state machine, affect,
  memory extraction). New core logic should not *lower* the score.
- Wired into CI (backlog **F0.9**) — surviving mutants in core modules block merge.

## CI (backlog F0.8)

GitHub Actions on macOS runners:
1. `swift build`
2. `swift test`
3. `muter run` on core modules (once F0.9 lands)
4. Secret scan / ensure no model weights or user data are committed.

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
