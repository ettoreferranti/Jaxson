//! Local structured logging sink (NFR-4).
//!
//! The pure crates (notably `jaxson-agent`) emit `tracing` events — turn spans, retrieval
//! and learning counts, state transitions, extraction failures. This installs the sink:
//! human-readable lines to **stderr** and to a **daily rolling file** in Jaxson's data dir
//! (next to the encrypted memory DB). Nothing leaves the device; the files are git-ignored.
//!
//! Control verbosity with `JAXSON_LOG` (e.g. `JAXSON_LOG=debug`, or
//! `JAXSON_LOG=jaxson_agent=debug,info`); the default is `info`. The ONNX Runtime (`ort`,
//! used by Piper TTS) logs reams of graph-optimization detail at INFO, so it's pinned to
//! `warn` unless `JAXSON_LOG` mentions `ort` explicitly.

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Install the global subscriber. Returns a guard that must stay alive for the whole
/// process (dropping it flushes the non-blocking file writer). If no data dir is
/// available the file sink is skipped and only stderr is used.
#[must_use]
pub fn init() -> Option<tracing_appender::non_blocking::WorkerGuard> {
    let filter = || {
        let directives = std::env::var("JAXSON_LOG").unwrap_or_else(|_| "info".to_string());
        let mut filter = EnvFilter::new(&directives);
        // Quiet ORT's chatty graph-optimization INFO logs by default, unless the user has
        // said something about `ort` themselves.
        if !directives.contains("ort") {
            filter = filter.add_directive("ort=warn".parse().expect("valid directive"));
        }
        filter
    };

    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(filter());

    let (file_layer, guard) = match log_dir() {
        Some(dir) => {
            let (writer, guard) =
                tracing_appender::non_blocking(tracing_appender::rolling::daily(dir, "jaxson.log"));
            let layer = fmt::layer()
                .with_ansi(false)
                .with_writer(writer)
                .with_filter(filter());
            (Some(layer), Some(guard))
        }
        None => (None, None),
    };

    tracing_subscriber::registry()
        .with(stderr_layer)
        .with(file_layer)
        .init();

    if guard.is_none() {
        tracing::warn!("no data directory available; logging to stderr only");
    }
    guard
}

/// The directory the log file lives in (the shared data dir), created if missing.
fn log_dir() -> Option<std::path::PathBuf> {
    let dir = crate::persist::data_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}
