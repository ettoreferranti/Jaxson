//! Persistence for Jaxson's memory graph: load on launch, save after every change.
//!
//! Encrypted-at-rest via SQLCipher (`jaxson-memory`'s `sqlite` feature). The encryption
//! key lives in the **macOS Keychain** — generated on first run, fetched thereafter — so
//! the on-disk DB is unreadable without the logged-in user's Keychain (ADR A7 /
//! docs/PRIVACY-SECURITY.md). For development, `$JAXSON_DB_KEY` supplies the key directly
//! and skips the Keychain prompt (see [`db_key`]). Without the `sqlite` feature the app
//! still runs, just ephemerally: memory lives in RAM and is lost on quit.
//!
//! Failures here are never fatal — they degrade to an ephemeral session and surface a
//! message through [`Persistence::status`] so the UI can show it.

use jaxson_memory::{Edge, MemoryGraph, MemoryNode};

/// Jaxson's per-user data directory (`~/Library/Application Support/Jaxson` on macOS).
/// Shared with the logging sink so memory and logs sit side by side.
pub(crate) fn data_dir() -> Option<std::path::PathBuf> {
    directories::ProjectDirs::from("com", "jaxson", "Jaxson").map(|d| d.data_dir().to_path_buf())
}

/// The memory store, plus a human-readable status describing where (or whether) it
/// persists. Holds an open encrypted DB when the `sqlite` feature is on and opening
/// succeeded; otherwise it's a no-op that keeps the app running ephemerally.
pub struct Persistence {
    #[cfg(feature = "sqlite")]
    store: Option<jaxson_memory::SqliteStore>,
    status: String,
}

impl Persistence {
    /// Open the encrypted store, falling back to an ephemeral session on any failure
    /// (missing feature, Keychain denied, I/O error). Never panics.
    pub fn open() -> Self {
        #[cfg(feature = "sqlite")]
        {
            match open_store() {
                Ok((store, path)) => {
                    tracing::info!(path = %path.display(), "opened encrypted memory store");
                    Persistence {
                        store: Some(store),
                        status: format!("memory: {}", path.display()),
                    }
                }
                Err(e) => {
                    let key_from_env = std::env::var(KEY_ENV)
                        .map(|v| !v.is_empty())
                        .unwrap_or(false);
                    let status = ephemeral_status(&e, key_from_env);
                    tracing::warn!(
                        error = %e,
                        "could not open memory store; running EPHEMERALLY — memories will NOT survive a restart: {status}"
                    );
                    Persistence {
                        store: None,
                        status,
                    }
                }
            }
        }
        #[cfg(not(feature = "sqlite"))]
        {
            Persistence {
                status: "memory not persisted (build with --features sqlite)".to_string(),
            }
        }
    }

    /// A short message about where memory persists, or why it doesn't.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// The graph loaded from disk — an empty graph if nothing is stored or persistence
    /// is off. A load error degrades to empty and is reported via [`status`](Self::status).
    pub fn load(&mut self) -> MemoryGraph {
        #[cfg(feature = "sqlite")]
        if let Some(store) = &self.store {
            use jaxson_memory::MemoryStore;
            match store.load() {
                Ok(graph) => return graph,
                Err(e) => self.status = format!("load failed: {e}"),
            }
        }
        MemoryGraph::new()
    }

    /// Persist the whole graph. Failures are surfaced via [`status`](Self::status) but
    /// never interrupt the conversation.
    pub fn save(&mut self, graph: &MemoryGraph) {
        #[cfg(feature = "sqlite")]
        if let Some(store) = &mut self.store {
            use jaxson_memory::MemoryStore;
            match store.save(graph) {
                Ok(()) => tracing::debug!(nodes = graph.node_count(), "saved memory"),
                Err(e) => {
                    tracing::error!(error = %e, "failed to save memory");
                    self.status = format!("save failed: {e}");
                }
            }
        }
        let _ = graph;
    }
}

/// A readable, serializable snapshot of the graph for the JSON export (the DB itself is
/// encrypted, so this is the way to eyeball what Jaxson remembers).
#[derive(serde::Serialize)]
struct GraphExport<'a> {
    nodes: Vec<&'a MemoryNode>,
    edges: &'a [Edge],
}

/// Dump the graph as pretty JSON into the data directory and return the file path.
/// Works regardless of the `sqlite` feature — it's a plaintext debug view.
pub fn export_json(graph: &MemoryGraph) -> Result<std::path::PathBuf, String> {
    let dir = data_dir().ok_or("no data directory")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("memory-export-{stamp}.json"));
    let export = GraphExport {
        nodes: graph.nodes().collect(),
        edges: graph.edges(),
    };
    let json = serde_json::to_string_pretty(&export).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(feature = "sqlite")]
const KEYCHAIN_SERVICE: &str = "com.jaxson.Jaxson";
#[cfg(feature = "sqlite")]
const KEYCHAIN_ACCOUNT: &str = "memory-db-key";

/// Open (creating if needed) the encrypted DB under the data dir, keyed from Keychain.
#[cfg(feature = "sqlite")]
fn open_store() -> Result<(jaxson_memory::SqliteStore, std::path::PathBuf), String> {
    let dir = data_dir().ok_or("no data directory")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("memory.jaxsondb");
    let key = db_key()?;
    let store = jaxson_memory::SqliteStore::open(&path, &key).map_err(|e| e.to_string())?;
    Ok((store, path))
}

/// Environment variable that supplies the DB key directly, bypassing the Keychain.
#[cfg(feature = "sqlite")]
const KEY_ENV: &str = "JAXSON_DB_KEY";

/// Turn an open failure into a clear, user-facing status line. SQLCipher reports a wrong
/// key as "file is not a database" (it can't decrypt the header) — overwhelmingly because
/// the DB was created under a *different* key than the one in play now (e.g. the Keychain
/// vs a `$JAXSON_DB_KEY` passphrase). That used to degrade silently to an in-memory session
/// and quietly lose memories on quit, so name the cause and the fix instead of echoing a
/// cryptic backend error. `key_from_env` says which key we tried, to point the right way.
#[cfg(feature = "sqlite")]
fn ephemeral_status(error: &str, key_from_env: bool) -> String {
    if error.contains("not a database") {
        let fix = if key_from_env {
            "it was likely made with a different key (e.g. the Keychain). Unset $JAXSON_DB_KEY to use the original, or delete memory.jaxsondb to start fresh under this key"
        } else {
            "it was likely made with a different key (e.g. a $JAXSON_DB_KEY passphrase). Set that key, or delete memory.jaxsondb to start fresh"
        };
        format!("⚠ memory NOT saved — wrong DB key: {fix}.")
    } else {
        format!("memory not persisted ({error})")
    }
}

/// The DB encryption key. In development, set `$JAXSON_DB_KEY` to supply the key directly
/// and skip the Keychain — macOS re-prompts for Keychain access on every launch of an
/// unsigned `cargo build` binary (its identity changes each rebuild), which is tedious
/// while iterating. The DB stays encrypted either way; this only changes where the key
/// comes from. **Don't** set it for a real install: a key in the environment is far weaker
/// than one sealed in the Keychain (see docs/PRIVACY-SECURITY.md). Otherwise the key is
/// fetched from the Keychain (generated on first run).
#[cfg(feature = "sqlite")]
fn db_key() -> Result<String, String> {
    match std::env::var(KEY_ENV) {
        Ok(key) if !key.is_empty() => {
            tracing::warn!(
                "using DB key from ${KEY_ENV} (Keychain bypassed) — dev only, not for a real install"
            );
            Ok(key)
        }
        _ => keychain_key(),
    }
}

/// Fetch the DB encryption key from the Keychain, generating and storing a fresh random
/// one on first run.
#[cfg(feature = "sqlite")]
fn keychain_key() -> Result<String, String> {
    let entry = keyring::Entry::new(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
        .map_err(|e| format!("keychain: {e}"))?;
    match entry.get_password() {
        Ok(key) => Ok(key),
        Err(keyring::Error::NoEntry) => {
            let key = new_key();
            entry
                .set_password(&key)
                .map_err(|e| format!("keychain set: {e}"))?;
            Ok(key)
        }
        Err(e) => Err(format!("keychain get: {e}")),
    }
}

/// A fresh ~256-bit random passphrase: two random v4 UUIDs, hex, no dashes.
#[cfg(feature = "sqlite")]
fn new_key() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::ephemeral_status;

    #[test]
    fn wrong_key_with_env_key_suggests_unset_or_delete() {
        // SQLCipher's wrong-key symptom, while a $JAXSON_DB_KEY is in play.
        let s = ephemeral_status("storage backend error: file is not a database", true);
        assert!(s.contains("wrong DB key"), "{s}");
        assert!(
            s.contains("Unset $JAXSON_DB_KEY") || s.contains("delete"),
            "{s}"
        );
    }

    #[test]
    fn wrong_key_without_env_key_suggests_setting_it() {
        let s = ephemeral_status("file is not a database", false);
        assert!(s.contains("wrong DB key"), "{s}");
        assert!(s.contains("$JAXSON_DB_KEY"), "{s}");
    }

    #[test]
    fn unrelated_errors_pass_through_verbatim() {
        let s = ephemeral_status("no data directory", false);
        assert!(s.contains("no data directory"), "{s}");
        assert!(!s.contains("wrong DB key"), "{s}");
    }
}
