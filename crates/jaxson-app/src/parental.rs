//! Parental controls (FR-S3): a parent passcode (stored only as a salted hash via
//! [`jaxson_safety::PasscodeHash`]) and the chosen guardrail [`Strictness`], persisted as
//! JSON in the data dir so they survive restarts. Entering the passcode unlocks reviewing
//! memories and tuning strictness; without it a child session can't weaken its own
//! guardrails. The file holds no plaintext passcode, so it isn't sensitive.

use jaxson_safety::{PasscodeHash, Strictness};
use serde::{Deserialize, Serialize};

/// The persisted parental-control settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParentalConfig {
    /// The parent passcode hash, or `None` until first-run setup.
    pub passcode: Option<PasscodeHash>,
    /// Guardrail strictness applied to Jaxson's replies.
    #[serde(default)]
    pub strictness: Strictness,
}

impl ParentalConfig {
    /// Whether a parent passcode has been set yet.
    pub fn has_passcode(&self) -> bool {
        self.passcode.is_some()
    }

    /// Whether `attempt` matches the set passcode (always `false` if none is set).
    pub fn unlocks(&self, attempt: &str) -> bool {
        self.passcode.as_ref().is_some_and(|p| p.verify(attempt))
    }
}

fn config_path() -> Option<std::path::PathBuf> {
    crate::persist::data_dir().map(|d| d.join("parental.json"))
}

/// Load the parental config, or defaults (no passcode, Standard strictness) if absent or
/// unreadable — never fatal.
pub fn load() -> ParentalConfig {
    let Some(path) = config_path() else {
        return ParentalConfig::default();
    };
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persist the parental config as JSON in the data dir. Errors are returned for the UI to
/// surface (never panics).
pub fn save(config: &ParentalConfig) -> Result<(), String> {
    let dir = crate::persist::data_dir().ok_or("no data directory")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("parental.json");
    let json = serde_json::to_string_pretty(config).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}
