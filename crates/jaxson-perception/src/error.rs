use thiserror::Error;

/// Errors from the speech-perception layer.
#[derive(Debug, Error)]
pub enum PerceptionError {
    /// Loading the model failed (missing file, bad format, …).
    #[error("model load error: {0}")]
    ModelLoad(String),
    /// The backend failed while transcribing.
    #[error("transcription error: {0}")]
    Backend(String),
    /// The supplied audio was unusable (e.g. empty).
    #[error("invalid audio: {0}")]
    Audio(String),
}
