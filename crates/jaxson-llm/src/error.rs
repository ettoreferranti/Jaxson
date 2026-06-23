use thiserror::Error;

/// Errors surfaced by a [`TextGenerator`](crate::TextGenerator).
#[derive(Debug, Error)]
pub enum LlmError {
    /// The model file could not be found or loaded.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Generation failed mid-stream (tokenization, decode, or sampling).
    #[error("generation failed: {0}")]
    Generation(String),

    /// A backend-specific or otherwise uncategorized failure.
    #[error("backend error: {0}")]
    Backend(String),
}
