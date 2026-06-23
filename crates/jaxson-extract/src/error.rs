use thiserror::Error;

/// Failures while extracting memories from a conversation.
#[derive(Debug, Error, PartialEq)]
pub enum ExtractError {
    /// The model produced nothing parseable.
    #[error("the model returned no usable content")]
    EmptyResponse,
    /// The model's JSON could not be parsed.
    #[error("could not parse extraction JSON: {0}")]
    Json(String),
    /// A relation referenced a memory index that doesn't exist.
    #[error("relation references memory index {index} but only {count} were extracted")]
    BadRelationIndex { index: usize, count: usize },
    /// The underlying text generator failed.
    #[error("generation failed: {0}")]
    Generation(String),
}
