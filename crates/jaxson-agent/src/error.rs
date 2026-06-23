use thiserror::Error;

use jaxson_extract::ExtractError;

/// Failures during a conversation turn.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The model failed to produce a reply.
    #[error("generation failed: {0}")]
    Generation(String),
    /// Extracting memories from the turn failed.
    #[error(transparent)]
    Extraction(#[from] ExtractError),
    /// A learned edge couldn't be inserted into the graph.
    #[error("memory graph error: {0}")]
    Graph(String),
}
