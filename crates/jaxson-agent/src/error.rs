use thiserror::Error;

/// Failures during a conversation turn.
///
/// Note: failing to *extract* memories is **not** an error — a model returning
/// imperfect JSON just means nothing was learned that turn (see `Agent::respond`).
/// Only failures that prevent producing a reply surface here.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The model failed to produce a reply.
    #[error("generation failed: {0}")]
    Generation(String),
    /// A learned edge couldn't be inserted into the graph.
    #[error("memory graph error: {0}")]
    Graph(String),
}
