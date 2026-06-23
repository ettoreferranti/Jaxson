//! # jaxson-core
//!
//! The deterministic, UI-free heart of Jaxson: the affect model and the
//! relationship state machine. Memories and interactions are translated into
//! [`RelationshipEvent`]s, and applying an event is a pure, clamped transition over
//! [`RelationshipState`]. Because state is a pure function of the event history,
//! Jaxson's behavior is deterministic and testable — which is exactly why the
//! agent's "algorithms" live here rather than in the LLM.

mod mood;
mod relationship;

pub use mood::{Emotion, MoodVector};
pub use relationship::{RelationshipEvent, RelationshipState};
