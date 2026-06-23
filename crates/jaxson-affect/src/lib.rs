//! # jaxson-affect
//!
//! The affect engine: it turns Jaxson's relationship state plus the sentiment of the
//! latest exchange into the mood that drives the face — deliberately separate from the
//! LLM, so personality stays consistent regardless of the model's wording (FR-E4).
//!
//! [`analyze`] reads sentiment from text (a deterministic lexicon stand-in), and
//! [`AffectEngine`] maps state + sentiment to a target [`MoodVector`], smoothed over
//! time. Smoothing reuses [`jaxson_core::MoodVector::blended`], so the orchestrator can
//! drive mood through the state machine's `MoodObserved` event.
//!
//! ```
//! use jaxson_affect::{analyze, AffectEngine};
//! use jaxson_core::{MoodVector, RelationshipState};
//!
//! let engine = AffectEngine::default();
//! let sentiment = analyze("I really love this!");
//! let mood = engine.update(MoodVector::NEUTRAL, &RelationshipState::INITIAL, sentiment);
//! assert!(mood.valence() > 0.0);
//! ```

mod engine;
mod sentiment;

pub use engine::AffectEngine;
pub use sentiment::{action_sentiment, analyze, Sentiment};
