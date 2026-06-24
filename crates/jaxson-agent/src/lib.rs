//! # jaxson-agent
//!
//! The conversation loop. [`Agent::respond`] ties the pieces together for one turn:
//!
//! 1. retrieve memories relevant to the user's input ([`jaxson_memory::retrieve`]),
//! 2. build the prompt (persona + state-driven hints + memories + history) and
//!    generate a reply ([`jaxson_llm`]),
//! 3. learn from the exchange ([`jaxson_extract`]) — new nodes/edges, embedded and
//!    merged into the graph,
//! 4. advance the relationship state machine ([`jaxson_core::RelationshipState`]).
//!
//! The model and [`Embedder`] are injected per turn, so the same agent runs against
//! the deterministic mock backends or the real model. Persistence is the caller's job
//! (load a graph via [`Agent::with_graph`], persist [`Agent::graph`] through a
//! [`jaxson_memory::MemoryStore`]).
//!
//! ```
//! use jaxson_agent::{Agent, HashEmbedder};
//! use jaxson_llm::backends::ScriptedGenerator;
//!
//! let mut model = ScriptedGenerator::new([
//!     "Hi! Nice to meet you.",
//!     r#"{"memories":[{"kind":"person","content":"User is Ettore","confidence":0.9}],"relations":[]}"#,
//! ]);
//! let embedder = HashEmbedder::default();
//! let mut agent = Agent::new("You are Jaxson, a kind companion.");
//!
//! let turn = agent.respond(&mut model, &embedder, 0, "Hi, I'm Ettore").unwrap();
//! assert_eq!(turn.learned, 1);
//! assert_eq!(agent.graph().node_count(), 1);
//! ```

mod agent;
mod curiosity;
mod embedder;
mod error;

pub use agent::{Agent, AgentConfig, Turn};
pub use embedder::{Embedder, HashEmbedder};
pub use error::AgentError;
