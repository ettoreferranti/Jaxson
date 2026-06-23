//! # jaxson-extract
//!
//! Turns conversation into memory: it prompts an LLM to emit structured JSON
//! describing durable facts about the user, then parses that into
//! [`jaxson_memory`] nodes and edges (stamped with provenance and confidence).
//!
//! The prompt building and JSON parsing are pure and fully mutation-tested; the
//! model call goes through [`jaxson_llm::TextGenerator`], so the whole pipeline is
//! testable with the deterministic mock.
//!
//! ```
//! use jaxson_extract::Extractor;
//! use jaxson_llm::{backends::MockGenerator, Message};
//! use jaxson_memory::MemoryId;
//!
//! let mut model = MockGenerator::new(
//!     r#"{"memories":[{"kind":"preference","content":"likes hiking","confidence":0.8}],"relations":[]}"#,
//! );
//! let mut n = 0u128;
//! let (nodes, _edges) = Extractor::default()
//!     .extract_into_graph(&mut model, &[Message::user("I love hiking")], 0, || {
//!         let id = MemoryId::from_u128(n);
//!         n += 1;
//!         id
//!     })
//!     .unwrap();
//! assert_eq!(nodes.len(), 1);
//! ```

mod error;
mod extractor;
mod parse;
mod prompt;

pub use error::ExtractError;
pub use extractor::Extractor;
pub use parse::{parse_extraction, ExtractedMemory, ExtractedRelation, Extraction};
pub use prompt::{extraction_messages, transcript, EXTRACTION_SYSTEM};
