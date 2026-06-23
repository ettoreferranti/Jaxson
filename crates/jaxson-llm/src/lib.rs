//! # jaxson-llm
//!
//! Local LLM orchestration for Jaxson. This crate owns everything *around* token
//! generation — chat messages, prompt/chat-template assembly, and decoding config —
//! plus a pluggable [`TextGenerator`] backend.
//!
//! The orchestration layer is pure and deterministic so it is fully unit- and
//! mutation-tested. The actual model runs through a backend:
//!
//! - [`backends::MockGenerator`] — deterministic, always available, used by tests,
//!   demos, and UI development.
//! - `backends::LlamaGenerator` — `llama.cpp` with Metal offload, behind the `llama`
//!   cargo feature (needs cmake at build time and a GGUF model at runtime).
//!
//! ```
//! use jaxson_llm::{assemble, ChatTemplate, GenerationConfig, Message, TextGenerator};
//! use jaxson_llm::backends::MockGenerator;
//!
//! let history = [Message::user("Hello!")];
//! let messages = assemble("You are Jaxson, a kind companion.", &[], &history);
//! let prompt = ChatTemplate::ChatMl.render(&messages);
//!
//! let mut model = MockGenerator::friendly();
//! let reply = model.complete(&prompt, &GenerationConfig::default()).unwrap();
//! assert!(!reply.is_empty());
//! ```

pub mod backends;
pub mod ollama;

mod config;
mod error;
mod generator;
mod message;
mod prompt;

pub use config::GenerationConfig;
pub use error::LlmError;
pub use generator::TextGenerator;
pub use message::{Message, Role};
pub use prompt::{assemble, build_system_message, ChatTemplate};
