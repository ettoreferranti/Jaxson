//! Text-generation backends.
//!
//! [`MockGenerator`] is always available and deterministic. The real `llama.cpp`
//! backend lives behind the `llama` cargo feature because it pulls in a native build
//! (cmake + C/C++ toolchain) and needs a local GGUF model at runtime.

mod mock;
pub use mock::MockGenerator;

#[cfg(feature = "llama")]
mod llama;
#[cfg(feature = "llama")]
pub use llama::{LlamaConfig, LlamaGenerator};
