//! Text-generation backends.
//!
//! [`MockGenerator`] and [`ScriptedGenerator`] are always available and deterministic
//! (for tests, demos, and UI development). The real `llama.cpp` backend lives behind
//! the `llama` cargo feature because it pulls in a native build (cmake + C/C++
//! toolchain) and needs a local GGUF model at runtime.

mod mock;
mod scripted;
pub use mock::MockGenerator;
pub use scripted::ScriptedGenerator;

#[cfg(feature = "llama")]
mod llama;
#[cfg(feature = "llama")]
pub use llama::{LlamaConfig, LlamaGenerator};

use crate::config::GenerationConfig;

/// Stream a canned `reply` word-by-word through `on_token`, honoring `max_tokens`, and
/// return the concatenated text. Shared by the deterministic test/demo backends.
pub(crate) fn stream_words(
    reply: &str,
    config: &GenerationConfig,
    on_token: &mut dyn FnMut(&str),
) -> String {
    let limit = config.max_tokens.max(1);
    let mut out = String::new();
    for (i, word) in reply.split_whitespace().take(limit).enumerate() {
        // Re-introduce the spaces that `split_whitespace` removed.
        let piece = if i == 0 {
            word.to_string()
        } else {
            format!(" {word}")
        };
        on_token(&piece);
        out.push_str(&piece);
    }
    out
}
