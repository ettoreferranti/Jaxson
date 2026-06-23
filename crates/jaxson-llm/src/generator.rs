use crate::config::GenerationConfig;
use crate::error::LlmError;

/// A text-generation backend. Implementations stream output token-by-token through
/// `on_token` and return the full generated text.
///
/// Keeping this trait object-safe (no generic methods; `on_token` is a trait object)
/// lets the rest of Jaxson depend on `dyn TextGenerator` and swap the deterministic
/// [`MockGenerator`](crate::backends::MockGenerator) for the real llama.cpp backend.
pub trait TextGenerator {
    /// Generate a completion for `prompt`, invoking `on_token` for each streamed
    /// piece, and returning the concatenated text.
    fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError>;

    /// Convenience: generate without observing the stream.
    fn complete(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        self.generate(prompt, config, &mut |_| {})
    }
}
