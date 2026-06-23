use std::collections::VecDeque;

use crate::config::GenerationConfig;
use crate::error::LlmError;
use crate::generator::TextGenerator;

/// A deterministic generator that returns a queue of canned replies, one per call.
///
/// Useful for testing multi-call flows (e.g. an agent turn that generates a reply and
/// then runs an extraction pass): queue the responses in the order they'll be
/// requested. Once the queue is exhausted it returns an empty string.
#[derive(Debug, Clone)]
pub struct ScriptedGenerator {
    replies: VecDeque<String>,
}

impl ScriptedGenerator {
    pub fn new<I, S>(replies: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        ScriptedGenerator {
            replies: replies.into_iter().map(Into::into).collect(),
        }
    }

    /// Number of replies still queued.
    pub fn remaining(&self) -> usize {
        self.replies.len()
    }
}

impl TextGenerator for ScriptedGenerator {
    fn generate(
        &mut self,
        _prompt: &str,
        config: &GenerationConfig,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError> {
        let reply = self.replies.pop_front().unwrap_or_default();
        Ok(super::stream_words(&reply, config, on_token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GenerationConfig {
        GenerationConfig::default()
    }

    #[test]
    fn returns_queued_replies_in_order() {
        let mut g = ScriptedGenerator::new(["first", "second"]);
        assert_eq!(g.remaining(), 2);
        assert_eq!(g.complete("p", &cfg()).unwrap(), "first");
        assert_eq!(g.complete("p", &cfg()).unwrap(), "second");
        assert_eq!(g.remaining(), 0);
    }

    #[test]
    fn returns_empty_when_exhausted() {
        let mut g = ScriptedGenerator::new(["only"]);
        assert_eq!(g.complete("p", &cfg()).unwrap(), "only");
        assert_eq!(g.complete("p", &cfg()).unwrap(), "");
    }
}
