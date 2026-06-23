use crate::config::GenerationConfig;
use crate::error::LlmError;
use crate::generator::TextGenerator;

/// A deterministic generator that streams a canned reply word-by-word.
///
/// It ignores the prompt entirely, which makes it perfect for tests, demos, and
/// developing the UI/orchestration before the native model backend is wired up.
#[derive(Debug, Clone)]
pub struct MockGenerator {
    reply: String,
}

impl MockGenerator {
    pub fn new(reply: impl Into<String>) -> Self {
        MockGenerator {
            reply: reply.into(),
        }
    }

    /// A friendly, on-character canned reply.
    pub fn friendly() -> Self {
        MockGenerator::new(
            "Hi! I'm Jaxson. I don't know much about you yet, but I'd love to learn.",
        )
    }
}

impl TextGenerator for MockGenerator {
    fn generate(
        &mut self,
        _prompt: &str,
        config: &GenerationConfig,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError> {
        Ok(super::stream_words(&self.reply, config, on_token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            ..Default::default()
        }
    }

    #[test]
    fn returns_the_full_reply_when_unbounded() {
        let mut g = MockGenerator::new("hello there friend");
        let out = g.complete("ignored", &cfg(100)).unwrap();
        assert_eq!(out, "hello there friend");
    }

    #[test]
    fn streamed_pieces_concatenate_to_the_return_value() {
        let mut g = MockGenerator::new("one two three");
        let mut streamed = String::new();
        let returned = g
            .generate("p", &cfg(100), &mut |t| streamed.push_str(t))
            .unwrap();
        assert_eq!(streamed, returned);
        assert_eq!(streamed, "one two three");
    }

    #[test]
    fn streams_one_piece_per_word() {
        let mut g = MockGenerator::new("a b c d");
        let mut count = 0;
        g.generate("p", &cfg(100), &mut |_| count += 1).unwrap();
        assert_eq!(count, 4);
    }

    #[test]
    fn respects_max_tokens() {
        let mut g = MockGenerator::new("one two three four five");
        let out = g.complete("p", &cfg(2)).unwrap();
        assert_eq!(out, "one two");
    }

    #[test]
    fn max_tokens_zero_is_treated_as_one() {
        let mut g = MockGenerator::new("one two three");
        let out = g.complete("p", &cfg(0)).unwrap();
        assert_eq!(out, "one");
    }

    #[test]
    fn ignores_the_prompt() {
        let mut g = MockGenerator::new("fixed");
        assert_eq!(g.complete("anything at all", &cfg(100)).unwrap(), "fixed");
    }
}
