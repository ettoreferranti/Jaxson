use jaxson_llm::{ChatTemplate, GenerationConfig, Message, TextGenerator};
use jaxson_memory::{Edge, MemoryId, MemoryNode, Provenance};

use crate::error::ExtractError;
use crate::parse::{parse_extraction, Extraction};
use crate::prompt::extraction_messages;

/// Extracts memories from conversation by prompting a [`TextGenerator`] and parsing
/// its JSON. Generic over the backend via `dyn TextGenerator`, so it works with the
/// real model or the deterministic mock.
#[derive(Debug, Clone)]
pub struct Extractor {
    /// Chat template matching the loaded model.
    pub template: ChatTemplate,
    /// Provenance stamped on extracted memories.
    pub provenance: Provenance,
    /// Decoding config. Extraction wants determinism, so temperature defaults to 0.
    pub config: GenerationConfig,
}

impl Default for Extractor {
    fn default() -> Self {
        Extractor {
            template: ChatTemplate::ChatMl,
            provenance: Provenance::InferredFromConversation,
            config: GenerationConfig {
                temperature: 0.0,
                ..GenerationConfig::default()
            },
        }
    }
}

impl Extractor {
    /// Run an extraction pass over the recent conversation turns.
    pub fn extract(
        &self,
        generator: &mut dyn TextGenerator,
        recent: &[Message],
    ) -> Result<Extraction, ExtractError> {
        let prompt = self.template.render(&extraction_messages(recent));
        let raw = generator
            .complete(&prompt, &self.config)
            .map_err(|e| ExtractError::Generation(e.to_string()))?;
        parse_extraction(&raw)
    }

    /// Extract and convert straight into graph nodes and edges. `next_id` supplies
    /// ids (a counter in tests, [`MemoryId::new`] in production).
    pub fn extract_into_graph(
        &self,
        generator: &mut dyn TextGenerator,
        recent: &[Message],
        created_at: i64,
        next_id: impl FnMut() -> MemoryId,
    ) -> Result<(Vec<MemoryNode>, Vec<Edge>), ExtractError> {
        self.extract(generator, recent)?
            .into_graph(created_at, self.provenance, next_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jaxson_llm::backends::MockGenerator;
    use jaxson_memory::MemoryKind;

    fn counter() -> impl FnMut() -> MemoryId {
        let mut n = 0u128;
        move || {
            let id = MemoryId::from_u128(n);
            n += 1;
            id
        }
    }

    #[test]
    fn extracts_from_a_mocked_model_response() {
        let mut gen = MockGenerator::new(
            r#"{"memories":[{"kind":"preference","content":"likes hiking","confidence":0.8}],"relations":[]}"#,
        );
        let ex = Extractor::default()
            .extract(&mut gen, &[Message::user("I love hiking")])
            .unwrap();
        assert_eq!(ex.memories.len(), 1);
        assert_eq!(ex.memories[0].kind, MemoryKind::Preference);
        assert_eq!(ex.memories[0].content, "likes hiking");
    }

    #[test]
    fn extract_into_graph_produces_nodes_with_inferred_provenance() {
        let mut gen = MockGenerator::new(
            r#"{"memories":[{"kind":"person","content":"friend Sam","confidence":0.9}],"relations":[]}"#,
        );
        let (nodes, edges) = Extractor::default()
            .extract_into_graph(&mut gen, &[Message::user("my friend Sam")], 7, counter())
            .unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].provenance, Provenance::InferredFromConversation);
        assert_eq!(nodes[0].created_at, 7);
        assert!(edges.is_empty());
    }

    #[test]
    fn surfaces_parse_errors_from_the_model() {
        let mut gen = MockGenerator::new("sorry, I can't do that");
        let result = Extractor::default().extract(&mut gen, &[Message::user("hi")]);
        assert!(matches!(result, Err(ExtractError::Json(_))));
    }

    #[test]
    fn default_extractor_uses_deterministic_temperature() {
        assert_eq!(Extractor::default().config.temperature, 0.0);
    }
}
