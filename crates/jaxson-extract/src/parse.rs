//! Parse the model's structured output into memory-graph additions.
//!
//! Pure and deterministic: this is where the value lives, so it's heavily tested and
//! mutation-graded. The LLM call itself happens in [`Extractor`](crate::Extractor).

use serde::Deserialize;

use jaxson_memory::{Edge, MemoryId, MemoryKind, MemoryNode, Provenance, Relation};

use crate::error::ExtractError;

/// One memory the model proposed.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedMemory {
    pub kind: MemoryKind,
    pub content: String,
    pub confidence: f32,
}

/// A relation between two extracted memories, referenced by their index in
/// [`Extraction::memories`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtractedRelation {
    pub from: usize,
    pub to: usize,
    pub relation: Relation,
    pub weight: f32,
}

/// The structured result of an extraction pass.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Extraction {
    pub memories: Vec<ExtractedMemory>,
    pub relations: Vec<ExtractedRelation>,
}

impl Extraction {
    /// Whether nothing was extracted.
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty() && self.relations.is_empty()
    }

    /// Convert into concrete graph nodes and edges.
    ///
    /// `created_at` stamps every node, `provenance` records how they were learned,
    /// and `next_id` supplies ids — inject a deterministic counter in tests, or
    /// [`MemoryId::new`](jaxson_memory::MemoryId::new) in production. Returns
    /// [`ExtractError::BadRelationIndex`] if a relation points outside the memory list.
    pub fn into_graph(
        self,
        created_at: i64,
        provenance: Provenance,
        mut next_id: impl FnMut() -> MemoryId,
    ) -> Result<(Vec<MemoryNode>, Vec<Edge>), ExtractError> {
        let ids: Vec<MemoryId> = (0..self.memories.len()).map(|_| next_id()).collect();

        let nodes = self
            .memories
            .into_iter()
            .zip(&ids)
            .map(|(m, &id)| {
                MemoryNode::new(id, m.kind, m.content, created_at, provenance, m.confidence)
            })
            .collect();

        let count = ids.len();
        let resolve = |index: usize| {
            ids.get(index)
                .copied()
                .ok_or(ExtractError::BadRelationIndex { index, count })
        };

        let mut edges = Vec::with_capacity(self.relations.len());
        for r in self.relations {
            edges.push(Edge::new(
                resolve(r.from)?,
                resolve(r.to)?,
                r.relation,
                r.weight,
            ));
        }
        Ok((nodes, edges))
    }
}

// --- Wire format (what the model is asked to emit) ---

fn default_confidence() -> f32 {
    0.6
}

fn default_weight() -> f32 {
    0.5
}

#[derive(Deserialize)]
struct ResponseDto {
    #[serde(default)]
    memories: Vec<MemoryDto>,
    #[serde(default)]
    relations: Vec<RelationDto>,
}

#[derive(Deserialize)]
struct MemoryDto {
    kind: MemoryKind,
    content: String,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

#[derive(Deserialize)]
struct RelationDto {
    from: usize,
    to: usize,
    relation: Relation,
    #[serde(default = "default_weight")]
    weight: f32,
}

/// Strip an optional Markdown code fence (```json … ```) that models often add.
fn strip_code_fences(s: &str) -> &str {
    let t = s.trim();
    let Some(inner) = t.strip_prefix("```").and_then(|x| x.strip_suffix("```")) else {
        return t;
    };
    let inner = inner.trim_start();
    match inner.split_once('\n') {
        // Drop a leading language tag line (``` or ```json).
        Some((first, rest))
            if first.trim().eq_ignore_ascii_case("json") || first.trim().is_empty() =>
        {
            rest.trim()
        }
        _ => inner.trim(),
    }
}

/// The outermost `{ … }` object in `s` (first `{` to last `}`), tolerating prose a model
/// wraps around its JSON (e.g. "Here is the JSON: { … }.").
fn extract_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    let candidate = s.get(start..=end)?;
    // Reject a stray `}` before `{` (which yields an empty slice).
    candidate.starts_with('{').then_some(candidate)
}

/// Parse the model's raw response into an [`Extraction`].
pub fn parse_extraction(raw: &str) -> Result<Extraction, ExtractError> {
    // Drop reasoning + chat-control tokens, then any code fence, then isolate the JSON
    // object (models often wrap it in prose).
    let cleaned = jaxson_llm::clean_output(raw);
    let fenced = strip_code_fences(&cleaned);
    let json = extract_json_object(fenced).unwrap_or(fenced).trim();
    if json.is_empty() {
        return Err(ExtractError::EmptyResponse);
    }
    let dto: ResponseDto =
        serde_json::from_str(json).map_err(|e| ExtractError::Json(e.to_string()))?;

    Ok(Extraction {
        memories: dto
            .memories
            .into_iter()
            .map(|m| ExtractedMemory {
                kind: m.kind,
                content: m.content,
                confidence: m.confidence,
            })
            .collect(),
        relations: dto
            .relations
            .into_iter()
            .map(|r| ExtractedRelation {
                from: r.from,
                to: r.to,
                relation: r.relation,
                weight: r.weight,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counter() -> impl FnMut() -> MemoryId {
        let mut n = 0u128;
        move || {
            let id = MemoryId::from_u128(n);
            n += 1;
            id
        }
    }

    #[test]
    fn parses_memories_and_relations() {
        let raw = r#"{
            "memories": [
                {"kind": "person", "content": "sister Mia", "confidence": 0.9},
                {"kind": "preference", "content": "likes hiking", "confidence": 0.7}
            ],
            "relations": [
                {"from": 0, "to": 1, "relation": "related_to", "weight": 0.4}
            ]
        }"#;
        let ex = parse_extraction(raw).unwrap();
        assert_eq!(ex.memories.len(), 2);
        assert_eq!(ex.memories[0].kind, MemoryKind::Person);
        assert_eq!(ex.memories[1].content, "likes hiking");
        assert_eq!(ex.relations.len(), 1);
        assert_eq!(ex.relations[0].relation, Relation::RelatedTo);
    }

    #[test]
    fn applies_defaults_for_missing_confidence_and_weight() {
        let raw = r#"{"memories":[{"kind":"fact","content":"x"}],"relations":[{"from":0,"to":0,"relation":"related_to"}]}"#;
        let ex = parse_extraction(raw).unwrap();
        assert_eq!(ex.memories[0].confidence, 0.6);
        assert_eq!(ex.relations[0].weight, 0.5);
    }

    #[test]
    fn missing_arrays_default_to_empty() {
        let ex = parse_extraction("{}").unwrap();
        assert!(ex.is_empty());
    }

    #[test]
    fn is_empty_only_when_both_lists_are_empty() {
        assert!(Extraction::default().is_empty());
        let only_memory = Extraction {
            memories: vec![ExtractedMemory {
                kind: MemoryKind::Fact,
                content: "x".to_string(),
                confidence: 0.5,
            }],
            relations: vec![],
        };
        assert!(!only_memory.is_empty());
        // Only a relation present — guards against `&&` becoming `||`.
        let only_relation = Extraction {
            memories: vec![],
            relations: vec![ExtractedRelation {
                from: 0,
                to: 0,
                relation: Relation::Knows,
                weight: 0.5,
            }],
        };
        assert!(!only_relation.is_empty());
    }

    #[test]
    fn strips_json_code_fence() {
        let raw = "```json\n{\"memories\":[{\"kind\":\"fact\",\"content\":\"x\"}]}\n```";
        let ex = parse_extraction(raw).unwrap();
        assert_eq!(ex.memories.len(), 1);
    }

    #[test]
    fn strip_code_fences_drops_the_language_tag() {
        assert_eq!(strip_code_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("{\"a\":1}"), "{\"a\":1}");
    }

    #[test]
    fn extract_json_object_handles_braces_out_of_order() {
        assert_eq!(extract_json_object("a{\"x\":1}b"), Some("{\"x\":1}"));
        assert_eq!(extract_json_object("no braces"), None);
        assert_eq!(extract_json_object("}{"), None); // stray close before open
    }

    #[test]
    fn parses_json_wrapped_in_prose() {
        let raw = "Sure! Here is the JSON:\n{\"memories\":[{\"kind\":\"preference\",\"content\":\"likes hiking\"}]}\nLet me know if you need more.";
        let ex = parse_extraction(raw).unwrap();
        assert_eq!(ex.memories.len(), 1);
        assert_eq!(ex.memories[0].content, "likes hiking");
    }

    #[test]
    fn strips_reasoning_before_parsing() {
        let raw =
            "<think>let me extract</think>{\"memories\":[{\"kind\":\"fact\",\"content\":\"x\"}]}";
        let ex = parse_extraction(raw).unwrap();
        assert_eq!(ex.memories.len(), 1);
    }

    #[test]
    fn strips_bare_code_fence() {
        let raw = "```\n{}\n```";
        assert!(parse_extraction(raw).unwrap().is_empty());
    }

    #[test]
    fn empty_response_is_an_error() {
        assert_eq!(parse_extraction("   "), Err(ExtractError::EmptyResponse));
        assert_eq!(
            parse_extraction("```\n\n```"),
            Err(ExtractError::EmptyResponse)
        );
    }

    #[test]
    fn invalid_json_is_an_error() {
        assert!(matches!(
            parse_extraction("not json"),
            Err(ExtractError::Json(_))
        ));
    }

    #[test]
    fn unknown_kind_is_a_json_error() {
        let raw = r#"{"memories":[{"kind":"banana","content":"x"}]}"#;
        assert!(matches!(parse_extraction(raw), Err(ExtractError::Json(_))));
    }

    #[test]
    fn into_graph_assigns_ids_and_resolves_relation_indices() {
        let ex = parse_extraction(
            r#"{"memories":[{"kind":"person","content":"a","confidence":0.9},
                            {"kind":"preference","content":"b","confidence":0.5}],
                "relations":[{"from":0,"to":1,"relation":"likes","weight":0.8}]}"#,
        )
        .unwrap();
        let (nodes, edges) = ex
            .into_graph(42, Provenance::InferredFromConversation, counter())
            .unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].id, MemoryId::from_u128(0));
        assert_eq!(nodes[0].created_at, 42);
        assert_eq!(nodes[0].provenance, Provenance::InferredFromConversation);
        assert_eq!(nodes[1].id, MemoryId::from_u128(1));
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, MemoryId::from_u128(0));
        assert_eq!(edges[0].to, MemoryId::from_u128(1));
        assert_eq!(edges[0].relation, Relation::Likes);
    }

    #[test]
    fn into_graph_rejects_out_of_range_relation_index() {
        let ex = Extraction {
            memories: vec![ExtractedMemory {
                kind: MemoryKind::Fact,
                content: "only one".to_string(),
                confidence: 0.5,
            }],
            relations: vec![ExtractedRelation {
                from: 0,
                to: 5,
                relation: Relation::Knows,
                weight: 0.5,
            }],
        };
        assert_eq!(
            ex.into_graph(0, Provenance::StatedByUser, counter()),
            Err(ExtractError::BadRelationIndex { index: 5, count: 1 })
        );
    }
}
